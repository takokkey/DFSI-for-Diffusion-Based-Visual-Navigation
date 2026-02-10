#!/usr/bin/env python3
"""
map_adapter_rclpy_esdf.py

CPU-friendly Local 2D TSDF/ESDF adapter for VisualNav (方案 C).

Inputs:
  - LaserScan (/scan)  [default]
  - PointCloud2 (optional)

Core pipeline:
  1) Accumulate N frames of points in target_frame (default: base_link)
  2) Build local 2D TSDF (TsdfCostMap)
  3) TSDF -> Occupancy (conservative thresholds)
  4) Occupancy -> 2D ESDF (explicit EDT if OpenCV available; safe fallback otherwise)
  5) ESDF -> costmap (0~100 int8) publish as nav_msgs/OccupancyGrid on /tsdf_costmap

Adaptive update (Algorithmic contribution):
  - Motion-aware staleness rho_motion (world_frame->target_frame TF)
  - Residual-aware geometric consistency R_mean/R95 (scan vs previous occupancy)
  - Dual-criteria FSM: S0 Normal / S1 Soft / S2 Hard

Debug/Interface topics:
  - /tsdf_residual_score  (Float32): default publishes R95
  - /map_fsm_state        (String): "S0"/"S1"/"S2"
"""

import math
import time
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, MapMetaData
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header, Float32, String

# Optional OpenCV for fast EDT
try:
    import cv2
except Exception:
    cv2 = None

# Optional Open3D for voxel downsample
try:
    import open3d as o3d
except Exception:
    o3d = None

# PointCloud2 helper
try:
    import sensor_msgs_py.point_cloud2 as pc2_read
except Exception:
    try:
        from sensor_msgs import point_cloud2 as pc2_read
    except Exception:
        pc2_read = None

# ROS2 tf2
try:
    import tf2_ros
except Exception:
    tf2_ros = None

# Your TSDF implementation & config
from tsdf_cost_map import TsdfCostMap
from costmap_cfg import CostMapConfig


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def quat_to_rot_matrix(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        return np.eye(3)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def transform_points(pts: np.ndarray, transform):
    """Apply geometry_msgs/TransformStamped to Nx3 points."""
    t = transform.transform.translation
    q = transform.transform.rotation

    R = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
    t_vec = np.array([t.x, t.y, t.z], dtype=float)
    return (R @ pts.T).T + t_vec


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
class LidarTsdfEsdfAdapter(Node):
    def __init__(self):
        super().__init__("lidar_tsdf_esdf_adapter")

        # ---------------- Parameters ----------------
        self.declare_parameter("input_mode", "scan")  # 'scan' or 'pointcloud'
        self.declare_parameter("input_topic_scan", "/scan")
        self.declare_parameter("input_topic_pc", "/points")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("world_frame", "odom")

        self.declare_parameter("accum_frames", 6)
        self.declare_parameter("max_range", 10.0)
        self.declare_parameter("min_range", 0.01)
        self.declare_parameter("publish_rate", 1.0)  # Hz

        # Downsample
        self.declare_parameter("voxel_downsample_factor", 0.8)

        # TSDF->Occ thresholds
        self.declare_parameter("d_occ", 0.080)
        self.declare_parameter("d_free", 0.160)
        self.declare_parameter("treat_unknown_as_obstacle", True)

        # TSDF value convention (important!)
        # - auto: detect signed TSDF vs normalized 0~1 vs unsigned distance (meters)
        # - signed: TSDF is signed meters (negative inside obstacle)
        # - norm01: TSDF is normalized to [0,1] where small=obstacle, large=free
        # - unsigned_meter: TSDF is unsigned distance in meters (small=obstacle, large=free)
        self.declare_parameter("tsdf_mode", "auto")
        self.declare_parameter("tsdf_mode_strict", False)  # never override tsdf_mode even if range mismatches
        self.declare_parameter("occ_inflate_cells", 2)     # inflate occupied seeds (grid cells)
        self.declare_parameter("occ_min_count_warn", 80)   # warn if occupied seeds too few
        self.declare_parameter("tsdf_neg_eps", 1e-6)  # decide whether TSDF has negative values
        self.declare_parameter("tsdf_obstacle_threshold01", 0.01)  # only for norm01
        self.declare_parameter("tsdf_free_space_threshold01", 0.8)  # only for norm01
        self.declare_parameter("tsdf_stat_log_sec", 2.0)  # log TSDF stats every N seconds (0 disables)

        # ESDF->cost parameters
        self.declare_parameter("d_max_safe", 1.00)  # meters
        self.declare_parameter("unknown_cost", 100)

        # Residual parameters
        self.declare_parameter("residual_stride", 3)
        self.declare_parameter("residual_eps_small", 1.3)
        self.declare_parameter("residual_eps_bad", 2.0)
        self.declare_parameter("residual_max_beams", 360)

        # FSM parameters
        self.declare_parameter("tau_low", 0.2)
        self.declare_parameter("tau_high", 0.6)
        self.declare_parameter("lambda_theta", 0.5)

        # Debug/interface
        self.declare_parameter("publish_debug_topics", True)

        # Read params
        self.input_mode = (
            self.get_parameter("input_mode").get_parameter_value().string_value.lower()
        )
        self.scan_topic = self.get_parameter("input_topic_scan").value
        self.pc_topic = self.get_parameter("input_topic_pc").value
        self.target_frame = self.get_parameter("target_frame").value
        self.world_frame = self.get_parameter("world_frame").value

        self.accum_frames = int(self.get_parameter("accum_frames").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.voxel_downsample_factor = float(
            self.get_parameter("voxel_downsample_factor").value
        )

        self.d_occ = float(self.get_parameter("d_occ").value)
        self.d_free = float(self.get_parameter("d_free").value)
        self.treat_unknown_as_obstacle = bool(
            self.get_parameter("treat_unknown_as_obstacle").value
        )

        # TSDF convention controls (see _tsdf_to_occ_state)
        self.tsdf_mode = str(self.get_parameter("tsdf_mode").value).strip().lower()
        self.tsdf_mode_strict = bool(self.get_parameter("tsdf_mode_strict").value)
        self.occ_inflate_cells = int(self.get_parameter("occ_inflate_cells").value)
        self.occ_min_count_warn = int(self.get_parameter("occ_min_count_warn").value)
        self.tsdf_neg_eps = float(self.get_parameter("tsdf_neg_eps").value)
        self.tsdf_obstacle_threshold01 = float(
            self.get_parameter("tsdf_obstacle_threshold01").value
        )
        self.tsdf_free_space_threshold01 = float(
            self.get_parameter("tsdf_free_space_threshold01").value
        )
        self.tsdf_stat_log_sec = float(self.get_parameter("tsdf_stat_log_sec").value)

        self._tsdf_last_stat_log_wall = 0.0
        self._tsdf_last_mode = None

        self.d_max_safe = float(self.get_parameter("d_max_safe").value)
        self.unknown_cost = int(self.get_parameter("unknown_cost").value)

        self.residual_stride = int(self.get_parameter("residual_stride").value)
        self.residual_eps_small = float(self.get_parameter("residual_eps_small").value)
        self.residual_eps_bad = float(self.get_parameter("residual_eps_bad").value)
        self.residual_max_beams = int(self.get_parameter("residual_max_beams").value)

        self._tau_low = float(self.get_parameter("tau_low").value)
        self._tau_high = float(self.get_parameter("tau_high").value)
        self._lambda_theta = float(self.get_parameter("lambda_theta").value)

        self.publish_debug_topics = bool(
            self.get_parameter("publish_debug_topics").value
        )

        # ---------------- TSDF init ----------------
        self.cfg = CostMapConfig()
        self.tsdf = TsdfCostMap(self.cfg.general, self.cfg.tsdf_cost_map)
        self.resolution = float(self.cfg.general.resolution)

        # motion staleness
        self._last_pose_world: Optional[Tuple[float, float, float]] = None
        self._accum_travel = 0.0
        self._accum_yaw = 0.0
        self._rho_motion = 0.0
        self._L_ref = float(self.cfg.general.x_max - self.cfg.general.x_min)

        # residual stats
        self._R_mean = 0.0
        self._R95 = 0.0
        # Residual validity & map-mode autodetect (for robust raycasting)
        self._residual_valid = False
        self._residual_hits = 0
        self._residual_total = 0
        # mode = (use_yx, swap_origin_xy, flip_x, flip_y)
        self._residual_mode = None
        self._residual_mode_last_update_s = 0.0

        # FSM
        self._state = "S0"

        # caches
        self.accum_points: List[np.ndarray] = []      # obstacle/end points
        self.accum_free_points: List[np.ndarray] = [] # free-space ray samples
        self._last_pub_time = 0.0

        # previous map for residual prediction
        self._prev_occ_state: Optional[np.ndarray] = None  # -1 unknown, 0 free, 1 occ
        self._prev_origin_xy: Tuple[float, float] = (0.0, 0.0)

        # ---------------- TF ----------------
        if tf2_ros is not None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        else:
            self.tf_buffer = None
            self.get_logger().warn("tf2_ros not available; TF lookup disabled.")

        # ---------------- Subscriptions ----------------
        if self.input_mode == "scan":
            self.get_logger().info(f"Using LaserScan mode, subscribing to {self.scan_topic}")
            self.scan_sub = self.create_subscription(
                LaserScan, self.scan_topic, self.scan_callback, 10
            )
        elif self.input_mode == "pointcloud":
            self.get_logger().info(f"Using PointCloud2 mode, subscribing to {self.pc_topic}")
            self.pc_sub = self.create_subscription(
                PointCloud2, self.pc_topic, self.pc_callback, 10
            )
        else:
            raise ValueError("input_mode must be 'scan' or 'pointcloud'")

        # ---------------- Publishers ----------------
        self.map_pub = self.create_publisher(OccupancyGrid, "/tsdf_costmap", 1)
        # Publish ESDF normalization distance (d_max_safe) for downstream safety filter
        self.dmax_pub = self.create_publisher(Float32, "/esdf_d_max_safe", 1)


        if self.publish_debug_topics:
            self.residual_pub = self.create_publisher(Float32, "/tsdf_residual_score", 1)
            self.state_pub = self.create_publisher(String, "/map_fsm_state", 1)
        else:
            self.residual_pub = None
            self.state_pub = None

        # ---------------- Logs ----------------
        self.get_logger().info(
            f"Local window: res={self.resolution:.3f}, "
            f"x[{self.cfg.general.x_min},{self.cfg.general.x_max}], "
            f"y[{self.cfg.general.y_min},{self.cfg.general.y_max}]"
        )
        self.get_logger().info(
            f"Occ thresholds: d_occ={self.d_occ:.3f}, d_free={self.d_free:.3f}, "
            f"unknown_as_obs={self.treat_unknown_as_obstacle}"
        )
        self.get_logger().info(
            f"Residual: stride={self.residual_stride}, "
            f"eps_small={self.residual_eps_small:.3f}, eps_bad={self.residual_eps_bad:.3f}"
        )
        if cv2 is None:
            self.get_logger().warn("OpenCV not found; ESDF uses a safe fallback approximation.")
        if o3d is None:
            self.get_logger().warn("Open3D not found; point downsample uses a simple fallback.")

    # ------------------------------------------------------------------
    # Debug publish
    # ------------------------------------------------------------------
    def _publish_debug(self):
        if not self.publish_debug_topics:
            return

        if self.residual_pub is not None:
            msg = Float32()
            msg.data = float(self._R95)
            self.residual_pub.publish(msg)

        if self.state_pub is not None:
            s = String()
            s.data = str(self._state)
            self.state_pub.publish(s)

        # Keep /esdf_d_max_safe alive even if /tsdf_costmap is skipped
        try:
            dm = Float32()
            dm.data = float(self.d_max_safe)
            self.dmax_pub.publish(dm)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        # 1) residual update (scan vs previous occupancy)
        self._update_residual_from_scan(msg)

        # 2) convert scan to points for TSDF accumulation
        ranges = np.array(msg.ranges, dtype=float)
        finite = np.isfinite(ranges)
        valid = finite & (ranges > self.min_range) & (ranges < self.max_range)
        if int(np.sum(valid)) == 0:
            self._publish_debug()
            return

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        zs = np.zeros_like(xs)
        pts = np.stack([xs, ys, zs], axis=1)[valid]

        # Also sample free-space points along each valid ray (cheap ray-marching)
        free_margin = max(self.resolution * 1.0, 0.05)  # meters
        free_step = max(self.resolution * 1.5, 0.08)    # meters
        max_samples_per_ray = 8
        free_pts_list = []
        rv = ranges[valid]
        av = angles[valid]
        for r_i, a_i in zip(rv, av):
            r_free = float(r_i) - free_margin
            if r_free <= self.min_range + 1e-3:
                continue
            # choose a small number of samples; cap for CPU
            n_s = int(min(max_samples_per_ray, max(2, math.ceil((r_free - self.min_range) / free_step) + 1)))
            ds = np.linspace(self.min_range, r_free, n_s, dtype=float)
            fx = ds * math.cos(float(a_i))
            fy = ds * math.sin(float(a_i))
            fz = np.zeros_like(ds)
            free_pts_list.append(np.stack([fx, fy, fz], axis=1))
        free_pts = (
            np.vstack(free_pts_list) if free_pts_list else np.zeros((0, 3), dtype=float)
        )

        # TF to target_frame if needed
        if self.tf_buffer is not None:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                pts = transform_points(pts, t)
                if free_pts.shape[0] > 0:
                    free_pts = transform_points(free_pts, t)
            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed ({msg.header.frame_id}->{self.target_frame}): {e}"
                )

        self.accum_points.append(pts)
        if free_pts.shape[0] > 0:
            self.accum_free_points.append(free_pts)

        if len(self.accum_points) >= self.accum_frames:
            self._process_accumulated_points()
        else:
            self._publish_debug()

    def pc_callback(self, msg: PointCloud2):
        if pc2_read is None:
            self.get_logger().error("PointCloud2 reader not available.")
            return

        points = []
        for p in pc2_read.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
            if len(points) >= 200000:
                break
        if not points:
            self._publish_debug()
            return

        pts = np.array(points, dtype=float)
        free_pts = np.empty((0, 3), dtype=float)  # PointCloud2: no free-space rays

        if self.tf_buffer is not None:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                pts = transform_points(pts, t)
                if free_pts.shape[0] > 0:
                    free_pts = transform_points(free_pts, t)
            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed ({msg.header.frame_id}->{self.target_frame}): {e}"
                )

        self.accum_points.append(pts)
        if free_pts.shape[0] > 0:
            self.accum_free_points.append(free_pts)

        if len(self.accum_points) >= self.accum_frames:
            self._process_accumulated_points()
        else:
            self._publish_debug()

    # ------------------------------------------------------------------
    # Motion staleness
    # ------------------------------------------------------------------
    def _update_motion_staleness(self):
        if self.tf_buffer is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.target_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup ({self.world_frame}->{self.target_frame}) failed: {e}"
            )
            return

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation

        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if self._last_pose_world is not None:
            lx, ly, lyaw = self._last_pose_world
            dtrans = math.hypot(tx - lx, ty - ly)
            dyaw = (yaw - lyaw + math.pi) % (2.0 * math.pi) - math.pi
            self._accum_travel += dtrans
            self._accum_yaw += abs(dyaw)

        self._last_pose_world = (tx, ty, yaw)

        rho_trans = 0.0 if self._L_ref <= 1e-6 else (self._accum_travel / self._L_ref)
        rho_rot = min(self._accum_yaw / (2.0 * math.pi), 1.0)
        self._rho_motion = float(min(1.0, rho_trans + self._lambda_theta * rho_rot))

    # ------------------------------------------------------------------
    # Residual (scan vs previous occupancy)
    # ------------------------------------------------------------------
    def _quat_to_yaw(self, q) -> float:
        """Yaw (rotation around Z) from geometry_msgs.msg.Quaternion-like."""
        x = float(getattr(q, 'x', 0.0))
        y = float(getattr(q, 'y', 0.0))
        z = float(getattr(q, 'z', 0.0))
        w = float(getattr(q, 'w', 1.0))
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _lookup_pose2d(self, target_frame: str, source_frame: str, stamp) -> tuple:
        """Return (x, y, yaw, ok) for TF target<-source at given stamp."""
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
            t = tf.transform.translation
            r = tf.transform.rotation
            return float(t.x), float(t.y), self._quat_to_yaw(r), True
        except Exception:
            return 0.0, 0.0, 0.0, False

    def _raycast_pred(
        self,
        occ: np.ndarray,
        origin_xy: tuple,
        res: float,
        x0: float,
        y0: float,
        theta: float,
        r_end: float,
        *,
        use_yx: bool = False,
        swap_origin_xy: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        hit_unknown: bool = False,
        min_hit_dist: float = 0.0,
    ):
        """Raycast in occupancy to find predicted hit distance. Return None if no hit / out of bounds."""
        ox, oy = origin_xy
        if swap_origin_xy:
            ox, oy = oy, ox

        # Coordinate flips (in target_frame) to handle map conventions
        def _flip_coords(x: float, y: float):
            return (-x if flip_x else x), (-y if flip_y else y)

        # Determine bounds & indexing convention
        if use_yx:
            ny, nx = occ.shape
            def get_state(ix: int, iy: int):
                return occ[iy, ix]
        else:
            nx, ny = occ.shape
            def get_state(ix: int, iy: int):
                return occ[ix, iy]

        step = float(res)
        r0 = max(float(self.min_range), 2.0 * step)
        r = r0
        while r <= r_end + 1e-6:
            x = x0 + r * math.cos(theta)
            y = y0 + r * math.sin(theta)
            x, y = _flip_coords(x, y)
            ix = int(math.floor((x - ox) / step))
            iy = int(math.floor((y - oy) / step))
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
                return None
            s = int(get_state(ix, iy))
            if s == 1 or (hit_unknown and s == -1):
                if r < min_hit_dist:
                    return None
                return r
            r += step
        return None

    def _update_residual_from_scan(self, msg: LaserScan):
        """
        2D geometric consistency residual (robust):
          r = z_meas - z_pred
        z_pred is obtained by raycasting the *previous* occupancy map in target_frame.
        We auto-detect the map indexing/origin conventions (xy/yx, origin swap, axis flips)
        to avoid 'always huge residual' when conventions mismatch.
        """
        if self._prev_occ_state is None or self._prev_origin_xy is None:
            self._R_mean = 0.0
            self._R95 = 0.0
            self._residual_valid = False
            self._residual_hits = 0
            self._residual_total = 0
            return

        ranges = np.array(msg.ranges, dtype=float)
        finite = np.isfinite(ranges)
        valid = finite & (ranges > self.min_range) & (ranges < self.max_range)
        if int(np.sum(valid)) == 0:
            self._R_mean = 0.0
            self._R95 = 0.0
            self._residual_valid = False
            self._residual_hits = 0
            self._residual_total = 0
            return

        idx_all = np.where(valid)[0]
        if self.residual_stride > 1:
            idx_all = idx_all[:: self.residual_stride]
        if idx_all.size > self.residual_max_beams:
            idx_all = idx_all[: self.residual_max_beams]

        angles = msg.angle_min + idx_all * msg.angle_increment
        z_meas = ranges[idx_all]

        # TF: target_frame <- scan_frame
        scan_frame = msg.header.frame_id if msg.header.frame_id else self.target_frame
        try:
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        except Exception:
            stamp = self.get_clock().now()

        x0, y0, yaw0, ok_pose = self._lookup_pose2d(self.target_frame, scan_frame, stamp)

        occ = self._prev_occ_state
        origin_xy = self._prev_origin_xy
        res = float(self.resolution)

        # Residual raycast settings
        min_hit_dist = max(float(self.min_range), 2.0 * res)
        hit_unknown = False  # for residual, unknown should NOT be treated as obstacle
        inflate_corr = float(getattr(self, 'occ_inflate_cells', 0)) * res

        # --- auto-detect mapping mode (cheap probe) ---
        now_s = self.get_clock().now().nanoseconds * 1e-9
        need_reselect = (self._residual_mode is None) or ((now_s - self._residual_mode_last_update_s) > 3.0) or (self._residual_hits < 10)
        if need_reselect:
            # Probe ~80 beams
            probe_n = int(min(80, max(10, idx_all.size)))
            step_i = max(1, idx_all.size // probe_n)
            probe_angles = angles[::step_i]
            probe_meas = z_meas[::step_i]

            best = None  # (hits, med, mode_tuple)
            modes = []
            for use_yx in (False, True):
                for swap_origin_xy in (False, True):
                    for flip_x in (False, True):
                        for flip_y in (False, True):
                            modes.append((use_yx, swap_origin_xy, flip_x, flip_y))

            for mode in modes:
                use_yx, swap_origin_xy, flip_x, flip_y = mode
                abs_res = []
                hits = 0
                for a, zm in zip(probe_angles, probe_meas):
                    theta = yaw0 + float(a)
                    r_end = min(float(self.max_range), float(zm) + 2.0 * res)
                    z_pred = self._raycast_pred(
                        occ, origin_xy, res, x0, y0, theta, r_end,
                        use_yx=use_yx, swap_origin_xy=swap_origin_xy, flip_x=flip_x, flip_y=flip_y,
                        hit_unknown=hit_unknown, min_hit_dist=min_hit_dist,
                    )
                    if z_pred is None:
                        continue
                    z_pred = min(float(self.max_range), float(z_pred) + inflate_corr)
                    abs_res.append(abs(float(zm) - z_pred))
                    hits += 1
                if hits == 0:
                    med = 1e9
                else:
                    med = float(np.median(np.array(abs_res, dtype=float)))
                cand = (hits, med, mode)
                if best is None:
                    best = cand
                else:
                    # Prefer more hits; tie-break by smaller median residual
                    if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                        best = cand

            if best is not None:
                self._residual_mode = best[2]
                self._residual_mode_last_update_s = now_s

        # Use selected mode (fallback to default)
        if self._residual_mode is None:
            self._residual_mode = (False, False, False, False)

        use_yx, swap_origin_xy, flip_x, flip_y = self._residual_mode

        residuals_abs = []
        hits = 0
        total = int(len(angles))

        for a, zm in zip(angles, z_meas):
            theta = yaw0 + float(a)
            r_end = min(float(self.max_range), float(zm) + 2.0 * res)
            z_pred = self._raycast_pred(
                occ, origin_xy, res, x0, y0, theta, r_end,
                use_yx=use_yx, swap_origin_xy=swap_origin_xy, flip_x=flip_x, flip_y=flip_y,
                hit_unknown=hit_unknown, min_hit_dist=min_hit_dist,
            )
            if z_pred is None:
                continue
            z_pred = min(float(self.max_range), float(z_pred) + inflate_corr)
            residuals_abs.append(abs(float(zm) - z_pred))
            hits += 1

        self._residual_hits = int(hits)
        self._residual_total = int(total)

        # Require enough hits to consider the residual meaningful
        min_hits = 20
        if hits < min_hits:
            self._R_mean = 0.0
            self._R95 = 0.0
            self._residual_valid = False
        else:
            arr = np.array(residuals_abs, dtype=float)
            self._R_mean = float(np.mean(arr))
            self._R95 = float(np.percentile(arr, 95))
            self._residual_valid = True

        # Debug (throttled by mode update interval, but still useful when things go wrong)
        self.get_logger().info(
            f"[residual] hits={hits}/{total}, mode(use_yx={use_yx}, swap_origin={swap_origin_xy}, flip_x={flip_x}, flip_y={flip_y}), "
            f"Rmean={self._R_mean:.3f}, R95={self._R95:.3f}, scan_frame={scan_frame}, target={self.target_frame}, pose_ok={ok_pose}"
        )
    def _decide_state(self) -> str:
        # Decide map quality state S0/S1/S2 based on motion staleness (rho) and scan-map residual (R95).
        # S0: OK, S1: suspicious (slow-update), S2: bad/stale (hard-reset / stop).
        rho = float(getattr(self, "_rho_motion", 0.0))
        r95 = float(getattr(self, "_R95", 0.0))
        rmean = float(getattr(self, "_R_mean", 0.0))
        residual_valid = bool(getattr(self, "_residual_valid", False))

        eps_small = float(getattr(self, "residual_eps_small", 0.05))
        eps_bad = float(getattr(self, "residual_eps_bad", 0.30))

        # If residual is unavailable (e.g., no prev map), fall back to motion only.
        if not residual_valid:
            if rho < self._tau_low:
                return "S0"
            elif rho < self._tau_high:
                return "S1"
            else:
                return "S2"

        # Coupled policy:
        # - Residual alone should NOT immediately force S2 when the robot is stationary/fresh (rho low),
        #   otherwise you get 'always S2' even at standstill due to minor map imperfections.
        # - Escalate to S2 when motion is stale OR residual is extremely large.


        extreme_bad = (r95 >= max(2.0 * eps_bad, eps_bad + 0.7, 3.0))

        if r95 >= eps_bad:
            if extreme_bad or rho >= self._tau_high:
                return "S2"
            else:
                return "S1"

        if r95 >= eps_small:
            if rho >= self._tau_high:
                return "S2"
            else:
                return "S1"

        # Residual looks fine -> use motion only.
        if rho < self._tau_low:
            return "S0"
        elif rho < self._tau_high:
            return "S1"
        else:
            return "S2"
    def _tsdf_to_occ_state(self, tsdf: np.ndarray) -> np.ndarray:
        """Convert TSDF scalar field to occupancy state grid.

        Output encoding:
          -1 unknown, 0 free, 1 occupied

        Supports:
          - signed TSDF (inside negative)
          - normalized 0~1 TSDF-like confidence
          - unsigned distance-to-surface in meters (your current logs look like this: TSDF in [~0, 5+])
        """
        tsdf = np.asarray(tsdf)
        occ_state = np.full(tsdf.shape, -1, dtype=np.int8)

        finite = np.isfinite(tsdf)
        if not np.any(finite):
            return occ_state

        vmin = float(np.min(tsdf[finite]))
        vmax = float(np.max(tsdf[finite]))
        has_neg = bool(np.any(tsdf[finite] < -float(self.tsdf_neg_eps)))

        looks_01 = (vmin >= -1e-3) and (vmax <= 1.5)

        mode_req = str(getattr(self, "tsdf_mode", "auto")).strip().lower() or "auto"
        mode_used = mode_req

        if mode_used in ("auto", "detect", "adaptive"):
            if has_neg:
                mode_used = "signed"
            elif looks_01:
                mode_used = "norm01"
            else:
                mode_used = "unsigned_meter"

        if not bool(getattr(self, "tsdf_mode_strict", False)):
            if mode_used == "norm01" and (not looks_01):
                mode_used = "signed" if has_neg else "unsigned_meter"
            if mode_used == "signed" and (not has_neg):
                mode_used = "norm01" if looks_01 else "unsigned_meter"

        if mode_used == "signed":
            occ = finite & (tsdf <= -abs(self.d_occ))
            fre = finite & (tsdf >=  abs(self.d_free))
        elif mode_used == "norm01":
            occ_th = float(self.tsdf_obstacle_threshold01)
            fre_th = float(self.tsdf_free_space_threshold01)
            occ = finite & (tsdf <= occ_th)
            fre = finite & (tsdf >= fre_th)
        else:  # unsigned_meter
            occ = finite & (tsdf <= abs(self.d_occ))
            fre = finite & (tsdf >= abs(self.d_free))

        occ_state[fre] = 0
        occ_state[occ] = 1

        inflate = int(getattr(self, "occ_inflate_cells", 0))
        if inflate > 0:
            occ_mask = (occ_state == 1)
            for _ in range(inflate):
                m0 = occ_mask
                m1 = m0.copy()
                m1[1:, :]   |= m0[:-1, :]
                m1[:-1, :]  |= m0[1:, :]
                m1[:, 1:]   |= m0[:, :-1]
                m1[:, :-1]  |= m0[:, 1:]
                m1[1:, 1:]  |= m0[:-1, :-1]
                m1[1:, :-1] |= m0[:-1, 1:]
                m1[:-1, 1:] |= m0[1:, :-1]
                m1[:-1, :-1]|= m0[1:, 1:]
                occ_mask = m1
            occ_state[occ_mask] = 1

        n_finite = int(np.count_nonzero(finite))
        n_occ = int(np.count_nonzero(occ_state == 1))
        n_free = int(np.count_nonzero(occ_state == 0))
        n_unk = int(np.count_nonzero(occ_state == -1))

        try:
            self.get_logger().info(
                f"[tsdf->occ] mode={mode_used} (req={mode_req}), tsdf[min,max]=({vmin:.4g},{vmax:.4g}), "
                f"finite={n_finite}/{tsdf.size}, occ/free/unk={n_occ}/{n_free}/{n_unk}, "
                f"d_occ={float(self.d_occ):.3f}, d_free={float(self.d_free):.3f}, "
                f"th01_occ={float(self.tsdf_obstacle_threshold01):.3f}, th01_free={float(self.tsdf_free_space_threshold01):.3f}, "
                f"inflate={inflate}"
            )
            warn_th = int(getattr(self, "occ_min_count_warn", 0))
            if warn_th > 0 and n_occ < warn_th:
                self.get_logger().warn(
                    f"[tsdf->occ] occupied seeds too few (occ={n_occ} < {warn_th}). "
                    f"Likely thresholds mismatch or obstacles too thin. "
                    f"Try: tsdf_mode=unsigned_meter (or auto), increase d_occ (e.g. 0.08), "
                    f"increase occ_inflate_cells (e.g. 2), and/or increase d_max_safe (e.g. 1.0)."
                )
        except Exception:
            pass

        return occ_state

    def _occ_to_esdf(self, occ_state: np.ndarray) -> np.ndarray:
        """
        Compute signed-like 2D ESDF in meters.
        OpenCV distanceTransform preferred.
        Fallback is conservative approximate distance.
        """
        res = self.resolution
        num_x, num_y = occ_state.shape

        if self.treat_unknown_as_obstacle:
            obstacle = (occ_state == 1) | (occ_state == -1)
        else:
            obstacle = (occ_state == 1)

        if cv2 is not None:
            binary_free = (~obstacle).astype(np.uint8)
            dist_free_pix = cv2.distanceTransform(binary_free, cv2.DIST_L2, 3)

            binary_obs = obstacle.astype(np.uint8)
            dist_obs_pix = cv2.distanceTransform(binary_obs, cv2.DIST_L2, 3)

            esdf = dist_free_pix * res
            esdf[obstacle] = -dist_obs_pix[obstacle] * res
            return esdf.astype(np.float32)

        # Fallback
        esdf = np.zeros((num_x, num_y), dtype=np.float32)
        INF = 1e6
        esdf[:] = INF
        esdf[obstacle] = 0.0

        for _ in range(2):
            for i in range(num_x):
                for j in range(num_y):
                    v = esdf[i, j]
                    if i > 0:
                        v = min(v, esdf[i - 1, j] + 1.0)
                    if j > 0:
                        v = min(v, esdf[i, j - 1] + 1.0)
                    esdf[i, j] = v
            for i in range(num_x - 1, -1, -1):
                for j in range(num_y - 1, -1, -1):
                    v = esdf[i, j]
                    if i + 1 < num_x:
                        v = min(v, esdf[i + 1, j] + 1.0)
                    if j + 1 < num_y:
                        v = min(v, esdf[i, j + 1] + 1.0)
                    esdf[i, j] = v

        esdf = esdf * res
        esdf[obstacle] = -0.0
        return esdf.astype(np.float32)

    def _esdf_to_cost(self, esdf: np.ndarray, occ_state: np.ndarray) -> np.ndarray:
        """
        ESDF (meters) -> cost [0,100] int8 for /tsdf_costmap.
        """
        dmax = max(self.d_max_safe, 1e-3)
        d = np.clip(esdf, 0.0, dmax)
        cost = (1.0 - (d / dmax)) * 100.0

        cost[occ_state == 1] = 100.0
        cost[occ_state == -1] = float(self.unknown_cost)

        return np.clip(cost, 0.0, 100.0).astype(np.int8)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def _process_accumulated_points(self):
        all_pts = (
            np.vstack(self.accum_points) if self.accum_points else np.zeros((0, 3))
        )
        all_free = (
            np.vstack(self.accum_free_points)
            if getattr(self, 'accum_free_points', None)
            else np.zeros((0, 3))
        )
        self.accum_points = []
        self.accum_free_points = []

        if all_pts.shape[0] == 0:
            self.get_logger().warn("Accumulated points empty, skip.")
            self._publish_debug()
            return

        # 1) update motion
        self._update_motion_staleness()

        # 2) decide FSM
        prev_state = self._state
        self._state = self._decide_state()
        if self._state != prev_state:
            self.get_logger().info(
                f"FSM: {prev_state} -> {self._state} "
                f"(rho={self._rho_motion:.3f}, R95={self._R95:.3f})"
            )

        # 3) skip condition
        now = time.time()
        min_dt = 1.0 / max(self.publish_rate, 1e-3)
        if (
            self._state == "S0"
            and self._rho_motion < 0.1
            and self._R95 < self.residual_eps_small
            and (now - self._last_pub_time) < min_dt
        ):
            self.get_logger().info(
                "Skip rebuild (S0): "
                f"rho={self._rho_motion:.3f}, R95={self._R95:.3f}, "
                f"dt={now - self._last_pub_time:.2f}s"
            )
            self._publish_debug()
            return

        # 4) downsample
        pts_ds = all_pts
        if o3d is not None:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                voxel = max(1e-6, self.resolution * self.voxel_downsample_factor)
                pcd = pcd.voxel_down_sample(voxel_size=voxel)
                pts_ds = np.asarray(pcd.points)
            except Exception as e:
                self.get_logger().warn(f"Open3D downsample failed, use raw points: {e}")
                pts_ds = all_pts
        else:
            # simple random downsample
            n = all_pts.shape[0]
            keep = int(max(1000, min(n, n * 0.5)))
            if n > keep:
                idx = np.random.choice(n, size=keep, replace=False)
                pts_ds = all_pts[idx]

        if pts_ds.shape[0] == 0:
            self.get_logger().warn("Downsampled points empty, skip.")
            self._publish_debug()
            return

        # 4b) downsample free-space samples (optional)
        free_ds = all_free
        if free_ds.shape[0] > 0:
            # cap raw free points to keep CPU stable
            n = free_ds.shape[0]
            keep = int(max(800, min(n, n * 0.35)))
            if n > keep:
                idx = np.random.choice(n, size=keep, replace=False)
                free_ds = free_ds[idx]

        # 5) build TSDF / prepare indices
        self.get_logger().info(
            f"Build TSDF with pts: raw={all_pts.shape[0]}, ds={pts_ds.shape[0]}"
        )

        try:
            if o3d is not None:
                if free_ds.shape[0] > 0:
                    # LaserScan mode: bypass TerrainAnalysis; use explicit obs/free evidence.
                    self.tsdf.UpdatePoints(pts_ds, free_ds, downsample=True)
                    self.tsdf.UpdateMapParams()
                else:
                    # PointCloud2 mode: keep original TerrainAnalysis split inside TsdfCostMap.
                    pcd_o3d = o3d.geometry.PointCloud()
                    pcd_o3d.points = o3d.utility.Vector3dVector(pts_ds)
                    self.tsdf.LoadPointCloud(pcd_o3d)
            else:
                self.get_logger().warn(
                    "Open3D missing but TsdfCostMap may require it. "
                    "If TSDF creation fails, please install open3d."
                )
                # Without Open3D we can't reliably downsample/split. Fallback to old path.
                pcd_o3d = None
                try:
                    self.tsdf.LoadPointCloud(pcd_o3d)
                except Exception:
                    pass

            data, coord = self.tsdf.CreateTSDFMap()
        except Exception as e:
            self.get_logger().warn(f"TSDF creation failed: {e}")
            self._publish_debug()
            return

        if data is None or data[0] is None:
            self.get_logger().warn("TSDF returned None.")
            self._publish_debug()
            return

        tsdf_array = data[0]
        if tsdf_array.ndim != 2:
            self.get_logger().warn(f"TSDF not 2D: shape={tsdf_array.shape}")
            self._publish_debug()
            return

        start_x, start_y = float(coord[0]), float(coord[1])

        # 6) TSDF -> Occ -> ESDF -> cost
        occ_state = self._tsdf_to_occ_state(tsdf_array)
        # Debug: occupancy composition
        try:
            n_occ = int(np.sum(occ_state == 1))
            n_free = int(np.sum(occ_state == 0))
            n_unk = int(np.sum(occ_state == -1))
            self.get_logger().info(
                f"OccState counts: occ={n_occ}, free={n_free}, unk={n_unk}"
            )
        except Exception:
            pass

        esdf = self._occ_to_esdf(occ_state)
        cost = self._esdf_to_cost(esdf, occ_state)

        # 7) publish OccupancyGrid (costmap)
        num_x, num_y = cost.shape

        occ_msg = OccupancyGrid()
        occ_msg.header = Header()
        occ_msg.header.stamp = self.get_clock().now().to_msg()
        occ_msg.header.frame_id = self.target_frame

        meta = MapMetaData()
        meta.resolution = float(self.resolution)
        meta.width = int(num_x)
        meta.height = int(num_y)

        meta.origin = Pose()
        meta.origin.position.x = start_x
        meta.origin.position.y = start_y
        meta.origin.position.z = 0.0
        meta.origin.orientation.w = 1.0

        occ_msg.info = meta

        # Keep same convention with navigate side:
        # publish cost.T.flatten()
        occ_msg.data = cost.T.flatten().tolist()

        self.map_pub.publish(occ_msg)

        # Also publish d_max_safe so the navigation side can convert cost->meters
        try:
            dm = Float32()
            dm.data = float(self.d_max_safe)
            self.dmax_pub.publish(dm)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish /esdf_d_max_safe: {e}")

        # 8) cache for next residual prediction
        self._prev_occ_state = occ_state.copy()
        self._prev_origin_xy = (start_x, start_y)

        # 9) reset accumulators
        self._last_pub_time = now
        self._accum_travel = 0.0
        self._accum_yaw = 0.0
        self._rho_motion = 0.0

        # Residual after rebuild cleared
        self._R_mean = 0.0
        self._R95 = 0.0

        # 10) publish debug topics
        self._publish_debug()

        tmin = float(np.nanmin(tsdf_array))
        tmax = float(np.nanmax(tsdf_array))
        self.get_logger().info(
            f"Published /tsdf_costmap (ESDF-based): "
            f"shape=({num_x},{num_y}), origin=({start_x:.2f},{start_y:.2f}), "
            f"TSDF[min,max]=({tmin:.3f},{tmax:.3f}), "
            f"d_max_safe={self.d_max_safe:.2f}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LidarTsdfEsdfAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
