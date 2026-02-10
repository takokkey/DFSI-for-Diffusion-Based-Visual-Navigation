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
        self.declare_parameter("d_occ", 0.05)
        self.declare_parameter("d_free", 0.10)
        self.declare_parameter("treat_unknown_as_obstacle", True)

        # ESDF->cost parameters
        self.declare_parameter("d_max_safe", 0.50)  # meters
        self.declare_parameter("unknown_cost", 100)

        # Residual parameters
        self.declare_parameter("residual_stride", 3)
        self.declare_parameter("residual_eps_small", 0.05)
        self.declare_parameter("residual_eps_bad", 0.30)
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

        # FSM
        self._state = "S0"

        # caches
        self.accum_points: List[np.ndarray] = []
        self._last_pub_time = 0.0

        # previous map for residual prediction
        self._prev_occ_state: Optional[np.ndarray] = None  # -1 unknown, 0 free, 1 occ
        self._prev_origin_xy: Tuple[float, float] = (0.0, 0.0)

        # ---------------- TF ----------------
        if tf2_ros is not None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
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
            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed ({msg.header.frame_id}->{self.target_frame}): {e}"
                )

        self.accum_points.append(pts)

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

        if self.tf_buffer is not None:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                pts = transform_points(pts, t)
            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed ({msg.header.frame_id}->{self.target_frame}): {e}"
                )

        self.accum_points.append(pts)

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
    def _update_residual_from_scan(self, msg: LaserScan):
        """
        2D geometric consistency residual:
          r = z_meas - z_pred
        z_pred obtained by simple raycast into previous occupancy.
        """
        if self._prev_occ_state is None:
            self._R_mean = 0.0
            self._R95 = 0.0
            return

        ranges = np.array(msg.ranges, dtype=float)
        finite = np.isfinite(ranges)
        valid = finite & (ranges > self.min_range) & (ranges < self.max_range)
        if int(np.sum(valid)) == 0:
            self._R_mean = 0.0
            self._R95 = 0.0
            return

        idx_all = np.where(valid)[0]
        if self.residual_stride > 1:
            idx_all = idx_all[:: self.residual_stride]
        if idx_all.size > self.residual_max_beams:
            idx_all = idx_all[: self.residual_max_beams]

        angles = msg.angle_min + idx_all * msg.angle_increment
        z_meas = ranges[idx_all]

        origin_x, origin_y = self._prev_origin_xy
        occ = self._prev_occ_state
        num_x, num_y = occ.shape
        res = self.resolution

        residuals = []

        for a, zm in zip(angles, z_meas):
            ca, sa = math.cos(a), math.sin(a)

            z_pred = self.max_range
            step = res
            r = self.min_range
            while r <= min(self.max_range, zm + res * 2.0):
                x = r * ca
                y = r * sa
                ix = int(math.floor((x - origin_x) / res))
                iy = int(math.floor((y - origin_y) / res))
                if ix < 0 or ix >= num_x or iy < 0 or iy >= num_y:
                    r += step
                    continue

                state = occ[ix, iy]
                if state == 1:
                    z_pred = r
                    break
                if self.treat_unknown_as_obstacle and state == -1:
                    z_pred = r
                    break

                r += step

            residuals.append(zm - z_pred)

        if not residuals:
            self._R_mean = 0.0
            self._R95 = 0.0
            return

        abs_r = np.abs(np.array(residuals, dtype=float))
        self._R_mean = float(np.mean(abs_r))
        self._R95 = float(np.percentile(abs_r, 95))

    # ------------------------------------------------------------------
    # FSM decision
    # ------------------------------------------------------------------
    def _decide_state(self):
        """
        Dual-criteria FSM:
          - Primary: residual R95
          - Secondary: motion rho_motion
        """
        if self._prev_occ_state is None:
            return "S2"

        if self._R95 >= self.residual_eps_bad:
            return "S2"

        if self._R95 >= self.residual_eps_small:
            if self._rho_motion >= self._tau_high:
                return "S2"
            return "S1"

        if self._rho_motion < self._tau_low:
            return "S0"
        elif self._rho_motion < self._tau_high:
            return "S1"
        else:
            return "S2"

    # ------------------------------------------------------------------
    # TSDF -> Occ -> ESDF -> costmap
    # ------------------------------------------------------------------
    def _tsdf_to_occ_state(self, tsdf: np.ndarray) -> np.ndarray:
        """
        Return occ_state with values:
          -1 unknown
           0 free
           1 occupied
        """
        occ_state = np.full(tsdf.shape, -1, dtype=np.int8)
        occ_state[tsdf <= -self.d_occ] = 1
        occ_state[tsdf >= self.d_free] = 0
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
        self.accum_points = []

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

        # 5) build TSDF
        self.get_logger().info(
            f"Build TSDF with pts: raw={all_pts.shape[0]}, ds={pts_ds.shape[0]}"
        )

        try:
            if o3d is not None:
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pts_ds)
                self.tsdf.LoadPointCloud(pcd_o3d)
            else:
                # TsdfCostMap 可能要求 open3d PointCloud；
                # 如果你环境里没有 open3d，建议直接安装或改 TsdfCostMap 接口。
                self.get_logger().warn(
                    "Open3D missing but TsdfCostMap may require it. "
                    "If TSDF creation fails, please install open3d."
                )
                pcd_o3d = None
                self.tsdf.LoadPointCloud(pcd_o3d)

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
