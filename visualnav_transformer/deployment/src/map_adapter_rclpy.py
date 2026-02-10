#!/usr/bin/env python3
"""
map_adapter_rclpy.py (tf_transformations-free, ego-centric TSDF around base_link)

- 支持 LaserScan (/scan) 或 PointCloud2 (例如 /camera/.../depth/color/points)
- 将输入点变换到目标 frame (默认 'base_link') 并累积多帧 -> 生成 TSDF（使用 tsdf_cost_map.TsdfCostMap）
- 发布 nav_msgs/OccupancyGrid 到 /tsdf_costmap
- 使用纯 numpy 实现四元数->变换矩阵，无需 tf-transformations 包
- 在 TSDF 构建和 OccupancyGrid 发布时打出详细日志，方便与导航侧
  (navigate.tsdf_callback_local / PathGuide.set_cost_map) 做 shape/origin 一致性检查。

注意：现在 TSDF 是挂在 base_link 坐标系下的“局部地图”，
      也就是说 /tsdf_costmap 的 frame_id = 'base_link'（或你通过参数设置的 target_frame）。
"""

import math
import os
import sys
import time

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, MapMetaData
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header

# try pointcloud helper
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

# your TSDF implementation
from tsdf_cost_map import TsdfCostMap
from costmap_cfg import CostMapConfig


# ---------------------------------------------------------------------------
# 基础数学工具
# ---------------------------------------------------------------------------
def quat_to_rot_matrix(qx, qy, qz, qw):
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix (numpy)."""
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

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )
    return R


def transform_points(pts, trans_msg):
    """
    Build 4x4 transform from TransformStamped-like fields (translation + rotation),
    apply to pts (N,3) -> return transformed pts (N,3).

    trans_msg: object with .transform.translation.x/y/z and .transform.rotation.x/y/z/w
               or an object with .translation/.rotation fields.
    """
    if hasattr(trans_msg, "transform"):
        t = trans_msg.transform.translation
        r = trans_msg.transform.rotation
    else:
        # fallback: expect (.translation, .rotation)
        t = trans_msg.translation
        r = trans_msg.rotation

    R = quat_to_rot_matrix(r.x, r.y, r.z, r.w)
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.array([t.x, t.y, t.z], dtype=float)

    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=float)])
    pts_tf = (T @ homo.T).T[:, :3]
    return pts_tf


# ---------------------------------------------------------------------------
# 主节点：LiDAR / PointCloud → TSDF → OccupancyGrid
# ---------------------------------------------------------------------------
class LidarTsdfAdapter(Node):
    def __init__(self):
        super().__init__("lidar_tsdf_adapter")

        # ---------------- 参数 ----------------
        self.declare_parameter("input_mode", "scan")  # 'scan' or 'pointcloud'
        self.declare_parameter("input_topic_scan", "/scan")
        self.declare_parameter(
            "input_topic_pc",
            "/camera/realsense2_camera_node/depth/color/points",
        )
        # === 修改点 1：默认 target_frame 改为 'base_link'（原来是 'odom'） ===
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("accum_frames", 6)
        self.declare_parameter("max_range", 10.0)
        self.declare_parameter("min_range", 0.01)
        self.declare_parameter("publish_rate", 1.0)  # Hz
        self.declare_parameter("voxel_downsample_factor", 0.8)

        input_mode = (
            self.get_parameter("input_mode")
            .get_parameter_value()
            .string_value
        )
        self.input_mode = input_mode.lower()
        self.scan_topic = (
            self.get_parameter("input_topic_scan")
            .get_parameter_value()
            .string_value
        )
        self.pc_topic = (
            self.get_parameter("input_topic_pc")
            .get_parameter_value()
            .string_value
        )
        self.target_frame = (
            self.get_parameter("target_frame")
            .get_parameter_value()
            .string_value
        )
        self.accum_frames = (
            self.get_parameter("accum_frames")
            .get_parameter_value()
            .integer_value
        )
        self.max_range = float(
            self.get_parameter("max_range")
            .get_parameter_value()
            .double_value
        )
        self.min_range = float(
            self.get_parameter("min_range")
            .get_parameter_value()
            .double_value
        )
        self.publish_rate = float(
            self.get_parameter("publish_rate")
            .get_parameter_value()
            .double_value
        )
        self.voxel_downsample_factor = float(
            self.get_parameter("voxel_downsample_factor")
            .get_parameter_value()
            .double_value
        )

        # ---------------- TSDF 初始化 ----------------
        self.cfg = CostMapConfig()
        self.tsdf = TsdfCostMap(self.cfg.general, self.cfg.tsdf_cost_map)

        self._last_pub_time = 0.0
        self.accum_points = []  # list of numpy (N,3) arrays
        self.msg_count = 0

        self.get_logger().info(
            f"TSDF config: resolution={self.cfg.general.resolution:.3f}, "
            f"x_min={self.cfg.general.x_min}, x_max={self.cfg.general.x_max}, "
            f"y_min={self.cfg.general.y_min}, y_max={self.cfg.general.y_max}, "
            f"target_frame={self.target_frame}"
        )

        # ---------------- TF buffer/listener ----------------
        if tf2_ros is not None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self
            )
        else:
            self.tf_buffer = None
            self.tf_listener = None
            self.get_logger().warn(
                "tf2_ros not available; transformations will fail if needed."
            )

        # ---------------- 订阅输入 ----------------
        if self.input_mode == "scan":
            self.scan_sub = self.create_subscription(
                LaserScan, self.scan_topic, self.scan_callback, 10
            )
            self.get_logger().info(
                f"Subscribed to LaserScan topic {self.scan_topic}"
            )
        else:
            self.pc_sub = self.create_subscription(
                PointCloud2, self.pc_topic, self.pc_callback, 10
            )
            self.get_logger().info(
                f"Subscribed to PointCloud2 topic {self.pc_topic}"
            )

        # ---------------- 发布 costmap ----------------
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            "/tsdf_costmap",
            1,
        )
        self.get_logger().info(
            f"LidarTsdfAdapter initialized (mode={self.input_mode}), "
            f"target_frame={self.target_frame}, publishing to /tsdf_costmap"
        )

    # ------------------------------------------------------------------
    # LaserScan 回调
    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        finite_mask = np.isfinite(ranges)
        valid_mask = (
            finite_mask & (ranges > self.min_range) & (ranges < self.max_range)
        )
        valid_count = int(np.sum(valid_mask))
        self.get_logger().debug(
            f"Scan received: total={len(ranges)}, valid={valid_count}"
        )

        if valid_count == 0:
            return

        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        zs = np.zeros_like(xs)
        pts = np.stack([xs, ys, zs], axis=1)[valid_mask]

        # TF transform if available（从 scan frame → target_frame，一般是 base_link）
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
                    f"TF lookup failed ({msg.header.frame_id} -> "
                    f"{self.target_frame}): {e}"
                )

        self.accum_points.append(pts)
        self.msg_count += 1

        if len(self.accum_points) >= self.accum_frames:
            self._process_accumulated_points()

    # ------------------------------------------------------------------
    # PointCloud2 回调
    # ------------------------------------------------------------------
    def pc_callback(self, msg: PointCloud2):
        if pc2_read is None:
            self.get_logger().error(
                "point_cloud2 read module not available; "
                "cannot parse PointCloud2"
            )
            return

        points = []
        for p in pc2_read.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            points.append([p[0], p[1], p[2]])
            # 防止一次云过大
            if len(points) >= 200000:
                break

        if len(points) == 0:
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
                    f"TF lookup failed for PointCloud2 "
                    f"({msg.header.frame_id} -> {self.target_frame}): {e}"
                )

        self.accum_points.append(pts)
        self.msg_count += 1

        if len(self.accum_points) >= self.accum_frames:
            self._process_accumulated_points()

    # ------------------------------------------------------------------
    # 累积点云 → TSDF → OccupancyGrid
    # ------------------------------------------------------------------
    def _process_accumulated_points(self):
        all_pts = (
            np.vstack(self.accum_points)
            if len(self.accum_points) > 0
            else np.zeros((0, 3))
        )
        self.accum_points = []

        if all_pts.shape[0] == 0:
            self.get_logger().warn(
                "Accumulated points empty, skipping TSDF."
            )
            return

        # 下采样
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            voxel = max(
                1e-6,
                self.cfg.general.resolution * self.voxel_downsample_factor,
            )
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            pts_ds = np.asarray(pcd.points)
        except Exception as e:
            self.get_logger().warn(
                f"Downsample failed, using raw pts: {e}"
            )
            pts_ds = all_pts

        if pts_ds.shape[0] == 0:
            self.get_logger().warn(
                "Downsampled produced 0 points, skipping TSDF."
            )
            return

        self.get_logger().info(
            f"Accumulated {all_pts.shape[0]} pts -> "
            f"downsampled {pts_ds.shape[0]} pts; building TSDF..."
        )

        # 构建 TSDF（现在是在 target_frame = base_link 坐标系下）
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pts_ds)

        try:
            self.tsdf.LoadPointCloud(pcd_o3d)
            data, coord = self.tsdf.CreateTSDFMap()
        except Exception as e:
            self.get_logger().warn(f"TSDF creation failed: {e}")
            return

        if data is None or data[0] is None:
            self.get_logger().warn("TSDF returned None")
            return

        tsdf_array = data[0]
        if tsdf_array.ndim != 2:
            self.get_logger().warn(
                f"TSDF array is not 2D (shape={tsdf_array.shape}), "
                f"skipping."
            )
            return

        start_x, start_y = float(coord[0]), float(coord[1])

        # 频率控制
        now = time.time()
        if now - self._last_pub_time < 1.0 / max(self.publish_rate, 1e-3):
            return
        self._last_pub_time = now

        # TSDF 数值范围和 NaN 检查
        tmin = float(np.nanmin(tsdf_array))
        tmax = float(np.nanmax(tsdf_array))
        nan_count = int(np.isnan(tsdf_array).sum())

        self.get_logger().info(
            f"TSDF map ready: shape={tsdf_array.shape}, "
            f"origin=({start_x:.2f},{start_y:.2f}) in {self.target_frame}, "
            f"tmin={tmin:.3f}, tmax={tmax:.3f}, "
            f"nan_count={nan_count}, "
            f"resolution={self.cfg.general.resolution:.3f}"
        )

        # 正规化：TSDF -> [0,1] -> cost [0,1]
        if tmax - tmin <= 1e-6:
            norm = np.clip(tsdf_array, 0.0, 1.0)
        else:
            norm = (tsdf_array - tmin) / (tmax - tmin)

        cost = 1.0 - norm
        cost = np.nan_to_num(cost, nan=1.0)

        # [0,1] -> [0,100] int8
        occ = (np.clip(cost, 0.0, 1.0) * 100.0).astype(np.int8)

        # ---------------- OccupancyGrid 构造 ----------------
        occ_msg = OccupancyGrid()
        occ_msg.header = Header()
        occ_msg.header.stamp = self.get_clock().now().to_msg()
        occ_msg.header.frame_id = self.target_frame  # 一般是 'base_link'

        meta = MapMetaData()
        meta.resolution = float(self.cfg.general.resolution)

        # 注意：tsdf_array 形状约定为 [num_x, num_y]，其中：
        #   - x 方向对应 world x，y 方向对应 world y
        #   - 在 OccupancyGrid 中，data 按 row-major (y,x) 存储。
        # 为了与导航侧 (navigate.tsdf_callback_local -> reshape((h,w)).T) 完全对齐：
        #   这里将 tsdf_array 看作 [H,W] = [num_x, num_y]，
        #   width = num_x, height = num_y，
        #   发布时使用 occ.T.flatten()。
        num_x, num_y = tsdf_array.shape
        meta.width = int(num_x)
        meta.height = int(num_y)

        origin = Pose()
        origin.position.x = start_x
        origin.position.y = start_y
        origin.position.z = 0.0
        meta.origin = origin
        occ_msg.info = meta

        # 关键：使用 .T 再 flatten，保证导航端用 reshape((h,w)).T 能还原原始栅格。
        data_flat = occ.T.flatten()
        occ_msg.data = data_flat.tolist()

        self.costmap_pub.publish(occ_msg)
        self.get_logger().info(
            f"Published /tsdf_costmap (size {num_x}x{num_y}), "
            f"origin=({start_x:.2f},{start_y:.2f}) in {self.target_frame}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LidarTsdfAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
