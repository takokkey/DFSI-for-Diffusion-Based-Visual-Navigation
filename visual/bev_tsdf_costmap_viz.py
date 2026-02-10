#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image

# 优先用 OpenCV 做 colormap（速度快，颜色接近 RViz 的“热力图”）
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# 没有 cv2 时用 matplotlib 兜底
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _bgr_from_mpl_colormap(gray01: np.ndarray, name: str) -> np.ndarray:
    # gray01: HxW float in [0,1]
    cmap = cm.get_cmap(name)
    rgba = cmap(gray01)  # HxWx4 float
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    bgr = rgb[..., ::-1].copy()
    return bgr


class TsdfCostmapBevViz(Node):
    def __init__(self):
        super().__init__("bev_tsdf_costmap_viz_color")

        # ---- params ----
        self.declare_parameter("topic", "/tsdf_costmap")
        self.declare_parameter("save_png", True)
        self.declare_parameter("save_dir", "/tmp/tsdf_bev")
        self.declare_parameter("save_every_n", 5)

        self.declare_parameter("publish_image", False)
        self.declare_parameter("image_topic", "/tsdf_costmap_bev")

        # 颜色与映射
        self.declare_parameter("colormap", "jet")     # "jet" / "turbo" / "plasma" ...
        self.declare_parameter("vmin", 0.0)           # OccupancyGrid 里有效值通常 0~100
        self.declare_parameter("vmax", 100.0)
        self.declare_parameter("alpha", 0.90)         # 伪彩色覆盖到白底的透明度
        self.declare_parameter("zero_transparent", True)  # 0 视作透明（白底）
        self.declare_parameter("unknown_white", True)      # -1 视作白色背景

        # 朝向（OccupancyGrid 行列与图像坐标通常要 flip 才符合 RViz）
        self.declare_parameter("flip_ud", True)       # 上下翻转
        self.declare_parameter("flip_lr", False)      # 左右翻转（一般不需要）

        # 可选：画网格线（更像 BEV）
        self.declare_parameter("draw_grid", False)
        self.declare_parameter("grid_step_m", 0.5)    # 每 0.5m 一条线

        self._cnt = 0
        self._last_saved = 0.0

        topic = self.get_parameter("topic").get_parameter_value().string_value
        self.sub = self.create_subscription(OccupancyGrid, topic, self.cb, 10)

        self.pub_img: Optional[rclpy.publisher.Publisher] = None
        if self.get_parameter("publish_image").get_parameter_value().bool_value:
            img_topic = self.get_parameter("image_topic").get_parameter_value().string_value
            self.pub_img = self.create_publisher(Image, img_topic, 10)

        self.get_logger().info(f"Subscribed: {topic}")
        if self.get_parameter("save_png").get_parameter_value().bool_value:
            d = self.get_parameter("save_dir").get_parameter_value().string_value
            _ensure_dir(d)
            self.get_logger().info(f"Saving PNGs to: {d}")

        if not _HAS_CV2 and not _HAS_MPL:
            self.get_logger().error("Neither cv2 nor matplotlib is available. Please install one of them.")

    def cb(self, msg: OccupancyGrid):
        self._cnt += 1

        w = int(msg.info.width)
        h = int(msg.info.height)
        res = float(msg.info.resolution)

        data = np.asarray(msg.data, dtype=np.int16).reshape((h, w))  # -1..100
        unknown = (data < 0)

        vmin = float(self.get_parameter("vmin").value)
        vmax = float(self.get_parameter("vmax").value)
        if vmax <= vmin:
            vmax = vmin + 1.0

        clipped = np.clip(data.astype(np.float32), vmin, vmax)
        norm01 = (clipped - vmin) / (vmax - vmin)  # 0..1

        # 伪彩色
        cmap_name = str(self.get_parameter("colormap").value).lower()
        if _HAS_CV2:
            gray8 = (norm01 * 255.0).astype(np.uint8)
            colored = cv2.applyColorMap(gray8, cv2.COLORMAP_JET) if cmap_name == "jet" else cv2.applyColorMap(gray8, cv2.COLORMAP_TURBO)
        else:
            # matplotlib colormap
            mpl_name = cmap_name if cmap_name in ["jet", "turbo", "plasma", "inferno", "magma", "viridis"] else "jet"
            colored = _bgr_from_mpl_colormap(norm01, mpl_name)

        # 背景白底 + alpha 覆盖（更像你图二，不会黑灰一坨）
        alpha = float(self.get_parameter("alpha").value)
        out = np.full_like(colored, 255, dtype=np.uint8)  # 白底

        zero_transparent = bool(self.get_parameter("zero_transparent").value)
        if zero_transparent:
            mask = (data > 0) & (~unknown)
        else:
            mask = (~unknown)

        if np.any(mask):
            # out[mask] = alpha*colored + (1-alpha)*white
            out_f = out.astype(np.float32)
            col_f = colored.astype(np.float32)
            out_f[mask] = alpha * col_f[mask] + (1.0 - alpha) * out_f[mask]
            out = np.clip(out_f, 0, 255).astype(np.uint8)

        unknown_white = bool(self.get_parameter("unknown_white").value)
        if unknown_white and np.any(unknown):
            out[unknown] = (255, 255, 255)

        # flip 以匹配 RViz 习惯显示
        if bool(self.get_parameter("flip_ud").value):
            out = np.flipud(out)
        if bool(self.get_parameter("flip_lr").value):
            out = np.fliplr(out)

        # 可选画网格线（给论文/报告更直观）
        if bool(self.get_parameter("draw_grid").value) and _HAS_CV2:
            step_m = float(self.get_parameter("grid_step_m").value)
            if step_m > 1e-6 and res > 1e-6:
                step_px = max(1, int(round(step_m / res)))
                # 轻灰色网格
                for y in range(0, out.shape[0], step_px):
                    cv2.line(out, (0, y), (out.shape[1]-1, y), (230, 230, 230), 1)
                for x in range(0, out.shape[1], step_px):
                    cv2.line(out, (x, 0), (x, out.shape[0]-1), (230, 230, 230), 1)

        # 发布 Image topic（可选）
        if self.pub_img is not None:
            imsg = Image()
            imsg.header = msg.header
            imsg.height = out.shape[0]
            imsg.width = out.shape[1]
            imsg.encoding = "bgr8"
            imsg.step = int(out.shape[1] * 3)
            imsg.data = out.tobytes()
            self.pub_img.publish(imsg)

        # 保存 PNG
        if bool(self.get_parameter("save_png").value):
            n = int(self.get_parameter("save_every_n").value)
            if n <= 0:
                n = 1
            if (self._cnt % n) == 0:
                save_dir = str(self.get_parameter("save_dir").value)
                _ensure_dir(save_dir)
                ts = time.time()
                fn = os.path.join(save_dir, f"tsdf_costmap_bev_{ts:.3f}.png")
                if _HAS_CV2:
                    cv2.imwrite(fn, out)
                elif _HAS_MPL:
                    # matplotlib 保存
                    import matplotlib.pyplot as plt
                    plt.imsave(fn, out[..., ::-1])  # BGR->RGB
                self.get_logger().info(f"Saved: {fn}  (w={w}, h={h}, res={res:.3f})")


def main():
    rclpy.init()
    node = TsdfCostmapBevViz()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
