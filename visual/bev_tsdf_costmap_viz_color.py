#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV visualization for nav_msgs/OccupancyGrid costmap (e.g., /tsdf_costmap).

- Uses a "RViz-like" rainbow colormap (JET) instead of grayscale.
- Treats value==0 as transparent by default (only show unsafe area).
- Unknown cells (<0) are transparent by default.
- Draws robot position (0,0) as a small circle by default.

Run (example):
  python3 bev_tsdf_costmap_viz_color.py --ros-args \
    -p topic:=/tsdf_costmap \
    -p save_png:=true \
    -p save_dir:=/tmp/tsdf_bev \
    -p save_every_n:=5

Notes:
- OccupancyGrid values are assumed in [-1, 0..100]:
  -1 = unknown
   0 = free (often cost 0)
 100 = max cost / occupied
"""

import os
import time
from datetime import datetime

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _grid_to_rgba(
    grid: np.ndarray,
    *,
    show_free: bool = False,
    show_unknown: bool = False,
    alpha_free: int = 0,
    alpha_unknown: int = 0,
    colormap: str = "jet",
) -> np.ndarray:
    """
    grid: int16 array (H,W) with values [-1,0..100]
    returns RGBA uint8 image (H,W,4)
    """
    h, w = grid.shape
    g = grid.astype(np.int16)

    unk = g < 0
    g = np.clip(g, 0, 100)

    # Normalize to 0..255
    u8 = (g.astype(np.float32) / 100.0 * 255.0).astype(np.uint8)

    # Colorize
    if cv2 is not None:
        # OpenCV uses BGR
        cm = cv2.COLORMAP_JET
        colored = cv2.applyColorMap(u8, cm)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: simple RGB ramp (blue->red)
        t = u8.astype(np.float32) / 255.0
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[..., 0] = t                    # R
        rgb[..., 2] = 1.0 - t              # B
        rgb = (rgb * 255.0).astype(np.uint8)

    # Alpha
    alpha = np.full((h, w), 255, dtype=np.uint8)

    if not show_free:
        alpha[g == 0] = np.uint8(alpha_free)

    if not show_unknown:
        alpha[unk] = np.uint8(alpha_unknown)

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return rgba


def _crop_rgba_by_mask(
    rgba: np.ndarray,
    mask: np.ndarray,
    *,
    margin: int = 8,
    min_size: int = 96,
) -> np.ndarray:
    """Crop RGBA image to the tight bounding box of mask==True (+margin).

    If the active region is very small, we enforce a minimum crop size
    (centered on the active region) to avoid excessive zoom.
    """
    if mask is None:
        return rgba
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if not np.any(mask):
        return rgba

    h, w, _ = rgba.shape
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Add margin
    y0 = max(0, y0 - margin)
    y1 = min(h - 1, y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(w - 1, x1 + margin)

    # Enforce minimum crop size (around the bbox center)
    def _expand(lo: int, hi: int, limit: int) :
        length = hi - lo + 1
        if length >= min_size:
            return lo, hi
        extra = min_size - length
        lo2 = lo - extra // 2
        hi2 = hi + (extra - extra // 2)
        if lo2 < 0:
            hi2 = min(limit - 1, hi2 - lo2)
            lo2 = 0
        if hi2 > limit - 1:
            lo2 = max(0, lo2 - (hi2 - (limit - 1)))
            hi2 = limit - 1
        return lo2, hi2

    y0, y1 = _expand(y0, y1, h)
    x0, x1 = _expand(x0, x1, w)

    return rgba[y0 : y1 + 1, x0 : x1 + 1, :]


def _resize_and_pad_rgba(
    rgba: np.ndarray,
    *,
    out_size: int = 512,
    pad_to_square: bool = False,
    keep_aspect: bool = True,
    pad_value: int = 255,
    resize_mode: str = "nearest",
) -> np.ndarray:
    """Resize (and optionally pad) RGBA to out_size.

    - If pad_to_square=True, keep aspect ratio and center-pad to (out_size,out_size).
    - If pad_to_square=False and keep_aspect=True, keep aspect ratio and resize so
      the longer edge becomes out_size (no padding, returns a rectangular image).
    - If pad_to_square=False and keep_aspect=False, directly resize to (out_size,out_size).
    """
    if out_size is None or int(out_size) <= 0:
        return rgba
    out_size = int(out_size)
    if Image is None:
        return rgba

    h, w, _ = rgba.shape
    if h <= 0 or w <= 0:
        return rgba

    mode = str(resize_mode).lower()
    if mode in ("linear", "bilinear"):
        resample = Image.BILINEAR
    else:
        resample = Image.NEAREST

    im = Image.fromarray(rgba, mode="RGBA")

    if not pad_to_square:
        if bool(keep_aspect):
            scale = float(out_size) / float(max(h, w))
            nh = max(1, int(round(h * scale)))
            nw = max(1, int(round(w * scale)))
            im = im.resize((nw, nh), resample=resample)
        else:
            im = im.resize((out_size, out_size), resample=resample)
        return np.array(im, dtype=np.uint8)

    scale = float(out_size) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    im_small = im.resize((nw, nh), resample=resample)

    canvas = Image.new("RGBA", (out_size, out_size), (pad_value, pad_value, pad_value, 255))
    ox = (out_size - nw) // 2
    oy = (out_size - nh) // 2
    canvas.paste(im_small, (ox, oy), im_small)
    return np.array(canvas, dtype=np.uint8)


def _rgba_to_rgb_on_background(rgba: np.ndarray, *, bg: int = 255) -> np.ndarray:
    """Composite RGBA onto a solid background (returns RGB uint8)."""
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgb * a + float(bg) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_robot_rgba(
    rgba: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    res: float,
    x: float = 0.0,
    y: float = 0.0,
    r_m: float = 0.12,
):
    """
    Draw a small circle at (x,y) in map coordinates.
    """
    if cv2 is None:
        return rgba  # keep simple without OpenCV
    h, w, _ = rgba.shape
    cx = int(round((x - origin_x) / max(res, 1e-6)))
    cy = int(round((y - origin_y) / max(res, 1e-6)))

    # OccupancyGrid origin is bottom-left in map coords; images usually top-left.
    # We'll flip vertically when saving (see below), so here use raw indexing.
    rad = int(round(r_m / max(res, 1e-6)))
    rad = max(2, min(rad, 30))

    # Draw in-place on RGB channels only; keep alpha
    rgb = rgba[..., :3].copy()
    a = rgba[..., 3].copy()
    cv2.circle(rgb, (cx, cy), rad, (255, 255, 255), 2)  # white ring
    rgba[..., :3] = rgb
    rgba[..., 3] = a
    return rgba


class BevCostmapViz(Node):
    def __init__(self):
        super().__init__("bev_tsdf_costmap_viz_color")

        # Parameters
        self.declare_parameter("topic", "/tsdf_costmap")
        self.declare_parameter("save_png", True)
        self.declare_parameter("save_dir", "/tmp/tsdf_bev")
        self.declare_parameter("save_every_n", 5)

        # Zoom/crop parameters (to reduce blank/transparent area in saved PNGs).
        # - auto_crop: detect active cost region and crop to it (plus margin).
        # - crop_use_alpha: if True, use alpha>0 as the active mask (best when show_free/show_unknown are False).
        #                  if False, use cost>=crop_cost_thresh as the active mask.
        # - output_size: if >0, resize/pad the cropped image to output_size x output_size for easier viewing.
        self.declare_parameter("auto_crop", True)
        self.declare_parameter("crop_use_alpha", True)
        self.declare_parameter("crop_cost_thresh", 1)  # 1~100, only used when crop_use_alpha=False
        self.declare_parameter("crop_margin_cells", 8)
        self.declare_parameter("crop_min_size_cells", 96)
        self.declare_parameter("output_size", 512)      # 0 keeps original size
        self.declare_parameter("pad_to_square", False)
        self.declare_parameter("keep_aspect", True)
        self.declare_parameter("opaque_background", True)
        self.declare_parameter("background_value", 255)  # 255=white, 0=black
        self.declare_parameter("resize_mode", "nearest") # nearest|linear

        # Visual params
        self.declare_parameter("show_free", False)
        self.declare_parameter("show_unknown", False)
        self.declare_parameter("alpha_free", 0)      # 0=transparent
        self.declare_parameter("alpha_unknown", 0)   # 0=transparent
        self.declare_parameter("draw_robot", True)
        self.declare_parameter("robot_radius_m", 0.12)

        # Geometry / orientation
        self.declare_parameter("flip_y", True)       # make +y up in the image
        self.declare_parameter("rotate_90", False)   # debug only

        self.topic = str(self.get_parameter("topic").value)
        self.save_png = bool(self.get_parameter("save_png").value)
        self.save_dir = str(self.get_parameter("save_dir").value)
        self.save_every_n = int(self.get_parameter("save_every_n").value)

        _ensure_dir(self.save_dir)

        self._count = 0
        self._last_info = None

        self.sub = self.create_subscription(OccupancyGrid, self.topic, self.cb, 10)
        self.get_logger().info(
            f"Subscribe {self.topic} -> save_png={self.save_png}, dir={self.save_dir}, every_n={self.save_every_n}"
        )

        if Image is None:
            self.get_logger().warn("PIL not found; PNG saving may fail. Try: pip install pillow")

    def cb(self, msg: OccupancyGrid):
        self._count += 1
        if (self._count % max(self.save_every_n, 1)) != 0:
            return

        w = int(msg.info.width)
        h = int(msg.info.height)
        if w <= 0 or h <= 0:
            return

        data = np.array(msg.data, dtype=np.int16).reshape((h, w))
        rgba = _grid_to_rgba(
            data,
            show_free=bool(self.get_parameter("show_free").value),
            show_unknown=bool(self.get_parameter("show_unknown").value),
            alpha_free=int(self.get_parameter("alpha_free").value),
            alpha_unknown=int(self.get_parameter("alpha_unknown").value),
        )

        origin_x = float(msg.info.origin.position.x)
        origin_y = float(msg.info.origin.position.y)
        res = float(msg.info.resolution)

        if bool(self.get_parameter("draw_robot").value):
            rgba = _draw_robot_rgba(
                rgba,
                origin_x=origin_x,
                origin_y=origin_y,
                res=res,
                x=0.0,
                y=0.0,
                r_m=float(self.get_parameter("robot_radius_m").value),
            )

        # Convert map->image coordinates: flip vertically so +y is up in the PNG.
        if bool(self.get_parameter("flip_y").value):
            rgba = np.flipud(rgba)

        if bool(self.get_parameter("rotate_90").value):
            rgba = np.rot90(rgba, k=1)

        # Optional: crop/zoom to reduce blank area.
        if bool(self.get_parameter("auto_crop").value):
            margin = int(self.get_parameter("crop_margin_cells").value)
            min_size = int(self.get_parameter("crop_min_size_cells").value)
            use_alpha = bool(self.get_parameter("crop_use_alpha").value)
            if use_alpha:
                mask = (rgba[..., 3] > 0)
            else:
                thresh = int(self.get_parameter("crop_cost_thresh").value)
                mask = (data >= max(1, thresh))
                # Apply the same transforms as RGBA.
                if bool(self.get_parameter("flip_y").value):
                    mask = np.flipud(mask)
                if bool(self.get_parameter("rotate_90").value):
                    mask = np.rot90(mask, k=1)
            rgba = _crop_rgba_by_mask(rgba, mask, margin=margin, min_size=min_size)

        # Optional: resize (and pad) so the result is easier to read.
        out_size = int(self.get_parameter("output_size").value)
        pad_sq = bool(self.get_parameter("pad_to_square").value)
        keep_aspect = bool(self.get_parameter("keep_aspect").value)
        # background value for padding / compositing
        bg_val = int(self.get_parameter("background_value").value)
        try:
            bg_val = int(self.get_parameter("pad_value").value)  # optional alias
        except Exception:
            pass
        mode = str(self.get_parameter("resize_mode").value)
        rgba = _resize_and_pad_rgba(
            rgba,
            out_size=out_size,
            pad_to_square=pad_sq,
            keep_aspect=keep_aspect,
            pad_value=bg_val,
            resize_mode=mode,
        )

        if self.save_png and Image is not None:
            out = os.path.join(self.save_dir, f"costmap_{_now_str()}.png")
            if bool(self.get_parameter("opaque_background").value):
                rgb = _rgba_to_rgb_on_background(rgba, bg=bg_val)
                Image.fromarray(rgb, mode="RGB").save(out)
            else:
                Image.fromarray(rgba, mode="RGBA").save(out)
            self.get_logger().info(
                f"Saved {out}  (raw={h}x{w}, out={rgba.shape[0]}x{rgba.shape[1]}, res={res:.3f}, origin=({origin_x:.2f},{origin_y:.2f}))"
            )


def main():
    rclpy.init()
    node = BevCostmapViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
