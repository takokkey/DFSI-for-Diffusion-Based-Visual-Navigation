#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_nomad_vs_pathguide_pairseq_fixed_v2.py

Fixes your current situation:
- Your log shows image frames stop at img=30, so the saver never passes the "fresh image" gate.
  This is commonly a QoS mismatch on the image topic.

This version adds:
1) --image-qos {reliable,best_effort}  (default: reliable)
2) --image-stale-sec (default: 10.0)
3) --ignore-image-stale  (if set, allow saving using last received image even if old)
4) More explicit debug logging when images stop.

Pairing:
- Uses seq from Float32MultiArray.layout.dim[0].label == 'seq:<int>'
- Same color per sample index across left/right.

Left : camera + NoMaD candidates (dashed) + waypoint_nomad (red star)
Right: camera + PathGuide candidates (solid) + waypoint (green star)
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


def make_qos(reliability: str, depth: int) -> QoSProfile:
    rel = ReliabilityPolicy.RELIABLE if reliability.lower() == "reliable" else ReliabilityPolicy.BEST_EFFORT
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=rel,
        durability=DurabilityPolicy.VOLATILE,
    )


def parse_seq(msg: Float32MultiArray) -> int:
    try:
        dims = getattr(msg, "layout", None).dim
        if dims and isinstance(dims[0].label, str) and dims[0].label.startswith("seq:"):
            return int(dims[0].label.split(":", 1)[1])
    except Exception:
        pass
    return -1


def parse_actions_xy(msg: Float32MultiArray) -> Optional[np.ndarray]:
    arr = np.asarray(msg.data, dtype=np.float32)
    if arr.size == 0:
        return None
    if (arr.size % 2) != 0:
        # tolerate leading dummy 0
        if arr.size >= 3 and abs(float(arr[0])) < 1e-6 and ((arr.size - 1) % 2 == 0):
            arr = arr[1:]
        else:
            return None
    return arr.reshape(-1, 2)


def infer_samples_horizon(n_points: int) -> Tuple[int, int]:
    candidates = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    best = None
    best_score = 1e18
    for ns in candidates:
        if n_points % ns != 0:
            continue
        h = n_points // ns
        if not (1 <= h <= 256):
            continue
        score = abs(ns - 8) + 0.05 * min(abs(h - 8), abs(h - 12), abs(h - 16), abs(h - 24))
        if score < best_score:
            best_score = score
            best = (ns, h)
    return best if best else (1, n_points)


@dataclass
class ActionPacket:
    t: float
    seq: int
    xy: np.ndarray  # (N,2)


@dataclass
class WaypointPacket:
    t: float
    seq: int
    wp: np.ndarray  # (2,)


class Saver(Node):
    def __init__(self, args):
        super().__init__("viz_nomad_vs_pathguide_pairseq_saver")
        self.args = args
        self.bridge = CvBridge()
        os.makedirs(self.args.out_dir, exist_ok=True)

        self.img_rgb: Optional[np.ndarray] = None
        self.t_img: float = 0.0
        self.img_count: int = 0
        self._last_img_count_seen: int = 0
        self._last_img_warn_t: float = 0.0

        self.nomad: Dict[int, ActionPacket] = {}
        self.pg: Dict[int, ActionPacket] = {}
        self.wp_nomad: Dict[int, WaypointPacket] = {}
        self.wp_final: Dict[int, WaypointPacket] = {}

        self.fig, (self.axL, self.axR) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("NoMaD raw vs PathGuide constrained (same color per sample index)")

        self.saved = 0
        self.last_saved_img_count = -1
        self._last_status_t = 0.0

        # QoS
        image_qos = make_qos(self.args.image_qos, self.args.image_depth)
        arr_qos = qos_profile_sensor_data  # best-effort, volatile

        self.create_subscription(Image, self.args.image_topic, self.cb_img, image_qos)
        self.create_subscription(Float32MultiArray, self.args.nomad_topic, self.cb_nomad, arr_qos)
        self.create_subscription(Float32MultiArray, self.args.pg_topic, self.cb_pg, arr_qos)
        self.create_subscription(Float32MultiArray, self.args.waypoint_nomad_topic, self.cb_wp_nomad, arr_qos)
        self.create_subscription(Float32MultiArray, self.args.waypoint_topic, self.cb_wp_final, arr_qos)

        self.timer = self.create_timer(1.0 / float(self.args.tick_hz), self.tick)

        self.get_logger().info(
            f"Headless saver started. out_dir={self.args.out_dir}, save_every={self.args.save_every}, "
            f"image_qos={self.args.image_qos}, image_stale_sec={self.args.image_stale_sec}, "
            f"require_nomad={self.args.require_nomad}, require_wp={self.args.require_wp}"
        )

    # ---- callbacks ----
    def cb_img(self, msg: Image):
        try:
            # Try bgr8 then rgb8 fallback
            try:
                cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                rgb = cv[..., ::-1]
            except Exception:
                cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                rgb = cv
            self.img_rgb = rgb
            self.t_img = time.time()
            self.img_count += 1
        except Exception as e:
            self.get_logger().warn(f"image decode failed: {e}")

    def cb_nomad(self, msg: Float32MultiArray):
        seq = parse_seq(msg)
        xy = parse_actions_xy(msg)
        if seq < 0 or xy is None:
            return
        self.nomad[seq] = ActionPacket(time.time(), seq, xy)

    def cb_pg(self, msg: Float32MultiArray):
        seq = parse_seq(msg)
        xy = parse_actions_xy(msg)
        if seq < 0 or xy is None:
            return
        self.pg[seq] = ActionPacket(time.time(), seq, xy)

    def cb_wp_nomad(self, msg: Float32MultiArray):
        seq = parse_seq(msg)
        if seq < 0 or len(msg.data) < 2:
            return
        wp = np.asarray([msg.data[0], msg.data[1]], dtype=np.float32)
        self.wp_nomad[seq] = WaypointPacket(time.time(), seq, wp)

    def cb_wp_final(self, msg: Float32MultiArray):
        seq = parse_seq(msg)
        if seq < 0 or len(msg.data) < 2:
            return
        wp = np.asarray([msg.data[0], msg.data[1]], dtype=np.float32)
        self.wp_final[seq] = WaypointPacket(time.time(), seq, wp)

    # ---- helpers ----
    def image_fresh(self) -> bool:
        if self.img_rgb is None:
            return False
        if self.args.ignore_image_stale:
            return True
        return (time.time() - self.t_img) <= float(self.args.image_stale_sec)

    def latest_complete_seq(self) -> Optional[int]:
        if not self.pg:
            return None
        for seq in sorted(self.pg.keys(), reverse=True):
            if self.args.require_nomad and (seq not in self.nomad):
                continue
            if self.args.require_wp:
                if (seq not in self.wp_nomad) or (seq not in self.wp_final):
                    continue
            return seq
        return None

    def split(self, xy: np.ndarray) -> np.ndarray:
        n_points = int(xy.shape[0])
        ns, h = infer_samples_horizon(n_points)
        if self.args.num_samples is not None and self.args.horizon is not None:
            expect = int(self.args.num_samples) * int(self.args.horizon)
            if expect == n_points:
                ns, h = int(self.args.num_samples), int(self.args.horizon)
        return xy.reshape(ns, h, 2)

    def to_pixels(self, traj_xy: np.ndarray, ih: int, iw: int):
        scale = float(ih) / max(float(self.args.cam_meters), 1e-6)
        x = traj_xy[:, 0] * float(self.args.traj_scale)
        y = traj_xy[:, 1] * float(self.args.traj_scale)
        u = (iw * 0.5) - (y * scale)
        v = (ih * 1.0) - (x * scale)
        return u, v

    def warn_if_images_stop(self):
        now = time.time()
        if self.img_count != self._last_img_count_seen:
            self._last_img_count_seen = self.img_count
            return
        # no new image since last status; warn every 3s
        if (now - self._last_img_warn_t) > 3.0:
            self._last_img_warn_t = now
            self.get_logger().warn(
                f"No new images arriving on {self.args.image_topic}. "
                f"Try: ros2 topic info -v {self.args.image_topic} then run with --image-qos reliable|best_effort. "
                f"(current image_qos={self.args.image_qos})"
            )

    def status_log(self):
        now = time.time()
        if (now - self._last_status_t) < float(self.args.status_period_sec):
            return
        self._last_status_t = now
        self.get_logger().info(
            f"rx: img={self.img_count} | pg={len(self.pg)} nomad={len(self.nomad)} "
            f"wpN={len(self.wp_nomad)} wpF={len(self.wp_final)} | saved={self.saved}"
        )
        self.warn_if_images_stop()

    # ---- tick ----
    def tick(self):
        self.status_log()

        if self.img_count <= 0:
            return
        if (self.img_count % int(self.args.save_every)) != 0:
            return
        if self.img_count == self.last_saved_img_count:
            return
        if not self.image_fresh():
            return

        seq = self.latest_complete_seq()
        if seq is None:
            return

        img = self.img_rgb
        ih, iw = img.shape[:2]

        pg = self.split(self.pg[seq].xy)
        nomad = self.split(self.nomad[seq].xy) if (seq in self.nomad) else None

        S = pg.shape[0] if nomad is None else min(pg.shape[0], nomad.shape[0])
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(S)]

        # Left: NoMaD dashed
        self.axL.clear()
        self.axL.set_title(f"NoMaD raw (seq={seq}) dashed + waypoint_nomad")
        self.axL.imshow(img)
        if nomad is not None:
            for i in range(S):
                u, v = self.to_pixels(nomad[i], ih, iw)
                self.axL.plot(u, v, linewidth=2.5, alpha=0.95, color=colors[i], linestyle="--")
        if seq in self.wp_nomad:
            u, v = self.to_pixels(self.wp_nomad[seq].wp.reshape(1, 2), ih, iw)
            self.axL.scatter(u, v, marker="*", s=220, color="red", edgecolors="white", linewidths=1.5)
        self.axL.set_axis_off()

        # Right: PG solid
        self.axR.clear()
        self.axR.set_title(f"PathGuide constrained (seq={seq}) solid + waypoint")
        self.axR.imshow(img)
        for i in range(S):
            u, v = self.to_pixels(pg[i], ih, iw)
            self.axR.plot(u, v, linewidth=2.5, alpha=0.95, color=colors[i], linestyle="-")
        if seq in self.wp_final:
            u, v = self.to_pixels(self.wp_final[seq].wp.reshape(1, 2), ih, iw)
            self.axR.scatter(u, v, marker="*", s=220, color="lime", edgecolors="black", linewidths=1.5)
        self.axR.set_axis_off()

        self.fig.tight_layout()

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fname = f"overlay_seq{seq:06d}_{ts}_frame{self.img_count:06d}.{self.args.format}"
        fpath = os.path.join(self.args.out_dir, fname)
        self.fig.savefig(fpath, dpi=int(self.args.dpi), bbox_inches="tight")

        self.last_saved_img_count = self.img_count
        self.saved += 1
        if int(self.args.log_every) > 0 and (self.saved % int(self.args.log_every) == 0):
            self.get_logger().info(f"Saved {self.saved} files. Latest: {fpath}")

        if self.args.max_saves is not None and self.saved >= int(self.args.max_saves):
            self.get_logger().info(f"Reached max_saves={self.args.max_saves}. Exiting.")
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image-topic", default="/camera/realsense2_camera_node/color/image_raw")
    ap.add_argument("--nomad-topic", default="/sampled_actions_nomad")
    ap.add_argument("--pg-topic", default="/sampled_actions_pg")
    ap.add_argument("--waypoint-nomad-topic", default="/waypoint_nomad")
    ap.add_argument("--waypoint-topic", default="/waypoint")

    ap.add_argument("--out-dir", default="viz_overlays")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--max-saves", type=int, default=None)
    ap.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--status-period-sec", type=float, default=2.0)

    ap.add_argument("--tick-hz", type=float, default=20.0)
    ap.add_argument("--require-nomad", action="store_true", default=True)
    ap.add_argument("--require-wp", action="store_true", default=False)

    ap.add_argument("--cam-meters", type=float, default=2.0)
    ap.add_argument("--traj-scale", type=float, default=6.0)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)

    # Image QoS + stale gate
    ap.add_argument("--image-qos", default="reliable", choices=["reliable", "best_effort"])
    ap.add_argument("--image-depth", type=int, default=5)
    ap.add_argument("--image-stale-sec", type=float, default=10.0)
    ap.add_argument("--ignore-image-stale", action="store_true", help="Allow saving using last image even if stale.")

    args = ap.parse_args()

    rclpy.init()
    node = Saver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
