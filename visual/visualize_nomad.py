#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_navigate_overlay_v2.py

Headless visualizer for navigate.py (both with --guide and without --guide).

Fix vs v1:
- v1 only saved when (img_count % save_every == 0). If the image topic pauses at a non-multiple (e.g., img=34),
  saving would *never* trigger even if /sampled_actions and /waypoint continue.
- v2 adds "save on update": whenever new actions/waypoint arrive (timestamps change), it will save using the latest image.

Overlays:
- /sampled_actions  (sampled trajectories; colored)
- /waypoint         (final chosen waypoint; star)
on the latest RGB image.

Usage:
  python3 -u visualize_navigate_overlay_v2.py --out-dir viz --image-qos best_effort
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)

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
    """Parse optional 'seq:<int>' embedded in layout.dim[0].label."""
    try:
        dims = getattr(msg, "layout", None).dim
        if dims and isinstance(dims[0].label, str) and dims[0].label.startswith("seq:"):
            return int(dims[0].label.split(":", 1)[1])
    except Exception:
        pass
    return -1


def parse_actions_xy(msg: Float32MultiArray) -> Optional[np.ndarray]:
    """Return (N,2) or None. Tolerates a leading dummy 0."""
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
    """Heuristic factorization: total points = num_samples * horizon."""
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
        super().__init__("viz_navigate_overlay_saver")
        self.args = args
        self.bridge = CvBridge()
        os.makedirs(self.args.out_dir, exist_ok=True)

        self.img_rgb: Optional[np.ndarray] = None
        self.t_img: float = 0.0
        self.img_count: int = 0

        self.latest_actions: Optional[ActionPacket] = None
        self.latest_wp: Optional[WaypointPacket] = None

        self.last_saved_img_count = -1
        self.last_saved_actions_t = 0.0
        self.last_saved_wp_t = 0.0
        self.saved = 0

        self._last_status_t = 0.0
        self._last_img_count_seen = 0
        self._last_img_warn_t = 0.0

        # QoS
        # Images are often sensor-data QoS; but allow override.
        if self.args.image_qos_profile == "sensor_data":
            image_qos = qos_profile_sensor_data
        else:
            image_qos = make_qos(self.args.image_qos, self.args.image_depth)
        arr_qos = qos_profile_sensor_data  # best-effort, volatile by default

        self.create_subscription(Image, self.args.image_topic, self.cb_img, image_qos)
        self.create_subscription(Float32MultiArray, self.args.actions_topic, self.cb_actions, arr_qos)
        self.create_subscription(Float32MultiArray, self.args.waypoint_topic, self.cb_wp, arr_qos)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
        self.timer = self.create_timer(1.0 / float(self.args.tick_hz), self.tick)

        self.get_logger().info(
            f"Headless saver started. out_dir={self.args.out_dir}, save_every={self.args.save_every}, "
            f"save_on_update={self.args.save_on_update}, image_qos={self.args.image_qos}, "
            f"image_qos_profile={self.args.image_qos_profile}, require_waypoint={self.args.require_waypoint}"
        )

    # ---- callbacks ----
    def cb_img(self, msg: Image):
        try:
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

    def cb_actions(self, msg: Float32MultiArray):
        xy = parse_actions_xy(msg)
        if xy is None:
            return
        self.latest_actions = ActionPacket(time.time(), parse_seq(msg), xy)

    def cb_wp(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        wp = np.asarray([msg.data[0], msg.data[1]], dtype=np.float32)
        self.latest_wp = WaypointPacket(time.time(), parse_seq(msg), wp)

    # ---- helpers ----
    def image_fresh(self) -> bool:
        if self.img_rgb is None:
            return False
        if self.args.ignore_image_stale:
            return True
        return (time.time() - self.t_img) <= float(self.args.image_stale_sec)

    def split(self, xy: np.ndarray) -> np.ndarray:
        n_points = int(xy.shape[0])
        ns, h = infer_samples_horizon(n_points)
        if self.args.num_samples is not None and self.args.horizon is not None:
            expect = int(self.args.num_samples) * int(self.args.horizon)
            if expect == n_points:
                ns, h = int(self.args.num_samples), int(self.args.horizon)
        return xy.reshape(ns, h, 2)

    def to_pixels(self, traj_xy: np.ndarray, ih: int, iw: int):
        """
        Simple BEV projection on the image:
          - x forward (meters), y left (meters)
          - image origin at bottom-center
        """
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
        if (now - self._last_img_warn_t) > 3.0:
            self._last_img_warn_t = now
            self.get_logger().warn(
                f"No new images on {self.args.image_topic}. "
                f"Try: ros2 topic info -v {self.args.image_topic} then run with "
                f"--image-qos reliable|best_effort or --image-qos-profile sensor_data. "
                f"(current image_qos={self.args.image_qos}, image_qos_profile={self.args.image_qos_profile})"
            )

    def status_log(self):
        now = time.time()
        if (now - self._last_status_t) < float(self.args.status_period_sec):
            return
        self._last_status_t = now
        a = 0 if self.latest_actions is None else int(self.latest_actions.xy.shape[0])
        w = "no" if self.latest_wp is None else "yes"
        self.get_logger().info(f"rx: img={self.img_count} | actions_pts={a} | waypoint={w} | saved={self.saved}")
        self.warn_if_images_stop()

    def should_save(self) -> bool:
        if self.img_rgb is None:
            return False
        if self.latest_actions is None:
            return False
        if self.args.require_waypoint and (self.latest_wp is None):
            return False
        if not self.image_fresh():
            # allow saving with stale image only if user opts in
            return False

        # Mode 1: frame-based (original behavior)
        frame_trigger = False
        if self.img_count > 0 and int(self.args.save_every) > 0:
            if (self.img_count % int(self.args.save_every)) == 0 and self.img_count != self.last_saved_img_count:
                frame_trigger = True

        # Mode 2: save on update (new actions/waypoint arrived since last save)
        update_trigger = False
        if self.args.save_on_update:
            if self.latest_actions.t > self.last_saved_actions_t:
                update_trigger = True
            if self.latest_wp is not None and self.latest_wp.t > self.last_saved_wp_t:
                update_trigger = True

        return frame_trigger or update_trigger

    # ---- tick ----
    def tick(self):
        self.status_log()

        if not self.should_save():
            return

        img = self.img_rgb
        ih, iw = img.shape[:2]

        actions = self.split(self.latest_actions.xy)  # (S,H,2)
        # navigate.py publishes raw sampled actions; for visualization we often want a *trajectory*.
        # Optionally interpret actions as per-step deltas and integrate (cumsum) to get positions.
        if self.args.actions_are_deltas:
            actions = np.cumsum(actions, axis=1)

        # Auto-scale: navigate.py may scale /waypoint (when model_params["normalize"]) but not /sampled_actions.
        # We infer a scalar factor so that actions[0, waypoint_index] matches the received waypoint.
        scale_factor = 1.0
        if self.args.auto_scale_actions and (self.latest_wp is not None):
            k = int(self.args.waypoint_index)
            k = max(0, min(k, actions.shape[1]-1))
            a_ref = actions[0, k].astype(np.float32)
            w_ref = self.latest_wp.wp.astype(np.float32)
            # robust component-wise ratio
            ratios = []
            for i in range(2):
                if abs(a_ref[i]) > 1e-6:
                    ratios.append(float(w_ref[i] / a_ref[i]))
            if ratios:
                # use median to reduce outliers
                scale_factor = float(np.median(np.asarray(ratios)))
                # avoid insane scales
                if not np.isfinite(scale_factor) or abs(scale_factor) > 1e3 or abs(scale_factor) < 1e-3:
                    scale_factor = 1.0
            actions = actions * scale_factor

        S = actions.shape[0]

        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(S)]

        if self.args.debug_stats:
            a_min = np.min(actions.reshape(-1,2), axis=0)
            a_max = np.max(actions.reshape(-1,2), axis=0)
            if self.latest_wp is not None:
                w = self.latest_wp.wp
                self.get_logger().info(
                    f"debug: actions_xy min={a_min.tolist()} max={a_max.tolist()} | waypoint={w.tolist()} | scale={scale_factor:.4f}"
                )
            else:
                self.get_logger().info(
                    f"debug: actions_xy min={a_min.tolist()} max={a_max.tolist()} | waypoint=None | scale={scale_factor:.4f}"
                )

        self.ax.clear()
        seq_a = self.latest_actions.seq
        seq_w = -1 if self.latest_wp is None else self.latest_wp.seq
        if not self.args.no_title:
            self.ax.set_title(f"navigate.py overlay | actions_seq={seq_a} waypoint_seq={seq_w}")
        self.ax.imshow(img)
        # IMPORTANT: prevent matplotlib autoscale from shrinking the camera image into a corner.
        # We lock axis limits to the image pixel bounds and disable autoscale.
        self.ax.set_xlim(0, iw)
        self.ax.set_ylim(ih, 0)
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect("auto")
        if self.args.full_frame:
            # Make the axes occupy the whole figure canvas (minimize white margins)
            self.ax.set_position([0.0, 0.0, 1.0, 1.0])
            self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        # trajectories
        for i in range(S):
            u, v = self.to_pixels(actions[i], ih, iw)
            self.ax.plot(u, v, linewidth=2.2, alpha=0.95, color=colors[i])

        # waypoint
        if self.latest_wp is not None:
            u, v = self.to_pixels(self.latest_wp.wp.reshape(1, 2), ih, iw)
            self.ax.scatter(u, v, marker="*", s=240, color="lime", edgecolors="black", linewidths=1.5)

        self.ax.set_axis_off()
        # No tight_layout: it can introduce extra margins; we want the image to fill the canvas.
        

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fname = f"navigate_overlay_{ts}_img{self.img_count:06d}.{self.args.format}"
        fpath = os.path.join(self.args.out_dir, fname)
        self.fig.savefig(fpath, dpi=int(self.args.dpi), bbox_inches="tight", pad_inches=0.0)

        # update "last saved" markers
        self.last_saved_img_count = self.img_count
        self.last_saved_actions_t = max(self.last_saved_actions_t, self.latest_actions.t if self.latest_actions else 0.0)
        self.last_saved_wp_t = max(self.last_saved_wp_t, self.latest_wp.t if self.latest_wp else 0.0)
        self.saved += 1

        if int(self.args.log_every) > 0 and (self.saved % int(self.args.log_every) == 0):
            self.get_logger().info(f"Saved {self.saved} files. Latest: {fpath}")

        if self.args.max_saves is not None and self.saved >= int(self.args.max_saves):
            self.get_logger().info(f"Reached max_saves={self.args.max_saves}. Exiting.")
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description="Headless overlay visualizer for navigate.py (guide or no-guide).")

    ap.add_argument("--image-topic", default="/camera/realsense2_camera_node/color/image_raw")
    ap.add_argument("--actions-topic", default="/sampled_actions")
    ap.add_argument("--waypoint-topic", default="/waypoint")

    ap.add_argument("--out-dir", default="viz_navigate")
    ap.add_argument("--save-every", type=int, default=10, help="Frame-based trigger: every N images.")
    ap.add_argument("--save-on-update", action="store_true", default=True,
                    help="Also save when new actions/waypoint arrive (recommended).")
    ap.add_argument("--max-saves", type=int, default=None)
    ap.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--status-period-sec", type=float, default=2.0)

    # QoS / staleness (camera topic often needs BEST_EFFORT or sensor_data)
    ap.add_argument("--image-qos", default="best_effort", choices=["reliable", "best_effort"])
    ap.add_argument("--image-qos-profile", default="custom", choices=["custom", "sensor_data"],
                    help="Use sensor_data QoS directly if needed.")
    ap.add_argument("--image-depth", type=int, default=5)
    ap.add_argument("--image-stale-sec", type=float, default=10.0)
    ap.add_argument("--ignore-image-stale", action="store_true", default=False)

    # plotting geometry
    ap.add_argument("--cam-meters", type=float, default=2.0, help="Vertical FOV meters to map to image height.")
    ap.add_argument("--traj-scale", type=float, default=6.0, help="Extra scale factor for traj/waypoint in pixel projection.")

    # optional: force split
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)


    ap.add_argument("--waypoint-index", type=int, default=2,
                    help="Index k used by navigate.py for chosen waypoint (default navigate.py --waypoint=2).")
    ap.add_argument("--actions-are-deltas", action="store_true", default=True,
                    help="Treat sampled actions as per-step deltas and cumsum to get a trajectory (recommended for navigate.py).")
    ap.add_argument("--auto-scale-actions", action="store_true", default=True,
                    help="Auto-scale sampled actions to match waypoint units using the chosen waypoint index.")
    ap.add_argument("--debug-stats", action="store_true", default=False,
                    help="Log min/max of actions/waypoint and the inferred scale factor.")

    ap.add_argument("--require-waypoint", action="store_true", default=False)
    ap.add_argument("--tick-hz", type=float, default=20.0)

    # layout
    ap.add_argument("--full-frame", action="store_true", default=True,
                    help="Fill the canvas with the camera image (minimize white margins).")
    ap.add_argument("--no-title", action="store_true", default=True,
                    help="Disable title to avoid extra top margin.")

    args = ap.parse_args()

    rclpy.init()
    node = Saver(args)
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
