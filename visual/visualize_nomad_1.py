#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_nomad_seqsync_async.py

Your checks show:
- /camera/realsense2_camera_node/color/image_raw is publishing steadily (~15–28 Hz)
- Publisher QoS is RELIABLE
So if the visualizer output keeps using the *same* frame, the camera is not the problem.
The common cause is: matplotlib savefig is heavy and blocks rclpy's single-threaded executor,
so the node stops processing new Image callbacks.

This async version fixes it by rendering/saving in a background worker thread.

No changes needed in navigate.py.
"""

import argparse
import os
import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Deque, List

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

import queue
import threading


def make_qos(reliability: str, depth: int) -> QoSProfile:
    rel = ReliabilityPolicy.RELIABLE if reliability.lower() == "reliable" else ReliabilityPolicy.BEST_EFFORT
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=rel,
        durability=DurabilityPolicy.VOLATILE,
    )


def parse_actions_xy(msg: Float32MultiArray) -> Optional[np.ndarray]:
    arr = np.asarray(msg.data, dtype=np.float32)
    if arr.size == 0:
        return None
    if (arr.size % 2) != 0:
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
    viz_seq: int
    xy: np.ndarray


@dataclass
class ImagePacket:
    t: float
    img_count: int
    rgb: np.ndarray


@dataclass
class SaveJob:
    viz_seq: int
    pair_dt: float
    img_count: int
    img_rgb: np.ndarray
    actions: np.ndarray  # (S,H,2)
    waypoint: np.ndarray


class Viz(Node):
    def __init__(self, args):
        super().__init__("viz_nomad_seqsync_async_saver")
        self.args = args
        self.bridge = CvBridge()
        os.makedirs(self.args.out_dir, exist_ok=True)

        self._viz_seq = 0
        self.actions_q: Deque[ActionPacket] = deque(maxlen=int(self.args.max_action_queue))
        self.images_q: Deque[ImagePacket] = deque(maxlen=int(self.args.max_image_queue))
        self.latest_img_count = 0

        # worker queue
        self.save_q: "queue.Queue[SaveJob]" = queue.Queue(maxsize=int(self.args.save_queue_size))
        self._stop_evt = threading.Event()
        self.worker = threading.Thread(target=self._save_worker, daemon=True)
        self.worker.start()

        # QoS: your publisher is RELIABLE (ros2 topic info -v confirms)
        if self.args.image_qos_profile == "sensor_data":
            image_qos = qos_profile_sensor_data
        else:
            image_qos = make_qos(self.args.image_qos, self.args.image_depth)
        arr_qos = qos_profile_sensor_data

        self.create_subscription(Image, self.args.image_topic, self.cb_img, image_qos)
        self.create_subscription(Float32MultiArray, self.args.actions_topic, self.cb_actions, arr_qos)
        self.create_subscription(Float32MultiArray, self.args.waypoint_topic, self.cb_wp, arr_qos)

        self.timer = self.create_timer(float(self.args.status_period_sec), self.status)

        self.get_logger().info(
            f"Async viz started. out_dir={self.args.out_dir} | image_qos={self.args.image_qos} "
            f"(profile={self.args.image_qos_profile}) | sync_dt={self.args.sync_dt}s | pair_mode={self.args.pair_mode} | dpi={self.args.dpi}"
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
            t = time.time()
            self.latest_img_count += 1
            self.images_q.append(ImagePacket(t=t, img_count=self.latest_img_count, rgb=rgb))
        except Exception as e:
            self.get_logger().warn(f"image decode failed: {e}")

    def cb_actions(self, msg: Float32MultiArray):
        xy = parse_actions_xy(msg)
        if xy is None:
            return
        self._viz_seq += 1
        self.actions_q.append(ActionPacket(t=time.time(), viz_seq=self._viz_seq, xy=xy))

    def cb_wp(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        wp = np.asarray([msg.data[0], msg.data[1]], dtype=np.float32)
        t_wp = time.time()
        pair = self.try_pair(t_wp, wp)
        if pair is not None:
            self.enqueue_job(*pair)

    # ---- pairing ----
    def try_pair(self, t_wp: float, wp: np.ndarray):
        if not self.actions_q:
            return None

        dt_max = float(self.args.sync_dt)
        candidates: List[ActionPacket] = list(self.actions_q)

        chosen: Optional[ActionPacket] = None
        chosen_dt = 1e9

        if self.args.pair_mode == "after":
            for a in reversed(candidates):
                dt = t_wp - a.t
                if dt < 0:
                    continue
                if dt <= dt_max:
                    chosen = a
                    chosen_dt = dt
                    break
        else:
            for a in candidates:
                dt = abs(t_wp - a.t)
                if dt <= dt_max and dt < chosen_dt:
                    chosen = a
                    chosen_dt = dt

        if chosen is None:
            return None

        # drop old actions up to chosen
        while self.actions_q and self.actions_q[0].viz_seq <= chosen.viz_seq:
            self.actions_q.popleft()

        # image snapshot
        img_pkt = None
        if self.args.match_image:
            img_pkt = self.match_image(chosen.t, float(self.args.image_sync_dt))
            if self.args.require_image and (img_pkt is None):
                return None
        else:
            img_pkt = self.images_q[-1] if self.images_q else None
            if self.args.require_image and (img_pkt is None):
                return None

        if img_pkt is None:
            return None

        return (chosen.viz_seq, float(chosen_dt), chosen.xy, wp, img_pkt)

    def match_image(self, target_t: float, dt_max: float) -> Optional[ImagePacket]:
        if not self.images_q:
            return None
        best = None
        best_dt = 1e9
        for im in self.images_q:
            dt = abs(im.t - target_t)
            if dt < best_dt:
                best_dt = dt
                best = im
        if best is None or best_dt > dt_max:
            return None
        return best

    # ---- action processing ----
    def split(self, xy: np.ndarray) -> np.ndarray:
        n_points = int(xy.shape[0])
        ns, h = infer_samples_horizon(n_points)
        if self.args.num_samples is not None and self.args.horizon is not None:
            expect = int(self.args.num_samples) * int(self.args.horizon)
            if expect == n_points:
                ns, h = int(self.args.num_samples), int(self.args.horizon)
        return xy.reshape(ns, h, 2)

    def prepare_actions(self, actions_xy: np.ndarray, waypoint: np.ndarray):
        actions = self.split(actions_xy)
        if self.args.actions_are_deltas:
            actions = np.cumsum(actions, axis=1)

        if self.args.auto_scale_actions:
            k = int(self.args.waypoint_index)
            k = max(0, min(k, actions.shape[1] - 1))
            a_ref = actions[0, k].astype(np.float32)
            w_ref = waypoint.astype(np.float32)
            ratios = []
            for i in range(2):
                if abs(a_ref[i]) > 1e-6:
                    ratios.append(float(w_ref[i] / a_ref[i]))
            if ratios:
                scale_factor = float(np.median(np.asarray(ratios)))
                if np.isfinite(scale_factor) and 1e-3 < abs(scale_factor) < 1e3:
                    actions = actions * scale_factor
        return actions

    def enqueue_job(self, viz_seq: int, pair_dt: float, actions_xy: np.ndarray, waypoint: np.ndarray, img_pkt: ImagePacket):
        actions = self.prepare_actions(actions_xy, waypoint)
        job = SaveJob(
            viz_seq=int(viz_seq),
            pair_dt=float(pair_dt),
            img_count=int(img_pkt.img_count),
            img_rgb=img_pkt.rgb.copy(),
            actions=actions.copy(),
            waypoint=waypoint.copy(),
        )
        try:
            self.save_q.put_nowait(job)
        except queue.Full:
            try:
                _ = self.save_q.get_nowait()
                self.save_q.put_nowait(job)
            except Exception:
                pass

    # ---- plotting (worker thread) ----
    def to_pixels(self, traj_xy: np.ndarray, ih: int, iw: int):
        scale = float(ih) / max(float(self.args.cam_meters), 1e-6)
        x = traj_xy[:, 0] * float(self.args.traj_scale)
        y = traj_xy[:, 1] * float(self.args.traj_scale)
        u = (iw * 0.5) - (y * scale)
        v = (ih * 1.0) - (x * scale)
        return u, v

    def _save_worker(self):
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
        cmap = plt.get_cmap("tab20")
        saved = 0

        while not self._stop_evt.is_set():
            try:
                job: SaveJob = self.save_q.get(timeout=0.2)
            except queue.Empty:
                continue

            img = job.img_rgb
            ih, iw = img.shape[:2]

            ax.clear()
            if not self.args.no_title:
                ax.set_title(f"viz_seq{job.viz_seq} | pair_dt={job.pair_dt*1000:.0f}ms | img#{job.img_count}")
            ax.imshow(img)

            ax.set_xlim(0, iw)
            ax.set_ylim(ih, 0)
            ax.set_autoscale_on(False)
            ax.set_aspect("auto")
            if self.args.full_frame:
                ax.set_position([0.0, 0.0, 1.0, 1.0])
                fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

            S = job.actions.shape[0]
            for i in range(S):
                u, v = self.to_pixels(job.actions[i], ih, iw)
                ax.plot(u, v, linewidth=2.2, alpha=0.95, color=cmap(i % 20))

            u, v = self.to_pixels(job.waypoint.reshape(1, 2), ih, iw)
            ax.scatter(u, v, marker="*", s=240, color="lime", edgecolors="black", linewidths=1.5)

            ax.set_axis_off()

            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            fname = f"navigate_overlay_vizseq{job.viz_seq:06d}_{ts}_img{job.img_count:06d}.{self.args.format}"
            fpath = os.path.join(self.args.out_dir, fname)
            fig.savefig(fpath, dpi=int(self.args.dpi), bbox_inches="tight", pad_inches=0.0)

            saved += 1
            self.save_q.task_done()

    def status(self):
        self.get_logger().info(
            f"rx: img_count={self.latest_img_count} | images_buf={len(self.images_q)} | actions_buf={len(self.actions_q)} | save_q={self.save_q.qsize()}"
        )

    def destroy_node(self):
        self._stop_evt.set()
        try:
            super().destroy_node()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="navigate.py visualizer with pairing + async saving.")

    ap.add_argument("--image-topic", default="/camera/realsense2_camera_node/color/image_raw")
    ap.add_argument("--actions-topic", default="/sampled_actions")
    ap.add_argument("--waypoint-topic", default="/waypoint")

    ap.add_argument("--out-dir", default="viz_nomad_sync")
    ap.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--dpi", type=int, default=60)
    ap.add_argument("--status-period-sec", type=float, default=2.0)

    # pairing
    ap.add_argument("--sync-dt", type=float, default=0.35)
    ap.add_argument("--pair-mode", default="after", choices=["after", "nearest"])
    ap.add_argument("--max-action-queue", type=int, default=80)

    # image matching
    ap.add_argument("--match-image", action="store_true", default=False)
    ap.add_argument("--image-sync-dt", type=float, default=0.25)
    ap.add_argument("--require-image", action="store_true", default=False)
    ap.add_argument("--max-image-queue", type=int, default=20)

    # QoS
    ap.add_argument("--image-qos", default="reliable", choices=["reliable", "best_effort"])
    ap.add_argument("--image-qos-profile", default="custom", choices=["custom", "sensor_data"])
    ap.add_argument("--image-depth", type=int, default=3)

    # trajectory
    ap.add_argument("--waypoint-index", type=int, default=2)
    ap.add_argument("--actions-are-deltas", action="store_true", default=True)
    ap.add_argument("--auto-scale-actions", action="store_true", default=True)

    # projection
    ap.add_argument("--cam-meters", type=float, default=2.0)
    ap.add_argument("--traj-scale", type=float, default=6.0)

    # plot layout
    ap.add_argument("--full-frame", action="store_true", default=True)
    ap.add_argument("--no-title", action="store_true", default=True)

    # optional force split
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)

    # async
    ap.add_argument("--save-queue-size", type=int, default=8)

    args = ap.parse_args()

    rclpy.init()
    node = Viz(args)
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
