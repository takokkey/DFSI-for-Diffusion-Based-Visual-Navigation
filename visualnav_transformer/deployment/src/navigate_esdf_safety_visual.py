#!/usr/bin/env python3
import argparse
import os
import time
import threading
import queue
import copy

import numpy as np
import rclpy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from rclpy.node import Node
from PIL import Image as PILImage

# ROS
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String, Float32
from nav_msgs.msg import OccupancyGrid

# UTILS
from visualnav_transformer.deployment.src.topic_names import (
    IMAGE_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    WAYPOINT_TOPIC,
)

# ---- debug/safety globals (module-level defaults; may be overridden by CLI) ----
_DEBUG_COSTMAP_STATS = False
_SAFETY_DEBUG = False
_COSTMAP_LAST_LOG_T = 0.0
# Throttle for costmap debug logs (used when --debug-tsdf is enabled).
_COSTMAP_LOG_PERIOD_SEC = 1.0

from visualnav_transformer.deployment.src.utils import (
    load_model,
    msg_to_pil,
    to_numpy,
    transform_images,
)
from visualnav_transformer.train.vint_train.training.train_utils import get_action

# PathGuide / TSDF
from guide import PathGuide
from costmap_cfg import CostMapConfig  # 保留用于与 TSDF 配置对齐

# CONSTANTS
MODEL_WEIGHTS_PATH = "model_weights"
ROBOT_CONFIG_PATH = "config/robot.yaml"
MODEL_CONFIG_PATH = "config/models.yaml"
TOPOMAP_IMAGES_DIR = "topomaps/images"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# GLOBALS
context_queue = []
context_size = None

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------------------
# TSDF costmap globals
# ---------------------------------------------------------------------
_tsdf_lock = threading.Lock()
_latest_cost_map = {"arr": None, "origin": (0.0, 0.0), "res": None, "t_wall": 0.0}
pathguide = None  # will init later

# Map FSM state (from map_adapter_rclpy_esdf)
_state_lock = threading.Lock()
_latest_map_state = "S0"  # "S0" | "S1" | "S2"

# ESDF normalization distance (published by map_adapter_rclpy_esdf.py)
_dmax_lock = threading.Lock()
_latest_dmax_safe = None  # meters
_latest_dmax_wall = 0.0

# Front scan emergency signal (optional hard safety backstop)
_scan_lock = threading.Lock()
_latest_front_range = None  # meters
_latest_scan_wall = 0.0
_scan_front_half_angle_rad = np.deg2rad(20.0)  # will be overwritten by args



def map_fsm_state_callback(msg: String):
    global _latest_map_state
    data = msg.data.strip() if msg.data is not None else "S0"
    # Only accept known states to be safe
    with _state_lock:
        _latest_map_state = data if data in ("S0", "S1", "S2") else "S0"



def esdf_dmax_callback(msg: Float32):
    """Receive ESDF normalization distance d_max_safe (meters)."""
    global _latest_dmax_safe, _latest_dmax_wall
    try:
        v = float(msg.data)
        with _dmax_lock:
            _latest_dmax_safe = v
            _latest_dmax_wall = time.time()
    except Exception:
        pass


def scan_front_callback(msg: LaserScan):
    """Compute min range in a small front sector as a last-resort safety backstop."""
    global _latest_front_range, _latest_scan_wall, _scan_front_half_angle_rad
    try:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return
        angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
        finite = np.isfinite(ranges)
        in_front = np.abs(angles) <= float(_scan_front_half_angle_rad)
        valid = finite & in_front & (ranges > 0.0)
        if not np.any(valid):
            return
        rmin = float(np.min(ranges[valid]))
        with _scan_lock:
            _latest_front_range = rmin
            _latest_scan_wall = time.time()
    except Exception:
        pass

def tsdf_callback_local(msg: OccupancyGrid):
    """
    把 OccupancyGrid 转成 cost_map numpy 并注入 PathGuide。

    约定（与 map_adapter_rclpy / map_adapter_rclpy_esdf 保持一致）：
      - map_adapter 发布的 OccupancyGrid.data 是 occ.T.flatten()
        其中 occ.shape = [num_x, num_y]，flatten 前内存布局是 [num_y, num_x]。
      - 这里 reshape((h,w)).T 后得到 cost_grid.shape = [w,h] = [num_x,num_y]，
        与 PathGuide.set_cost_map 里 tsdf_cost_map.num_x/num_y 完全对齐。
    """
    global pathguide
    try:
        w = int(msg.info.width)
        h = int(msg.info.height)
        arr = np.array(msg.data, dtype=np.int8)
        if arr.size == 0:
            print("[navigate] /tsdf_costmap received empty data, skip.")
            return

        if arr.size != w * h:
            print(
                f"[navigate] tsdf_callback_local size mismatch: "
                f"len(data)={arr.size}, width*height={w*h}, try best-effort reshape."
            )

        try:
            # data 是 row-major [h,w]（发布端用了 .T），所以 reshape(h,w).T -> [w,h]
            cost_grid = arr.reshape((h, w)).T.astype(np.float32) / 100.0
        except Exception as e:
            print(
                f"[navigate] reshape(h,w).T failed ({e}), "
                f"fallback reshape(w,h) without T."
            )
            cost_grid = arr.reshape((w, h)).astype(np.float32) / 100.0

        origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        with _tsdf_lock:
            _latest_cost_map["arr"] = cost_grid
            _latest_cost_map["origin"] = origin
            _latest_cost_map["res"] = float(msg.info.resolution) if msg.info.resolution else None
            _latest_cost_map["t_wall"] = time.time()

        if pathguide is not None:
            try:
                pathguide.set_cost_map(cost_grid, start_x=origin[0], start_y=origin[1])
                cmin = float(np.min(cost_grid))
                cmax = float(np.max(cost_grid))
                print(
                    f"[navigate] PathGuide cost_map set shape={cost_grid.shape}, "
                    f"origin=({origin[0]:.2f},{origin[1]:.2f}), "
                    f"cost[min,max]=({cmin:.3f},{cmax:.3f})"
                )

                if _DEBUG_COSTMAP_STATS:
                    global _COSTMAP_LAST_LOG_T
                    now = time.time()
                    if (now - _COSTMAP_LAST_LOG_T) > _COSTMAP_LOG_PERIOD_SEC:
                        _COSTMAP_LAST_LOG_T = now
                        cg = cost_grid
                        flat = cg.reshape(-1)
                        flat = flat[np.isfinite(flat)]
                        if flat.size > 0:
                            nz = float(np.mean(flat > 0.0))
                            mid = float(np.mean((flat > 0.02) & (flat < 0.98)))
                            occ = float(np.mean(flat >= 0.999))
                            mean = float(np.mean(flat))
                            p50, p90, p99 = [float(x) for x in np.percentile(flat, [50, 90, 99])]
                            # Gradient magnitude (rough) to sanity-check smoothness.
                            gx, gy = np.gradient(cg.astype(np.float32))
                            gmean = float(np.mean(np.sqrt(gx * gx + gy * gy)))
                            uniq = int(np.unique(np.round(flat, 3)).size)
                            print(
                                f"[navigate][costmap] nz={nz:.4f} mid={mid:.4f} occ={occ:.4f} "
                                f"mean={mean:.4f} p50={p50:.4f} p90={p90:.4f} p99={p99:.4f} "
                                f"grad_mean={gmean:.6g} uniq~={uniq}"
                            )
                        else:
                            print("[navigate][costmap] empty/NaN costmap")
            except Exception as e:
                print("[navigate] set_cost_map failed:", e)
    except Exception as e:
        print("[navigate] tsdf_callback_local exception:", e)


def callback_obs(msg: Image):
    global context_queue
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        context_queue.append(obs_img)
        if len(context_queue) > context_size + 1:
            context_queue.pop(0)


class NavigationNode(Node):
    def __init__(self):
        super().__init__("navigation_node")
        self.create_subscription(Image, IMAGE_TOPIC, callback_obs, 10)
        self.create_subscription(OccupancyGrid, "/tsdf_costmap", tsdf_callback_local, 1)
        # Subscribe to TSDF/ESDF map FSM state (S0/S1/S2)
        self.create_subscription(String, "/map_fsm_state", map_fsm_state_callback, 10)
        # ESDF normalization distance for safety filter
        self.create_subscription(Float32, "/esdf_d_max_safe", esdf_dmax_callback, 10)
        # Optional scan-based emergency backstop
        self.create_subscription(LaserScan, "/scan", scan_front_callback, 10)

        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )
        # Compare/ablation: publish raw NoMaD vs PathGuide candidates (for visualization)
        self.sampled_actions_nomad_pub = self.create_publisher(
            Float32MultiArray, "/sampled_actions_nomad", 1
        )
        self.sampled_actions_pg_pub = self.create_publisher(
            Float32MultiArray, "/sampled_actions_pg", 1
        )
        self.waypoint_nomad_pub = self.create_publisher(
            Float32MultiArray, "/waypoint_nomad", 1
        )




# ---------------------------------------------------------------------
# Compare helper: compute raw NoMaD candidates asynchronously
# ---------------------------------------------------------------------
class CompareNomadWorker:
    """Background worker that computes raw NoMaD (no guidance) trajectories.

    Doing an extra denoise pass every planning step can significantly reduce the
    control rate. This worker keeps the *executed* (PathGuide-guided) trajectory
    in the main loop, while computing raw NoMaD in a background thread for viz.
    """

    def __init__(self, node: Node, model, scheduler, num_diffusion_iters: int, device, args):
        self.node = node
        self.model = model
        self.scheduler = scheduler
        self.num_diffusion_iters = int(num_diffusion_iters)
        self.device = device
        self.args = args

        self._q: "queue.Queue" = queue.Queue(maxsize=1)  # keep latest only
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name="compare_nomad_worker", daemon=True)
        self._started = False

    def start(self):
        if not self._started:
            self._started = True
            self._th.start()

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        try:
            if self._started:
                self._th.join(timeout=1.0)
        except Exception:
            pass

    def submit(self, noisy_action: torch.Tensor, obs_cond: torch.Tensor, waypoint_idx: int, normalize: bool, seq: int = -1) -> bool:
        """Submit a job (drops older job if busy)."""
        if not self._started:
            self.start()

        job = (noisy_action.detach().clone(), obs_cond.detach().clone(), int(waypoint_idx), bool(normalize), int(seq))

        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            # drop old, keep latest
            try:
                _ = self._q.get_nowait()
            except Exception:
                pass
            try:
                self._q.put_nowait(job)
                return True
            except Exception:
                return False

    def _denoise_nomad_raw(self, noisy_init: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        naction_local = noisy_init
        self.scheduler.set_timesteps(self.num_diffusion_iters)
        with torch.inference_mode():
            for k in self.scheduler.timesteps[:]:
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction_local,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction_local = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction_local,
                ).prev_sample
        return naction_local

    def _run(self):
        while not self._stop.is_set():
            job = self._q.get()
            if job is None:
                continue
            noisy_action, obs_cond, waypoint_idx, normalize, seq = job
            try:
                naction_nomad = self._denoise_nomad_raw(noisy_action, obs_cond)

                # Publish raw NoMaD candidates for visualization
                na_nomad_np = to_numpy(get_action(naction_nomad))
                if normalize:
                    _scale = MAX_V / RATE
                    na_nomad_pub = na_nomad_np * _scale
                else:
                    na_nomad_pub = na_nomad_np

                msg_nomad = Float32MultiArray()
                if seq >= 0: msg_nomad.layout.dim = [MultiArrayDimension(label=f"seq:{seq}", size=0, stride=0)]
                msg_nomad.data = na_nomad_pub.astype(np.float32).reshape(-1).tolist()
                self.node.sampled_actions_nomad_pub.publish(msg_nomad)

                wp_nomad = na_nomad_pub[0, waypoint_idx].copy()
                msg_wp_nomad = Float32MultiArray()
                if seq >= 0: msg_wp_nomad.layout.dim = [MultiArrayDimension(label=f"seq:{seq}", size=0, stride=0)]
                msg_wp_nomad.data = [float(wp_nomad[0]), float(wp_nomad[1])]
                self.node.waypoint_nomad_pub.publish(msg_wp_nomad)

            except Exception as e:
                # Never crash the control loop due to compare
                try:
                    print("[compare_nomad_worker] exception:", e)
                except Exception:
                    pass

# ---------------------------------------------------------------------
# Safety filter helpers (ESDF-first) + a simple recovery primitive
# ---------------------------------------------------------------------
def _get_latest_map_snapshot(max_stale_sec: float):
    """Return (cost01, origin_xy, res, age_sec) or (None, None, None, None) if unavailable/stale."""
    with _tsdf_lock:
        arr = _latest_cost_map.get("arr", None)
        origin = _latest_cost_map.get("origin", (0.0, 0.0))
        res = _latest_cost_map.get("res", None)
        t_wall = float(_latest_cost_map.get("t_wall", 0.0))
    if arr is None or res is None:
        return None, None, None, None
    age = time.time() - t_wall
    if max_stale_sec is not None and max_stale_sec > 0.0 and age > max_stale_sec:
        return None, None, None, None
    return arr, origin, float(res), float(age)


def _get_latest_dmax(default_dmax: float):
    with _dmax_lock:
        v = _latest_dmax_safe
    if v is None or not np.isfinite(v) or v <= 0.0:
        return float(default_dmax)
    return float(v)


def _get_latest_front_range(max_stale_sec: float):
    with _scan_lock:
        r = _latest_front_range
        t = _latest_scan_wall
    if r is None:
        return None
    age = time.time() - float(t)
    if max_stale_sec is not None and max_stale_sec > 0.0 and age > max_stale_sec:
        return None
    return float(r)


def _inside_map(x: float, y: float, origin_xy, res: float, shape_xy):
    ox, oy = origin_xy
    nx, ny = int(shape_xy[0]), int(shape_xy[1])
    x_min = ox
    y_min = oy
    x_max = ox + nx * res
    y_max = oy + ny * res
    return (x >= x_min) and (x < x_max) and (y >= y_min) and (y < y_max)


def _sample_cost01(cost01: np.ndarray, origin_xy, res: float, x: float, y: float):
    """cost01 in [0,1]. Out-of-bounds treated as obstacle (1.0)."""
    ox, oy = origin_xy
    nx, ny = cost01.shape  # [num_x, num_y]
    ix = int((x - ox) / res)
    iy = int((y - oy) / res)
    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
        return 1.0
    v = float(cost01[ix, iy])
    if not np.isfinite(v):
        return 1.0
    return float(np.clip(v, 0.0, 1.0))


def _cost01_to_clearance_m(cost01: float, dmax: float):
    # map_adapter_rclpy_esdf: cost = (1 - d/dmax)*100 -> d = dmax*(1 - cost01)
    return max(0.0, float(dmax) * (1.0 - float(cost01)))


class RecoveryPrimitive:
    """A tiny reflex: stop -> back -> small turn, then return control to NoMaD."""

    def __init__(self):
        self.active = False
        self.phase = 0
        self.phase_end = 0.0
        self.turn_sign = 1.0
        self.cooldown_until = 0.0

    def trigger(self, now: float, turn_sign: float, args):
        if now < self.cooldown_until:
            return False
        self.active = True
        self.phase = 0
        self.turn_sign = 1.0 if turn_sign >= 0 else -1.0
        self.phase_end = now + float(args.recover_stop_sec)
        self.cooldown_until = now + float(args.recover_cooldown_sec)
        return True

    def step(self, now: float, args):
        if not self.active:
            return None
        # advance phase if needed
        if now >= self.phase_end:
            self.phase += 1
            if self.phase == 1:
                self.phase_end = now + float(args.recover_back_sec)
            elif self.phase == 2:
                self.phase_end = now + float(args.recover_turn_sec)
            else:
                self.active = False
                return None

        if self.phase == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        elif self.phase == 1:
            return np.array([float(args.recover_back_x), 0.0], dtype=np.float32)
        elif self.phase == 2:
            return np.array([float(args.recover_turn_x), float(args.recover_turn_y) * self.turn_sign], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)


def _choose_turn_sign(cost01: np.ndarray, origin_xy, res: float, dmax: float):
    """Pick left/right based on which side has larger clearance at a short probe point."""
    # probe points in base_link frame (meters)
    probes = [(0.30, 0.18), (0.30, -0.18)]
    ds = []
    for x, y in probes:
        c = _sample_cost01(cost01, origin_xy, res, x, y)
        ds.append(_cost01_to_clearance_m(c, dmax))
    # ds[0]=left, ds[1]=right
    return 1.0 if ds[0] >= ds[1] else -1.0


def _apply_esdf_safety_filter(
    traj_xy: np.ndarray,
    waypoint_xy: np.ndarray,
    fsm_state: str,
    args,
):
    """Return (new_waypoint_xy, action) where action in {'OK','SLOW','RECOVER'}."""
    cost01, origin_xy, res, age = _get_latest_map_snapshot(args.map_stale_sec)
    dmax = _get_latest_dmax(args.d_max_safe_default)

    # thresholds (meters)
    d_warn = float(args.d_warn)
    d_stop = float(args.d_stop)
    if fsm_state == "S1":
        d_warn *= float(args.s1_thresh_scale)
        d_stop *= float(args.s1_thresh_scale)
    elif fsm_state == "S2":
        d_warn *= float(args.s2_thresh_scale)
        d_stop *= float(args.s2_thresh_scale)

    # 0) scan backstop
    fr = _get_latest_front_range(args.scan_stale_sec)
    if fr is not None:
        if fr <= float(args.scan_stop):
            return waypoint_xy, "RECOVER"
        if fr <= float(args.scan_warn):
            # simple speed reduction
            alpha = max(float(args.alpha_min), float(fr) / max(float(args.scan_warn), 1e-3))
            return (waypoint_xy * alpha).astype(np.float32), "SLOW"

    # 1) no map -> cannot ESDF-filter; still allow motion (vision-only).
    if cost01 is None:
        # No map available (missing or stale). Be conservative by default.
        pol = getattr(args, 'no_map_policy', 'slow')
        if pol == 'ok':
            return waypoint_xy, "OK"
        if pol == 'recover':
            return waypoint_xy, "RECOVER"
        # slow
        alpha = float(getattr(args, 'no_map_alpha', 0.35))
        return (waypoint_xy * alpha), "SLOW"

    # 2) ensure waypoint inside map; if not, pull it back along trajectory.
    idx = int(args.waypoint)
    idx = max(0, min(idx, int(traj_xy.shape[0]) - 1))

    # build candidate indices (from far to near) that are inside map
    candidates = [i for i in range(idx, -1, -1) if _inside_map(float(traj_xy[i, 0]), float(traj_xy[i, 1]), origin_xy, res, cost01.shape)]
    if not candidates:
        # everything out of local window -> conservative shrink
        return (waypoint_xy * float(args.oob_shrink)).astype(np.float32), "SLOW"

    def dmin_upto(i):
        dmins = []
        for j in range(0, i + 1):
            x, y = float(traj_xy[j, 0]), float(traj_xy[j, 1])
            c = _sample_cost01(cost01, origin_xy, res, x, y)
            dmins.append(_cost01_to_clearance_m(c, dmax))
        return float(np.min(dmins)) if dmins else float("inf")

    # 3) pick the furthest waypoint that satisfies dmin >= d_warn
    chosen_i = None
    chosen_dmin = None
    for i in candidates:
        dm = dmin_upto(i)
        if dm >= d_warn:
            chosen_i = i
            chosen_dmin = dm
            break

    if chosen_i is not None:
        # keep progress, but if it is close to warn boundary, mildly scale down
        wp = traj_xy[chosen_i].astype(np.float32)
        if chosen_dmin is not None and chosen_dmin < (d_warn * 1.2):
            alpha = np.clip((chosen_dmin - d_stop) / max((d_warn - d_stop), 1e-3), float(args.alpha_min), 1.0)
            wp = (wp * float(alpha)).astype(np.float32)
            return wp, "SLOW"
        return wp, "OK"

    # 4) no waypoint satisfies d_warn: check chosen index for d_stop vs recover
    dm_far = dmin_upto(candidates[0])
    if dm_far <= d_stop:
        return waypoint_xy, "RECOVER"

    # 5) between d_stop and d_warn: scale down
    alpha = np.clip((dm_far - d_stop) / max((d_warn - d_stop), 1e-3), float(args.alpha_min), 1.0)
    wp = (traj_xy[candidates[0]] * float(alpha)).astype(np.float32)
    return wp, "SLOW"

def load_topomap_images(topomap_dir):
    filenames = sorted(
        os.listdir(topomap_dir),
        key=lambda x: int(x.split(".")[0]),
    )
    print(f"Loaded {len(filenames)} topo nodes from {topomap_dir}")
    return filenames


def resolve_goal_index(goal_node, filenames):
    if goal_node >= 0:
        if goal_node >= len(filenames):
            print(
                f"[navigate] goal-node {goal_node} out of range "
                f"(#nodes={len(filenames)}), will clamp to last."
            )
            return len(filenames) - 1
        return goal_node
    else:
        return len(filenames) - 1


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(args):
    global context_size, pathguide

    # Configure scan callback sector
    global _scan_front_half_angle_rad
    _scan_front_half_angle_rad = np.deg2rad(float(args.scan_front_half_angle_deg))

    # Safety toggle: controls ESDF-first hard rules + recovery primitive (default enabled).
    # Safety is OFF by default to keep the original navigation behavior.
    safety_enabled = bool(getattr(args, "safety_enable", False)) and (not bool(getattr(args, "safety_disable", False)))

    rclpy.init()
    node = NavigationNode()

    # ------------------------------------------------------------------
    # load model parameters + weights (兼容原始 TSDF 项目的 config/models.yaml 结构)
    # ------------------------------------------------------------------
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_cfg = yaml.safe_load(f) or {}

    if args.model not in model_cfg:
        raise ValueError(f"Model '{args.model}' not found in {MODEL_CONFIG_PATH}")

    model_entry = model_cfg[args.model]

    # 两种常见结构：
    #   A) 原始 navigate.py 风格：models.yaml 里只有路径 {config_path, ckpt_path}
    #   B) ESDF 版本风格：models.yaml 里直接就是完整参数 {context_size, diffusion, model_path, }
    if isinstance(model_entry, dict) and ("config_path" in model_entry or "ckpt_path" in model_entry):
        model_config_path = model_entry.get("config_path", None)
        if not model_config_path:
            raise KeyError(f"{MODEL_CONFIG_PATH} for model '{args.model}' missing 'config_path'")
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f) or {}

        ckpt_path = model_entry.get("ckpt_path", None)
        if not ckpt_path:
            # 兜底：如果只提供了 model_path，就按当前 repo 的 model_weights 目录拼起来
            mp = model_entry.get("model_path", None)
            if not mp:
                raise KeyError(f"{MODEL_CONFIG_PATH} for model '{args.model}' missing 'ckpt_path'/'model_path'")
            ckpt_path = os.path.join(MODEL_WEIGHTS_PATH, mp)
    else:
        model_params = model_entry
        if "model_path" not in model_params:
            raise KeyError(
                f"{MODEL_CONFIG_PATH} for model '{args.model}' must contain either 'config_path'/'ckpt_path' or 'model_path'"
            )
        ckpt_path = os.path.join(MODEL_WEIGHTS_PATH, model_params["model_path"])

    # 关键：context_size 必须来自“真正的 model_params”
    if "context_size" not in model_params:
        raise KeyError(
            f"model_params missing 'context_size'. Your {MODEL_CONFIG_PATH} is likely the path-only format; in that case make sure it contains config_path and that config YAML contains context_size."
        )
    context_size = model_params["context_size"]

    # load model weights
    # 兼容两种 load_model 签名：load_model(path, params, device) 或 load_model(path, device=)
    try:
        model = load_model(ckpt_path, model_params, device)
        model = model.to(device)
    except TypeError:
        model = load_model(ckpt_path, device=device)
    model.eval()

    # diffusion scheduler：兼容两种 config（带 diffusion 字段 / 只有 num_diffusion_iters）
    if isinstance(model_params, dict) and ("diffusion" in model_params):
        diff = model_params["diffusion"]
        num_diffusion_iters = int(diff.get("num_diffusion_iters", diff.get("num_train_steps", 10)))
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(diff.get("num_train_steps", num_diffusion_iters)),
            beta_start=float(diff.get("beta_start", 0.0001)),
            beta_end=float(diff.get("beta_end", 0.02)),
            beta_schedule=str(diff.get("beta_schedule", "squaredcos_cap_v2")),
            prediction_type=str(diff.get("prediction_type", "epsilon")),
            clip_sample=bool(diff.get("clip_sample", True)),
        )
    else:
        # 原始 TSDF 项目（navigate.py）默认用 squaredcos + epsilon
        num_diffusion_iters = int(model_params.get("num_diffusion_iters", 10))
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )


    # PathGuide（与 guide.py 中的 ACTION_STATS 匹配）
    ACTION_STATS = {"min": np.array([-2.5, -4.0]), "max": np.array([5.0, 4.0])}
    pathguide = PathGuide(device, ACTION_STATS)

    # load topomap
    topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    topomap_filenames = load_topomap_images(topomap_dir)
    goal_node = resolve_goal_index(args.goal_node, topomap_filenames)
    # Preload topomap images (PIL) once to avoid repeatedly opening files in the loop.
    # This follows the original TSDF navigate.py behavior.
    topomap = []
    for fn in topomap_filenames:
        try:
            img_path = os.path.join(topomap_dir, fn)
            topomap.append(PILImage.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"[navigate] failed to open topomap image {fn}: {e}")
            topomap.append(PILImage.new("RGB", (model_params["image_size"], model_params["image_size"])))


    # planning step counter (used for compare throttling)


    plan_step = 0



    # start navigation loop
    waypoint_msg = Float32MultiArray()
    waypoint_msg.layout.dim = [MultiArrayDimension(label=f"seq:{plan_step}", size=0, stride=0)]

    global context_queue
    closest_node = 0

    # Safety-related state
    prev_fsm_state = "S0"
    s2_hold_until = 0.0
    recovery = RecoveryPrimitive()
    turn_sign_fallback = 1.0


    # num_diffusion_iters 已在加载 config 时确定（兼容两种 config 结构）
    noise_scheduler.set_timesteps(num_diffusion_iters)

    while rclpy.ok():
        loop_start_time = time.time()

        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                # ------------------------------------------------------
                # 视觉编码 & 最近拓扑节点选择
                # ------------------------------------------------------
                obs_images = transform_images(
                    context_queue,
                    model_params["image_size"],
                    center_crop=False,
                )
                # 把时间维的多帧 [3T,H,W] 重新拼成通道维
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1).to(device)
                mask = torch.zeros(1).long().to(device)

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)

                # Candidate goal images from the topomap window [start, end]
                goal_image = [
                    transform_images(
                        topomap[i],
                        model_params["image_size"],
                        center_crop=False,
                    ).to(device)
                    for i in range(start, end + 1)
                ]
                goal_image = torch.cat(goal_image, dim=0)

                # Original TSDF navigate.py model API: vision_encoder + dist_pred_net
                obsgoal_cond = model(
                    "vision_encoder",
                    obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                    goal_img=goal_image,
                    input_goal_mask=mask.repeat(len(goal_image)),
                )
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = int(np.argmin(dists))
                closest_node = min_idx + start

                # Select a "sub-goal" conditioning that is slightly ahead if we're close.
                sg_idx = min(
                    min_idx + int(dists[min_idx] < args.close_threshold),
                    len(obsgoal_cond) - 1,
                )
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
            else:
                raise Exception("Model type not implemented")

            # ----------------------------------------------------------
            # diffusion-based trajectory sampling
            # ----------------------------------------------------------
            with torch.no_grad():
                # obs_cond is produced by vision_encoder in the closest-node step above.
                # Expand to match (num_samples, ) for diffusion sampling.
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)


                def _denoise(noisy_init: torch.Tensor, obs_cond_in: torch.Tensor, guide_enabled: bool) -> torch.Tensor:
                    naction_local = noisy_init
                    noise_scheduler.set_timesteps(num_diffusion_iters)
                
                    for step_idx, k in enumerate(noise_scheduler.timesteps[:]):
                        # 1) diffusion denoise step
                        noise_pred = model(
                            "noise_pred_net",
                            sample=naction_local,
                            timestep=k,
                            global_cond=obs_cond_in,
                        )
                        naction_local = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction_local,
                        ).prev_sample
                
                        # 2) PathGuide TSDF/ESDF gradient injection (gated by FSM state)
                        if guide_enabled:
                            # Read current map state from /map_fsm_state
                            with _state_lock:
                                fsm_state = _latest_map_state
                
                            # Adaptive guidance strength based on FSM:
                            #   S0: full grad_scale
                            #   S1: half grad_scale
                            #   S2: disable geometry guidance (vision-only)
                            if fsm_state == "S2":
                                local_grad_scale = 0.0
                            elif fsm_state == "S1":
                                local_grad_scale = args.grad_scale * 0.5
                            else:
                                local_grad_scale = args.grad_scale
                
                            with _tsdf_lock:
                                has_map = pathguide.cost_map is not None
                
                            if has_map and local_grad_scale > 0.0:
                                try:
                                    # Temporarily enable grad; PathGuide builds its own graph internally
                                    with torch.enable_grad():
                                        grad, _ = pathguide.get_gradient(
                                            naction_local,
                                            scale_factor=args.guide_scale_factor,
                                        )
                
                                    # Apply correction in no_grad world (numerical update only)
                                    if grad is not None:
                                        naction_local = (naction_local - local_grad_scale * grad).detach()
                                except Exception as e:
                                    print("[navigate] pathguide guidance exception:", e)
                
                            # Optional: online health check (every 50 steps)
                            if args.debug_tsdf and (step_idx % 50 == 0) and has_map:
                                try:
                                    with torch.enable_grad():
                                        test_traj = torch.randn_like(naction_local[:1])
                                        test_grad, _ = pathguide.get_gradient(
                                            test_traj,
                                            scale_factor=args.guide_scale_factor,
                                        )
                                    if test_grad is not None:
                                        print(
                                            "[DEBUG_TSDF] grad_test.abs().mean() =",
                                            float(test_grad.abs().mean()),
                                        )
                                except Exception as e:
                                    print("[DEBUG_TSDF] test_grad exception:", e)
                
                    return naction_local
                
                # ------------------------------------------------------
                # compare/ablation: NoMaD (raw) vs PathGuide (guided)
                #   - main loop computes ONLY the executed (guided) trajectory to keep control rate
                #   - raw NoMaD is computed asynchronously in a background worker for visualization
                # ------------------------------------------------------
                if getattr(args, 'compare_nomad', False):
                    # Deterministic noise stream (seeded once); do NOT reseed every step.
                    if 'compare_rng' not in locals():
                        try:
                            compare_rng = torch.Generator(device=device)
                        except Exception:
                            compare_rng = torch.Generator()
                        compare_rng.manual_seed(int(getattr(args, 'compare_seed', 0)))

                    noisy_action = torch.randn(
                        (args.num_samples, model_params['len_traj_pred'], 2),
                        device=device,
                        generator=compare_rng,
                    )

                    # PathGuide-guided candidates (used for execution)
                    naction = _denoise(noisy_action.clone(), obs_cond, guide_enabled=bool(args.guide))

                    # Publish PathGuide candidates for visualization (metric-scaled if model normalizes)
                    na_pg_np = to_numpy(get_action(naction))
                    if model_params['normalize']:
                        _scale = MAX_V / RATE
                        na_pg_pub = na_pg_np * _scale
                    else:
                        na_pg_pub = na_pg_np

                    msg_pg = Float32MultiArray()
                    msg_pg.layout.dim = [MultiArrayDimension(label=f"seq:{plan_step}", size=0, stride=0)]
                    msg_pg.data = na_pg_pub.astype(np.float32).reshape(-1).tolist()
                    node.sampled_actions_pg_pub.publish(msg_pg)

                    # Lazy init worker with its own scheduler instance (thread-safety)
                    if 'compare_worker' not in locals():
                        _sched_worker = copy.deepcopy(noise_scheduler)
                        compare_worker = CompareNomadWorker(
                            node=node,
                            model=model,
                            scheduler=_sched_worker,
                            num_diffusion_iters=num_diffusion_iters,
                            device=device,
                            args=args,
                        )
                        compare_worker.start()

                    # Throttle compare jobs to avoid stealing control compute
                    ce = int(getattr(args, 'compare_every', 10))
                    do_submit = (ce <= 1) or (plan_step % ce == 0)

                    if getattr(args, 'compare_sync', False):
                        # Debug path: do it synchronously (may reduce control rate)
                        naction_nomad = _denoise(noisy_action.clone(), obs_cond, guide_enabled=False)
                        na_nomad_np = to_numpy(get_action(naction_nomad))
                        if model_params['normalize']:
                            _scale = MAX_V / RATE
                            na_nomad_pub = na_nomad_np * _scale
                        else:
                            na_nomad_pub = na_nomad_np
                        msg_nomad = Float32MultiArray()
                        msg_nomad.layout.dim = [MultiArrayDimension(label=f"seq:{plan_step}", size=0, stride=0)]
                        msg_nomad.data = na_nomad_pub.astype(np.float32).reshape(-1).tolist()
                        node.sampled_actions_nomad_pub.publish(msg_nomad)
                        wp_nomad = na_nomad_pub[0, args.waypoint].copy()
                        msg_wp_nomad = Float32MultiArray()
                        msg_wp_nomad.layout.dim = [MultiArrayDimension(label=f"seq:{plan_step}", size=0, stride=0)]
                        msg_wp_nomad.data = [float(wp_nomad[0]), float(wp_nomad[1])]
                        node.waypoint_nomad_pub.publish(msg_wp_nomad)
                    else:
                        if do_submit:
                            compare_worker.submit(
                                noisy_action=noisy_action,
                                obs_cond=obs_cond,
                                waypoint_idx=args.waypoint,
                                seq=plan_step,
                                normalize=bool(model_params['normalize']),
                            )
                else:
                    noisy_action = torch.randn(
                        (args.num_samples, model_params['len_traj_pred'], 2),
                        device=device,
                    )
                    naction = _denoise(noisy_action, obs_cond, guide_enabled=bool(args.guide))
                
                # ------------------------------------------------------
                # publish sampled actions and waypoint
                # ------------------------------------------------------
                naction_np = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate(
                    (np.array([0]), naction_np.flatten())
                ).tolist()
                node.sampled_actions_pub.publish(sampled_actions_msg)

                # Choose waypoint from predicted trajectory
                chosen_waypoint = naction_np[0, args.waypoint].copy()
                traj_xy = naction_np[0].copy()

                if model_params["normalize"]:
                    scale = MAX_V / RATE
                    chosen_waypoint = chosen_waypoint * scale
                    traj_xy = traj_xy * scale

                with _state_lock:
                    fsm_state = _latest_map_state

                # Entering S2: do a short hold (let the local map rebuild), but never freeze forever
                now_wall = time.time()
                if fsm_state != prev_fsm_state and fsm_state == "S2":
                    s2_hold_until = now_wall + float(args.s2_pause_sec)
                prev_fsm_state = fsm_state

                # 1) Recovery primitive overrides all (stop->back->turn) [only when safety is enabled]
                wp_out = None
                if safety_enabled and recovery.active:
                    wp_out = recovery.step(now_wall, args)

                # 2) Otherwise: (optional) ESDF-first safety filter + FSM soft throttling
                if wp_out is None:
                    if safety_enabled:
                        wp_candidate, action = _apply_esdf_safety_filter(traj_xy, chosen_waypoint, fsm_state, args)

                        if action == "RECOVER":
                            # choose a turn direction (prefer the side with larger clearance if map exists)
                            cost01, origin_xy, res, _age = _get_latest_map_snapshot(args.map_stale_sec)
                            if cost01 is not None:
                                dmax = _get_latest_dmax(args.d_max_safe_default)
                                turn_sign = _choose_turn_sign(cost01, origin_xy, res, dmax)
                            else:
                                turn_sign = -turn_sign_fallback
                                turn_sign_fallback = turn_sign
                            recovery.trigger(now_wall, turn_sign, args)
                            wp_out = recovery.step(now_wall, args)
                        else:
                            wp_out = wp_candidate
                    else:
                        # Safety disabled: run vision-first waypoint (no ESDF/scan hard constraints, no recovery).
                        wp_out = np.array([float(chosen_waypoint[0]), float(chosen_waypoint[1])], dtype=np.float32)

                    # FSM-based soft throttling (S1/S2) - do NOT force zero forever
                    if fsm_state == "S1":
                        wp_out = (wp_out * float(args.s1_alpha)).astype(np.float32)
                    elif fsm_state == "S2":
                        if now_wall < s2_hold_until:
                            wp_out = np.array([0.0, 0.0], dtype=np.float32)
                        else:
                            wp_out = (wp_out * float(args.s2_alpha)).astype(np.float32)

                waypoint_msg.layout.dim = [MultiArrayDimension(label=f"seq:{plan_step}", size=0, stride=0)]


                waypoint_msg.data = [float(wp_out[0]), float(wp_out[1])]
                node.waypoint_pub.publish(waypoint_msg)

                elapsed_time = time.time() - loop_start_time
                sleep_time = max(0.0, (1.0 / RATE) - elapsed_time)
                time.sleep(sleep_time)
        # increment planning step counter (for compare throttling)
        plan_step += 1


        reached_goal = closest_node == goal_node
        if reached_goal:
            print("Reached goal! Stopping")
            break

        rclpy.spin_once(node)

    # Gracefully stop compare worker (if any)
    try:
        if 'compare_worker' in locals() and compare_worker is not None:
            compare_worker.stop()
    except Exception:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LangDiffusor / NoMaD navigation with optional TSDF/ESDF guidance."
    )
    parser.add_argument("--model", "-m", default="nomad", type=str)
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--dir", "-d", default="topomap", type=str)
    parser.add_argument("--goal-node", "-g", default=-1, type=int)
    parser.add_argument("--close-threshold", "-t", default=3, type=int)
    parser.add_argument("--radius", "-r", default=4, type=int)
    parser.add_argument("--num-samples", "-n", default=8, type=int)

    # TSDF/ESDF guidance 相关
    parser.add_argument("--guide", action="store_true", help="Enable TSDF/ESDF-based PathGuide.")
    parser.add_argument(
        "--grad-scale",
        type=float,
        default=0.5,
        help="Step size for TSDF/ESDF gradient correction.",
    )
    parser.add_argument(
        "--guide-scale-factor",
        type=float,
        default=1.0,
        help="Scale factor applied to world-space trajs inside PathGuide.",
    )
    parser.add_argument(
        "--debug-tsdf",
        action="store_true",
        help="Enable periodic online TSDF gradient health-check logs.",
    )


    # ---------------- visualization / ablation ----------------
    parser.add_argument(
        "--compare-nomad",
        action="store_true",
        help="Publish raw NoMaD (no guidance) and PathGuide-constrained trajectories on separate topics for visualization.",
    )
    parser.add_argument(
        "--compare-seed",
        type=int,
        default=0,
        help="Seed for --compare-nomad so raw/guided share the same initial noise.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=10,
        help="When --compare-nomad is on, compute/publish raw NoMaD once every N planning steps (default: 10).",
    )
    parser.add_argument(
        "--compare-sync",
        action="store_true",
        help="Compute raw NoMaD synchronously (debug only; may reduce control rate).",
    )


    # ---------------- ESDF safety filter (hard rules) ----------------
    parser.add_argument("--d-warn", type=float, default=0.55, help="ESDF warn distance (m): start shortening/slowdown")
    parser.add_argument("--d-stop", type=float, default=0.30, help="ESDF stop distance (m): trigger recovery primitive")
    parser.add_argument("--d-max-safe-default", type=float, default=0.50, help="Fallback d_max_safe (m) if /esdf_d_max_safe not received")
    parser.add_argument("--map-stale-sec", type=float, default=0.6, help="If /tsdf_costmap older than this, skip ESDF filter")
    parser.add_argument("--no-map-policy", type=str, default="slow", choices=["ok","slow","recover"],
                        help="When /tsdf_costmap is missing/stale: ok=do nothing, slow=shrink waypoint, recover=force recovery")
    parser.add_argument("--no-map-alpha", type=float, default=0.35, help="Waypoint shrink factor when no-map-policy==slow")
    parser.add_argument("--oob-shrink", type=float, default=0.35, help="If waypoint out of local map window, shrink factor")
    parser.add_argument("--alpha-min", type=float, default=0.15, help="Min slowdown factor when in warn zone")

    # Scan emergency backstop (optional)
    parser.add_argument("--scan-warn", type=float, default=0.45, help="Front scan warn distance (m)")
    parser.add_argument("--scan-stop", type=float, default=0.25, help="Front scan stop distance (m)")
    parser.add_argument("--scan-stale-sec", type=float, default=0.4, help="Ignore scan if older than this")
    parser.add_argument("--scan-front-half-angle-deg", type=float, default=20.0, help="Front sector half-angle (deg)")

    # FSM soft throttling (from map_adapter_rclpy_esdf)
    parser.add_argument("--s1-alpha", type=float, default=0.85, help="Scale waypoint when FSM==S1")
    parser.add_argument("--s2-alpha", type=float, default=0.35, help="Scale waypoint when FSM==S2")
    parser.add_argument("--s2-pause-sec", type=float, default=0.20, help="Short pause when entering S2 (sec)")
    parser.add_argument("--s1-thresh-scale", type=float, default=1.0, help="Scale d_warn/d_stop in S1")
    parser.add_argument("--s2-thresh-scale", type=float, default=1.2, help="Scale d_warn/d_stop in S2")

    # Recovery primitive: stop -> back -> turn
    parser.add_argument("--recover-stop-sec", type=float, default=0.12, help="Recovery phase-0: stop duration")
    parser.add_argument("--recover-back-sec", type=float, default=0.25, help="Recovery phase-1: back duration")
    parser.add_argument("--recover-turn-sec", type=float, default=0.35, help="Recovery phase-2: turn duration")
    parser.add_argument("--recover-back-x", type=float, default=-0.12, help="Recovery back waypoint x (negative = backward)")
    parser.add_argument("--recover-turn-x", type=float, default=0.05, help="Recovery turn waypoint x")
    parser.add_argument("--recover-turn-y", type=float, default=0.12, help="Recovery turn waypoint y magnitude")
    parser.add_argument("--recover-cooldown-sec", type=float, default=0.9, help="Min time between recovery triggers")

    # Safety toggle (hard ESDF rules + recovery). Default is enabled for backward compatibility.
    parser.add_argument("--safety-enable", action="store_true", default=True, help="Enable ESDF hard safety filter + recovery (default: enabled)")
    parser.add_argument("--safety-debug", action="store_true", help="Print safety action/tag each planning step")
    parser.add_argument("--safety-disable", action="store_true", help="Disable ESDF hard safety filter + recovery; run vision navigation without hard constraints")
    args = parser.parse_args()

    # set debug flags
    _DEBUG_COSTMAP_STATS = bool(args.debug_tsdf)
    _SAFETY_DEBUG = bool(getattr(args, 'safety_debug', False))
    if args.safety_enable and (args.d_warn > args.d_max_safe_default):
        print(f"[WARN] d_warn({args.d_warn:.2f}) > d_max_safe_default({args.d_max_safe_default:.2f}). Consider increasing d_max_safe_default or lowering d_warn.")
    print(f"Using {device}")
    main(args)
