#!/usr/bin/env python3
"""
navigate_esdf.py

Monocular VisualNav-Transformer / NoMaD navigation with an optional
LiDAR->(2D TSDF)->(explicit 2D ESDF)->costmap safety auxiliary.

This is the *navigation-side* companion of `map_adapter_rclpy_esdf.py`.

Key goals:
- Keep the ORIGINAL VisualNav/NoMaD visual navigation as the main driver.
- Use LiDAR-derived local distance-field ONLY as a safety auxiliary.
- Preserve legacy CLI:
    --dir topomap
    --goal-node 255
- Consume /map_fsm_state (S0/S1/S2) published by the mapping node to
  modulate PathGuide strength in a minimal, robust way.
"""

import argparse
import os
import time
import threading
from pathlib import Path
from typing import List

import numpy as np
import rclpy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage
from rclpy.node import Node

# ROS msgs
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

# UTILS
from visualnav_transformer.deployment.src.topic_names import (
    IMAGE_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    WAYPOINT_TOPIC,
)
from visualnav_transformer.deployment.src.utils import (
    load_model,
    msg_to_pil,
    to_numpy,
    transform_images,
)
from visualnav_transformer.train.vint_train.training.train_utils import get_action

# PathGuide / TSDF
from guide import PathGuide
from costmap_cfg import CostMapConfig  # kept to align geometry config


# ---------------------------------------------------------------------
# Repo-relative path helpers
# ---------------------------------------------------------------------
def find_repo_root() -> Path:
    """Find directory that contains config/robot.yaml by searching upward."""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config" / "robot.yaml").exists():
            return p
    return Path.cwd()


REPO_ROOT = find_repo_root()

MODEL_WEIGHTS_PATH = str(REPO_ROOT / "model_weights")
ROBOT_CONFIG_PATH = str(REPO_ROOT / "config" / "robot.yaml")
MODEL_CONFIG_PATH = str(REPO_ROOT / "config" / "models.yaml")
TOPOMAP_IMAGES_ROOT = str(REPO_ROOT / "topomaps" / "images")


# ---------------------------------------------------------------------
# Robot limits
# ---------------------------------------------------------------------
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = float(robot_config.get("max_v", 0.2))
MAX_W = float(robot_config.get("max_w", 0.1))
RATE = float(robot_config.get("frame_rate", 6.0))


# ---------------------------------------------------------------------
# Global visual context
# ---------------------------------------------------------------------
context_queue: List[PILImage.Image] = []
context_size = 0


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cpu":
    print("Using cpu")


# ---------------------------------------------------------------------
# TSDF/ESDF costmap globals
# ---------------------------------------------------------------------
_tsdf_lock = threading.Lock()
_latest_cost_map = {"arr": None, "origin": (0.0, 0.0)}
pathguide: PathGuide = None  # initialized later


# ---------------------------------------------------------------------
# Map FSM state globals (from map_adapter_rclpy_esdf)
# ---------------------------------------------------------------------
_state_lock = threading.Lock()
_latest_map_state = "S0"  # "S0" | "S1" | "S2"


def map_fsm_state_callback(msg: String):
    global _latest_map_state
    data = (msg.data or "").strip()
    if data not in ("S0", "S1", "S2"):
        data = "S0"
    with _state_lock:
        _latest_map_state = data


def tsdf_callback_local(msg: OccupancyGrid):
    """
    Convert OccupancyGrid -> numpy cost grid and inject into PathGuide.

    Assumption aligned with map_adapter's convention:
      - publisher packs the array as occ.T.flatten()
      - so here reshape(h, w).T yields [num_x, num_y]
    """
    global pathguide

    try:
        w = int(msg.info.width)
        h = int(msg.info.height)
        arr = np.array(list(msg.data), dtype=np.int8)

        if arr.size == 0 or w <= 0 or h <= 0:
            print("[navigate_esdf] /tsdf_costmap received empty data, skip.")
            return

        if arr.size != w * h:
            print(
                f"[navigate_esdf] costmap size mismatch: len={arr.size}, w*h={w*h}. "
                "Attempting best-effort reshape."
            )

        try:
            cost_grid = arr.reshape((h, w)).T.astype(np.float32) / 100.0
        except Exception:
            cost_grid = arr.reshape((w, h)).astype(np.float32) / 100.0

        origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        with _tsdf_lock:
            _latest_cost_map["arr"] = cost_grid
            _latest_cost_map["origin"] = origin

            if pathguide is not None:
                try:
                    pathguide.set_cost_map(cost_grid, start_x=origin[0], start_y=origin[1])
                except Exception as e:
                    print("[navigate_esdf] PathGuide.set_cost_map failed:", e)

    except Exception as e:
        print("[navigate_esdf] tsdf_callback_local exception:", e)


def callback_obs(msg: Image):
    global context_queue
    obs_img = msg_to_pil(msg)
    context_queue.append(obs_img)
    if context_size > 0 and len(context_queue) > context_size + 1:
        context_queue.pop(0)


class NavigationNode(Node):
    def __init__(self):
        super().__init__("navigation_node_esdf")

        self.create_subscription(Image, IMAGE_TOPIC, callback_obs, 10)
        self.create_subscription(OccupancyGrid, "/tsdf_costmap", tsdf_callback_local, 1)
        self.create_subscription(String, "/map_fsm_state", map_fsm_state_callback, 10)

        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )


# ---------------------------------------------------------------------
# Topomap helpers
# ---------------------------------------------------------------------
def load_topomap_images(topomap_dir: str) -> List[str]:
    if not os.path.isdir(topomap_dir):
        raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")

    filenames = [f for f in os.listdir(topomap_dir) if f.endswith(".png")]
    filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))
    print(f"Loaded {len(filenames)} topo nodes from {topomap_dir}")
    return filenames


def resolve_goal_id(goal_node: int, filenames: List[str]) -> int:
    """
    Interpret --goal-node as image id.
    If id not present, fall back to nearest valid id.
    """
    if not filenames:
        return -1

    if goal_node < 0:
        return int(filenames[-1].split(".")[0])

    ids = {int(f.split(".")[0]) for f in filenames}
    if goal_node in ids:
        return goal_node

    sorted_ids = sorted(ids)
    smaller = [i for i in sorted_ids if i <= goal_node]
    if smaller:
        return smaller[-1]
    return sorted_ids[0]


def build_goal_index_window(closest_id: int, goal_id: int, radius: int) -> List[int]:
    start = max(closest_id - radius, 0)
    end = max(start, min(closest_id + radius, goal_id))
    return list(range(start, end + 1))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VisualNav/NoMaD with optional TSDF/ESDF safety guidance."
    )

    parser.add_argument("--model", "-m", default="nomad", type=str)
    parser.add_argument("--waypoint", "-w", default=2, type=int)

    # Legacy topomap args
    parser.add_argument("--dir", "-d", default="topomap", type=str)
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="Topomap image id (e.g., 255 means 255.png). -1 means last.",
    )
    parser.add_argument("--radius", "-r", default=5, type=int)

    # Diffusion sampling
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # PathGuide options
    parser.add_argument("--guide", action="store_true")
    parser.add_argument("--grad-scale", default=0.5, type=float)
    parser.add_argument(
        "--guide-scale-factor",
        default=1.0,
        type=float,
        help="Internal scale factor passed to PathGuide.",
    )
    parser.add_argument(
        "--debug-tsdf",
        action="store_true",
        help="Print occasional TSDF/PathGuide debug info.",
    )

    # Safety behaviour
    parser.add_argument(
        "--collision-cost-stop",
        type=float,
        default=0.06,
        help="Publish zero waypoint when min collision cost exceeds this value.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main(args: argparse.Namespace):
    global context_size, pathguide

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    rclpy.init()
    node = NavigationNode()

    # -----------------------------------------------------------------
    # Load models.yaml (RobotecAI VisualNav legacy layout)
    # -----------------------------------------------------------------
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    if args.model not in model_paths:
        raise ValueError(f"Model '{args.model}' not found in {MODEL_CONFIG_PATH}")

    ckpt_rel = model_paths[args.model].get("ckpt_path")
    model_config_path = model_paths[args.model].get("config_path")
    if not ckpt_rel or not model_config_path:
        raise KeyError(
            f"{MODEL_CONFIG_PATH} entry for '{args.model}' must include "
            "'ckpt_path' and 'config_path'."
        )

    ckpt_path = ckpt_rel
    if not os.path.isabs(ckpt_rel):
        ckpt_path = os.path.join(MODEL_WEIGHTS_PATH, ckpt_rel)

    if not os.path.isabs(model_config_path):
        model_config_path = str(REPO_ROOT / model_config_path)

    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    # robust defaults
    context_size = int(model_params.get("context_size", 0))
    if context_size <= 0:
        print(
            f"[navigate_esdf] WARNING: 'context_size' not found in {model_config_path}. "
            "Will run with context_size=0 (single-frame)."
        )

    image_size = model_params.get("image_size", 224)
    len_traj_pred = int(model_params.get("len_traj_pred", 8))
    normalize = bool(model_params.get("normalize", True))
    model_type = model_params.get("model_type", args.model)

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    diffusion_cfg = model_params.get("diffusion", {})
    num_train_steps = int(diffusion_cfg.get("num_train_steps", 1000))
    num_diffusion_iters = int(diffusion_cfg.get("num_diffusion_iters", 10))
    beta_start = float(diffusion_cfg.get("beta_start", 0.0001))
    beta_end = float(diffusion_cfg.get("beta_end", 0.02))
    beta_schedule = diffusion_cfg.get("beta_schedule", "linear")
    prediction_type = diffusion_cfg.get("prediction_type", "epsilon")
    clip_sample = bool(diffusion_cfg.get("clip_sample", False))

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        clip_sample=clip_sample,
    )

    # -----------------------------------------------------------------
    # PathGuide init
    # -----------------------------------------------------------------
    ACTION_STATS = {"min": np.array([-2.5, -4.0]), "max": np.array([5.0, 4.0])}
    pathguide = PathGuide(device, ACTION_STATS)

    # -----------------------------------------------------------------
    # Topomap
    # -----------------------------------------------------------------
    topomap_dir = os.path.join(TOPOMAP_IMAGES_ROOT, args.dir)
    topomap_filenames = load_topomap_images(topomap_dir)
    goal_id = resolve_goal_id(args.goal_node, topomap_filenames)

    # -----------------------------------------------------------------
    # Navigation loop
    # -----------------------------------------------------------------
    waypoint_msg = Float32MultiArray()
    sampled_actions_msg = Float32MultiArray()

    closest_id = 0
    noise_scheduler.set_timesteps(num_diffusion_iters)

    print(
        f"[navigate_esdf] model={args.model}, model_type={model_type}, "
        f"context_size={context_size}, len_traj_pred={len_traj_pred}"
    )
    print(f"[navigate_esdf] topomap='{args.dir}', goal_id={goal_id}")

    while rclpy.ok():
        loop_start = time.time()
        rclpy.spin_once(node, timeout_sec=0.0)

        if context_size > 0 and len(context_queue) <= context_size:
            time.sleep(0.01)
            continue
        if context_size == 0 and len(context_queue) == 0:
            time.sleep(0.01)
            continue

        # --------------------------------------------------------------
        # 1) Topological matching (NoMaD style)
        # --------------------------------------------------------------
        if model_type == "nomad":
            obs_src = (
                context_queue[-(context_size + 1) :]
                if context_size > 0
                else [context_queue[-1]]
            )
            obs_images = transform_images(obs_src, image_size, center_crop=False)

            try:
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1).to(device)
            except Exception:
                obs_images = obs_images.to(device)

            mask = torch.zeros(1).long().to(device)

            candidate_ids = build_goal_index_window(closest_id, goal_id, args.radius)
            goal_imgs = []
            goal_idx = []

            for i in candidate_ids:
                png = os.path.join(topomap_dir, f"{i}.png")
                if not os.path.exists(png):
                    continue
                g = transform_images(
                    PILImage.open(png).convert("RGB"),
                    image_size,
                    center_crop=False,
                )
                goal_imgs.append(g.to(device))
                goal_idx.append(i)

            if goal_imgs:
                goal_image = torch.cat(goal_imgs, dim=0)
                goal_index = torch.tensor(goal_idx, device=device).long().unsqueeze(0)

                with torch.no_grad():
                    _, image_features = model(
                        "encode_images",
                        obs_images,
                        goal_image,
                        output_hidden_states=True,
                    )
                    _, closest_node = model(
                        "navigation",
                        image_features=image_features,
                        goal_index=goal_index,
                        mask=mask,
                        testing=True,
                    )

                    try:
                        closest_id = int(goal_idx[int(closest_node.item())])
                    except Exception:
                        closest_id = int(goal_idx[0])

        # --------------------------------------------------------------
        # 2) Diffusion-based trajectory sampling
        # --------------------------------------------------------------
        with torch.no_grad():
            obs_img = context_queue[-1]
            obs_images = transform_images([obs_img], image_size, center_crop=False).to(device)

            if model_type == "nomad":
                obs_cond = model("encode_observation", obs_images)
            else:
                raise Exception("Model type not implemented in navigate_esdf")

            naction = torch.randn((args.num_samples, len_traj_pred, 2), device=device)
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for step_idx, k in enumerate(noise_scheduler.timesteps[:]):

                noise_pred = model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                ).prev_sample

                # PathGuide gradient injection (gated by /map_fsm_state)
                if args.guide:
                    with _state_lock:
                        fsm_state = _latest_map_state

                    if fsm_state == "S2":
                        local_grad_scale = 0.0
                    elif fsm_state == "S1":
                        local_grad_scale = args.grad_scale * 0.5
                    else:
                        local_grad_scale = args.grad_scale

                    with _tsdf_lock:
                        has_map = pathguide is not None and pathguide.cost_map is not None

                    if has_map and local_grad_scale > 0.0:
                        try:
                            with torch.enable_grad():
                                grad, _ = pathguide.get_gradient(
                                    naction,
                                    scale_factor=args.guide_scale_factor,
                                )
                            if grad is not None:
                                naction = (naction - local_grad_scale * grad).detach()
                        except Exception as e:
                            print("[navigate_esdf] pathguide guidance exception:", e)

                    if args.debug_tsdf and (step_idx % 50 == 0) and has_map:
                        try:
                            with torch.enable_grad():
                                test_traj = torch.randn_like(naction[:1])
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

            # ----------------------------------------------------------
            # 3) Publish sampled actions and waypoint
            # ----------------------------------------------------------
            naction_np = to_numpy(get_action(naction))
            sampled_actions_msg.data = np.concatenate(
                (np.array([0]), naction_np.flatten())
            ).tolist()
            node.sampled_actions_pub.publish(sampled_actions_msg)

            wp_idx = int(np.clip(args.waypoint, 0, naction_np.shape[1] - 1))
            chosen_waypoint = naction_np[0, wp_idx]

            if normalize:
                chosen_waypoint *= MAX_V / RATE

            with _state_lock:
                if _latest_map_state == "S2":
                    waypoint_msg.data = [0.0, 0.0]
                else:
                    waypoint_msg.data = chosen_waypoint.tolist()

            node.waypoint_pub.publish(waypoint_msg)

        elapsed = time.time() - loop_start
        sleep_time = max(0.0, (1.0 / max(RATE, 1e-3)) - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    args = parse_args()
    print(f"[navigate_esdf] args: {args}")
    main(args)
