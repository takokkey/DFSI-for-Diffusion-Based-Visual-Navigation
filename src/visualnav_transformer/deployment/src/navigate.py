#!/usr/bin/env python3
import argparse
import os
import time
import threading

import numpy as np
import rclpy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from rclpy.node import Node
from PIL import Image as PILImage

# ROS
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid

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
_latest_cost_map = {"arr": None, "origin": (0.0, 0.0)}
pathguide = None  # will init later


def tsdf_callback_local(msg: OccupancyGrid):
    """
    把 OccupancyGrid 转成 cost_map numpy 并注入 PathGuide。

    约定（与 map_adapter_rclpy.py 保持一致）：
      - map_adapter_rclpy 发布的 OccupancyGrid.data 是 occ.T.flatten()
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
            print(f"[navigate] reshape(h,w).T failed ({e}), fallback reshape(w,h) without T.")
            cost_grid = arr.reshape((w, h)).astype(np.float32) / 100.0

        origin = (msg.info.origin.position.x, msg.info.origin.position.y)

        with _tsdf_lock:
            _latest_cost_map["arr"] = cost_grid
            _latest_cost_map["origin"] = origin

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
                except Exception as e:
                    print("[navigate] set_cost_map failed:", e)
    except Exception as e:
        print("[navigate] tsdf_callback_local exception:", e)


def callback_obs(msg: Image):
    global context_queue
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


class NavigationNode(Node):
    def __init__(self):
        super().__init__("navigation_node")
        self.create_subscription(Image, IMAGE_TOPIC, callback_obs, 10)
        self.create_subscription(OccupancyGrid, "/tsdf_costmap", tsdf_callback_local, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)


def main(args: argparse.Namespace):
    global context_size, pathguide

    # ------------------------------------------------------------------
    # load model parameters
    # ------------------------------------------------------------------
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # ------------------------------------------------------------------
    # load model weights
    # ------------------------------------------------------------------
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(ckpth_path, model_params, device)
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 初始化 PathGuide（与 guide.py 中的 ACTION_STATS 匹配）
    # ------------------------------------------------------------------
    ACTION_STATS = {"min": np.array([-2.5, -4.0]), "max": np.array([5.0, 4.0])}
    pathguide = PathGuide(device, ACTION_STATS)

    # ------------------------------------------------------------------
    # load topomap
    # ------------------------------------------------------------------
    topomap_filenames = sorted(
        os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, args.dir)),
        key=lambda x: int(x.split(".")[0]),
    )
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node
    reached_goal = False

    # ROS 初始化
    rclpy.init()
    node = NavigationNode()

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while rclpy.ok():
        loop_start_time = time.time()
        waypoint_msg = Float32MultiArray()

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
                goal_image = [
                    transform_images(
                        g_img,
                        model_params["image_size"],
                        center_crop=False,
                    ).to(device)
                    for g_img in topomap[start : end + 1]
                ]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model(
                    "vision_encoder",
                    obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                    goal_img=goal_image,
                    input_goal_mask=mask.repeat(len(goal_image)),
                )
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start

                sg_idx = min(
                    min_idx + int(dists[min_idx] < args.close_threshold),
                    len(obsgoal_cond) - 1,
                )
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # ------------------------------------------------------
                # Diffusion loop + TSDF Guidance
                # ------------------------------------------------------
                with torch.no_grad():
                    # 扩展到多个 sample
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2),
                        device=device,
                    )
                    naction = noisy_action
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for step_idx, k in enumerate(noise_scheduler.timesteps[:]):
                        # 1) diffusion denoise step
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

                        # 2) PathGuide TSDF 梯度注入
                        if args.guide:
                            with _tsdf_lock:
                                has_map = pathguide.cost_map is not None

                            if has_map:
                                try:
                                    # 暂时开启梯度以便 PathGuide 内部构建计算图
                                    with torch.enable_grad():
                                        # 直接把当前 sample 传入，get_gradient 内部会 detach + requires_grad_(True)
                                        grad, _ = pathguide.get_gradient(
                                            naction,
                                            scale_factor=args.guide_scale_factor,
                                        )

                                    # 在 no_grad 世界里应用修正（只对数值做更新，不保留图）
                                    if grad is not None:
                                        naction = (naction - args.grad_scale * grad).detach()
                                except Exception as e:
                                    print("[navigate] pathguide guidance exception:", e)

                            # 可选：在线 health check（每 50 步一次）
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

                # ------------------------------------------------------
                # publish sampled actions and waypoint
                # ------------------------------------------------------
                naction_np = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate(
                    (np.array([0]), naction_np.flatten())
                ).tolist()
                node.sampled_actions_pub.publish(sampled_actions_msg)

                chosen_waypoint = naction_np[0, args.waypoint]
                if model_params["normalize"]:
                    chosen_waypoint *= MAX_V / RATE
                waypoint_msg.data = chosen_waypoint.tolist()
                node.waypoint_pub.publish(waypoint_msg)

                elapsed_time = time.time() - loop_start_time
                sleep_time = max(0.0, (1.0 / RATE) - elapsed_time)
                time.sleep(sleep_time)

        reached_goal = closest_node == goal_node
        if reached_goal:
            print("Reached goal! Stopping...")
            break

        rclpy.spin_once(node, timeout_sec=0.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LangDiffusor / NoMaD navigation with optional TSDF guidance."
    )
    parser.add_argument("--model", "-m", default="nomad", type=str)
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--dir", "-d", default="topomap", type=str)
    parser.add_argument("--goal-node", "-g", default=-1, type=int)
    parser.add_argument("--close-threshold", "-t", default=3, type=int)
    parser.add_argument("--radius", "-r", default=4, type=int)
    parser.add_argument("--num-samples", "-n", default=8, type=int)

    # TSDF guidance 相关
    parser.add_argument("--guide", action="store_true", help="Enable TSDF-based PathGuide.")
    parser.add_argument(
        "--grad-scale",
        type=float,
        default=0.5,
        help="Step size for TSDF gradient correction.",
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

    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
