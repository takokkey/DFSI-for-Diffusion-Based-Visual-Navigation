#!/usr/bin/env python3
from typing import Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from visualnav_transformer.deployment.src.topic_names import WAYPOINT_TOPIC

# ============ 配置 ============

CONFIG_PATH = "config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

# 线速度 / 角速度上限
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
DT = 1.0 / robot_config["frame_rate"]

# 这里用 vel_navi_topic；如果 config 里没有，就默认 "/cmd_vel"
VEL_TOPIC = robot_config.get("vel_navi_topic", "/cmd_vel")

EPS = 1e-8
LINEAR_VEL = 1e-2   # 线速度缩放系数，可之后再调


def clip_angle(a: float) -> float:
    """把角度裁剪到 [-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def pd_controller(waypoint: np.ndarray) -> Tuple[float, float]:
    """
    简单 PD 控制器：
    waypoint: 长度 2 或 4 的 numpy 向量:
        [dx, dy] 或 [dx, dy, hx, hy]
    返回: (v, w)
    """
    assert (
        len(waypoint) == 2 or len(waypoint) == 4
    ), "waypoint must be a 2D or 4D vector"

    if len(waypoint) == 2:
        dx, dy = waypoint
        hx, hy = None, None
    else:
        dx, dy, hx, hy = waypoint

    # 1) dx,dy 很小且给了 heading：用 heading 调整角度
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0.0
        w = clip_angle(np.arctan2(hy, hx)) / DT

    # 2) dx≈0：原地转向
    elif np.abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * DT)

    # 3) 其他情况：根据 dx,dy 简单计算
    else:
        v = LINEAR_VEL * dx / (np.abs(dy) + EPS)
        w = np.arctan(dy / dx)

    # 裁剪速度到允许范围
    v = float(np.clip(v, -MAX_V, MAX_V))
    w = float(np.clip(w, -MAX_W, MAX_W))
    return v, w


class VelPublisher(Node):
    def __init__(self):
        super().__init__("cmd_vel_publisher")

        # 订阅 navigate.py 输出的 /waypoint
        self.waypoint_subscription = self.create_subscription(
            Float32MultiArray,
            WAYPOINT_TOPIC,
            self.waypoint_callback,
            10,
        )

        # 发布到 vel_navi_topic（当前 config 里是 /cmd_vel_mux/input/navi）
        self.publisher = self.create_publisher(Twist, VEL_TOPIC, 10)

        self.get_logger().info(
            f"VelPublisher initialized, subscribing {WAYPOINT_TOPIC}, "
            f"publishing Twist to {VEL_TOPIC}"
        )

    def waypoint_callback(self, msg: Float32MultiArray):
        wp = np.array(msg.data, dtype=np.float32)
        v, w = pd_controller(wp)

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.publisher.publish(twist)

        # 打一点调试日志方便确认
        self.get_logger().info(f"WAYPOINT {wp.tolist()} -> cmd: v={v:.3f}, w={w:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = VelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
