#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Central topic registry for visualnav-transformer + ESDF extensions.

这个文件是「原始 visualnav-transformer 的 topic 定义」和
「现在 ESDF / PathGuide 扩展」的合并版：

- 保留了你最初的所有常量，用于 explore / topomap 等老代码；
- 新增了一组 ESDF / TSDF / PathGuide / Create3 相关的 topic & frame，
  给 map_adapter_rclpy_esdf.py、navigate_esdf.py、cmd_vel_navi_bridge.py 等使用。

这样旧代码不会炸，新代码也能直接 import 这些名字。
"""

# ----------------------------------------------------
# 原始：image obs topics
# ----------------------------------------------------
FRONT_IMAGE_TOPIC = "/usb_cam_front/image_raw"
REVERSE_IMAGE_TOPIC = "/usb_cam_reverse/image_raw"
IMAGE_TOPIC = "/camera/realsense2_camera_node/color/image_raw"

# ----------------------------------------------------
# 原始：exploration topics
# ----------------------------------------------------
SUBGOALS_TOPIC = "/subgoals"
GRAPH_NAME_TOPIC = "/graph_name"
WAYPOINT_TOPIC = "/waypoint"
REVERSE_MODE_TOPIC = "/reverse_mode"
SAMPLED_OUTPUTS_TOPIC = "/sampled_outputs"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"
SAMPLED_WAYPOINTS_GRAPH_TOPIC = "/sampled_waypoints_graph"
BACKTRACKING_IMAGE_TOPIC = "/backtracking_image"
FRONTIER_IMAGE_TOPIC = "/frontier_image"
SUBGOALS_SHAPE_TOPIC = "/subgoal_shape"
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"
ANNOTATED_IMAGE_TOPIC = "/annotated_image"
CURRENT_NODE_IMAGE_TOPIC = "/current_node_image"
FLIP_DIRECTION_TOPIC = "/flip_direction"
TURNING_TOPIC = "/turning"
SUBGOAL_GEN_RATE_TOPIC = "/subgoal_gen_rate"
MARKER_TOPIC = "/visualization_marker_array"
VIZ_NAV_IMAGE_TOPIC = "/nav_image"

# 原始：visualization topics
CHOSEN_SUBGOAL_TOPIC = "/chosen_subgoal"

# 原始：recorded on the robot
ODOM_TOPIC = "/odom"
BUMPER_TOPIC = "/mobile_base/events/bumper"
JOY_BUMPER_TOPIC = "/joy_bumper"

# ----------------------------------------------------
# ESDF / TSDF / PathGuide 扩展部分
# ----------------------------------------------------

# Robot frames（用在 TF / map_adapter_rclpy_esdf 等）
FRAME_BASE = "base_link"
FRAME_ODOM = "odom"
FRAME_LASER = "laser_link"

# Sensing
SCAN_TOPIC = "/scan"  # 2D LiDAR
COLOR_IMAGE_TOPIC = IMAGE_TOPIC  # 等价别名
DEPTH_IMAGE_TOPIC = "/camera/realsense2_camera_node/depth/image_rect_raw"

# ESDF / TSDF 相关 topic
TSDF_COSTMAP_TOPIC = "/tsdf_costmap"          # ESDF 编码的局部 costmap
TSDF_RESIDUAL_TOPIC = "/tsdf_residual_score"  # 残差 / 置信度评分
MAP_FSM_TOPIC = "/map_fsm_state"              # S0/S1/S2 ... map_fsm 状态机

# Navigation IO（与 Create3 / cmd_vel_bridge / publish_cmd 对齐）
CMD_VEL_NAVI_TOPIC = "/cmd_vel_mux/input/navi"
CMD_VEL_TOPIC = "/cmd_vel"
STOP_STATUS_TOPIC = "/stop_status"            # create3 stop_state
DOCK_STATUS_TOPIC = "/dock"                   # dock_visible / is_docked
