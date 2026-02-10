# DFSI: Diffusion-based Visual Navigation with LiDAR Execution-Time Safety (ROS2)

This repository provides a ROS2 deployment of a **frozen NoMaD-style diffusion navigator** with **DFSI**, a plug-and-play **execution-time safety** layer that injects **LiDAR-derived distance-to-obstacle constraints** **without retraining** or modifying the backbone policy.

DFSI includes:
- **R-DFI (TSDF → ESDF mapping + refresh manager)**: builds a robot-centric local ESDF from planar LiDAR scans and regulates updates via a reliability FSM (ARM-FSM) to handle motion staleness and scan–map mismatch.
- **DSG (Dual-Stage Safety Gate)**: (i) reranks diffusion-sampled trajectory candidates using ESDF clearance; (ii) applies an action-level safety supervisor that may override commands when safety margins are violated.

<p align="center">
  <img src="pip_esdf.png" width="720" alt="System overview of DFSI">
</p>
<p align="center"><em>System overview of DFSI.</em></p>
---

## Demo Videos

### Indoor experiment
<p align="center">
  <img src="Indoor.gif" width="360" alt="Indoor experiment (GIF)">
</p>

Full video (mp4): [Indoor.mp4](Indoor.mp4)

## 1) Deployment (Docker)

### 1.1 Clone
```bash
git clone git@github.com:takokkey/DFSI-for-Diffusion-Based-Visual-Navigation.git
cd DFSI-for-Diffusion-Based-Visual-Navigation
```

### 1.2 Build the Docker image
```bash
docker build -t visualnav_transformer:latest .
```

### 1.3 Run the Docker container
```bash
docker run -it --env ROS_DOMAIN_ID=$ROS_DOMAIN_ID --rm --gpus=all --net=host visualnav_transformer:latest
```

> Notes
> - `--net=host` is required for ROS2 communication between host and container.
> - Prepare a `topomap/` directory (or change `--dir`) before navigation.

---

## 2) Startup (ESDF + Safety)

### 2.1 Host side (outside Docker)

**(1) Robot bringup**
```bash
ros2 launch iqr_turtlebot_bringup bringup.launch.xml use_rvzi:=false
```

**(2) Twist bridge (forward Twist to robot)**
```bash
python3 cmd_vel_navi_bridge.py
```

### 2.2 Container side (inside Docker)

Open **3 terminals** in the container (or use tmux). In **each** terminal, set `PYTHONPATH`:
```bash
export PYTHONPATH=/visualnav-transformer/src/visualnav_transformer/deployment/src:/visualnav-transformer/src:/visualnav-transformer/diffusion_policy:$PYTHONPATH
```

### 2.3 Recommended daily run (Scheme 2)

**Terminal A — ESDF map_adapter (safety)**
```bash
python -u src/visualnav_transformer/deployment/src/map_adapter_rclpy_esdf_safety.py   --ros-args   -p world_frame:=laser_link   -p target_frame:=laser_link
```

**Terminal B — Navigation (ESDF safety enabled)**
```bash
python -u src/visualnav_transformer/deployment/src/navigate_esdf_safety.py   --model nomad   --dir topomap   --guide   --grad-scale 0.5   --goal-node 255   --safety-enable   --d-warn 0.45   --d-stop 0.25   --s2-pause-sec 0.20   --s2-alpha 0.35   --recover-stop-sec 0.10   --recover-back-sec 0.30   --recover-turn-sec 0.40   --debug-tsdf
```

**Terminal C — Waypoint → Twist executor**
```bash
python -u scripts/publish_cmd.py
```

---

## 3) Parameters (Safety & Recovery)

| Flag | Meaning | Typical / Suggested | Notes |
|---|---|---|---|
| `--safety-enable` | enable action-level safety supervisor | on (Scheme 2/3) | overrides nominal command when unsafe |
| `--d-warn` | warning distance threshold (m) | 0.45 (daily) / 0.70 (conservative) | larger = more conservative |
| `--d-stop` | stop distance threshold (m) | 0.25 (daily) / 0.40 (conservative) | larger = earlier stop |
| `--s2-pause-sec` | S2 pause duration (s) | 0.20 (daily) / 0.25 (conservative) | “unreliable” state hold |
| `--s2-alpha` | S2 blending / margin factor | 0.35 (daily) / 0.25 (conservative) | project-specific; verify in code |
| `--recover-stop-sec` | recovery: stop time (s) | 0.10 | Scheme 2 |
| `--recover-back-sec` | recovery: back time (s) | 0.30 / 0.35 | Scheme 2 / 3 |
| `--recover-turn-sec` | recovery: turn time (s) | 0.40 / 0.50 | Scheme 2 / 3 |
| `--debug-tsdf` | debug TSDF/ESDF pipeline | optional | useful for diagnosing mapping |
| `--safety-debug` | verbose safety debugging | optional | Scheme 3 example |

---

## 4) Quick pre-run checks 

After launching the host + container nodes, verify that the critical ROS2 topics are present:
```bash
ros2 topic list | grep -E "tsdf_costmap|esdf_d_max_safe|map_fsm_state|waypoint|cmd_vel_mux/input/navi"
```

You should at least see:
- Mapping
  - `/tsdf_costmap`
  - `/esdf_d_max_safe`
  - `/map_fsm_state`
- Control
  - `/waypoint`
  - `/cmd_vel_mux/input/navi`

(Optional) inspect message flow:
```bash
ros2 topic hz /tsdf_costmap
ros2 topic echo -n 1 /map_fsm_state
```

---

<details>
<summary><b>Scheme 1 — Minimal pipeline (first bring-up)</b></summary>

**Host**
```bash
ros2 launch iqr_turtlebot_bringup bringup.launch.xml use_rvzi:=false
python3 cmd_vel_navi_bridge.py
```

**Docker (3 terminals)**
```bash
export PYTHONPATH=/visualnav-transformer/src/visualnav_transformer/deployment/src:/visualnav-transformer/src:/visualnav-transformer/diffusion_policy:$PYTHONPATH
```

A) map_adapter:
```bash
python -u src/visualnav_transformer/deployment/src/map_adapter_rclpy_esdf_safety.py
```

B) navigate (minimal args):
```bash
python -u src/visualnav_transformer/deployment/src/navigate_esdf_safety.py   --model nomad   --dir topomap   --guide   --grad-scale 0.5   --goal-node 255
```

C) publish_cmd:
```bash
python -u scripts/publish_cmd.py
```

</details>

<details>
<summary><b>Scheme 3 — Conservative indoor anti-collision (more recovery)</b></summary>

**Host**
```bash
ros2 launch iqr_turtlebot_bringup bringup.launch.xml use_rvzi:=false
python3 cmd_vel_navi_bridge.py
```

**Docker**
```bash
export PYTHONPATH=/visualnav-transformer/src/visualnav_transformer/deployment/src:/visualnav-transformer/src:/visualnav-transformer/diffusion_policy:$PYTHONPATH
```

A) map_adapter:
```bash
python -u src/visualnav_transformer/deployment/src/map_adapter_rclpy_esdf_safety.py   --ros-args -p world_frame:=laser_link -p target_frame:=laser_link
```

B) navigate (conservative thresholds + slower S2):
```bash
python -u src/visualnav_transformer/deployment/src/navigate_esdf_safety.py   --model nomad --dir topomap --guide --grad-scale 0.5 --goal-node 255   --safety-enable   --d-warn 0.70 --d-stop 0.40   --s2-pause-sec 0.25 --s2-alpha 0.25   --recover-back-sec 0.35 --recover-turn-sec 0.50   --safety-debug
```

C) publish_cmd:
```bash
python -u scripts/publish_cmd.py
```

</details>

<details>
<summary><b>Debug & Visualization (optional)</b></summary>

### TSDF costmap BEV visualization (save PNGs)
```bash
python3 bev_tsdf_costmap_viz_color.py --ros-args   -p topic:=/tsdf_costmap   -p save_png:=true   -p save_dir:=/tmp/tsdf_bev   -p save_every_n:=5   -p colormap:=jet   -p vmin:=0.0 -p vmax:=100.0   -p alpha:=0.90   -p zero_transparent:=true   -p unknown_white:=true   -p flip_ud:=true
```

### Navigation visualization (safe)
```bash
python -u src/visualnav_transformer/deployment/src/navigate_esdf_safety_visual.py   --model nomad   --dir topomap   --guide --grad-scale 0.5   --goal-node 255   --safety-enable   --d-warn 0.70 --d-stop 0.40   --s2-pause-sec 0.25 --s2-alpha 0.25   --recover-back-sec 0.35 --recover-turn-sec 0.50   --safety-debug
```

### Compare with nominal (example)
```bash
python -u src/visualnav_transformer/deployment/src/navigate_esdf_safety_visual.py   --model nomad --dir topomap --guide --grad-scale 0.5 --goal-node 255   --d-warn 0.70 --d-stop 0.40 --s2-pause-sec 0.25 --s2-alpha 0.25   --recover-back-sec 0.35 --recover-turn-sec 0.50 --safety-debug   --compare-nomad --compare-seed 0 --compare-every 5
```

### Overlay visualization scripts
```bash
# Script 1
python3 visualize_nomad_vs_pathguide.py   --out-dir ./viz_overlays   --save-every 10   --cam-meters 2.0   --traj-scale 6.0   --require-wp

# Script 2 (odom-based)
python3 visualize_nomad_vs_pathguide_odom.py   --out-dir ./viz_overlays/test3   --save-every 10   --cam-meters 2.0   --traj-scale 6.0   --image-topic /camera/realsense2_camera_node/color/image_raw   --nomad-topic /sampled_actions_nomad   --pg-topic /sampled_actions_pg   --waypoint-nomad-topic /waypoint_nomad   --waypoint-topic /waypoint   --image-qos reliable   --image-stale-sec 10.0

# Script 3
python3 -u visualize_nomad1.py   --out-dir ./viz_overlays/nomad_test1   --save-every 10   --traj-steps 4
```

</details>

---

## Acknowledgements

Parts of this repository reuse or adapt code from:

- **visualnav-transformer** — https://github.com/robodhruv/visualnav-transformer  
- **NaviD** — https://github.com/SYSU-RoboticsLab/NaviD  

Please refer to the original repositories for full details and licenses.
