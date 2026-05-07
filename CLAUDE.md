# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous drone search-and-tracking system built with ROS 2, ArduPilot SITL, Gazebo Harmonic simulation, and computer vision (YOLO-World + LLM grounding). The drone performs a boustrophedon (lawnmower) search pattern, detects targets via camera, and locks on using gimbal control.

## Build Commands

**Gazebo plugin (C++, one-time build):**
```bash
cd ardupilot_gazebo && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4
```

**ROS 2 packages (Python, after any package changes):**
```bash
cd drone_rosws
colcon build
source install/setup.bash
```

**Environment setup (required before ROS commands):**
```bash
source /opt/ros/humble/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=<repo>/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=<repo>/ardupilot_gazebo/models:<repo>/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH
```

**Python environment for ground station:**
```bash
source /home/jlr3/.venv-alt/bin/activate
```

## Full Simulation Startup Sequence

Each step runs in a separate terminal:

```bash
# 1. Gazebo simulation
gz sim -v4 -r ardupilot_gazebo/worlds/iris_runway_15x15_walls.sdf

# 2. ArduPilot SITL (then type: output add 127.0.0.1:14550)
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console

# 3. ROS-Gazebo camera bridge
ros2 run ros_gz_bridge parameter_bridge \
  "/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"

# 4. UDP relay (bridges ROS camera to ground station)
python3 sim_interface/udp_relay.py

# 5. Ground station GUI
python3 sim_interface/GroundStation_sim.py

# 6. MAVROS bridge
ros2 launch mavros apm.launch fcu_url:=udp://127.0.0.1:14550@
```

## Architecture

### Communication Flow

```
Gazebo → ROS topic (/world/.../camera/image)
       → [udp_relay.py ROS node]
           → UDP:8500  → GroundStation_sim.py (raw video frames)
           ← UDP:8501  ← GroundStation (track command + target class)
           ← UDP:8502  ← GroundStation (mission start/stop)
           → /target_object (String ROS topic) → drone_searching_behavior
           → /object_found  (Bool ROS topic)   → lawnmower node

ArduPilot SITL ↔ MAVROS ↔ ROS topics
  /mavros/setpoint_position/local  ← lawnmower.py (waypoints during search)
  /mavros/setpoint_velocity/cmd_vel_unstamped ← search_node.py (follow target)
  /mavros/mount_control/command    ← search_node.py (gimbal lock-on)
```

### ROS Nodes (`drone_rosws/src/`)

| Package | Node | Role |
|---------|------|------|
| `lawnmower` | `lawnmower` | Boustrophedon search pattern; halts when `/object_found=True` |
| `drone_searching_behavior` | `search_node` | AprilTag tracking via velocity commands + gimbal control |
| `drone_searching_behavior` | `tag_overlay` | Annotates camera feed with AprilTag bounding boxes |
| `yolo_detector` | (pending) | On-drone YOLO detection (placeholder) |

### Ground Station (`sim_interface/`)

| File | Role |
|------|------|
| `GroundStation_sim.py` | PyQt6 GUI; YOLO-World detection; ReID + HSV person tracking; LLM grounding |
| `udp_relay.py` | ROS node; bridges Gazebo camera ↔ ground station; handles mission commands |
| `llm_client.py` | Wraps Vision LLM API (OpenAI-compatible); returns bbox + YOLO class terms |
| `RealGround.py` | Real-hardware variant of the ground station |

### Detection / Tracking Pipeline in Ground Station

1. Receive raw frame over UDP:8500
2. Run YOLO-World with on-demand object classes
3. Optionally call LLM grounding (`llm_client.py`) for semantic disambiguation
4. Match detections using ReID embeddings (OSNet) + HSV histogram similarity
5. Send target class back to drone via UDP:8501; set `/target_object` ROS topic
6. State machine: `IDLE → TRACKING → FOUND`

## Key Configuration

- **`sim_interface/.env`** — OpenAI API key for LLM grounding (`OPENAI_API_KEY=...`)
- **`udp_relay.py`** accepts a `camera_topic` ROS parameter to reconfigure the image source without code changes
- Lawnmower waypoint tolerance: 0.5 m; setpoint publish rate: 20 Hz
- `search_node` target configurable via `target_tag_id` ROS parameter

## Dependencies

**Python (`.venv-alt`):** PyQt6, opencv-python, ultralytics, torch/torchvision, torchreid, transformers, timm, mavproxy

**ROS packages:** mavros, ros_gz_bridge, cv_bridge, apriltag_msgs, sensor_msgs, geometry_msgs

**System:** ROS 2 Humble/Jazzy, Gazebo Harmonic, ArduPilot SITL
