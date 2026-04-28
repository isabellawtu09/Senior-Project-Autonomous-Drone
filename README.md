## Senior Project Autonomous Drone

This repository contains the Gazebo world assets, ROS 2 nodes, and ground-station tools used for autonomous search + tracking simulation with ArduPilot SITL.

The latest flow includes:
- a `15m x 15m` walled world (`iris_runway_15x15_walls.sdf`)
- a boustrophedon ("lawnmower") path-planning mission node
- YOLO-based target search/tracking support

## Repository Components

- `ardupilot_gazebo/`: Gazebo models/worlds/plugins (including wall environments)
- `drone_rosws/src/lawnmower/`: path-planning mission node
- `drone_rosws/src/yolo_detector/`: YOLO processor + UDP relay node
- `sim_interface/`: ground-station GUIs (`RealGround.py`, `ground_station.py`)

## Prerequisites

- ArduPilot SITL installed and working
- Gazebo Harmonic/Garden installed
- ROS 2 installation sourced
- Python environments for UI / YOLO dependencies (as used in your local setup)

Helpful documentation:
- ArduPilot Dev: https://ardupilot.org/dev/index.html
- MAVProxy: https://ardupilot.org/mavproxy/index.html
- Gazebo docs: https://gazebosim.org/docs/harmonic
- ROS 2 docs: https://docs.ros.org/en/humble/index.html

## One-Time Setup

1) Export Gazebo resource/plugin paths (adjust for your machine):

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=/home/$USER/Senior-Project-Autonomous-Drone/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=/home/$USER/Senior-Project-Autonomous-Drone/ardupilot_gazebo/models:/home/$USER/Senior-Project-Autonomous-Drone/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH
```

2) Source ROS 2 and build/source ROS workspace:

```bash
source /opt/ros/humble/setup.bash
cd /home/$USER/Senior-Project-Autonomous-Drone/drone_rosws
colcon build
source install/setup.bash
```

## Run Flow (15x15 Walls + Path Planning)

Use separate terminals.

### 1) Start Gazebo with the 15x15 wall world

```bash
cd /home/$USER/Senior-Project-Autonomous-Drone
gz sim -v4 -r ardupilot_gazebo/worlds/iris_runway_15x15_walls.sdf
```

### 2) Start ArduPilot SITL

From your ArduPilot directory:

```bash
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
```

### 3) Start ROS <-> Gazebo camera bridge

```bash
source /opt/ros/humble/setup.bash
source /home/$USER/Senior-Project-Autonomous-Drone/drone_rosws/install/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
"/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
"/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
```

### 4) Start mission and perception nodes

```bash
source /opt/ros/humble/setup.bash
source /home/$USER/Senior-Project-Autonomous-Drone/drone_rosws/install/setup.bash
ros2 run yolo_detector udp_relay
```

Optional:

```bash
ros2 run yolo_detector yolo_node
```

### 5) Start ground station

For state-machine based tracking (`IDLE`, `TRACKING`, `FOUND`) that can trigger mission behavior:

```bash
cd /home/$USER/Senior-Project-Autonomous-Drone/sim_interface
python3 RealGround.py
```

### 6) Run/verify path-planning node manually (optional)

```bash
source /opt/ros/humble/setup.bash
source /home/$USER/Senior-Project-Autonomous-Drone/drone_rosws/install/setup.bash
ros2 run lawnmower lawnmower
```

This node executes a boustrophedon sweep and stops pattern progression when `/object_found` becomes `True`.

## Notes

- The `15x15` world contains four perimeter wall models (`wall_north`, `wall_south`, `wall_east`, `wall_west`).
- If you switch worlds, update topic names that include the world name (for example, camera topics under `/world/<world_name>/...`). In this repo, camera topic strings are currently set directly in `drone_rosws/src/yolo_detector/yolo_detector/udp_relay.py` and `drone_rosws/src/yolo_detector/yolo_detector/yolo_node.py`.
- `sim_interface/ground_station.py` and `sim_interface/RealGround.py` are different interfaces; `RealGround.py` is currently aligned with the tracking state machine used by `udp_relay`.

