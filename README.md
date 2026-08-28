## Senior Project Autonomous Drone



## Prerequisites
- Python environments for UI / YOLO dependencies (as used in your local setup)


```bash
cd /home/$USER/Senior-Project-Autonomous-Drone
python3 -m venv .venv-alt --system-site-packages
source .venv-alt/bin/activate
pip install --upgrade pip
pip install "numpy<2" opencv-python==4.9.0.80 transformers accelerate timm ultralytics PyQt6 MAVProxy future
```

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

# In SITL, paste the following:
output add 127.0.0.1:14550
```

### 3) Start ROS <-> Gazebo camera bridge

```bash
ros2 run ros_gz_bridge parameter_bridge \
"/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
"/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
```

### 4) Start mission and perception nodes in drone_rosws yolo_detector package

```bash
source /home/$USER/Senior-Project-Autonomous-Drone/.venv-alt/bin/activate
python3 udp_relay.py
```



### 5) Start ground station

For state-machine based tracking (`IDLE`, `TRACKING`, `FOUND`) that can trigger mission behavior:

```bash
source /home/$USER/Senior-Project-Autonomous-Drone/.venv-alt/bin/activate
python3 GroundStation_sim.py
```



### 6) Run mavros


```bash
ros2 launch mavros apm.launch fcu_url:=udp://127.0.0.1:14550@

```
