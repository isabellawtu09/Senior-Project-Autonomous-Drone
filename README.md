## Senior Project Autonomous Drone 
### Note:
* User should have ardupilot installed prior to cloning this repository
* Ensure that paths match your own local setup eg:
  `export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/home/joanna/seniorproject_repo/ardupilot_gazebo/models`
  then,
  `source ~./bashrc `
* Also, ensure that you have sourced your ROS2 installation
* Necessary virtual environments have been created for mavproxy, the yolo and udp relay node within the `yolo_detector` package
* 

### Current Interface Setup Involves the following:

#### 1. World

`gz sim -v4 -r iris_objects_runway.sdf`

- Models imported: green and red coke can, mug

#### 2. Run camera bridge

`ros2 run ros_gz_bridge parameter_bridge \
"/world/iris_objects_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image@sensor_msgs/msg/Imag
e[gz.msgs.Image" \
"/world/iris_objects_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/camera_info@sensor_msgs/ms
g/CameraInfo[gz.msgs.CameraInfo"`

#### 3. Optional - run image view

 `ros2 run image_view image_view --ros-args --remap image:=/ultralytics/detection/image`

#### 4. Run yolo node

Ran in this directory

`srproject_repo/sim/drone_rosws/src/yolo_detector/yolo_detector`

Command:

`python3 yolo_node.py`

#### 5. Run ground station

activate `ardupilot-venv`

Command:

`python ground_station.py`

#### 6. Run udp relay node ( companion_script)

activate `.venv` 

Command:

`ros2 run yolo_detector udp_relay`
or
`python3 src/yolo_detector/yolo_detector/udp_relay.py` from `drone_rosws`

