# Grounding DINO + MobileSAM/FastSAM (Alternate Node)

This is an experimental, alternate detection path that does not modify the existing YOLO scripts.

## What this node does

1. Subscribes to camera frames.
2. Uses Grounding DINO to find objects from a natural-language phrase.
3. Optionally refines the best detection with MobileSAM or FastSAM mask.
4. Publishes:
   - `/object_found` (`std_msgs/Bool`)
   - `/grounding_sam_alt/annotated_image` (`sensor_msgs/Image`)
   - `/grounding_sam_alt/status` (`std_msgs/String`)

## File

- `grounding_sam_alt.py`

## Install dependencies

The existing repo already uses `ultralytics` and ROS image tools. Add:

```bash
pip install transformers accelerate timm
```

Optional segmentation backends:

```bash
pip install ultralytics
```

## Run

From workspace root:

```bash
source /opt/ros/humble/setup.bash
source /home/$USER/Senior-Project-Autonomous-Drone/drone_rosws/install/setup.bash
python3 /home/$USER/Senior-Project-Autonomous-Drone/drone_rosws/src/yolo_detector/yolo_detector/grounding_sam_alt.py
```

## Set natural-language target phrase

Publish to the same topic used by your UI:

```bash
ros2 topic pub /target_object std_msgs/msg/String "{data: 'woman with blue shirt'}" -1
```

## Optional parameters

Set via ROS arguments:

```bash
python3 grounding_sam_alt.py --ros-args \
  -p segmenter_backend:=mobile_sam \
  -p segmenter_model_path:=mobile_sam.pt \
  -p frame_stride:=5 \
  -p found_threshold:=0.35
```

Available `segmenter_backend` values:
- `none` (default)
- `mobile_sam`
- `fastsam`

If the segmenter backend fails to load, the node automatically falls back to boxes-only mode.

## Why this helps

- Grounding DINO is better at phrase-level grounding than class-only prompts.
- Optional SAM refinement gives a mask when you need tighter target localization.
- Keeping this in an alternate script lets you compare performance without breaking the current pipeline.
