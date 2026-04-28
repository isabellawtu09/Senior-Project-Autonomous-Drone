#!/usr/bin/env python3
"""
Experimental alternate pipeline:
Grounding DINO (text-to-box) + optional MobileSAM/FastSAM mask refinement.

This script is intentionally separate from the existing YOLO workflow.
Run it directly while keeping current nodes unchanged.
"""

import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundingSamAltNode(Node):
    def __init__(self) -> None:
        super().__init__("grounding_sam_alt")

        # Topics
        self.declare_parameter(
            "camera_topic",
            "/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image",
        )
        self.declare_parameter("target_topic", "/target_object")
        self.declare_parameter("found_topic", "/object_found")
        self.declare_parameter("annotated_topic", "/grounding_sam_alt/annotated_image")
        self.declare_parameter("status_topic", "/grounding_sam_alt/status")

        # Detection behavior
        self.declare_parameter("frame_stride", 5)
        self.declare_parameter("box_threshold", 0.30)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("found_threshold", 0.35)
        self.declare_parameter("model_id", "IDEA-Research/grounding-dino-tiny")

        # Segmentation backend: none | mobile_sam | fastsam
        self.declare_parameter("segmenter_backend", "none")
        self.declare_parameter("segmenter_model_path", "mobile_sam.pt")

        self.camera_topic = self.get_parameter("camera_topic").value
        self.target_topic = self.get_parameter("target_topic").value
        self.found_topic = self.get_parameter("found_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.status_topic = self.get_parameter("status_topic").value

        self.frame_stride = int(self.get_parameter("frame_stride").value)
        self.box_threshold = float(self.get_parameter("box_threshold").value)
        self.text_threshold = float(self.get_parameter("text_threshold").value)
        self.found_threshold = float(self.get_parameter("found_threshold").value)
        self.model_id = str(self.get_parameter("model_id").value)
        self.segmenter_backend = str(self.get_parameter("segmenter_backend").value).lower().strip()
        self.segmenter_model_path = str(self.get_parameter("segmenter_model_path").value).strip()

        self.bridge = CvBridge()
        # No hardcoded detection target: run detection only after explicit prompt.
        self.target_phrase = ""
        self.frame_count = 0
        self.latest_image: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.segmenter_warned = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading Grounding DINO model '{self.model_id}' on {self.device}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

        # Optional segmenter handles (lazy-loaded)
        self.mobile_sam = None
        self.fast_sam = None

        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 5)
        self.target_sub = self.create_subscription(String, self.target_topic, self.target_callback, 10)
        self.found_pub = self.create_publisher(Bool, self.found_topic, 10)
        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 5)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        # Run detector on timer so image callback stays lightweight.
        self.timer = self.create_timer(0.05, self.process_latest_frame)

        self.get_logger().info(
            "grounding_sam_alt started. Set /target_object to phrase prompts, "
            "e.g. 'woman with blue shirt'."
        )
        self.publish_status("Waiting for target prompt on /target_object")

    def target_callback(self, msg: String) -> None:
        phrase = msg.data.strip().lower()
        if not phrase or phrase == "stop":
            self.target_phrase = ""
            self.publish_status("Target cleared. Waiting for prompt.")
            return
        self.target_phrase = phrase
        self.publish_status(f"Target updated: {self.target_phrase}")

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_image = frame

    def process_latest_frame(self) -> None:
        self.frame_count += 1
        if self.frame_count % max(self.frame_stride, 1) != 0:
            return

        with self.lock:
            if self.latest_image is None:
                return
            frame = self.latest_image.copy()

        annotated, found = self.detect_and_annotate(frame, self.target_phrase)

        found_msg = Bool()
        found_msg.data = found
        self.found_pub.publish(found_msg)
        self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

    def detect_and_annotate(self, frame_bgr: np.ndarray, phrase: str) -> Tuple[np.ndarray, bool]:
        phrase = phrase.strip()
        if not phrase:
            return frame_bgr, False

        # Grounding DINO performs better with sentence-like prompts.
        prompt = phrase if phrase.endswith(".") else f"{phrase}."

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=frame_rgb, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [frame_rgb.shape[:2]]
        # Transformers changed this API in different versions:
        # some use box_threshold=..., others use threshold=...
        try:
            detections = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            detections = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]

        boxes = detections.get("boxes", [])
        scores = detections.get("scores", [])
        labels = detections.get("labels", [])

        if len(boxes) == 0:
            self.publish_status(f"No match: {phrase}")
            return frame_bgr, False

        best_idx = int(torch.argmax(scores).item()) if isinstance(scores, torch.Tensor) else 0
        best_box = boxes[best_idx].tolist() if isinstance(boxes[best_idx], torch.Tensor) else boxes[best_idx]
        best_score = float(scores[best_idx].item() if isinstance(scores[best_idx], torch.Tensor) else scores[best_idx])
        best_label = str(labels[best_idx])

        x1, y1, x2, y2 = [int(v) for v in best_box]
        x1 = max(0, min(x1, frame_bgr.shape[1] - 1))
        x2 = max(0, min(x2, frame_bgr.shape[1] - 1))
        y1 = max(0, min(y1, frame_bgr.shape[0] - 1))
        y2 = max(0, min(y2, frame_bgr.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            return frame_bgr, False

        annotated = frame_bgr.copy()
        found = best_score >= self.found_threshold
        if found:
            mask = self.segment_from_box(frame_bgr, (x1, y1, x2, y2))
            if mask is not None:
                overlay = np.zeros_like(annotated, dtype=np.uint8)
                overlay[:, :, 1] = 180
                alpha = 0.35
                annotated = np.where(mask[:, :, None], (1 - alpha) * annotated + alpha * overlay, annotated).astype(
                    np.uint8
                )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            text = f"{best_label} {best_score:.2f}"
            cv2.putText(annotated, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        self.publish_status(
            f"Best match '{best_label}' score={best_score:.2f} found={found} backend={self.segmenter_backend}"
        )
        return annotated, found

    def segment_from_box(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self.segmenter_backend == "none":
            return None
        if self.segmenter_backend == "mobile_sam":
            return self.segment_mobile_sam(frame_bgr, box)
        if self.segmenter_backend == "fastsam":
            return self.segment_fastsam(frame_bgr, box)
        return None

    def segment_mobile_sam(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            if self.mobile_sam is None:
                from ultralytics import SAM

                self.mobile_sam = SAM(self.segmenter_model_path)

            results = self.mobile_sam.predict(frame_bgr, bboxes=[list(box)], verbose=False)
            if not results or results[0].masks is None or results[0].masks.data is None:
                return None
            data = results[0].masks.data[0].detach().cpu().numpy()
            return data > 0.5
        except Exception as exc:
            self.warn_once(f"MobileSAM unavailable, using boxes only. Details: {exc}")
            return None

    def segment_fastsam(self, frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        try:
            if self.fast_sam is None:
                from ultralytics import FastSAM

                self.fast_sam = FastSAM(self.segmenter_model_path)

            results = self.fast_sam(frame_bgr, verbose=False)
            if not results or results[0].masks is None or results[0].masks.data is None:
                return None
            masks = results[0].masks.data.detach().cpu().numpy() > 0.5
            if masks.size == 0:
                return None
            x1, y1, x2, y2 = box
            # Pick the mask with largest overlap in the prompted box.
            best_mask = None
            best_overlap = 0
            for mask in masks:
                overlap = int(mask[y1:y2, x1:x2].sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_mask = mask
            return best_mask
        except Exception as exc:
            self.warn_once(f"FastSAM unavailable, using boxes only. Details: {exc}")
            return None

    def warn_once(self, message: str) -> None:
        if not self.segmenter_warned:
            self.segmenter_warned = True
            self.get_logger().warn(message)

    def publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundingSamAltNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
