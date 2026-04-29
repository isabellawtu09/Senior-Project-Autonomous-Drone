#!/usr/bin/env python3
"""
Experimental alternate pipeline:
OpenAI Vision (frame + prompt) -> detection decision.

This file is a copy-style alternative to grounding_sam_alt.py that keeps
the same ROS architecture (target topic, found topic, annotated image, status,
and target_info publication), but delegates the decision to an LLM.
"""

import base64
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String


class LlmFrameAltNode(Node):
    def __init__(self) -> None:
        super().__init__("llm_frame_alt")

        # Topics (kept compatible with current relay/UI flow by default).
        self.declare_parameter(
            "camera_topic",
            "/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image",
        )
        self.declare_parameter("target_topic", "/target_object")
        self.declare_parameter("found_topic", "/object_found")
        self.declare_parameter("annotated_topic", "/grounding_sam_alt/annotated_image")
        self.declare_parameter("status_topic", "/grounding_sam_alt/status")
        self.declare_parameter("target_info_topic", "/grounding_sam_alt/target_info")

        # LLM behavior
        self.declare_parameter("frame_stride", 8)
        self.declare_parameter("found_threshold", 0.65)
        self.declare_parameter("model", "5.2")
        self.declare_parameter("request_timeout_s", 15.0)
        self.declare_parameter("max_image_width", 640)
        self.declare_parameter("jpeg_quality", 70)
        self.declare_parameter("openai_api_key", "")
        # Prefer custom base URL / endpoint. No default to OpenAI-hosted URL.
        self.declare_parameter("openai_base_url", "")
        self.declare_parameter("openai_endpoint", "")
        self.declare_parameter("env_file", ".env")
        # Some models/providers reject custom temperature or json response_format.
        self.declare_parameter("use_temperature", True)
        self.declare_parameter("temperature", 0.0)
        self.declare_parameter("use_json_response_format", True)

        self.camera_topic = self.get_parameter("camera_topic").value
        self.target_topic = self.get_parameter("target_topic").value
        self.found_topic = self.get_parameter("found_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.target_info_topic = self.get_parameter("target_info_topic").value

        self.frame_stride = int(self.get_parameter("frame_stride").value)
        self.found_threshold = float(self.get_parameter("found_threshold").value)
        self.model = str(self.get_parameter("model").value)
        self.request_timeout_s = float(self.get_parameter("request_timeout_s").value)
        self.max_image_width = int(self.get_parameter("max_image_width").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.openai_base_url = str(self.get_parameter("openai_base_url").value).strip()
        self.openai_endpoint = str(self.get_parameter("openai_endpoint").value).strip()
        self.env_file = str(self.get_parameter("env_file").value).strip() or ".env"
        self.use_temperature = bool(self.get_parameter("use_temperature").value)
        self.temperature = float(self.get_parameter("temperature").value)
        self.use_json_response_format = bool(self.get_parameter("use_json_response_format").value)

        self.env_file_loaded_from = self.load_env_file(self.env_file)

        # Param wins; otherwise fall back to env (which may have been populated by .env above).
        key_from_param = str(self.get_parameter("openai_api_key").value).strip()
        self.openai_api_key = key_from_param if key_from_param else os.environ.get("OPENAI_API_KEY", "").strip()
        self.openai_api_key_source = "param" if key_from_param else ("env" if self.openai_api_key else "missing")
        if not self.openai_base_url:
            self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get("OPENAI_API_BASE", "").strip()
        self.openai_endpoint = self.resolve_openai_endpoint()

        self.bridge = CvBridge()
        self.target_phrase = ""
        self.frame_count = 0
        self.latest_image: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 5)
        self.target_sub = self.create_subscription(String, self.target_topic, self.target_callback, 10)
        self.found_pub = self.create_publisher(Bool, self.found_topic, 10)
        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 5)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.target_info_pub = self.create_publisher(Float32MultiArray, self.target_info_topic, 10)

        self.timer = self.create_timer(0.08, self.process_latest_frame)

        self.log_startup_config()
        self.publish_status("llm_frame_alt ready. Waiting for target prompt on /target_object")

    def log_startup_config(self) -> None:
        """Print a one-shot summary so you can verify key/base URL wiring at boot."""
        log = self.get_logger()
        if self.env_file_loaded_from:
            log.info(f".env loaded from: {self.env_file_loaded_from}")
        else:
            log.info(
                f".env not found (looked for '{self.env_file}'). Using only existing process env."
            )

        masked_key = self.mask_secret(self.openai_api_key)
        log.info(f"OPENAI_API_KEY ({self.openai_api_key_source}): {masked_key}")
        log.info(f"OPENAI_BASE_URL: '{self.openai_base_url or '(unset)'}'")
        log.info(f"Resolved endpoint: '{self.openai_endpoint or '(unset)'}'")
        log.info(f"Model: {self.model}")

        if not self.openai_api_key:
            log.warn("OPENAI_API_KEY is empty. LLM decisions will be disabled.")
        if not self.openai_endpoint:
            log.warn(
                "OpenAI endpoint/base URL is empty. "
                "Set openai_endpoint or openai_base_url (or OPENAI_ENDPOINT / OPENAI_BASE_URL)."
            )
            return

        # Sanity-check the resolved URL.
        if not (self.openai_endpoint.startswith("http://") or self.openai_endpoint.startswith("https://")):
            log.error(
                f"Endpoint missing http(s) scheme: '{self.openai_endpoint}'. "
                "Set OPENAI_BASE_URL to something like https://host/v1"
            )
        if self.openai_endpoint.endswith("/chat/completions/chat/completions"):
            log.error(
                "Endpoint already contains /chat/completions twice. "
                "OPENAI_BASE_URL should be the API root (e.g. https://host/v1), not include /chat/completions."
            )

    @staticmethod
    def mask_secret(secret: str) -> str:
        if not secret:
            return "(empty)"
        if len(secret) <= 8:
            return f"len={len(secret)} {'*' * len(secret)}"
        return f"len={len(secret)} {secret[:4]}...{secret[-4:]}"

    def load_env_file(self, env_file: str) -> str:
        """
        Lightweight .env loader (KEY=VALUE), used to avoid extra dependency.
        Existing environment variables are preserved. Returns the path that was
        loaded, or an empty string if no .env file was found.
        """
        env_path = Path(env_file)
        if not env_path.is_absolute():
            env_path = Path.cwd() / env_path
        if not env_path.exists():
            return ""
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                # Tolerate `export KEY=value` lines.
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return str(env_path)
        except Exception as exc:
            self.get_logger().warn(f"Failed reading env file '{env_path}': {exc}")
            return ""

    def target_callback(self, msg: String) -> None:
        phrase = msg.data.strip().lower()
        if not phrase or phrase == "stop":
            self.target_phrase = ""
            self.publish_status("Target cleared. Waiting for prompt.")
            return
        self.target_phrase = phrase
        self.publish_status(f"Target updated: {self.target_phrase}")

    def resolve_openai_endpoint(self) -> str:
        # 1) explicit endpoint param
        if self.openai_endpoint:
            return self.openai_endpoint
        # 2) env explicit endpoint
        env_endpoint = os.environ.get("OPENAI_ENDPOINT", "").strip()
        if env_endpoint:
            return env_endpoint

        # 3) base URL param or env
        base = self.openai_base_url or os.environ.get("OPENAI_BASE_URL", "").strip()
        if not base:
            base = os.environ.get("OPENAI_API_BASE", "").strip()
        if not base:
            return ""
        return f"{base.rstrip('/')}/chat/completions"

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

        try:
            annotated, found, info = self.detect_with_llm(frame, self.target_phrase)
        except Exception as exc:
            self.get_logger().error(f"LLM detection error: {exc}")
            self.publish_status(f"LLM error: {exc}")
            annotated = frame
            found = False
            info = self.make_target_info(False, 0.0, 0.5, 0.5, 0.0)

        found_msg = Bool()
        found_msg.data = found
        self.found_pub.publish(found_msg)
        self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        self.publish_target_info(info)

    def detect_with_llm(self, frame_bgr: np.ndarray, phrase: str) -> Tuple[np.ndarray, bool, dict]:
        phrase = phrase.strip()
        if not phrase:
            return frame_bgr, False, self.make_target_info(False, 0.0, 0.5, 0.5, 0.0)

        if not self.openai_api_key:
            self.publish_status("No OpenAI API key available; cannot run LLM detection.")
            return frame_bgr, False, self.make_target_info(False, 0.0, 0.5, 0.5, 0.0)
        if not self.openai_endpoint:
            self.publish_status("No OpenAI endpoint/base URL configured; cannot run LLM detection.")
            return frame_bgr, False, self.make_target_info(False, 0.0, 0.5, 0.5, 0.0)

        small = self.resize_for_api(frame_bgr)
        payload = self.build_openai_payload(small, phrase)
        response = self.call_openai(payload)
        decision = self.parse_decision(response)

        confidence = float(decision.get("confidence", 0.0))
        found_flag = bool(decision.get("found", False)) and confidence >= self.found_threshold
        cx = float(decision.get("cx", 0.5))
        cy = float(decision.get("cy", 0.5))
        area = float(decision.get("area", 0.0))
        reason = str(decision.get("reason", "")).strip()

        info = self.make_target_info(True, confidence, cx, cy, area)
        annotated = self.draw_overlay(frame_bgr, phrase, found_flag, confidence, cx, cy, area, reason)
        self.publish_status(
            f"LLM decision found={found_flag} conf={confidence:.2f} prompt='{phrase}' reason='{reason[:80]}'"
        )
        return annotated, found_flag, info

    def resize_for_api(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if w <= self.max_image_width:
            return frame_bgr
        scale = float(self.max_image_width) / float(max(w, 1))
        nh = max(1, int(h * scale))
        return cv2.resize(frame_bgr, (self.max_image_width, nh), interpolation=cv2.INTER_AREA)

    def build_openai_payload(self, frame_bgr: np.ndarray, phrase: str) -> dict:
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("Failed to encode frame for OpenAI request")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        instruction = (
            "You are a strict vision detector for a drone search task. "
            f"Target prompt: '{phrase}'. "
            "Return ONLY valid JSON (no markdown, no prose) with keys: "
            "found (boolean), confidence (0..1), "
            "cx (0..1), cy (0..1), area (0..1), reason (short string). "
            "If target is not visible, set found=false and confidence near 0."
        )
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only. No markdown."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
        }
        if self.use_temperature:
            payload["temperature"] = self.temperature
        if self.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def call_openai(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }
        # One-line per-request audit: confirms which endpoint/model/auth-prefix is in use.
        self.get_logger().debug(
            f"POST {self.openai_endpoint} model={payload.get('model')} "
            f"auth=Bearer {self.mask_secret(self.openai_api_key)} bytes={len(body)}"
        )
        req = urllib.request.Request(
            self.openai_endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI HTTP error {exc.code} from {self.openai_endpoint}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed to {self.openai_endpoint}: {exc}") from exc

    def parse_decision(self, api_response: dict) -> dict:
        content = ""
        decision: dict = {}
        try:
            content = api_response["choices"][0]["message"]["content"]
            if isinstance(content, list):
                # Some providers return content as a list of parts; join text parts.
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            decision = json.loads(content)
        except Exception:
            decision = self.try_extract_json(str(content))

        if not isinstance(decision, dict):
            decision = {}

        return {
            "found": bool(decision.get("found", False)),
            "confidence": float(max(0.0, min(float(decision.get("confidence", 0.0) or 0.0), 1.0))),
            "cx": float(max(0.0, min(float(decision.get("cx", 0.5) or 0.5), 1.0))),
            "cy": float(max(0.0, min(float(decision.get("cy", 0.5) or 0.5), 1.0))),
            "area": float(max(0.0, min(float(decision.get("area", 0.0) or 0.0), 1.0))),
            "reason": str(decision.get("reason", "") or ""),
        }

    def try_extract_json(self, text: str) -> dict:
        """Best-effort JSON extraction from model output (handles ```json fences)."""
        if not text:
            return {}
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        phrase: str,
        found: bool,
        confidence: float,
        cx: float,
        cy: float,
        area: float,
        reason: str,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        color = (0, 220, 0) if found else (0, 180, 255)
        status = "FOUND" if found else "SEARCHING"
        cv2.putText(out, f"{status} {phrase} conf={confidence:.2f}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out, reason[:80], (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Draw a coarse estimated box from normalized center/area, if provided.
        if area > 0.0:
            box_w = int(max(12, min(w - 1, w * np.sqrt(area))))
            box_h = int(max(12, min(h - 1, h * np.sqrt(area))))
            px = int(cx * w)
            py = int(cy * h)
            x1 = max(0, min(w - 1, px - box_w // 2))
            y1 = max(0, min(h - 1, py - box_h // 2))
            x2 = max(0, min(w - 1, x1 + box_w))
            y2 = max(0, min(h - 1, y1 + box_h))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        return out

    def publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def make_target_info(self, visible: bool, score: float, cx: float, cy: float, area: float) -> dict:
        return {
            "visible": 1.0 if visible else 0.0,
            "score": float(max(0.0, min(score, 1.0))),
            "cx": float(max(0.0, min(cx, 1.0))),
            "cy": float(max(0.0, min(cy, 1.0))),
            "area": float(max(0.0, min(area, 1.0))),
        }

    def publish_target_info(self, info: dict) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(info.get("visible", 0.0)),
            float(info.get("score", 0.0)),
            float(info.get("cx", 0.5)),
            float(info.get("cy", 0.5)),
            float(info.get("area", 0.0)),
        ]
        self.target_info_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LlmFrameAltNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
