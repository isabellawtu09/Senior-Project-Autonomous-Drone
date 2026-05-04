import os
import json
import base64
import urllib.request
import numpy as np
import cv2
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")

class VisionLLMClient:
    def is_configured(self):
        """Check if the necessary API keys are loaded."""
        import os
        # Change 'OPENAI_API_KEY' to whichever key your script actually uses
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base = os.environ.get("OPENAI_BASE_URL", "").strip()
        if base:
            self.endpoint = f"{base.rstrip('/')}/chat/completions"
        else:
            self.endpoint = "https://api.openai.com/v1/chat/completions"

    def is_ready(self):
        return bool(self.api_key and self.endpoint)

    # ----------------------------
    # MAIN LLM CALL (Spatial + Semantic)
    # ----------------------------
    def analyze(self, frame: np.ndarray, prompt: str) -> dict:
        if frame is None or not self.is_ready():
            return self._empty()

        # Lower quality slightly to reduce network latency
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return self._empty()

        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        instruction = f"""
You are a robotics vision expert for a drone system.

Task: Locate "{prompt}" in the image and provide visual descriptors.

1. Spatial: Find the bounding box using normalized coordinates (0.0 to 1.0).
   x1/y1 = top-left corner, x2/y2 = bottom-right corner.
2. Semantic: Provide 3-4 YOLO-World class strings. These must be FULL OBJECT DESCRIPTIONS,
   not single attributes. Good: "pink over-ear headphones", "pink audio headset".
   Bad: "pink", "headband", "rounded ear cups".

Return ONLY JSON:
{{
  "found": true/false,
  "confidence": 0-1,
  "x1": 0-1,
  "y1": 0-1,
  "x2": 0-1,
  "y2": 0-1,
  "yolo_terms": ["full object term 1", "full object term 2", "full object term 3"],
  "reason": "short explanation"
}}

If not visible: found=false, confidence=0, x1=0, y1=0, x2=0, y2=0, yolo_terms=[]
"""

        payload = {
            "model": "gpt-4o",
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            },
                        },
                    ],
                },
            ],
        }

        try:
            resp = self._call(payload)
            raw = resp["choices"][0]["message"]["content"]
            data = json.loads(raw) if isinstance(raw, str) else raw
            return self._normalize(data)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return self._empty()

    # ----------------------------
    # BBOX WRAPPER
    # ----------------------------
    def get_bbox(self, prompt: str, frame: np.ndarray):
        result = self.analyze(frame, prompt)

        if not result["found"]:
            return None

        h, w = frame.shape[:2]

        x1 = int(result["x1"] * w)
        y1 = int(result["y1"] * h)
        x2 = int(result["x2"] * w)
        y2 = int(result["y2"] * h)

        # Guard against degenerate boxes
        box_w = max(x2 - x1, 20)
        box_h = max(y2 - y1, 20)

        return [x1, y1, box_w, box_h], result["yolo_terms"]

    # ----------------------------
    # HTTP CALL
    # ----------------------------
    def _call(self, payload):
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception as e:
            print("[LLM CALL FAILED]", e)
            return {"choices": [{"message": {"content": "{}"}}]}

    # ----------------------------
    # DATA HANDLING
    # ----------------------------
    def _normalize(self, d):
        return {
            "found": bool(d.get("found", False)),
            "confidence": float(d.get("confidence", 0.0)),
            "x1": float(d.get("x1", 0.0)),
            "y1": float(d.get("y1", 0.0)),
            "x2": float(d.get("x2", 0.0)),
            "y2": float(d.get("y2", 0.0)),
            "yolo_terms": list(d.get("yolo_terms", [])),
            "reason": str(d.get("reason", "")),
        }

    def _empty(self):
        return {
            "found": False,
            "confidence": 0.0,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 0.0,
            "y2": 0.0,
            "yolo_terms": [],
            "reason": "",
        }