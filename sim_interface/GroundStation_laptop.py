#!/usr/bin/env python3
"""
Standalone GCS for laptop camera testing — no drone or ROS required.
Press START TRACKING to begin; mission timer starts automatically.
Logs time-to-detection and records video to metrics/.
"""
import os
import sys
import threading
import time
import queue

import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

from llm_client import VisionLLMClient

LOG_PATH = 'metrics/mission_times.md'
CAMERA_INDEX = 0

global_latest_frame = None
frame_write_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)


# --- HSV colour histogram helpers ---
def compute_hue_histogram(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    if mask.sum() < 100:
        return None
    h_hist = cv2.calcHist([hsv], [0], mask, [90], [0, 180])
    total = h_hist.sum()
    if total < 1e-6:
        return None
    return h_hist.flatten() / total


def hue_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(np.sum(np.sqrt(h1 * h2)))


def dominant_hue(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return -1
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)
    hues = hsv[:, :, 0][mask]
    if len(hues) == 0:
        return -1
    return int(np.median(hues))


# --- 1. WEBCAM CAPTURE THREAD ---
class WebcamThread(QThread):
    def __init__(self, camera_index=CAMERA_INDEX):
        super().__init__()
        self._camera_index = camera_index

    def run(self):
        global global_latest_frame
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            print(f'[ERROR] Cannot open camera {self._camera_index}')
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            with frame_write_lock:
                global_latest_frame = frame.copy()
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        cap.release()


# --- 2. INFERENCE THREAD ---
class InferenceThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    state_change_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    ACCEPT_THRESHOLD = 0.55
    UPDATE_GATE = 0.85

    def __init__(self):
        super().__init__()
        self.model = None
        self.target_bbox = None
        self.tracks = {}
        self.next_track_id = 1
        self.master_hsv_hist = None
        self.master_dominant_hue = -1
        self.is_tracking = False
        self.use_hsv = False
        self.track_lock = threading.Lock()

    def _load_model(self):
        self.log_signal.emit('[YOLO] Loading YOLO-World...')
        self.model = YOLO('yolov8m-world.pt')
        self.log_signal.emit('[YOLO] Model ready.')

    def _hue_veto(self, cand_hue):
        if self.master_dominant_hue < 0 or cand_hue < 0:
            return False
        diff = abs(self.master_dominant_hue - cand_hue)
        diff = min(diff, 180 - diff)
        return diff > 20

    def _update_references(self, crop_bgr):
        new_hue = compute_hue_histogram(crop_bgr)
        if new_hue is not None and self.master_hsv_hist is not None:
            if hue_similarity(self.master_hsv_hist, new_hue) > self.UPDATE_GATE:
                self.master_hsv_hist = 0.95 * self.master_hsv_hist + 0.05 * new_hue
                n = self.master_hsv_hist.sum()
                if n > 1e-6:
                    self.master_hsv_hist /= n

    def _set_classes(self, class_list):
        if self.model is not None and len(class_list) > 0:
            combined = list(dict.fromkeys(class_list[:2] + ['object']))
            if torch.backends.mps.is_available():
                self.model.to('cpu')
            self.model.set_classes(combined)
            if global_latest_frame is not None:
                dummy = cv2.resize(global_latest_frame, (640, 640))
                device = 'mps' if torch.backends.mps.is_available() else 'cpu'
                self.model.to(device)
                self.model.predict(dummy, verbose=False, conf=0.25, device=device)
            self.log_signal.emit(f'[YOLO] Classes set: {combined}')

    def start_cv_tracker(self, frame, bbox, yolo_terms, preserve_appearance=False, use_hsv=False):
        with self.track_lock:
            self.target_bbox = tuple(map(int, bbox)) if bbox else None
            self.tracks = {}
            self.next_track_id = 1
            self.use_hsv = use_hsv
            if not preserve_appearance:
                self.master_hsv_hist = None
                self.master_dominant_hue = -1
            self._set_classes(yolo_terms)
            self.is_tracking = True
            search_area = self.target_bbox if self.target_bbox else 'Full Frame'
            self.log_signal.emit(
                f"[TRACKER] Multi-Object Tracking | Initial Latch Area: {search_area} "
                f"| HSV Color Filtering: {'ON' if use_hsv else 'OFF'}"
            )

    def stop_cv_tracker(self):
        with self.track_lock:
            self.is_tracking = False
            self.target_bbox = None
            self.tracks = {}
            self.master_hsv_hist = None
            self.master_dominant_hue = -1
            self._set_classes([])

    def run(self):
        global global_latest_frame
        current_state = 'IDLE'
        self._load_model()
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        while True:
            frame = frame_queue.get()
            if frame is None:
                continue
            with frame_write_lock:
                global_latest_frame = frame.copy()

            found_any = False
            recovering_any = False

            with self.track_lock:
                if self.is_tracking and self.model is not None:
                    try:
                        self.model.to(device)
                        frame_h, frame_w = frame.shape[:2]
                        small = cv2.resize(frame, (640, 640))
                        scale_x = frame_w / 640.0
                        scale_y = frame_h / 640.0

                        results = self.model.predict(small, verbose=False, conf=0.25, device=device)[0]
                        num_raw = len(results.boxes) if results.boxes is not None else 0

                        valid_boxes = []
                        if num_raw > 0:
                            raw_boxes = results.boxes.xyxy.cpu().numpy()
                            all_boxes = raw_boxes * [scale_x, scale_y, scale_x, scale_y]
                            all_confs = results.boxes.conf.cpu().numpy()
                            for i in range(len(all_boxes)):
                                if all_confs[i] > 0.05:
                                    bx1, by1, bx2, by2 = map(int, all_boxes[i])
                                    crop = frame[max(0, by1):min(frame_h, by2), max(0, bx1):min(frame_w, bx2)]
                                    valid_boxes.append((bx1, by1, bx2 - bx1, by2 - by1, all_confs[i], crop))

                        # Stage 0: latch master HSV colour on initial target bbox
                        if self.use_hsv and self.master_hsv_hist is None and self.target_bbox is not None:
                            tx = self.target_bbox[0] + self.target_bbox[2] // 2
                            ty = self.target_bbox[1] + self.target_bbox[3] // 2
                            best_i, best_conf = None, -1
                            for i, (bx1, by1, bw, bh, conf, crop) in enumerate(valid_boxes):
                                margin_x = max(30, self.target_bbox[2] // 3)
                                margin_y = max(30, self.target_bbox[3] // 3)
                                in_zone = (
                                    (bx1 - margin_x) <= tx <= (bx1 + bw + margin_x) and
                                    (by1 - margin_y) <= ty <= (by1 + bh + margin_y)
                                )
                                if in_zone and conf > best_conf:
                                    best_conf, best_i = conf, i
                            if best_i is not None:
                                _, _, _, _, _, crop = valid_boxes[best_i]
                                self.master_hsv_hist = compute_hue_histogram(crop)
                                self.master_dominant_hue = dominant_hue(crop)
                                self.log_signal.emit('[TRACKER] Master HSV Signature Extracted. Building swarm profile.')

                        has_ref = self.use_hsv and (self.master_hsv_hist is not None)
                        matched_track_ids = set()
                        matched_box_indices = set()

                        # Stage 1: update existing tracks
                        for box_idx, (x, y, w, h, conf, crop) in enumerate(valid_boxes):
                            best_iou = 0
                            best_tid = None
                            for tid, track in self.tracks.items():
                                if tid in matched_track_ids:
                                    continue
                                tx, ty, tw, th = track['bbox']
                                ix1, iy1 = max(x, tx), max(y, ty)
                                ix2, iy2 = min(x + w, tx + tw), min(y + h, ty + th)
                                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                                iou = inter / (w * h + tw * th - inter + 1e-6)
                                if iou > best_iou:
                                    best_iou, best_tid = iou, tid
                            if best_iou > 0.10:
                                self.tracks[best_tid]['bbox'] = (x, y, w, h)
                                self.tracks[best_tid]['last_seen'] = time.time()
                                self.tracks[best_tid]['conf'] = conf
                                matched_track_ids.add(best_tid)
                                matched_box_indices.add(box_idx)
                                if has_ref:
                                    self._update_references(crop)

                        # Stage 2: register new objects
                        for box_idx, (x, y, w, h, conf, crop) in enumerate(valid_boxes):
                            if box_idx in matched_box_indices:
                                continue
                            is_valid = False
                            if has_ref:
                                cand_hue = dominant_hue(crop)
                                if not self._hue_veto(cand_hue):
                                    if hue_similarity(self.master_hsv_hist, compute_hue_histogram(crop)) > self.ACCEPT_THRESHOLD:
                                        is_valid = True
                            elif conf > 0.10:
                                is_valid = True

                            if not self.tracks and self.target_bbox is not None and not is_valid:
                                tx, ty, tw, th = self.target_bbox
                                ix1, iy1 = max(x, tx), max(y, ty)
                                ix2, iy2 = min(x + w, tx + tw), min(y + h, ty + th)
                                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                                if inter / (w * h + tw * th - inter + 1e-6) > 0.10:
                                    is_valid = True

                            if is_valid:
                                self.tracks[self.next_track_id] = {'bbox': (x, y, w, h), 'last_seen': time.time(), 'conf': conf}
                                self.next_track_id += 1

                        # Stage 3: cleanup and draw
                        current_time = time.time()
                        stale_ids = []
                        for tid, track in list(self.tracks.items()):
                            time_since = current_time - track['last_seen']
                            x, y, w, h = track['bbox']
                            if time_since < 0.2:
                                found_any = True
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(frame, f"ID:{tid} ({track['conf']:.2f})", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            elif time_since < 3.0:
                                recovering_any = True
                            else:
                                stale_ids.append(tid)
                        for tid in stale_ids:
                            del self.tracks[tid]
                            self.log_signal.emit(f'[TRACKER] Dropped ID:{tid} due to occlusion.')

                        if not found_any and not recovering_any and self.is_tracking:
                            cv2.putText(frame, 'RECOVERING / SEARCHING...', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    except Exception as e:
                        self.log_signal.emit(f'[TRACKER ERROR] {e}')

            if found_any:
                active_count = len([t for t in self.tracks.values() if time.time() - t['last_seen'] < 0.2])
                current_state = f'FOUND ({active_count} Targets)'
            elif recovering_any:
                current_state = 'LOST (RECOVERING)'
            else:
                current_state = 'TRACKING' if self.is_tracking else 'IDLE'

            self.state_change_signal.emit(current_state)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_img)


# --- 3. AI FINDER THREAD ---
class AIFinderThread(QThread):
    bbox_found_signal = pyqtSignal(list, list)
    log_signal = pyqtSignal(str)

    def __init__(self, llm_client, prompt):
        super().__init__()
        self.llm_client = llm_client
        self.prompt = prompt
        self.running = True
        self.last_call = 0

    def run(self):
        self.log_signal.emit(f"[LLM] Background scanning for: '{self.prompt}'...")
        while self.running:
            with frame_write_lock:
                snap = global_latest_frame.copy() if global_latest_frame is not None else None
            if snap is None:
                time.sleep(0.1)
                continue
            if time.time() - self.last_call < 0.5:
                time.sleep(0.2)
                continue
            self.last_call = time.time()
            try:
                small_snap = cv2.resize(snap, (640, 640))
                result = self.llm_client.get_bbox(self.prompt, small_snap)
                if result:
                    bbox_small, yolo_terms = result
                    self.log_signal.emit(f'[LLM] Target Reasoned. Handing off to YOLO: {yolo_terms}')
                    self.bbox_found_signal.emit(bbox_small, yolo_terms)
                    return
            except Exception as e:
                self.log_signal.emit(f'[LLM ERROR] {e}')
            time.sleep(0.3)

    def stop(self):
        self.running = False


# --- 4. MAIN GUI CLASS ---
class GroundStation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Drone Tracker GCS — Laptop Camera')
        self.resize(1100, 850)

        self.llm_client = VisionLLMClient()
        self.hunter_thread = None
        self.current_use_hsv = False

        self._mission_start = None
        self._mission_found = False
        self._prompt = ''
        self._ai_mode = False

        self.setStyleSheet("""
            QWidget { background-color: #1e1e2e; font-family: 'Segoe UI', Arial, sans-serif; }
            QLabel#statusLabel { font-size: 16px; font-weight: bold; padding: 10px; background-color: #313244; border-radius: 8px; color: #cdd6f4; }
            QLabel#videoFeed { background-color: #11111b; border: 2px solid #45475a; border-radius: 12px; }
            QLineEdit { background-color: #313244; color: #cdd6f4; font-size: 16px; padding: 10px 15px; border: 2px solid #45475a; border-radius: 8px; }
            QLineEdit:focus { border: 2px solid #89b4fa; }
            QPushButton#startBtn { background-color: #89b4fa; color: #11111b; font-size: 16px; font-weight: bold; padding: 10px 20px; border-radius: 8px; }
            QPushButton#startBtn:hover { background-color: #b4befe; }
            QPushButton#startBtn:pressed { background-color: #74c7ec; }
            QPushButton#stopBtn { background-color: #f38ba8; color: #11111b; font-size: 16px; font-weight: bold; padding: 10px 20px; border-radius: 8px; }
            QPushButton#stopBtn:hover { background-color: #eba0ac; }
            QPushButton#stopBtn:pressed { background-color: #e64553; }
            QTextEdit { background-color: #11111b; color: #a6e3a1; font-family: 'Courier New', monospace; font-size: 14px; border: 2px solid #45475a; border-radius: 8px; padding: 10px; }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        self.setLayout(layout)

        self.status_label = QLabel('STATUS: IDLE | Laptop Camera')
        self.status_label.setObjectName('statusLabel')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.video_label = QLabel('Starting camera...')
        self.video_label.setObjectName('videoFeed')
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet('color: #6c7086; font-size: 20px; font-weight: bold;')
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label, stretch=5)

        controls = QHBoxLayout()
        controls.setSpacing(15)

        self.ai_toggle = QCheckBox('AI Mode')
        self.ai_toggle.setStyleSheet('color: #cdd6f4; font-size: 16px; font-weight: bold;')
        controls.addWidget(self.ai_toggle)

        self.textBox = QLineEdit()
        self.textBox.setPlaceholderText("Enter Target Description (check 'AI Mode' for abstract requests)")
        self.textBox.setFixedHeight(50)
        controls.addWidget(self.textBox, stretch=4)

        self.track_button = QPushButton('START TRACKING')
        self.track_button.setObjectName('startBtn')
        self.track_button.setFixedHeight(50)
        self.track_button.clicked.connect(self.start_tracking)
        controls.addWidget(self.track_button, stretch=1)

        self.stop_button = QPushButton('STOP TRACKING')
        self.stop_button.setObjectName('stopBtn')
        self.stop_button.setFixedHeight(50)
        self.stop_button.clicked.connect(self.stop_tracking)
        controls.addWidget(self.stop_button, stretch=1)

        layout.addLayout(controls)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(120)
        layout.addWidget(self.log_console, stretch=1)

        self.cam_thread = WebcamThread(CAMERA_INDEX)
        self.cam_thread.start()

        self.inf = InferenceThread()
        self.inf.change_pixmap_signal.connect(self.update_image)
        self.inf.state_change_signal.connect(self.update_status)
        self.inf.log_signal.connect(self.append_log)
        self.inf.start()

        if not self.llm_client.is_configured():
            self.append_log('[WARNING] OPENAI_API_KEY not found. AI Mode will fall back to direct text.')
        self.append_log('[SYSTEM] Laptop camera mode. Enter a target and press START TRACKING.')

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        scaled = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.video_label.setPixmap(scaled)

    @pyqtSlot(str)
    def update_status(self, new_state):
        if 'FOUND' in new_state:
            color = '#a6e3a1'
            if self._mission_start is not None and not self._mission_found:
                self._mission_found = True
                elapsed = time.time() - self._mission_start
                mins, secs = divmod(elapsed, 60)
                self.append_log(f'[TIMER] Object found! Time to detection: {int(mins)}m {secs:.1f}s ({elapsed:.1f}s)')
                self._save_result(elapsed, mins, secs, found=True)
        elif 'LOST' in new_state:
            color = '#fab387'
        elif 'TRACKING' in new_state:
            color = '#f9e2af'
        else:
            color = '#cdd6f4'

        self.status_label.setText(f'STATUS: {new_state} | Laptop Camera')
        self.status_label.setStyleSheet(
            f'color: {color}; background-color: #313244; border-radius: 8px; '
            f'font-size: 16px; font-weight: bold; padding: 10px;'
        )

    @pyqtSlot(str)
    def append_log(self, message):
        timestamp = time.strftime('%H:%M:%S')
        self.log_console.append(f'[{timestamp}] {message}')
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    @pyqtSlot(list, list)
    def handle_ai_targets(self, bbox, yolo_terms):
        with frame_write_lock:
            if global_latest_frame is None:
                return
            live_h, live_w = global_latest_frame.shape[:2]
            snap_copy = global_latest_frame.copy()

        scale_w = live_w / 640
        scale_h = live_h / 640
        x, y, w, h = bbox
        scaled_bbox = [int(x * scale_w), int(y * scale_h), int(w * scale_w), int(h * scale_h)]
        self.append_log(f'[SYSTEM] Scaling bounding box: 640x640 -> {live_w}x{live_h}')
        has_ref = self.inf.master_hsv_hist is not None
        self.inf.start_cv_tracker(snap_copy, scaled_bbox, yolo_terms, preserve_appearance=has_ref, use_hsv=self.current_use_hsv)
        self.hunter_thread = None

    def start_tracking(self):
        user_input = self.textBox.text().strip()
        if not user_input:
            self.append_log('[ERROR] Target description cannot be empty.')
            return

        self.stop_tracking()

        self._prompt = user_input
        self._ai_mode = self.ai_toggle.isChecked()
        self._mission_start = time.time()
        self._mission_found = False
        self.append_log('[TIMER] Mission started.')

        KNOWN_COLORS = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'brown', 'grey', 'gray', 'cyan', 'magenta', 'maroon',
        ]
        user_words = user_input.lower().replace(',', ' ').replace('.', ' ').split()
        self.current_use_hsv = any(c in user_words for c in KNOWN_COLORS)

        if not self.ai_toggle.isChecked() or not self.llm_client.is_configured():
            if self.ai_toggle.isChecked() and not self.llm_client.is_configured():
                self.append_log('[WARNING] No API key. Falling back to direct input.')
            self.append_log(f"[SYSTEM] Direct mode. Searching globally for: '{user_input}'")
            with frame_write_lock:
                snap_copy = global_latest_frame.copy() if global_latest_frame is not None else None
            self.inf.start_cv_tracker(snap_copy, None, [user_input.lower()], preserve_appearance=False, use_hsv=self.current_use_hsv)
            return

        self.hunter_thread = AIFinderThread(self.llm_client, user_input)
        self.hunter_thread.log_signal.connect(self.append_log)
        self.hunter_thread.bbox_found_signal.connect(self.handle_ai_targets)
        self.hunter_thread.start()

    def stop_tracking(self):
        if self.hunter_thread is not None and self.hunter_thread.isRunning():
            self.hunter_thread.stop()
            self.hunter_thread.wait()
            self.append_log('[SYSTEM] Aborting background AI Hunter.')
            self.hunter_thread = None

        self.inf.stop_cv_tracker()

        if self._mission_start is not None and not self._mission_found:
            elapsed = time.time() - self._mission_start
            mins, secs = divmod(elapsed, 60)
            self.append_log(
                f'[TIMER] Stopped. Time: {int(mins)}m {secs:.1f}s ({elapsed:.1f}s) — object NOT found.'
            )
            self._save_result(elapsed, mins, secs, found=False)

        self._mission_start = None
        self._mission_found = False

        self.append_log('[SYSTEM] Tracker cleared. Standing by.')
        self.textBox.clear()

    def _save_result(self, elapsed, mins, secs, found=True):
        os.makedirs('metrics', exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        prompt_field = f'  prompt="{self._prompt}"' if self._prompt else ''
        ai_field = f'  ai_mode={"yes" if self._ai_mode else "no"}'
        outcome = 'Time to detection' if found else 'NOT FOUND — time searching'
        line = (
            f'* [{timestamp}]{prompt_field}{ai_field}'
            f'  {outcome}: {int(mins)}m {secs:.1f}s  ({elapsed:.1f}s)\n'
        )
        with open(LOG_PATH, 'a') as f:
            f.write(line)
        self.append_log(f'[TIMER] Result saved to {LOG_PATH}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GroundStation()
    window.show()
    sys.exit(app.exec())
