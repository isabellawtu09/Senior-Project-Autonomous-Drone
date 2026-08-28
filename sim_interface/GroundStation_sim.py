import sys
import os
import socket
import time
import queue
import threading
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import PyQt6
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit, QCheckBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

# Import custom LLM client
from llm_client import VisionLLMClient

# --- CONFIGURATION ---
DISCOVERY_PORT = 8499   
VIDEO_PORT = 8500       
TRACKPORT = 8501
COMMANDPORT = 8502
MAX_UDP_BUFFER = 65536 

drone_ip = None
global_latest_frame = None  
frame_write_lock = threading.Lock()

commandSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
frame_queue = queue.Queue(maxsize=2)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


# --- 1. NETWORK RECEIVER THREAD ---
class NetworkThread(QThread):
    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", VIDEO_PORT))
        except OSError:
            print(f"[ERROR] Port {VIDEO_PORT} is busy.")
            return

        datagram_buffer = b""
        maxFrameSize = 1024 * 1024  

        while True:
            try:
                packet, _ = sock.recvfrom(MAX_UDP_BUFFER)

                if packet == b"END":
                    if len(datagram_buffer) > 0:
                        np_arr = np.frombuffer(datagram_buffer, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            if frame_queue.full():
                                try:
                                    frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            frame_queue.put(frame)
                            
                    datagram_buffer = b""
                else:
                    datagram_buffer += packet
                    if len(datagram_buffer) > maxFrameSize:
                        datagram_buffer = b""
            except Exception:
                datagram_buffer = b""

# --- HSV COLOUR HISTOGRAM HELPERS ---
def compute_hue_histogram(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return None
    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    if mask.sum() < 100: 
        return None
    h_hist = cv2.calcHist([hsv], [0], mask, [90], [0, 180])
    total  = h_hist.sum()
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
    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 40)
    hues = hsv[:, :, 0][mask]
    if len(hues) == 0:
        return -1
    return int(np.median(hues))

# --- 2. INFERENCE THREAD (Multi-Object Tracker + HSV Color Filtering) ---
class InferenceThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    state_change_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    ACCEPT_THRESHOLD = 0.55
    UPDATE_GATE = 0.85

    def __init__(self):
        super().__init__()
        self.model               = None
        self.target_bbox         = None
        self.tracks              = {}  
        self.next_track_id       = 1
        self.master_hsv_hist     = None
        self.master_dominant_hue = -1
        self.is_tracking         = False
        self.use_hsv             = False  
        self.track_lock          = threading.Lock()

    def _load_model(self):
        self.log_signal.emit("[YOLO] Loading YOLO-World...")
        self.model = YOLO("yolov8m-world.pt")
        self.log_signal.emit("[YOLO] Model ready.")

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
            combined = list(dict.fromkeys(class_list[:2] + ["object"]))
            
            if torch.backends.mps.is_available():
                self.model.to('cpu')
                
            self.model.set_classes(combined)
            
            if global_latest_frame is not None:
                dummy = cv2.resize(global_latest_frame, (640, 640))
                device = 'mps' if torch.backends.mps.is_available() else 'cpu'
                self.model.to(device)
                self.model.predict(dummy, verbose=False, conf=0.25, device=device)
            self.log_signal.emit(f"[YOLO] Classes set: {combined}")

    def start_cv_tracker(self, frame, bbox, yolo_terms, preserve_appearance=False, use_hsv=False):
        with self.track_lock:
            self.target_bbox = tuple(map(int, bbox)) if bbox else None
            self.tracks = {}
            self.next_track_id = 1
            self.use_hsv = use_hsv
            
            if not preserve_appearance:
                self.master_hsv_hist     = None
                self.master_dominant_hue = -1
            self._set_classes(yolo_terms)
            self.is_tracking = True
            
            search_area = self.target_bbox if self.target_bbox else "Full Frame"
            self.log_signal.emit(f"[TRACKER] Multi-Object Tracking | Initial Latch Area: {search_area} | HSV Color Filtering: {'ON' if use_hsv else 'OFF'}")

    def stop_cv_tracker(self):
        with self.track_lock:
            self.is_tracking = False
            self.target_bbox = None
            self.tracks = {}
            self.master_hsv_hist = None
            self.master_dominant_hue = -1
            self._set_classes([])

    def run(self):
        global global_latest_frame, drone_ip
        current_state = "IDLE"
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
            any_last_seen = 0

            with self.track_lock:
                if self.is_tracking and self.model is not None:
                    try:
                        self.model.to(device)
                        frame_h, frame_w = frame.shape[:2]
                        small = cv2.resize(frame, (640, 640))
                        scale_x = frame_w / 640.0
                        scale_y = frame_h / 640.0
                        
                        results  = self.model.predict(small, verbose=False, conf=0.25, device=device)[0]
                        num_raw  = len(results.boxes) if results.boxes is not None else 0
                        
                        valid_boxes = []
                        if num_raw > 0:
                            raw_boxes = results.boxes.xyxy.cpu().numpy()
                            all_boxes = raw_boxes * [scale_x, scale_y, scale_x, scale_y]
                            all_confs = results.boxes.conf.cpu().numpy()
                            for i in range(len(all_boxes)):
                                if all_confs[i] > 0.05:
                                    bx1, by1, bx2, by2 = map(int, all_boxes[i])
                                    crop = frame[max(0,by1):min(frame_h,by2), max(0,bx1):min(frame_w,bx2)]
                                    valid_boxes.append((bx1, by1, bx2 - bx1, by2 - by1, all_confs[i], crop))

                        # --- STAGE 0: MASTER COLOR LATCHING ---
                        if self.use_hsv and self.master_hsv_hist is None and self.target_bbox is not None:
                            tx = self.target_bbox[0] + self.target_bbox[2] // 2
                            ty = self.target_bbox[1] + self.target_bbox[3] // 2
                            
                            best_i, best_conf = None, -1
                            for i, (bx1, by1, bw, bh, conf, crop) in enumerate(valid_boxes):
                                margin_x, margin_y = max(30, self.target_bbox[2] // 3), max(30, self.target_bbox[3] // 3)
                                in_zone = ((bx1 - margin_x) <= tx <= (bx1 + bw + margin_x) and
                                           (by1 - margin_y) <= ty <= (by1 + bh + margin_y))
                                if in_zone and conf > best_conf:
                                    best_conf, best_i = conf, i
                                    
                            if best_i is not None:
                                _, _, _, _, _, crop = valid_boxes[best_i]
                                self.master_hsv_hist = compute_hue_histogram(crop)
                                self.master_dominant_hue = dominant_hue(crop)
                                self.log_signal.emit(f"[TRACKER] Master HSV Signature Extracted. Building swarm profile.")

                        has_ref = self.use_hsv and (self.master_hsv_hist is not None)

                        matched_track_ids = set()
                        matched_box_indices = set()

                        # --- STAGE 1: UPDATE EXISTING TRACKS ---
                        for box_idx, (x, y, w, h, conf, crop) in enumerate(valid_boxes):
                            best_iou = 0
                            best_tid = None
                            for tid, track in self.tracks.items():
                                if tid in matched_track_ids: continue
                                
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

                        # --- STAGE 2: REGISTER NEW OBJECTS ENTERING FRAME ---
                        for box_idx, (x, y, w, h, conf, crop) in enumerate(valid_boxes):
                            if box_idx not in matched_box_indices:
                                is_valid_new_target = False
                                
                                if has_ref:
                                    cand_hue = dominant_hue(crop)
                                    if not self._hue_veto(cand_hue):
                                        app_score = hue_similarity(self.master_hsv_hist, compute_hue_histogram(crop))
                                        if app_score > self.ACCEPT_THRESHOLD:
                                            is_valid_new_target = True
                                elif conf > 0.10:
                                    is_valid_new_target = True

                                if not self.tracks and self.target_bbox is not None and not is_valid_new_target:
                                    tx, ty, tw, th = self.target_bbox
                                    ix1, iy1 = max(x, tx), max(y, ty)
                                    ix2, iy2 = min(x + w, tx + tw), min(y + h, ty + th)
                                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                                    iou = inter / (w * h + tw * th - inter + 1e-6)
                                    if iou > 0.10:
                                        is_valid_new_target = True

                                if is_valid_new_target:
                                    self.tracks[self.next_track_id] = {'bbox': (x,y,w,h), 'last_seen': time.time(), 'conf': conf}
                                    self.next_track_id += 1

                        # --- STAGE 3: CLEANUP AND DRAW ---
                        current_time = time.time()
                        stale_ids = []

                        for tid, track in list(self.tracks.items()):
                            time_since = current_time - track['last_seen']
                            x, y, w, h = track['bbox']
                            any_last_seen = max(any_last_seen, track['last_seen'])
                            
                            if time_since < 0.2: 
                                found_any = True
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                cv2.putText(frame, f"ID:{tid} ({track['conf']:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            elif time_since < 3.0: 
                                recovering_any = True
                            else: 
                                stale_ids.append(tid)

                        for tid in stale_ids:
                            del self.tracks[tid]
                            self.log_signal.emit(f"[TRACKER] Dropped ID:{tid} due to occlusion.")

                        if not found_any and not recovering_any and self.is_tracking:
                            cv2.putText(frame, "RECOVERING / SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    except Exception as e:
                        self.log_signal.emit(f"[TRACKER ERROR] {e}")

            # --- STATE LOGIC MACHINE ---
            if found_any:
                active_count = len([t for t in self.tracks.values() if time.time() - t['last_seen'] < 0.2])
                current_state = f"FOUND ({active_count} Targets)"
            elif recovering_any:
                current_state = "LOST (RECOVERING)"
            else:
                current_state = "TRACKING" if self.is_tracking else "IDLE"

            self.state_change_signal.emit(current_state)

            # --- DRONE COMMUNICATION ---
            if drone_ip is not None:
                if found_any and self.tracks:
                    active_tracks = [t for t in self.tracks.values() if time.time() - t['last_seen'] < 0.2]
                    if active_tracks:
                        primary = active_tracks[0]
                        frame_h, frame_w = frame.shape[:2]
                        bx1, by1, bw, bh = primary['bbox']
                        cx = bx1 + bw / 2.0
                        cy = by1 + bh / 2.0
                        offset_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
                        offset_y = (cy - frame_h / 2.0) / (frame_h / 2.0)
                        
                        # CHANGED: YOLO now sends CENTERING instead of FOUND to avoid script conflicts
                        cmd = f"CENTERING:{offset_x:.4f}:{offset_y:.4f}".encode()
                        commandSock.sendto(cmd, (drone_ip, COMMANDPORT))

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
                    self.log_signal.emit(f"[LLM] Target Reasoned. Handing off to YOLO: {yolo_terms}")
                    self.bbox_found_signal.emit(bbox_small, yolo_terms)
                    return
            except Exception as e:
                self.log_signal.emit(f"[LLM ERROR] {e}")
            time.sleep(0.3)

    def stop(self):
        self.running = False

# --- MAIN GUI CLASS ---
class GroundStation(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drone Tracker GCS") 
        self.resize(1100, 850)
        
        self.llm_client = VisionLLMClient()
        self.hunter_thread = None
        self.current_use_hsv = False

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

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)
        
        self.connected = False
        print("[STARTUP] Looking for Drone...")
        while not self.connected:
            self.connected = self.connect_to_drone()
            if not self.connected:
                QApplication.processEvents() 
        
        self.status_label = QLabel(f"STATUS: IDLE | Connected to {drone_ip}")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        self.video_label = QLabel("Waiting for Video Feed...")
        self.video_label.setObjectName("videoFeed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("color: #6c7086; font-size: 20px; font-weight: bold;")
        self.video_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_label, stretch=5) 

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(15)

        self.ai_toggle = QCheckBox("AI Mode")
        self.ai_toggle.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold;")
        self.ai_toggle.setChecked(False) 
        self.controls_layout.addWidget(self.ai_toggle)

        self.textBox = QLineEdit()
        self.textBox.setPlaceholderText("Enter Target Description (check 'AI Mode' for abstract requests)")
        self.textBox.setFixedHeight(50)
        self.controls_layout.addWidget(self.textBox, stretch=4)

        self.track_button = QPushButton("START TRACKING")
        self.track_button.setObjectName("startBtn")
        self.track_button.setFixedHeight(50) 
        self.track_button.clicked.connect(self.start_tracking)
        self.controls_layout.addWidget(self.track_button, stretch=1)

        self.stop_button = QPushButton("STOP TRACKING")
        self.stop_button.setObjectName("stopBtn")
        self.stop_button.setFixedHeight(50)
        self.stop_button.clicked.connect(self.stop_tracking)
        self.controls_layout.addWidget(self.stop_button, stretch=1)

        self.main_layout.addLayout(self.controls_layout)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True) 
        self.log_console.setFixedHeight(120)  
        self.main_layout.addWidget(self.log_console, stretch=1)
        self.append_log(f"[SYSTEM] Connection established with drone at {drone_ip}.")

        if not self.llm_client.is_configured():
            self.append_log("[WARNING] OPENAI_API_KEY not found in environment. AI Mode will default to direct text.")

        if self.connected:
            self.net_thread = NetworkThread()
            self.net_thread.start()

            self.inf = InferenceThread()
            self.inf.change_pixmap_signal.connect(self.update_image)
            self.inf.state_change_signal.connect(self.update_status)
            self.inf.log_signal.connect(self.append_log)
            self.inf.start()

    def connect_to_drone(self):
        global drone_ip, VIDEO_PORT
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
            sock.settimeout(1.0) 

            msg, addr = sock.recvfrom(1024)
            
            if msg.strip() == b"DISCOVER_STREAMING_SERVER":
                drone_ip = addr[0]
                print(f"[DISCOVERY] Found Drone at {drone_ip}")
                
                my_ip = get_local_ip()
                reply = f"{my_ip}:{VIDEO_PORT}".encode()
                sock.sendto(reply, addr)
                
                sock.close()
                return True
                
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[ERROR] {e}")

        sock.close()
        return False
            
    @pyqtSlot(QImage) 
    def update_image(self, qt_img):
        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), 
            self.video_label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(scaled_pixmap)

    @pyqtSlot(str) 
    def update_status(self, new_state):
        if "FOUND" in new_state:
            color = "#a6e3a1" 
        elif "LOST" in new_state:
            color = "#fab387" 
        elif "TRACKING" in new_state:
            color = "#f9e2af" 
        else:
            color = "#cdd6f4" 
            
        self.status_label.setText(f"STATUS: {new_state} | Connected to {drone_ip}")
        self.status_label.setStyleSheet(f"color: {color}; background-color: #313244; border-radius: 8px;")

    @pyqtSlot(str)
    def append_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_console.append(f"[{timestamp}] {message}")
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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
        scaled_bbox = [
            int(x * scale_w), int(y * scale_h),
            int(w * scale_w), int(h * scale_h)
        ]
        self.append_log(f"[SYSTEM] Scaling bounding box: 640x640 -> {live_w}x{live_h}")

        # CHANGED: Fire the exact string "FOUND" to the drone once the LLM API successfully grabs it
        try:
            commandSock.sendto(b"FOUND", (drone_ip, COMMANDPORT))
            self.append_log(f"[NETWORK] Atomic 'FOUND' command sent to drone at {drone_ip} (via LLM Handoff)")
        except Exception as e:
            self.append_log(f"[NETWORK ERROR] Could not contact drone: {e}")

        has_ref = self.inf.master_hsv_hist is not None
        self.inf.start_cv_tracker(snap_copy, scaled_bbox, yolo_terms, preserve_appearance=has_ref, use_hsv=self.current_use_hsv)
        self.hunter_thread = None 

    def start_tracking(self): 
        user_input = self.textBox.text().strip()
        
        if not user_input:
            self.append_log("[ERROR] Target description cannot be empty.")
            return

        self.stop_tracking()

        try:
            atomic_cmd = f"TRACKING:{user_input}".encode()
            commandSock.sendto(atomic_cmd, (drone_ip, COMMANDPORT))
            self.append_log(f"[NETWORK] Atomic TRACKING command sent to drone at {drone_ip}")
        except Exception as e:
            self.append_log(f"[NETWORK ERROR] Could not contact drone: {e}")

        KNOWN_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "grey", "gray", "cyan", "magenta", "maroon"]
        user_words = user_input.lower().replace(',', ' ').replace('.', ' ').split()
        self.current_use_hsv = any(color in user_words for color in KNOWN_COLORS)

        if not self.ai_toggle.isChecked() or not self.llm_client.is_configured():
            if self.ai_toggle.isChecked() and not self.llm_client.is_configured():
                self.append_log("[WARNING] No API key available. Falling back to direct input.")
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
            self.append_log("[SYSTEM] Aborting background AI Hunter.")
            self.hunter_thread = None

        self.inf.stop_cv_tracker()

        try:
            commandSock.sendto(b"RTL", (drone_ip, COMMANDPORT))
            self.append_log("[NETWORK] RTL command sent to drone.")
        except Exception as e:
            self.append_log(f"[NETWORK ERROR] Could not send RTL command: {e}")

        self.append_log("[SYSTEM] Tracker Cleared. Standing by.")
        self.textBox.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GroundStation()
    window.show()
    sys.exit(app.exec())
