import sys
import os
import socket
import time
import PyQt6
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import cv2
from ultralytics import YOLOWorld
import queue  
import torch 

# --- CONFIGURATION ---
DISCOVERY_PORT = 8499   
VIDEO_PORT = 8500       
TRACKPORT = 8501
COMMANDPORT = 8502
MAX_UDP_BUFFER = 65536 

drone_ip = None
Target = None

model = YOLOWorld('yolov8m-world.pt') 
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
                        print("[WARNING] Frame size exceeded maximum limit. Discarding buffer.")
                        datagram_buffer = b""
            except Exception as e:
                datagram_buffer = b""

# --- 2. INFERENCE THREAD ---
class InferenceThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    state_change_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def run(self):
        self.log_signal.emit("[INFERENCE] Waiting for frames...")
        current_state = "IDLE"
        last_seen_time = 0.0

        while True:
            frame = frame_queue.get()
            found_in_current_frame = False

            # --- YOLO LOGIC START ---
            if Target is not None:
                try:
                    device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
                    results = model.track(frame, persist=True, verbose=False, device=device)
                    
                    for r in results:
                        if r.boxes:
                            for i, box in enumerate(r.boxes):
                                cls_id = int(box.cls[0])
                                if cls_id not in model.names:
                                    continue
                                
                                cls_name = model.names[int(box.cls[0])]
                                conf = float(box.conf[0]) 

                                if cls_name == Target:
                                    found_in_current_frame = True
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    
                                    label = f"{cls_name} {conf * 100:.0f}%"
                                    
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    if r.masks is not None:
                                        mask_points = np.int32([r.masks.xy[i]])
                                        cv2.polylines(frame, mask_points, True, (0, 255, 0), 2)

                except RuntimeError:
                    self.log_signal.emit("[WARNING] Skipped corrupted frame.")
                    continue 

                # --- STATE MACHINE LOGIC ---
                if found_in_current_frame:
                    if current_state != "FOUND":
                        self.log_signal.emit(f"[TRACKER] Target '{Target}' ACQUIRED.")
                    current_state = "FOUND"
                    last_seen_time = time.time()
                else:
                    if current_state in ["FOUND", "LOST"]:
                        if time.time() - last_seen_time > 5.0:
                            if current_state != "TRACKING":
                                self.log_signal.emit(f"[TRACKER] Target '{Target}' LOST entirely. Resuming search.")
                            current_state = "TRACKING" 
                        else:
                            if current_state != "LOST":
                                self.log_signal.emit(f"[TRACKER] Target '{Target}' occluded. Waiting 5s...")
                            current_state = "LOST"     
                    else:
                        current_state = "TRACKING"
            else:
                if current_state != "IDLE":
                    self.log_signal.emit("[SYSTEM] Tracking stopped. Standing by.")
                current_state = "IDLE"

            # Update Drone and UI
            commandSock.sendto(current_state.encode(), (drone_ip, COMMANDPORT))
            self.state_change_signal.emit(current_state)

            # Process Image for GUI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.change_pixmap_signal.emit(qt_img) 

# --- MAIN GUI CLASS ---
class GroundStation(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drone Tracker GCS") 
        self.resize(1100, 850)

        # Apply Global Styling
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel#statusLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #313244;
                border-radius: 8px;
                color: #cdd6f4;
            }
            QLabel#videoFeed {
                background-color: #11111b;
                border: 2px solid #45475a;
                border-radius: 12px;
            }
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                font-size: 16px;
                padding: 10px 15px;
                border: 2px solid #45475a;
                border-radius: 8px;
            }
            QLineEdit:focus {
                border: 2px solid #89b4fa;
            }
            QPushButton#startBtn {
                background-color: #89b4fa;
                color: #11111b;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 8px;
            }
            QPushButton#startBtn:hover { background-color: #b4befe; }
            QPushButton#startBtn:pressed { background-color: #74c7ec; }
            
            QPushButton#stopBtn {
                background-color: #f38ba8;
                color: #11111b;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 8px;
            }
            QPushButton#stopBtn:hover { background-color: #eba0ac; }
            QPushButton#stopBtn:pressed { background-color: #e64553; }

            QTextEdit {
                background-color: #11111b;
                color: #a6e3a1;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)
        
        # Connection Boot Sequence
        self.connected = False
        print("[STARTUP] Looking for Drone...")
        while not self.connected:
            self.connected = self.connect_to_drone()
            if not self.connected:
                QApplication.processEvents() 
        
        # --- UI SETUP ---
        
        self.status_label = QLabel(f"STATUS: IDLE | Connected to {drone_ip}")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        # Middle: Video Feed
        self.video_label = QLabel("Waiting for Video Feed...")
        self.video_label.setObjectName("videoFeed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("color: #6c7086; font-size: 20px; font-weight: bold;")
        self.video_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_label, stretch=5) 

        # Bottom Row 1: Controls
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(15)

        self.textBox = QLineEdit()
        self.textBox.setPlaceholderText("Enter Target Description (e.g., 'person', 'red car')")
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

        # Bottom Row 2: Action Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True) 
        self.log_console.setFixedHeight(120)  
        self.main_layout.addWidget(self.log_console, stretch=1)
        self.append_log(f"[SYSTEM] Connection established with drone at {drone_ip}.")

        # --- START THREADS ---
        if self.connected:
            self.net_thread = NetworkThread()
            self.net_thread.start()

            self.inf_thread = InferenceThread()
            self.inf_thread.change_pixmap_signal.connect(self.update_image)
            self.inf_thread.state_change_signal.connect(self.update_status)
            self.inf_thread.log_signal.connect(self.append_log)
            self.inf_thread.start()

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
        if new_state == "FOUND":
            color = "#a6e3a1" 
        elif new_state == "LOST":
            color = "#fab387" 
        elif new_state == "TRACKING":
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

    def start_tracking(self): 
        user_input = self.textBox.text()
        global Target
        if user_input.strip() == "":
            self.append_log("[ERROR] Target description cannot be empty.")
            return
        Target = user_input.strip()
        model.set_classes([Target])
        self.append_log(f"[SYSTEM] Uploaded new target parameters: '{Target}'")

    def stop_tracking(self):
        global Target
        if Target is not None:
            self.append_log(f"[SYSTEM] Aborting track for '{Target}'.")
            Target = None
            commandSock.sendto(b"STOP", (drone_ip, COMMANDPORT))
        else:
            self.append_log("[SYSTEM] Already in IDLE state.")
        self.textBox.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GroundStation()
    window.show()
    sys.exit(app.exec())
