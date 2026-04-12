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

# --- CONFIGURATION ---
DISCOVERY_PORT = 8499   
VIDEO_PORT = 8500        
TRACKPORT = 8501        # Sends target description to ROS node
COMMANDPORT = 8502      # Sends GCS commands (Start/Stop)
MAX_UDP_BUFFER = 65536 
drone_ip = None

# --- NETWORK RECEIVER THREAD ---
# This thread handles incoming video frames (likely already processed by ROS)
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(("0.0.0.0", VIDEO_PORT))
            self.log_signal.emit(f"[VIDEO] Listening on port {VIDEO_PORT}")
        except OSError:
            self.log_signal.emit(f"[ERROR] Port {VIDEO_PORT} is busy.")
            return

        datagram_buffer = b""

        while True:
            try:
                packet, _ = sock.recvfrom(MAX_UDP_BUFFER)

                if packet == b"END":
                    if len(datagram_buffer) > 0:
                        # Decode the JPEG/PNG stream coming from the ROS Node
                        np_arr = np.frombuffer(datagram_buffer, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            # Convert BGR (OpenCV) to RGB (Qt)
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, ch = rgb_frame.shape
                            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                            self.change_pixmap_signal.emit(qt_img)
                    
                    datagram_buffer = b""
                else:
                    datagram_buffer += packet
            except Exception:
                datagram_buffer = b""

# --- MAIN GUI CLASS ---
class GroundStation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Tracker GCS (Remote Inference)") 
        self.resize(1100, 850)
        self.setup_ui()
        
        # Connection Sequence
        self.connected = False
        self.append_log("[STARTUP] Waiting for Drone/ROS Discovery...")
        while not self.connected:
            self.connected = self.connect_to_drone()
            if not self.connected:
                QApplication.processEvents() 

        # Start Video Thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.log_signal.connect(self.append_log)
        self.video_thread.start()

    def setup_ui(self):
        
        self.setStyleSheet("""
            QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI'; }
            QLabel#status { font-size: 16px; background-color: #313244; padding: 10px; border-radius: 8px; }
            QLabel#video { background-color: #11111b; border: 2px solid #45475a; border-radius: 12px; }
            QLineEdit { background-color: #313244; padding: 10px; border-radius: 8px; color: white; }
            QPushButton#start { background-color: #89b4fa; color: #11111b; font-weight: bold; border-radius: 8px; }
            QPushButton#stop { background-color: #f38ba8; color: #11111b; font-weight: bold; border-radius: 8px; }
            QTextEdit { background-color: #11111b; color: #a6e3a1; border-radius: 8px; }
        """)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("STATUS: CONNECTED")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.video_label = QLabel("Stream Offline")
        self.video_label.setObjectName("video")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label, stretch=5)

        ctrl_layout = QHBoxLayout()
        self.textBox = QLineEdit()
        self.textBox.setPlaceholderText("Target Description (e.g., 'blue backpack')")
        ctrl_layout.addWidget(self.textBox, stretch=4)

        self.track_button = QPushButton("SEND TARGET")
        self.track_button.setObjectName("start")
        self.track_button.setFixedHeight(45)
        self.track_button.clicked.connect(self.send_target)
        ctrl_layout.addWidget(self.track_button, stretch=1)

        self.stop_button = QPushButton("STOP")
        self.stop_button.setObjectName("stop")
        self.stop_button.setFixedHeight(45)
        self.stop_button.clicked.connect(self.stop_tracking)
        ctrl_layout.addWidget(self.stop_button, stretch=1)

        layout.addLayout(ctrl_layout)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(120)
        layout.addWidget(self.log_console)

    def connect_to_drone(self):
        global drone_ip
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
            sock.settimeout(1.0) 
            msg, addr = sock.recvfrom(1024)
            if msg.strip() == b"DISCOVER_STREAMING_SERVER":
                drone_ip = addr[0]
                # Get local IP to tell ROS node where to send video
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('10.255.255.255', 1))
                my_ip = s.getsockname()[0]
                s.close()
                
                sock.sendto(f"{my_ip}:{VIDEO_PORT}".encode(), addr)
                self.append_log(f"[SYSTEM] Linked to Drone at {drone_ip}")
                return True
        except:
            pass
        finally:
            sock.close()
        return False

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def append_log(self, message):
        self.log_console.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def send_target(self): 
        global drone_ip
        target_text = self.textBox.text().strip()
        if not target_text:
            return

        try:
            # Tell the ROS Node what to track
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(target_text.encode(), (drone_ip, TRACKPORT))
            sock.close()
            
            self.append_log(f"[COMMAND] Target '{target_text}' sent to ROS Node.")
            self.status_label.setText(f"TRACKING: {target_text.upper()}")
            self.status_label.setStyleSheet("color: #a6e3a1; background-color: #313244;")
        except Exception as e:
            self.append_log(f"[ERROR] Failed to send: {e}")

    def stop_tracking(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(b"STOP", (drone_ip, TRACKPORT))
            sock.close()
            self.append_log("[COMMAND] Stop signal sent.")
            self.status_label.setText("STATUS: IDLE")
            self.status_label.setStyleSheet("color: #cdd6f4; background-color: #313244;")
        except:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GroundStation()
    window.show()
    sys.exit(app.exec())