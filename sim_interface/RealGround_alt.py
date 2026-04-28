import socket
import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget

# Matches udp_relay.py
DISCOVERY_PORT = 8499
VIDEO_PORT = 8500
TRACK_PORT = 8501      # prompt text -> /target_object
COMMAND_PORT = 8502    # TRACKING / IDLE mission commands
MAX_UDP_BUFFER = 65536

drone_ip = None


class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", VIDEO_PORT))
            self.log_signal.emit(f"[VIDEO] Listening on {VIDEO_PORT}")
        except OSError:
            self.log_signal.emit(f"[ERROR] Video port {VIDEO_PORT} busy")
            return

        datagram_buffer = b""
        while True:
            packet, _ = sock.recvfrom(MAX_UDP_BUFFER)
            if packet == b"END":
                if datagram_buffer:
                    np_arr = np.frombuffer(datagram_buffer, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                        self.frame_signal.emit(img)
                datagram_buffer = b""
            else:
                datagram_buffer += packet


class RealGroundAlt(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealGround Alt - Grounding DINO Prompt UI")
        self.resize(1100, 820)
        self._build_ui()

        self.connected = False
        self.append_log("[STARTUP] Waiting for drone discovery...")
        while not self.connected:
            self.connected = self.connect_to_drone()
            if not self.connected:
                QApplication.processEvents()

        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.update_image)
        self.video_thread.log_signal.connect(self.append_log)
        self.video_thread.start()
        self.append_log("[SYSTEM] Ready. Send prompt and start mission.")

    def _build_ui(self):
        self.setStyleSheet(
            """
            QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI'; }
            QLabel#status { font-size: 16px; background-color: #313244; padding: 10px; border-radius: 8px; }
            QLabel#video { background-color: #11111b; border: 2px solid #45475a; border-radius: 12px; }
            QLineEdit { background-color: #313244; padding: 10px; border-radius: 8px; color: white; }
            QPushButton#primary { background-color: #89b4fa; color: #11111b; font-weight: bold; border-radius: 8px; }
            QPushButton#danger { background-color: #f38ba8; color: #11111b; font-weight: bold; border-radius: 8px; }
            QTextEdit { background-color: #11111b; color: #a6e3a1; border-radius: 8px; }
            """
        )

        layout = QVBoxLayout(self)
        self.status_label = QLabel("STATUS: CONNECTING")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.video_label = QLabel("Waiting for video...")
        self.video_label.setObjectName("video")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label, stretch=5)

        controls = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Prompt (e.g., woman with blue shirt)")
        controls.addWidget(self.prompt_input, stretch=4)

        self.send_target_btn = QPushButton("SEND PROMPT")
        self.send_target_btn.setObjectName("primary")
        self.send_target_btn.clicked.connect(self.send_prompt)
        controls.addWidget(self.send_target_btn, stretch=1)

        self.start_btn = QPushButton("START SEARCH")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_search)
        controls.addWidget(self.start_btn, stretch=1)

        self.stop_btn = QPushButton("STOP SEARCH")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.stop_search)
        controls.addWidget(self.stop_btn, stretch=1)
        layout.addLayout(controls)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(140)
        layout.addWidget(self.log_console)

    def append_log(self, message: str):
        self.log_console.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def connect_to_drone(self) -> bool:
        global drone_ip
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
            sock.settimeout(1.0)
            msg, addr = sock.recvfrom(1024)
            if msg.strip() == b"DISCOVER_STREAMING_SERVER":
                drone_ip = addr[0]
                self.append_log(f"[SYSTEM] Linked with drone at {drone_ip}")
                self.status_label.setText(f"STATUS: CONNECTED ({drone_ip})")
                return True
        except Exception:
            pass
        finally:
            sock.close()
        return False

    @pyqtSlot(QImage)
    def update_image(self, image: QImage):
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.video_label.setPixmap(pixmap)

    def _send_udp(self, payload: bytes, port: int):
        global drone_ip
        if not drone_ip:
            self.append_log("[ERROR] Not connected to drone")
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(payload, (drone_ip, port))
        sock.close()

    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.append_log("[ERROR] Prompt is empty")
            return
        self._send_udp(prompt.encode(), TRACK_PORT)
        self.append_log(f"[PROMPT] Sent: '{prompt}'")
        self.status_label.setText(f"STATUS: PROMPT '{prompt}' SENT")

    def start_search(self):
        self._send_udp(b"TRACKING", COMMAND_PORT)
        self.append_log("[COMMAND] TRACKING sent")
        self.status_label.setText("STATUS: SEARCH ACTIVE")

    def stop_search(self):
        # Clear prompt and command a controlled return to origin.
        self._send_udp(b"stop", TRACK_PORT)
        self._send_udp(b"RETURN_HOME", COMMAND_PORT)
        self.append_log("[COMMAND] stop + RETURN_HOME sent")
        self.status_label.setText("STATUS: RETURNING HOME")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RealGroundAlt()
    win.show()
    sys.exit(app.exec())
