"""
Bridge drone/sim camera to the ground station over UDP.

Subscribes to the raw ROS camera topic only — no on-drone detection in this path.
Run detectors (YOLOWorld, LLM client, etc.) on the ground station.
"""

import os
import time
import threading
import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
import cv2
from cv_bridge import CvBridge


class UdpRelay(Node):
    def __init__(self):
        super().__init__("udp_relay_node")
        self.declare_parameter(
            "camera_topic",
            "/world/iris_runway_15x15_walls/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image",
        )
        self.camera_topic = self.get_parameter("camera_topic").value
        self.bridge = CvBridge()
        self.control_tokens = {
            "tracking",
            "idle",
            "found",
            "start_boustrophedon",
        }
        self.active_prompt = ""

        self.UI_IP = "127.0.0.1"

        self.DISCOVERY_PORT = 8499
        self.VIDEO_PORT = 8500
        self.TRACK_PORT = 8501
        self.COMMAND_PORT = 8502
        self.MAX_UDP = 8000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Initialise mission state here so _terminate_mission is always safe to call
        self.mission_process = None
        self.tracking_started = False

        threading.Thread(target=self.shout_for_ui, daemon=True).start()
        threading.Thread(target=self.listen_for_ui_commands, daemon=True).start()
        threading.Thread(target=self.listen_for_mission_commands, daemon=True).start()

        self.target_pub = self.create_publisher(String, '/target_object', 10)
        self.tracking_pub = self.create_publisher(Bool, '/object_found', 10)
        self.create_subscription(
            Image,
            self.camera_topic,
            self.send_to_ui_callback,
            5,
        )

        self.get_logger().info(
            f"UDP Relay: camera '{self.camera_topic}' -> {self.UI_IP}:{self.VIDEO_PORT} "
            "(raw feed; detections on ground station). Discovery broadcasting."
        )
        self.publish_object_found(False)

    def shout_for_ui(self):
        shout_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        shout_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        message = b"DISCOVER_STREAMING_SERVER"
        while rclpy.ok():
            try:
                shout_sock.sendto(message, ("255.255.255.255", self.DISCOVERY_PORT))
                shout_sock.sendto(message, ("127.0.0.1", self.DISCOVERY_PORT))
            except Exception as e:
                self.get_logger().error(f"Shout error: {e}")
            time.sleep(1.0)

    def _log_proc(self, proc):
        """Watcher thread: pipes lawnmower stdout/stderr into the ROS logger."""
        for line in proc.stdout:
            self.get_logger().info(f"[LAWNMOWER] {line.decode(errors='ignore').strip()}")
        for line in proc.stderr:
            self.get_logger().error(f"[LAWNMOWER] STDERR: {line.decode(errors='ignore').strip()}")

    def _terminate_mission(self):
        """Safely terminate the lawnmower process."""
        if self.mission_process and self.mission_process.poll() is None:
            self.mission_process.terminate()
            self.mission_process = None
        self.tracking_started = False

    def _graceful_stop(self):
        """
        FIX #1 — Signal the lawnmower via the ROS topic FIRST so it can stop
        its pattern loop cleanly (tracking_active flag). Then wait up to 4 s
        for it to exit on its own before falling back to SIGTERM.

        This replaces the old pattern of terminating the process and publishing
        simultaneously, which killed the process before it could react.
        """
        # 1. Publish the found signal so BoustrophedonNode._tracking_cb fires.
        self.publish_object_found(True)
        self.get_logger().info("[FOUND] Published /object_found=True. Waiting for lawnmower to stop gracefully...")

        # 2. Give the lawnmower time to finish its current _go_to iteration,
        #    react to tracking_active=True, and exit run() cleanly.
        GRACE_PERIOD = 4.0   # seconds — tune if your waypoint loop is slower
        deadline = time.time() + GRACE_PERIOD
        while time.time() < deadline:
            if self.mission_process is None or self.mission_process.poll() is not None:
                self.get_logger().info("[FOUND] Lawnmower exited gracefully.")
                self.tracking_started = False
                return
            time.sleep(0.2)

        # 3. Fallback: process is still alive after grace period — force terminate.
        self.get_logger().warn("[FOUND] Lawnmower did not stop in time. Sending SIGTERM.")
        self._terminate_mission()

    def listen_for_mission_commands(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.COMMAND_PORT))

        while rclpy.ok():
            data, _ = sock.recvfrom(1024)
            cmd = data.strip()

            self.get_logger().info(f"[CMD] Received: {cmd}")

            # --- ATOMIC TRACKING:prompt command ---
            if cmd.startswith(b"TRACKING"):
                parts = cmd.split(b":", 1)
                if len(parts) == 2:
                    extracted_prompt = parts[1].decode(errors="ignore").strip()
                    if extracted_prompt:
                        self.active_prompt = extracted_prompt
                        prompt_msg = String()
                        prompt_msg.data = self.active_prompt
                        self.target_pub.publish(prompt_msg)
                        self.get_logger().info(f"TRACKING prompt set atomically: '{self.active_prompt}'")

                if not self.active_prompt:
                    self.get_logger().warn("Ignoring TRACKING: no active target prompt set yet.")
                    continue

                if not self.tracking_started:
                    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
                    ws = f"/home/{user}/src/Senior-Project-Autonomous-Drone/drone_rosws"
                    self.get_logger().info(f"[LAWNMOWER] Launching with USER='{user}', ws='{ws}'")

                    self.tracking_started = True
                    self.publish_object_found(False)

                    import subprocess
                    self.mission_process = subprocess.Popen(
                        f"source {ws}/install/setup.bash && ros2 run lawnmower lawnmower --ros-args -p mavros_ns:=/mavros",
                        shell=True,
                        executable="/bin/bash",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=os.environ.copy(),
                    )
                    threading.Thread(target=self._log_proc, args=(self.mission_process,), daemon=True).start()
                    self.get_logger().info(f"[LAWNMOWER] Process spawned with PID={self.mission_process.pid}")
                else:
                    self.get_logger().warn("[LAWNMOWER] Already running, ignoring duplicate TRACKING command.")

            elif cmd == b"FOUND":
                # FIX #1: Use graceful stop — topic first, SIGTERM only as fallback.
                self.get_logger().info("[FOUND] Object found! Starting graceful stop sequence.")
                threading.Thread(target=self._graceful_stop, daemon=True).start()

            elif cmd == b"RETURN_HOME":
                self.get_logger().info("RETURN_HOME requested.")
                self._terminate_mission()

                user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
                ws = f"/home/{user}/src/Senior-Project-Autonomous-Drone/drone_rosws"

                import subprocess
                self.mission_process = subprocess.Popen(
                    f"source {ws}/install/setup.bash && ros2 run lawnmower lawnmower --ros-args -p mavros_ns:=/mavros -p return_home_only:=true",
                    shell=True,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy(),
                )
                threading.Thread(target=self._log_proc, args=(self.mission_process,), daemon=True).start()

            elif cmd == b"IDLE":
                self.get_logger().info("[IDLE] Stopping mission.")
                self._terminate_mission()
                self.publish_object_found(False)

    def listen_for_ui_commands(self):
        cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd_sock.bind(("0.0.0.0", self.TRACK_PORT))
        while rclpy.ok():
            data, _ = cmd_sock.recvfrom(1024)
            text = data.decode(errors="ignore").strip()
            if not text:
                continue
            if text.lower() == "stop":
                self.active_prompt = ""
                msg = String()
                msg.data = "stop"
                self.target_pub.publish(msg)
                continue
            if text.lower() in self.control_tokens:
                self.get_logger().warn(f"Ignoring control token on prompt port: '{text}'")
                continue
            msg = String()
            msg.data = text
            self.active_prompt = text
            self.target_pub.publish(msg)

    def publish_object_found(self, value: bool):
        msg = Bool()
        msg.data = value
        self.tracking_pub.publish(msg)

    def send_to_ui_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        data = buffer.tobytes()

        for i in range(0, len(data), self.MAX_UDP):
            self.sock.sendto(data[i:i + self.MAX_UDP], (self.UI_IP, self.VIDEO_PORT))
        self.sock.sendto(b'END', (self.UI_IP, self.VIDEO_PORT))


def main(args=None):
    rclpy.init(args=args)
    node = UdpRelay()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
