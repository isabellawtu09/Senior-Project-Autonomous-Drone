import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import socket
import cv2
import threading
from cv_bridge import CvBridge

class UdpRelay(Node):
    def __init__(self):
        super().__init__('udp_relay_node')
        self.bridge = CvBridge()
        
        # Ports must match Ground Station
        # self.UI_IP = "127.0.0.1" 
        self.UI_IP = "172.20.10.2"
        
        self.DISCOVERY_PORT = 8499
        self.VIDEO_PORT = 8500
        self.TRACK_PORT = 8501
        self.MAX_UDP = 8000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 1. Start the Discovery Thread (The "Shouter")
        threading.Thread(target=self.shout_for_ui, daemon=True).start()

        # 2. Start the Command Thread
        threading.Thread(target=self.listen_for_ui_commands, daemon=True).start()

        self.target_pub = self.create_publisher(String, '/target_object', 10)
        # self.create_subscription(Image, '/ultralytics/detection/image', self.send_to_ui_callback, 5)
        self.create_subscription(Image, '/world/iris_objects_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image', self.send_to_ui_callback, 5)
        
        self.get_logger().info("UDP Relay started. Shouting for Ground Station...")

    def shout_for_ui(self):
        """Broadcasts DISCOVER message until a Ground Station responds"""
        shout_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        shout_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        message = b"DISCOVER_STREAMING_SERVER"
        
        while rclpy.ok():
            try:
                # Shout to the local network
                shout_sock.sendto(message, ("255.255.255.255", self.DISCOVERY_PORT))
                # If on the same PC, also shout specifically to localhost
                shout_sock.sendto(message, ("127.0.0.1", self.DISCOVERY_PORT))
            except Exception as e:
                self.get_logger().error(f"Shout error: {e}")
            
            import time
            time.sleep(1.0) # Shout once a second

    def listen_for_ui_commands(self):
        cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd_sock.bind(("0.0.0.0", self.TRACK_PORT))
        while rclpy.ok():
            data, _ = cmd_sock.recvfrom(1024)
            msg = String()
            msg.data = data.decode().strip()
            self.target_pub.publish(msg)

    def send_to_ui_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
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
