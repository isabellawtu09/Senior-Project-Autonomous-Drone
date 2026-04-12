#!/usr/bin/env python3
import rclpy, cv2, numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge

def tag_id_of(det):
    for attr in ('id','ids','fiducial_id','fiducial_ids'):
        if hasattr(det, attr):
            v = getattr(det, attr)
            try:
                return int(v[0]) if len(v)>0 else 0
            except TypeError:
                try: return int(v)
                except: pass
    return 0

def corners_of(det):
    # Common field names for pixel corners (x,y in image coordinates)
    for attr in ('corners','pixel_corners','px','points'):
        if hasattr(det, attr):
            c = getattr(det, attr)
            # expect 4 points; each has .x/.y or is [x,y]
            pts = []
            for p in c[:4]:
                try: pts.append((float(p.x), float(p.y)))
                except AttributeError:
                    try: pts.append((float(p[0]), float(p[1])))
                    except: pass
            if len(pts)==4: return pts
    # Fallback: try center + size if present (rare)
    if hasattr(det,'center') and hasattr(det,'size'):
        cx,cy = float(det.center.x), float(det.center.y)
        s = float(det.size) if hasattr(det,'size') else 40.0
        return [(cx-s,cy-s),(cx+s,cy-s),(cx+s,cy+s),(cx-s,cy+s)]
    return None

class TagOverlay(Node):
    def __init__(self):
        super().__init__('tag_overlay')
        self.bridge = CvBridge()
        self.last_img = None
        self.last_header = None

        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5,
                             durability=DurabilityPolicy.VOLATILE)
        qos_det = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=10,
                             durability=DurabilityPolicy.VOLATILE)

        img_topic = self.declare_parameter('image_topic','/image_raw').get_parameter_value().string_value
        det_topic = self.declare_parameter('detections_topic','/detections').get_parameter_value().string_value
        out_topic = self.declare_parameter('output_topic','/image_with_tags').get_parameter_value().string_value

        self.sub_img = self.create_subscription(Image, img_topic, self.on_image, qos_img)
        self.sub_det = self.create_subscription(AprilTagDetectionArray, det_topic, self.on_dets, qos_det)
        self.pub = self.create_publisher(Image, out_topic, 10)

        self.get_logger().info(f"Overlay listening image={img_topic}, detections={det_topic}, publishing {out_topic}")

    def on_image(self, msg: Image):
        try:
            self.last_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_header = msg.header
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")

    def on_dets(self, msg: AprilTagDetectionArray):
        if self.last_img is None:
            return
        frame = self.last_img.copy()
        for det in msg.detections:
            tid = tag_id_of(det)
            pts = corners_of(det)
            if pts:
                # draw polygon
                pts_i = np.array(pts, dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(frame, [pts_i], isClosed=True, thickness=2, color=(0,255,0))
                # label near first corner
                x,y = int(pts[0][0]), int(pts[0][1])
            else:
                # if no corners, just label center text in the middle
                h,w = frame.shape[:2]
                x,y = w//2, h//2
            cv2.putText(frame, f"tag {tid}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        if self.last_header:
            out.header = self.last_header
        self.pub.publish(out)

def main():
    rclpy.init()
    node = TagOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
