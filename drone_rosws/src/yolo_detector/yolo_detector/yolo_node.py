import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from std_msgs.msg import String

class YOLOProcessor(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        
        #self.det_model = YOLO("yolo26m.pt")
        self.det_model = YOLO("yolov8m-world.pt")
        
        # Publishers
        self.det_pub = self.create_publisher(Image, '/ultralytics/detection/image', 5)
        
        # Subscribers
        self.create_subscription(
            Image, 
            '/world/iris_objects_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image',
            self.camera_callback, 5)
        
        self.create_subscription(
            String,
            '/target_object',
            self.target_callback,
            10
        )

    def target_callback(self, msg):
        target = msg.data.lower().strip()
        
        if target == "stop" or not target:
            # clear the model's vocabulary entirely
            self.det_model.set_classes([])
            self.get_logger().info("Stopped searching. Vocabulary cleared.")
        else:
            self.det_model.set_classes([target])
            self.get_logger().info(f"Started searching for: {target}")

    def camera_callback(self, msg):
        # ros image to opencv
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # if vocabulary is empty, just show the raw video
        if not self.det_model.model.names:
            annotated_frame = cv_image
        else:
            # yolo with class detector
            results = self.det_model.predict(
                cv_image, 
                classes=[0],   
                conf=0.25,     
                verbose=False
            )
            
            annotated_frame = results[0].plot()

        output_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.det_pub.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

