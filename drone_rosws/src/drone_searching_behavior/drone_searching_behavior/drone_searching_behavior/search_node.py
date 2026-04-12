#!/usr/bin/env python3
import rclpy # library for ros functionalities
import cv2
from cv_bridge import CvBridge # converts between ros image messages
                               # and cv messages
from sensor_msgs.msg import Image
from rclpy.node import Node # node class from library
from geometry_msgs.msg import Twist
from apriltag_msgs.msg import AprilTagDetectionArray
from mavros_msgs.msg import MountControl


# after writing code, make executable by adding search node to console_scripts
# in pacakage's setup.py 

# class search_node inherits from Node class
class Search_Node(Node):
    def __init__(self):
        super().__init__("Search_Node") # initalize node class with name parameter
        # attributes
        self.tag_found = False 
        self.bridge = CvBridge()
        self.tag_x  = 0.0
        self.tag_y = 0.0
        self.current_gimbal_yaw = 0.0
        self.current_gimbal_pitch = 0.0
        self.latest_cv_image = None
        self.declare_parameter('target_tag_id', 0)
        
        # subscribe to detections package
        self.subscription = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.tag_callback, # function called by subscription
            10
        )
        

        self.publisher = self.create_publisher(
            Twist,
            # topic to publish to to control yaw and position through mavros link
            '/mavros/setpoint_velocity/cmd_vel_unstamped', 
            10
        )
        self.gimbal_pub = self.create_publisher( # publisher for gimball lock on behavior 
            MountControl,
            '/mavros/mount_control/command',
            10
        )
        self.timer = self.create_timer(0.05, self.timer_callback)

    def tag_callback(self, msg):
        # Initialize as not found for this specific frame
        found_id = False
        target_id = self.get_parameter('target_tag_id').get_parameter_value().integer_value
        for detection in msg.detections:
            # check if this specific detection is ID 2
            if detection.id == target_id:
                self.tag_found = True
                self.tag_x = detection.centre.x 
                self.tag_y = detection.centre.y
                found_id = True
                self.get_logger().info(f"Target ID {target_id} found at x: {self.tag_x}")
                break
            
        if not found_id:
            self.tag_found = False

        
    def timer_callback(self):
        msg = Twist()

        if not self.tag_found:
            
            msg.angular.z= 0.5
            self.get_logger().info("Still searching")

        else:
            image_center = 640.0 # camera frame is 1280 x 720

            error_x = image_center - self.tag_x

            kp = 0.001 

            msg.angular.x = 0.0
            image_center = 640.0
            error_x = image_center - self.tag_x
            
        
            # Adjust kp for gimbal movement sensitivity
            self.lock_gimbal(error_x)

            self.get_logger().info("found it")
        self.publisher.publish(msg)

    def lock_gimbal(self, error_x):
        msg = MountControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = 2 # targeting mode to lock in
        
        
        self.current_gimbal_yaw += error_x * 0.01 
        self.current_gimbal_pitch = 0.0

        msg.pitch = self.current_gimbal_pitch
        msg.yaw = self.current_gimbal_yaw
        msg.roll = 0.0
        self.gimbal_pub.publish(msg)

    
def main():
    rclpy.init()
    node = Search_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

