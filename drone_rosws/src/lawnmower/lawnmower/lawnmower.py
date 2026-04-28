#!/usr/bin/env python3
"""
Simple lawnmower mission for MAVROS + ArduPilot SITL.

This version intentionally keeps logic minimal:
- wait FCU + local pose stream
- pre-stream position setpoints
- set GUIDED, arm
- take off by commanding a local position setpoint
- fly boustrophedon waypoints
- stop pattern if /object_found becomes True
"""

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

TAKEOFF_ALT = 3.0
CRUISE_ALT = 3.0
AREA_WIDTH = 6.0
AREA_HEIGHT = 6.0
LANE_SPACING = 2.0
WAYPOINT_TOL = 0.6
SETPOINT_RATE = 20.0


def generate_boustrophedon(width, height, spacing, altitude):
    waypoints = []
    y = 0.0
    direction = 1
    while y <= height + 1e-6:
        x_start = 0.0 if direction == 1 else width
        x_end = width if direction == 1 else 0.0
        waypoints.append((x_start, y, altitude))
        waypoints.append((x_end, y, altitude))
        y += spacing
        direction *= -1
    return waypoints


class BoustrophedonNode(Node):
    def __init__(self):
        super().__init__("boustrophedon_node")
        self.declare_parameter("return_home_only", False)
        self.return_home_only = bool(self.get_parameter("return_home_only").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.state = State()
        self.current_pose = PoseStamped()
        self.has_pose = False
        self.tracking_active = False

        self.create_subscription(State, "/mavros/state", self._state_cb, qos)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self._pose_cb, qos)
        self.create_subscription(Bool, "/object_found", self._tracking_cb, 10)

        self.setpoint_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")

    def _state_cb(self, msg: State):
        self.state = msg

    def _pose_cb(self, msg: PoseStamped):
        self.current_pose = msg
        self.has_pose = True

    def _tracking_cb(self, msg: Bool):
        self.tracking_active = msg.data
        if self.tracking_active:
            self.get_logger().info("Object found. Stopping lawnmower.")

    def _make_setpoint(self, x, y, z):
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        sp.pose.position.x = float(x)
        sp.pose.position.y = float(y)
        sp.pose.position.z = float(z)
        sp.pose.orientation.w = 1.0
        return sp

    def _distance_to(self, x, y, z):
        p = self.current_pose.pose.position
        return math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2 + (p.z - z) ** 2)

    def _wait_connected(self, timeout=30.0):
        self.get_logger().info("Waiting for FCU connection...")
        start = time.time()
        while rclpy.ok() and not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.time() - start > timeout:
                raise RuntimeError("FCU connection timeout")
        self.get_logger().info("FCU connected.")

    def _wait_pose(self, timeout=20.0):
        self.get_logger().info("Waiting for local pose stream...")
        start = time.time()
        while rclpy.ok() and not self.has_pose:
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.time() - start > timeout:
                raise RuntimeError("Local pose timeout")
        self.get_logger().info("Local pose received.")

    def _set_mode(self, mode, retries=5):
        req = SetMode.Request()
        req.custom_mode = mode
        for _ in range(retries):
            self.mode_client.wait_for_service(timeout_sec=2.0)
            fut = self.mode_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=4.0)
            if fut.result() and fut.result().mode_sent:
                self.get_logger().info(f"Mode -> {mode}")
                return
            time.sleep(0.8)
        raise RuntimeError(f"Failed to set mode {mode}")

    def _arm(self, retries=5):
        req = CommandBool.Request()
        req.value = True
        for i in range(retries):
            self.arming_client.wait_for_service(timeout_sec=2.0)
            fut = self.arming_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=4.0)
            if fut.result() and fut.result().success:
                self.get_logger().info("Armed.")
                return
            self.get_logger().warn(f"Arm attempt {i + 1} failed")
            time.sleep(1.0)
        raise RuntimeError("Failed to arm")

    def _publish_setpoint_for(self, x, y, z, seconds):
        rate_sec = 1.0 / SETPOINT_RATE
        end = time.time() + seconds
        sp = self._make_setpoint(x, y, z)
        while rclpy.ok() and time.time() < end:
            sp.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_pub.publish(sp)
            rclpy.spin_once(self, timeout_sec=rate_sec)

    def _go_to(self, x, y, z, timeout=40.0):
        self.get_logger().info(f"Waypoint ({x:.1f}, {y:.1f}, {z:.1f})")
        sp = self._make_setpoint(x, y, z)
        start = time.time()
        rate_sec = 1.0 / SETPOINT_RATE

        while rclpy.ok():
            if self.tracking_active:
                return
            sp.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_pub.publish(sp)
            rclpy.spin_once(self, timeout_sec=rate_sec)

            if self._distance_to(x, y, z) < WAYPOINT_TOL:
                self.get_logger().info("Reached waypoint")
                return

            if time.time() - start > timeout:
                self.get_logger().warn("Waypoint timeout; moving on")
                return

    def run(self):
        self._wait_connected()
        self._wait_pose()

        # Pre-stream setpoints before GUIDED/arming for stable acceptance.
        self._publish_setpoint_for(0.0, 0.0, TAKEOFF_ALT, 2.0)

        self._set_mode("GUIDED")
        self._arm()

        if self.return_home_only:
            self.get_logger().info("Return-home mode active. Navigating to origin.")
            self._go_to(0.0, 0.0, CRUISE_ALT, timeout=45.0)
            self.get_logger().info("Return-home complete.")
            return

        # Take off by holding a higher local setpoint.
        self._go_to(0.0, 0.0, TAKEOFF_ALT, timeout=30.0)

        waypoints = generate_boustrophedon(AREA_WIDTH, AREA_HEIGHT, LANE_SPACING, CRUISE_ALT)
        self.get_logger().info(f"Starting lawnmower with {len(waypoints)} waypoints")

        for x, y, z in waypoints:
            self._go_to(x, y, z)
            if self.tracking_active:
                break

        self.get_logger().info("Lawnmower finished.")


def main():
    rclpy.init()
    node = BoustrophedonNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()