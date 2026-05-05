#!/usr/bin/env python3
"""
Boustrophedon (lawnmower) pattern flight script.
ROS2 Jazzy + MAVROS + ArduPilot SITL + Gazebo

Pattern layout (top view):
  →→→→→→→→→
            ↓
  ←←←←←←←←←
  ↓
  →→→→→→→→→
  ...

Uses /mavros/setpoint_position/local in GUIDED mode.
"""

from geometry_msgs import msg
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPointStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
import time
import math

# ─── Pattern Configuration ────────────────────────────────────────────────────
TAKEOFF_ALT   = 1.0    # meters
CRUISE_ALT    = 1.0    # meters AGL during pattern
AREA_WIDTH    = 14.0    # meters (X axis, direction of sweeps)
AREA_HEIGHT   = 14.0    # meters (Y axis, cross-track spacing total)
LANE_SPACING  = 2.0    # meters between rows
WAYPOINT_TOL  = 0.5    # meters - acceptance radius for each waypoint
SETPOINT_RATE = 20.0   # Hz - rate to publish setpoints
PATTERN_ORIGIN_X = -7.0   # start X of sweep area
PATTERN_ORIGIN_Y = -7.0   # start Y of sweep area
# ──────────────────────────────────────────────────────────────────────────────


def generate_boustrophedon(width, height, spacing, altitude):
    """
    Generate boustrophedon waypoints starting at local origin (0,0).
    Returns list of (x, y, z) tuples.
    """
    waypoints = []
    y = 0.0
    direction = 1  # 1 = forward (+X), -1 = reverse (-X)

    while y <= height + 1e-6:
        x_start = 0.0 if direction == 1 else width
        x_end   = width if direction == 1 else 0.0
        waypoints.append((x_start, y, altitude))
        waypoints.append((x_end,   y, altitude))
        y += spacing
        direction *= -1

    return waypoints


class BoustrophedonNode(Node):
    def __init__(self):
        super().__init__('boustrophedon_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state = State()
        self.current_pose = PoseStamped()
        self.tracking_active = False
        self.rtl_commanded = False
        self.target_offset = None

        self.create_subscription(Vector3, '/target_offset', self._offset_cb, 10)
        self.create_subscription(State, '/mavros/state', self._state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._pose_cb, qos)
        self.create_subscription(Bool, '/object_found', self._tracking_cb, 10)
        self.create_subscription(Bool, '/command_rtl', self._rtl_cb, 10)

        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        self.gp_origin_pub  = self.create_publisher(GeoPointStamped, '/mavros/global_position/set_gp_origin', 10)

        self.arming_client  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client    = self.create_client(SetMode,     '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL,  '/mavros/cmd/takeoff')

    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_setpoint(self, x, y, z):
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = 'map'
        sp.pose.position.x = x
        sp.pose.position.y = y
        sp.pose.position.z = z
        sp.pose.orientation.w = 1.0
        return sp
    
    def _offset_cb(self, msg):
        self.target_offset = (msg.x, msg.y)

    def _approach_target(self, step=0.12, dead_zone=0.12, max_steps=30):
        self.get_logger().info("Approaching target...")
        for _ in range(max_steps):
            if self.rtl_commanded or self.target_offset is None:
                return
            ox, oy = self.target_offset
            if abs(ox) < dead_zone and abs(oy) < dead_zone:
                self.get_logger().info("Target centered. Holding.")
                return
            p = self.current_pose.pose.position
            self._go_to(p.x + (-oy * step), p.y + (-ox * step), p.z, timeout=10, respect_tracking=False)
            self._spin_for(0.5)
    
    def _rtl_cb(self, msg):
        if msg.data:
            self.rtl_commanded = True
            self.get_logger().info('RTL commanded from GCS.')
    
    def _tracking_cb(self, msg):
        self.tracking_active = msg.data
        if self.tracking_active:
            self.get_logger().info('Object found! Stopping pattern.')

    def _distance_to(self, x, y, z):
        p = self.current_pose.pose.position
        return math.sqrt((p.x - x)**2 + (p.y - y)**2 + (p.z - z)**2)

    def _spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_connected(self, timeout=30):
        self.get_logger().info('Waiting for FCU connection...')
        start = time.time()
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.5)
            if time.time() - start > timeout:
                raise RuntimeError('FCU connection timeout')
        self.get_logger().info('FCU connected.')

    def _set_gp_origin(self, lat=47.3977, lon=8.5456, alt=488.0):
        """Publish GPS origin to help EKF initialize in SITL."""
        self.get_logger().info('Setting GPS origin...')
        msg = GeoPointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position.latitude  = lat
        msg.position.longitude = lon
        msg.position.altitude  = alt
        for _ in range(5):
            self.gp_origin_pub.publish(msg)
            time.sleep(0.2)
            rclpy.spin_once(self, timeout_sec=0.1)

    def _wait_for_position_estimate(self, timeout=60):
        """Wait until EKF provides a non-zero local position."""
        self.get_logger().info('Waiting for EKF position estimate...')
        start = time.time()
        while True:
            rclpy.spin_once(self, timeout_sec=0.5)
            p = self.current_pose.pose.position
            if abs(p.x) > 0.01 or abs(p.y) > 0.01 or abs(p.z) > 0.01:
                self.get_logger().info('Position estimate acquired.')
                return
            # Re-publish origin periodically to nudge EKF
            if (time.time() - start) % 5 < 0.6:
                self._set_gp_origin()
            if time.time() - start > timeout:
                raise RuntimeError('Timed out waiting for position estimate')


    def _set_mode(self, mode, retries=5):
        req = SetMode.Request()
        req.custom_mode = mode
        for _ in range(retries):
            self.mode_client.wait_for_service(timeout_sec=3.0)
            fut = self.mode_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            if fut.result() and fut.result().mode_sent:
                self.get_logger().info(f'Mode → {mode}')
                return
            time.sleep(1.0)
        raise RuntimeError(f'Failed to set mode {mode}')

    def _arm(self, retries=5):
        req = CommandBool.Request()
        req.value = True
        for i in range(retries):
            self.arming_client.wait_for_service(timeout_sec=3.0)
            fut = self.arming_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            if fut.result() and fut.result().success:
                self.get_logger().info('Armed.')
                return
            self.get_logger().warn(f'Arm attempt {i+1} failed, retrying...')
            time.sleep(2.0)
        raise RuntimeError('Failed to arm')

    def _takeoff(self, altitude):
        req = CommandTOL.Request()
        req.altitude = altitude
        self.takeoff_client.wait_for_service(timeout_sec=5.0)
        fut = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if not (fut.result() and fut.result().success):
            raise RuntimeError('Takeoff command failed')
        self.get_logger().info(f'Taking off to {altitude}m...')

    def _go_to(self, x, y, z, timeout=60, respect_tracking=True):
        """Publish setpoint and wait until within WAYPOINT_TOL."""
        self.get_logger().info(f'→ Waypoint ({x:.1f}, {y:.1f}, {z:.1f})')
        sp = self._make_setpoint(x, y, z)
        rate_sec = 1.0 / SETPOINT_RATE
        start = time.time()

        while True:
            if self.rtl_commanded:
                return
            if respect_tracking and self.tracking_active:
                return
            
            sp.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_pub.publish(sp)
            rclpy.spin_once(self, timeout_sec=rate_sec)

            dist = self._distance_to(x, y, z)
            if dist < WAYPOINT_TOL:
                self.get_logger().info(f'  ✓ Reached (dist={dist:.2f}m)')
                return

            if time.time() - start > timeout:
                self.get_logger().warn(f'  ⚠ Timeout reaching ({x:.1f},{y:.1f},{z:.1f}), continuing...')
                return

    def run(self):
        self._wait_connected()
        self._set_gp_origin()
        self._wait_for_position_estimate()
        self._set_mode('GUIDED')
        time.sleep(1.0)
    # Pre-stream setpoints so FCU accepts navigation after takeoff
        self.get_logger().info('Pre-streaming setpoints...')
        p = self.current_pose.pose.position
        for _ in range(50):  # ~2.5 seconds at 20Hz
            sp = self._make_setpoint(p.x, p.y, p.z + TAKEOFF_ALT)
            self.setpoint_pub.publish(sp)
            self._spin_for(0.05)

        self._arm()
        time.sleep(1.0)
        self._takeoff(TAKEOFF_ALT)

        # Wait to reach takeoff altitude
        self.get_logger().info('Waiting to reach takeoff altitude...')
        while self._distance_to(
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            TAKEOFF_ALT
        ) > 0.5:
            rclpy.spin_once(self, timeout_sec=0.2)

        # Generate and fly pattern
        # waypoints = generate_boustrophedon(AREA_WIDTH, AREA_HEIGHT, LANE_SPACING, CRUISE_ALT)
        waypoints = generate_boustrophedon(AREA_WIDTH, AREA_HEIGHT, LANE_SPACING, CRUISE_ALT)
        self.get_logger().info(
            f'Starting boustrophedon pattern: {len(waypoints)} waypoints, '
            f'{AREA_WIDTH}m x {AREA_HEIGHT}m, {LANE_SPACING}m lane spacing'
        )

        for i, (x, y, z) in enumerate(waypoints):
            self.get_logger().info(f'[{i+1}/{len(waypoints)}]')
            # self._go_to(x, y, z)
            self._go_to(x + PATTERN_ORIGIN_X, y + PATTERN_ORIGIN_Y, z)
            if self.rtl_commanded:
                self.get_logger().info('RTL commanded. Switching mode...')
                self._set_mode('RTL')
                return
            if self.tracking_active:
                self._approach_target()
                break

        if self.tracking_active:
            self._approach_target()
            self.get_logger().info("Holding final position...")
            p = self.current_pose.pose.position
            while rclpy.ok() and not self.rtl_commanded:
                sp = self._make_setpoint(p.x, p.y, p.z)
                self.setpoint_pub.publish(sp)
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.rtl_commanded:
                self._set_mode('RTL')
        else:
            self.get_logger().info('Pattern complete. Returning to launch...')
            self._go_to(0.0, 0.0, CRUISE_ALT)
            self._set_mode('RTL')
            self.get_logger().info('RTL initiated. Done.')

def main():
    rclpy.init()
    node = BoustrophedonNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    except RuntimeError as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()