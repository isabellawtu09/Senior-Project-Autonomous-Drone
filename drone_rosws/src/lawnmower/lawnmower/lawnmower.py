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

Uses MAVROS setpoint topics in GUIDED mode.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geometry_msgs.msg import PoseStamped, Twist
from geographic_msgs.msg import GeoPointStamped
from std_msgs.msg import Bool, Float32MultiArray

import time
import math
from collections import deque

# ─── Pattern Configuration ────────────────────────────────────────────────────
DEFAULT_TAKEOFF_ALT = 2.0     # meters
DEFAULT_CRUISE_ALT = 2.0      # meters
DEFAULT_LANE_SPACING = 2.0    # meters between rows
DEFAULT_WAYPOINT_TOL = 0.5    # meters
DEFAULT_SETPOINT_RATE = 20.0  # Hz
# 15x15 interior means nominal free half-extent ~7.5m from world origin.
DEFAULT_WORLD_HALF_EXTENT = 7.5
# Keep this clearance from all perimeter walls to avoid clipping at turns/overshoot.
DEFAULT_WALL_CLEARANCE = 1.5
# ──────────────────────────────────────────────────────────────────────────────


def generate_boustrophedon_bounds(x_min, x_max, y_min, y_max, spacing, altitude):
    """
    Generate boustrophedon waypoints over bounded XY extents.
    Returns list of (x, y, z) tuples.
    """
    if spacing <= 0:
        raise ValueError("Lane spacing must be > 0")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid search bounds")

    waypoints = []
    y = y_min
    direction = 1  # 1 = forward (+X), -1 = reverse (-X)

    while y <= y_max + 1e-6:
        x_start = x_min if direction == 1 else x_max
        x_end = x_max if direction == 1 else x_min
        waypoints.append((x_start, y, altitude))
        waypoints.append((x_end, y, altitude))
        y += spacing
        direction *= -1

    return waypoints


class BoustrophedonNode(Node):
    def __init__(self):
        super().__init__('boustrophedon_node')
        self.declare_parameter('mavros_ns', '/mavros')
        self.declare_parameter('takeoff_alt', DEFAULT_TAKEOFF_ALT)
        self.declare_parameter('cruise_alt', DEFAULT_CRUISE_ALT)
        self.declare_parameter('lane_spacing', DEFAULT_LANE_SPACING)
        self.declare_parameter('waypoint_tol', DEFAULT_WAYPOINT_TOL)
        self.declare_parameter('setpoint_rate', DEFAULT_SETPOINT_RATE)
        self.declare_parameter('world_half_extent', DEFAULT_WORLD_HALF_EXTENT)
        self.declare_parameter('wall_clearance', DEFAULT_WALL_CLEARANCE)
        self.declare_parameter('target_info_topic', '/grounding_sam_alt/target_info')
        self.declare_parameter('tracking_conf_threshold', 0.70)
        self.declare_parameter('tracking_consistency_required', 3)
        self.declare_parameter('tracking_consistency_window', 5)
        self.declare_parameter('tracking_lost_timeout', 2.5)
        self.declare_parameter('max_track_speed', 0.30)
        self.declare_parameter('max_yaw_rate', 0.40)
        self.declare_parameter('tracking_forward_kp', 0.8)
        self.declare_parameter('tracking_lateral_kp', 0.8)
        self.declare_parameter('tracking_yaw_kp', 2.0)
        self.declare_parameter('tracking_center_tolerance', 0.10)
        self.declare_parameter('tracking_target_area', 0.08)
        self.declare_parameter('position_wait_timeout', 25.0)
        self.declare_parameter('allow_position_timeout_continue', True)

        self.mavros_ns = self._normalize_ns(self.get_parameter('mavros_ns').value)
        self.takeoff_alt = float(self.get_parameter('takeoff_alt').value)
        self.cruise_alt = float(self.get_parameter('cruise_alt').value)
        self.lane_spacing = float(self.get_parameter('lane_spacing').value)
        self.waypoint_tol = float(self.get_parameter('waypoint_tol').value)
        self.setpoint_rate = float(self.get_parameter('setpoint_rate').value)
        self.world_half_extent = float(self.get_parameter('world_half_extent').value)
        self.wall_clearance = float(self.get_parameter('wall_clearance').value)
        self.target_info_topic = str(self.get_parameter('target_info_topic').value)
        self.tracking_conf_threshold = float(self.get_parameter('tracking_conf_threshold').value)
        self.tracking_consistency_required = int(self.get_parameter('tracking_consistency_required').value)
        self.tracking_consistency_window = int(self.get_parameter('tracking_consistency_window').value)
        self.tracking_lost_timeout = float(self.get_parameter('tracking_lost_timeout').value)
        self.max_track_speed = float(self.get_parameter('max_track_speed').value)
        self.max_yaw_rate = float(self.get_parameter('max_yaw_rate').value)
        self.tracking_forward_kp = float(self.get_parameter('tracking_forward_kp').value)
        self.tracking_lateral_kp = float(self.get_parameter('tracking_lateral_kp').value)
        self.tracking_yaw_kp = float(self.get_parameter('tracking_yaw_kp').value)
        self.tracking_center_tolerance = float(self.get_parameter('tracking_center_tolerance').value)
        self.tracking_target_area = float(self.get_parameter('tracking_target_area').value)
        self.position_wait_timeout = float(self.get_parameter('position_wait_timeout').value)
        self.allow_position_timeout_continue = bool(self.get_parameter('allow_position_timeout_continue').value)

        safe_half = max(self.world_half_extent - self.wall_clearance, 0.5)
        self.search_x_min = -safe_half
        self.search_x_max = safe_half
        self.search_y_min = -safe_half
        self.search_y_max = safe_half

        self.get_logger().info(f'Using MAVROS namespace: {self.mavros_ns}')
        self.get_logger().info(
            f'Search bounds: x[{self.search_x_min:.1f}, {self.search_x_max:.1f}] '
            f'y[{self.search_y_min:.1f}, {self.search_y_max:.1f}] '
            f'clearance={self.wall_clearance:.1f}m'
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state = State()
        self.current_pose = PoseStamped()
        self.tracking_active = False
        self.target_visible = False
        self.target_score = 0.0
        self.target_cx = 0.5
        self.target_cy = 0.5
        self.target_area = 0.0
        self.last_consistent_detection_time = None
        self.recent_detection_passes = deque(maxlen=max(self.tracking_consistency_window, 1))
        self.pose_msg_count = 0
        self.last_pose_wall_time = None

        self.create_subscription(State, self._topic('state'), self._state_cb, qos)
        self.create_subscription(PoseStamped, self._topic('local_position/pose'), self._pose_cb, qos)
        self.create_subscription(Bool, '/object_found', self._tracking_cb, 10)
        self.create_subscription(Float32MultiArray, self.target_info_topic, self._target_info_cb, 10)

        self.setpoint_pub = self.create_publisher(PoseStamped, self._topic('setpoint_position/local'), 10)
        self.velocity_pub = self.create_publisher(Twist, self._topic('setpoint_velocity/cmd_vel_unstamped'), 10)

        self.gp_origin_pub = self.create_publisher(GeoPointStamped, self._topic('global_position/set_gp_origin'), 10)

        self.arming_client = self.create_client(CommandBool, self._topic('cmd/arming'))
        self.mode_client = self.create_client(SetMode, self._topic('set_mode'))
        self.takeoff_client = self.create_client(CommandTOL, self._topic('cmd/takeoff'))

    def _normalize_ns(self, ns: str) -> str:
        ns = (ns or '/mavros').strip()
        if not ns.startswith('/'):
            ns = f'/{ns}'
        return ns.rstrip('/')

    def _topic(self, suffix: str) -> str:
        return f'{self.mavros_ns}/{suffix}'

    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.current_pose = msg
        self.pose_msg_count += 1
        self.last_pose_wall_time = time.time()

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
    
    def _tracking_cb(self, msg):
        self.tracking_active = msg.data
        if self.tracking_active:
            self.get_logger().info('Object found! Stopping pattern.')
        else:
            self._publish_zero_velocity()

    def _target_info_cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) < 5:
            return
        visible = data[0] >= 0.5
        score = float(data[1])
        self.target_visible = visible
        self.target_score = score
        self.target_cx = float(data[2])
        self.target_cy = float(data[3])
        self.target_area = float(data[4])
        pass_now = visible and (score >= self.tracking_conf_threshold)
        self.recent_detection_passes.append(1 if pass_now else 0)
        if pass_now:
            self.last_consistent_detection_time = time.time()

    def _distance_to(self, x, y, z):
        p = self.current_pose.pose.position
        return math.sqrt((p.x - x)**2 + (p.y - y)**2 + (p.z - z)**2)

    def _altitude_error(self, target_z: float) -> float:
        return abs(self.current_pose.pose.position.z - target_z)

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
            # Keep this non-blocking so startup cannot appear stuck at origin setup.
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_for_position_estimate(self, timeout=60):
        """Wait until local-position stream is actively publishing fresh messages."""
        self.get_logger().info('Waiting for EKF position estimate...')
        start = time.time()
        start_pose_count = self.pose_msg_count
        last_origin_retry = start
        while True:
            rclpy.spin_once(self, timeout_sec=0.5)
            # Use message freshness instead of non-zero coordinates.
            # In SITL, valid initial local position can legitimately stay near (0,0,0).
            has_fresh_stream = (
                self.pose_msg_count > start_pose_count + 2 and
                self.last_pose_wall_time is not None and
                (time.time() - self.last_pose_wall_time) < 1.5
            )
            if has_fresh_stream:
                self.get_logger().info('Position estimate acquired.')
                return
            # Re-publish origin only every 5s (throttled) to nudge EKF.
            now = time.time()
            if (now - last_origin_retry) >= 5.0:
                self._set_gp_origin()
                last_origin_retry = now
            if now - start > timeout:
                if self.allow_position_timeout_continue:
                    self.get_logger().warn(
                        'Timed out waiting for position estimate; continuing with conservative setpoints.'
                    )
                    return
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

    def _go_to(self, x, y, z, timeout=60):
        """Publish setpoint and wait until within WAYPOINT_TOL."""
        self.get_logger().info(f'→ Waypoint ({x:.1f}, {y:.1f}, {z:.1f})')
        sp = self._make_setpoint(x, y, z)
        rate_sec = 1.0 / max(self.setpoint_rate, 1.0)
        start = time.time()

        while True:
            if self.tracking_active:
                return
            
            sp.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_pub.publish(sp)
            rclpy.spin_once(self, timeout_sec=rate_sec)

            dist = self._distance_to(x, y, z)
            if dist < self.waypoint_tol:
                self.get_logger().info(f'  ✓ Reached (dist={dist:.2f}m)')
                return

            if time.time() - start > timeout:
                self.get_logger().warn(f'  ⚠ Timeout reaching ({x:.1f},{y:.1f},{z:.1f}), continuing...')
                return

    def _has_consistent_detection(self) -> bool:
        if len(self.recent_detection_passes) < self.tracking_consistency_required:
            return False
        return sum(self.recent_detection_passes) >= self.tracking_consistency_required

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    def _publish_zero_velocity(self):
        self.velocity_pub.publish(Twist())

    def _compute_tracking_velocity(self) -> Twist:
        cmd = Twist()
        now = time.time()
        consistent = self._has_consistent_detection()
        lost = (
            self.last_consistent_detection_time is None or
            (now - self.last_consistent_detection_time) > self.tracking_lost_timeout
        )
        if not consistent or lost:
            return cmd

        ex = self.target_cx - 0.5
        area_error = max(0.0, self.tracking_target_area - self.target_area)

        # First center the target, then move forward conservatively.
        if abs(ex) <= self.tracking_center_tolerance:
            forward = self.tracking_forward_kp * area_error
        else:
            forward = 0.0
        lateral = -self.tracking_lateral_kp * ex
        yaw_rate = -self.tracking_yaw_kp * ex

        cmd.linear.x = self._clamp(forward, 0.0, self.max_track_speed)
        cmd.linear.y = self._clamp(lateral, -self.max_track_speed, self.max_track_speed)
        cmd.linear.z = 0.0  # Keep altitude fixed in first tracking version.
        cmd.angular.z = self._clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        return cmd

    def _approach_target(self):
        self.get_logger().info(
            'Entering tracking approach mode (velocity clamp + confidence consistency + loss timeout).'
        )
        rate_sec = 1.0 / max(self.setpoint_rate, 1.0)
        while rclpy.ok() and self.tracking_active:
            rclpy.spin_once(self, timeout_sec=rate_sec)
            cmd = self._compute_tracking_velocity()
            self.velocity_pub.publish(cmd)
        self._publish_zero_velocity()

    def run(self):
        self._wait_connected()
        self._set_gp_origin()
        self._wait_for_position_estimate(timeout=self.position_wait_timeout)
        self._set_mode('GUIDED')
        time.sleep(1.0)
        self._arm()
        time.sleep(1.0)
        self._takeoff(self.takeoff_alt)

        # Wait to reach takeoff altitude
        self.get_logger().info('Waiting to reach takeoff altitude...')
        takeoff_wait_start = time.time()
        takeoff_wait_timeout = 25.0
        while self._altitude_error(self.takeoff_alt) > 0.35:
            # Keep publishing a hold setpoint while waiting; some stacks are more stable
            # when position setpoints continue during/after takeoff command.
            p = self.current_pose.pose.position
            hold = self._make_setpoint(p.x, p.y, self.takeoff_alt)
            self.setpoint_pub.publish(hold)
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.time() - takeoff_wait_start > takeoff_wait_timeout:
                self.get_logger().warn(
                    'Timed out waiting for takeoff altitude; continuing with current altitude.'
                )
                break

        # Generate and fly pattern
        waypoints = generate_boustrophedon_bounds(
            self.search_x_min,
            self.search_x_max,
            self.search_y_min,
            self.search_y_max,
            self.lane_spacing,
            self.cruise_alt,
        )
        self.get_logger().info(
            f'Starting boustrophedon pattern: {len(waypoints)} waypoints, '
            f'lane spacing={self.lane_spacing:.1f}m altitude={self.cruise_alt:.1f}m'
        )

        for i, (x, y, z) in enumerate(waypoints):
            self.get_logger().info(f'[{i+1}/{len(waypoints)}]')
            self._go_to(x, y, z)
            if self.tracking_active:
                break

        if self.tracking_active:
            self._approach_target()
        else:
            self.get_logger().info('Pattern complete. Returning to launch...')
            self._go_to(0.0, 0.0, self.cruise_alt)
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