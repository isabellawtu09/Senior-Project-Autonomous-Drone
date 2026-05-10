#!/usr/bin/env python3
"""
Measures elapsed time from takeoff to first object detection.

Run alongside the sim (with ROS environment sourced):
    python3 sim_interface/mission_timer.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
import time

TAKEOFF_ALT_THRESHOLD = 0.5  # meters — altitude above which the drone is considered airborne


class MissionTimer(Node):
    def __init__(self):
        super().__init__('mission_timer')
        self._takeoff_time = None
        self._airborne = False
        self._done = False
        self._prompt = ''
        self._ai_mode = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._pose_cb, qos)
        self.create_subscription(Bool, '/object_found', self._found_cb, 10)
        self.create_subscription(String, '/target_object', self._prompt_cb, 10)
        self.create_subscription(Bool, '/ai_mode', self._ai_mode_cb, 10)

        print('Mission timer ready. Waiting for takeoff...')

    def _prompt_cb(self, msg):
        if msg.data.lower() == 'stop':
            return
        self._prompt = msg.data

    def _ai_mode_cb(self, msg):
        self._ai_mode = msg.data

    def _pose_cb(self, msg):
        alt = msg.pose.position.z
        if not self._airborne and alt > TAKEOFF_ALT_THRESHOLD:
            self._airborne = True
            self._takeoff_time = time.time()
            print(f'[TAKEOFF] Detected at z={alt:.2f} m — timer started.')
        elif self._airborne and alt < TAKEOFF_ALT_THRESHOLD:
            self._airborne = False
            if not self._done and self._takeoff_time is not None:
                elapsed = time.time() - self._takeoff_time
                mins, secs = divmod(elapsed, 60)
                print()
                print('==============================')
                print('  RTL — object NOT found.')
                print(f'  Time airborne: {int(mins)}m {secs:.1f}s  ({elapsed:.1f}s total)')
                print('==============================')
                self._save_result(elapsed, mins, secs, found=False)
            self._takeoff_time = None
            self._done = False
            print('\n[LANDED] Ready for next run. Waiting for takeoff...')

    def _found_cb(self, msg):
        if msg.data and self._takeoff_time is not None and not self._done:
            elapsed = time.time() - self._takeoff_time
            self._done = True
            mins, secs = divmod(elapsed, 60)
            print()
            print('==============================')
            print('  Object found!')
            print(f'  Time from takeoff: {int(mins)}m {secs:.1f}s  ({elapsed:.1f}s total)')
            print('==============================')
            self._save_result(elapsed, mins, secs, found=True)

    def _save_result(self, elapsed, mins, secs, found=True):
        import os
        os.makedirs('metrics', exist_ok=True)
        log_path = 'metrics/mission_times.txt'
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        prompt_field = f'  prompt="{self._prompt}"' if self._prompt else ''
        ai_field = f'  ai_mode={"yes" if self._ai_mode else "no"}'
        outcome = 'Time to detection' if found else 'NOT FOUND — time airborne'
        line = f'[{timestamp}]{prompt_field}{ai_field}  {outcome}: {int(mins)}m {secs:.1f}s  ({elapsed:.1f}s)\n'
        with open(log_path, 'a') as f:
            f.write(line)
        print(f'Result saved to {log_path}')


def main():
    rclpy.init()
    node = MissionTimer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node._takeoff_time is None:
            print('\nTimer stopped — drone never left the ground.')
        elif not node._done:
            elapsed = time.time() - node._takeoff_time
            mins, secs = divmod(elapsed, 60)
            print(f'\nMission interrupted. Time airborne: {int(mins)}m {secs:.1f}s ({elapsed:.1f}s total)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
