#!/usr/bin/env python3
"""
Measures elapsed time from takeoff to first object detection.
Records a WebM video clip of each mission to metrics/videos/.

Run alongside the sim (with ROS environment sourced):
    python3 sim_interface/mission_timer.py
"""

import os
import subprocess
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

TAKEOFF_ALT_THRESHOLD = 0.5  # meters
CAMERA_TOPIC = (
    '/world/iris_runway_15x15_walls/model/iris_with_gimbal'
    '/model/gimbal/link/pitch_link/sensor/camera/image'
)
VIDEO_FPS = 15
VIDEOS_DIR = 'metrics/videos'
LOG_PATH = 'metrics/mission_times.md'


class MissionTimer(Node):
    def __init__(self):
        super().__init__('mission_timer')
        self._takeoff_time = None
        self._airborne = False
        self._done = False
        self._prompt = ''
        self._ai_mode = False

        self._writer: cv2.VideoWriter | None = None
        self._writer_pending = False
        self._video_path: str | None = None
        self._run_ts: str | None = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._pose_cb, qos)
        self.create_subscription(Bool, '/object_found', self._found_cb, 10)
        self.create_subscription(String, '/target_object', self._prompt_cb, 10)
        self.create_subscription(Bool, '/ai_mode', self._ai_mode_cb, 10)
        self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, qos)

        print('Mission timer ready. Waiting for takeoff...')

    def _prompt_cb(self, msg):
        if msg.data.lower() == 'stop':
            return
        self._prompt = msg.data

    def _ai_mode_cb(self, msg):
        self._ai_mode = msg.data

    def _image_cb(self, msg):
        if not self._writer_pending and self._writer is None:
            return
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            return
        if self._writer_pending:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self._video_path,
                cv2.VideoWriter_fourcc(*'VP80'),
                VIDEO_FPS,
                (w, h),
            )
            self._writer_pending = False
        if self._writer is not None:
            self._writer.write(frame)

    def _pose_cb(self, msg):
        alt = msg.pose.position.z
        if not self._airborne and alt > TAKEOFF_ALT_THRESHOLD:
            self._airborne = True
            self._run_ts = time.strftime('%Y-%m-%d_%H-%M-%S')
            self._takeoff_time = time.time()
            self._start_recording()
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
            self._stop_recording()
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
            self._stop_recording()

    def _start_recording(self):
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        self._video_path = os.path.join(VIDEOS_DIR, f'{self._run_ts}.webm')
        self._writer = None
        self._writer_pending = True

    def _stop_recording(self):
        self._writer_pending = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            mp4_path = self._video_path.replace('.webm', '.mp4')
            result = subprocess.run(
                [
                    '/usr/bin/ffmpeg', '-y',
                    '-i', self._video_path,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    mp4_path,
                ],
                capture_output=True,
            )
            if result.returncode == 0:
                os.remove(self._video_path)
                self._video_path = mp4_path
                print(f'Video saved → {mp4_path}')
            else:
                print(f'ffmpeg conversion failed, keeping WebM: {self._video_path}')

    def _save_result(self, elapsed, mins, secs, found=True):
        os.makedirs('metrics', exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        prompt_field = f'  prompt="{self._prompt}"' if self._prompt else ''
        ai_field = f'  ai_mode={"yes" if self._ai_mode else "no"}'
        outcome = 'Time to detection' if found else 'NOT FOUND — time airborne'
        video_field = (
            f'  [{self._run_ts}.mp4](videos/{self._run_ts}.mp4)'
            if self._run_ts else ''
        )
        line = (
            f'* [{timestamp}]{prompt_field}{ai_field}'
            f'  {outcome}: {int(mins)}m {secs:.1f}s  ({elapsed:.1f}s)'
            f'{video_field}\n'
        )
        with open(LOG_PATH, 'a') as f:
            f.write(line)
        print(f'Result saved to {LOG_PATH}')


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
        node._stop_recording()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
