#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

import socket
import json
import math
import time


class MoveItExecuteToReal(Node):
    def __init__(self):
        super().__init__("moveit_execute_to_real")

        self.trajectory_topic = "/arm_group_controller/joint_trajectory"

        self.pi_ip = "192.168.188.144"
        self.pi_port = 9090
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.joint_map = {
            "joint_6_base": "base",
            "joint_5_shoulder": "shoulder",
            "joint_4_elbow": "elbow",
            "joint_3_wrist_pitch": "wrist",
            "joint_2_wrist_roll": "wrist_rotation",
            "joint_1_gripper": "gripper",
        }

        # Your real straight-up offsets
        self.offset = {
            "base": 90.0,
            "shoulder": 91.6,
            "elbow": 90.4,
            "wrist": 95.1,
            "wrist_rotation": 90.0,
            "gripper": 80.0,
        }

        self.safe_range = {
            "base": (0.0, 180.0),
            "shoulder": (20.0, 180.0),
            "elbow": (10.0, 170.0),
            "wrist": (0.0, 180.0),
            "wrist_rotation": (10.0, 170.0),
            "gripper": (70.0, 130.0),
        }

        self.sub = self.create_subscription(
            JointTrajectory,
            self.trajectory_topic,
            self.trajectory_callback,
            10,
        )

        self.get_logger().info("✅ MoveIt EXECUTE → real arm bridge started")
        self.get_logger().info(f"Listening: {self.trajectory_topic}")
        self.get_logger().info(f"Sending to: {self.pi_ip}:{self.pi_port}")

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def duration_to_sec(self, d):
        return float(d.sec) + float(d.nanosec) * 1e-9

    def trajectory_callback(self, msg: JointTrajectory):
        if not msg.points:
            return

        self.get_logger().info(f"Execute received: {len(msg.points)} points")

        prev_time = 0.0

        for index, point in enumerate(msg.points):
            angles = {}

            for joint_name, position in zip(msg.joint_names, point.positions):
                if joint_name not in self.joint_map:
                    continue

                servo_name = self.joint_map[joint_name]

                rviz_deg = math.degrees(float(position))
                servo_deg = self.offset[servo_name] + rviz_deg

                low, high = self.safe_range[servo_name]
                servo_deg = self.clamp(servo_deg, low, high)

                angles[servo_name] = round(servo_deg, 2)

            if not angles:
                continue

            current_time = self.duration_to_sec(point.time_from_start)
            step_duration = current_time - prev_time
            prev_time = current_time

            if index == 0 or step_duration <= 0.0:
                step_duration = 0.15

            # New Raspberry Pi format
            packet = {
                "duration": max(0.12, step_duration),
                "angles": angles,
            }

            self.sock.sendto(
                json.dumps(packet).encode(),
                (self.pi_ip, self.pi_port),
            )

            self.get_logger().info(f"Sent: {packet}")

            time.sleep(max(0.12, step_duration))

        self.get_logger().info("✅ Execute trajectory sent to real arm")


def main(args=None):
    rclpy.init(args=args)
    node = MoveItExecuteToReal()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.sock.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()