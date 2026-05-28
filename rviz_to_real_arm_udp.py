#!/usr/bin/env python3

import json
import math
import socket
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class RVizToRealArmUDP(Node):
    def __init__(self):
        super().__init__("rviz_to_real_arm_udp")

        # ---------------------------------------------------------
        # Raspberry Pi UDP settings
        # ---------------------------------------------------------
        self.declare_parameter("pi_ip", "192.168.188.144")
        self.declare_parameter("pi_port", 9090)

        # Topic from RViz / MoveIt / ros2_control simulation
        self.declare_parameter("joint_state_topic", "/joint_states")

        # Send speed
        # This is the movement duration used by your Raspberry Pi executor.
        # Use big value like 20 or 30 to watch physical movement slowly.
        self.declare_parameter("move_duration", 8.0)

        # Minimum seconds between UDP sends
        self.declare_parameter("send_period", 0.20)

        # Only send if a servo changed by this many degrees
        self.declare_parameter("angle_deadband_deg", 0.5)

        # Safety
        self.declare_parameter("dry_run", False)

        # If True, sends gripper too.
        # If False, only sends arm joints.
        self.declare_parameter("send_gripper", True)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.pi_ip = self.get_parameter("pi_ip").value
        self.pi_port = int(self.get_parameter("pi_port").value)
        self.joint_state_topic = self.get_parameter("joint_state_topic").value

        self.move_duration = float(self.get_parameter("move_duration").value)
        self.send_period = float(self.get_parameter("send_period").value)
        self.angle_deadband_deg = float(self.get_parameter("angle_deadband_deg").value)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.send_gripper = bool(self.get_parameter("send_gripper").value)

        # ---------------------------------------------------------
        # URDF joint name -> Raspberry Pi servo key
        # ---------------------------------------------------------
        self.joint_to_servo = {
            "joint_6_base": "base",
            "joint_5_shoulder": "shoulder",
            "joint_4_elbow": "elbow",
            "joint_3_wrist_pitch": "wrist",
            "joint_2_wrist_roll": "wrist_rotation",
            "joint_1_gripper": "gripper",
        }

        # ---------------------------------------------------------
        # Servo safety limits, same idea as Raspberry Pi code
        # ---------------------------------------------------------
        self.safe_range = {
            "gripper": (70.0, 130.0),
            "wrist_rotation": (10.0, 170.0),
            "wrist": (0.0, 180.0),
            "elbow": (10.0, 170.0),
            "shoulder": (20.0, 160.0),
            "base": (0.0, 180.0),
        }

        # ---------------------------------------------------------
        # Per-joint calibration
        #
        # Current default conversion:
        #   servo_degree = joint_rad * 180/pi + 90
        #
        # If a real servo moves opposite direction, set scale to -1.0.
        # If a real servo center is wrong, adjust offset.
        # ---------------------------------------------------------
        self.servo_calibration = {
            "base": {"scale": 1.0, "offset": 90.0},
    "shoulder": {"scale": 1.0, "offset": 91.6},
    "elbow": {"scale": 1.0, "offset": 90.4},
    "wrist": {"scale": 1.0, "offset": 95.1},
    "wrist_rotation": {"scale": 1.0, "offset": 90.0},
    "gripper": {"scale": 1.0, "offset": 80.0},
        }

        # ---------------------------------------------------------
        # UDP
        # ---------------------------------------------------------
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.last_send_time = 0.0
        self.last_sent_angles: Optional[Dict[str, float]] = None

        self.subscription = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            10,
        )

        self.get_logger().info("✅ RViz → Raspberry Pi UDP bridge started")
        self.get_logger().info(f"Listening joint states: {self.joint_state_topic}")
        self.get_logger().info(f"Sending to Pi: {self.pi_ip}:{self.pi_port}")
        self.get_logger().info(f"Move duration: {self.move_duration:.2f}s")
        self.get_logger().info(f"Dry run: {self.dry_run}")

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def joint_rad_to_servo_degree(self, servo_key: str, joint_rad: float) -> float:
        cal = self.servo_calibration[servo_key]

        raw_degree = math.degrees(joint_rad) * cal["scale"] + cal["offset"]

        safe_min, safe_max = self.safe_range[servo_key]
        safe_degree = self.clamp(raw_degree, safe_min, safe_max)

        return safe_degree

    def should_send(self, angles: Dict[str, float]) -> bool:
        now = time.time()

        if now - self.last_send_time < self.send_period:
            return False

        if self.last_sent_angles is None:
            return True

        for key, value in angles.items():
            previous = self.last_sent_angles.get(key)

            if previous is None:
                return True

            if abs(value - previous) >= self.angle_deadband_deg:
                return True

        return False

    def send_udp(self, angles: Dict[str, float]):
        packet = {
            "duration": self.move_duration,
            "angles": angles,
        }

        data = json.dumps(packet).encode("utf-8")

        if self.dry_run:
            self.get_logger().info(f"[DRY RUN] Would send: {packet}")
            return

        self.sock.sendto(data, (self.pi_ip, self.pi_port))

        self.last_send_time = time.time()
        self.last_sent_angles = angles.copy()

        self.get_logger().info(f"Sent to real arm: {packet}")

    # ---------------------------------------------------------
    # Callback
    # ---------------------------------------------------------

    def joint_state_callback(self, msg: JointState):
        angles: Dict[str, float] = {}

        for joint_name, joint_pos in zip(msg.name, msg.position):
            if joint_name not in self.joint_to_servo:
                continue

            servo_key = self.joint_to_servo[joint_name]

            if servo_key == "gripper" and not self.send_gripper:
                continue

            servo_degree = self.joint_rad_to_servo_degree(
                servo_key,
                float(joint_pos),
            )

            angles[servo_key] = round(servo_degree, 2)

        if not angles:
            return

        if self.should_send(angles):
            self.send_udp(angles)


def main(args=None):
    rclpy.init(args=args)

    node = RVizToRealArmUDP()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()