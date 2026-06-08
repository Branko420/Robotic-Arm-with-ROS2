#!/usr/bin/env python3

import json
import math
import socket
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class MoveItExecuteLiveToRealArm(Node):
    def __init__(self):
        super().__init__("moveit_execute_live_to_real_arm")

        self.declare_parameter("joint_state_topic", "/joint_states")

        self.declare_parameter("pi_ip", "192.168.188.144")
        self.declare_parameter("pi_port", 9090)

        # Use "new" for:
        # {"duration": 0.25, "angles": {...}}
        #
        # Use "old" for:
        # {"base": 90, "shoulder": 90, ...}
        self.declare_parameter("packet_format", "new")

        # How often to send while MoveIt is executing.
        self.declare_parameter("send_hz", 8.0)

        # Duration for each physical mini-step.
        # For live following, this must be small, not 10 seconds.
        self.declare_parameter("step_duration", 0.18)

        # Ignore tiny joint noise.
        self.declare_parameter("motion_threshold_rad", 0.002)

        # Send only if servo angle changed enough.
        self.declare_parameter("angle_deadband_deg", 0.4)

        self.declare_parameter("send_gripper", True)
        self.declare_parameter("gripper_angle", 80.0)

        self.declare_parameter("dry_run", False)

        self.joint_state_topic = self.get_parameter("joint_state_topic").value

        self.pi_ip = self.get_parameter("pi_ip").value
        self.pi_port = int(self.get_parameter("pi_port").value)

        self.packet_format = self.get_parameter("packet_format").value

        self.send_hz = float(self.get_parameter("send_hz").value)
        self.step_duration = float(self.get_parameter("step_duration").value)

        self.motion_threshold_rad = float(self.get_parameter("motion_threshold_rad").value)
        self.angle_deadband_deg = float(self.get_parameter("angle_deadband_deg").value)

        self.send_gripper = bool(self.get_parameter("send_gripper").value)
        self.gripper_angle = float(self.get_parameter("gripper_angle").value)

        self.dry_run = bool(self.get_parameter("dry_run").value)

        if self.send_hz < 1.0:
            self.send_hz = 1.0

        self.send_period = 1.0 / self.send_hz

        self.joint_to_servo = {
            "joint_6_base": "base",
            "joint_5_shoulder": "shoulder",
            "joint_4_elbow": "elbow",
            "joint_3_wrist_pitch": "wrist",
            "joint_2_wrist_roll": "wrist_rotation",
        }

        # Your real straight-up neutral servo angles
        self.calibration = {
            "base": {"scale": 1.0, "offset": 90.0},
            "shoulder": {"scale": 1.0, "offset": 91.6},
            "elbow": {"scale": 1.0, "offset": 90.4},
            "wrist": {"scale": 1.0, "offset": 95.1},
            "wrist_rotation": {"scale": 1.0, "offset": 90.0},
        }

        self.safe_range = {
            "base": (0.0, 180.0),
            "shoulder": (20.0, 160.0),
            "elbow": (10.0, 170.0),
            "wrist": (0.0, 180.0),
            "wrist_rotation": (10.0, 170.0),
            "gripper": (70.0, 130.0),
        }

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.previous_positions: Optional[Dict[str, float]] = None
        self.last_sent_angles: Optional[Dict[str, float]] = None
        self.last_send_time = 0.0

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            10,
        )

        self.get_logger().info("✅ MoveIt EXECUTE live bridge active")
        self.get_logger().info(f"Listening to real executed /joint_states: {self.joint_state_topic}")
        self.get_logger().info(f"Sending to Raspberry Pi: {self.pi_ip}:{self.pi_port}")
        self.get_logger().info(f"packet_format={self.packet_format}")
        self.get_logger().info(f"send_hz={self.send_hz:.1f}, step_duration={self.step_duration:.2f}")
        self.get_logger().info(f"dry_run={self.dry_run}")

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def joint_state_to_dict(self, msg: JointState) -> Dict[str, float]:
        return {
            name: float(position)
            for name, position in zip(msg.name, msg.position)
        }

    def is_real_motion(self, current: Dict[str, float]) -> bool:
        if self.previous_positions is None:
            return False

        for joint_name in self.joint_to_servo.keys():
            if joint_name not in current:
                continue

            if joint_name not in self.previous_positions:
                continue

            diff = abs(current[joint_name] - self.previous_positions[joint_name])

            if diff >= self.motion_threshold_rad:
                return True

        return False

    def convert_to_servo_angles(self, current: Dict[str, float]) -> Optional[Dict[str, float]]:
        angles: Dict[str, float] = {}

        for joint_name, servo_key in self.joint_to_servo.items():
            if joint_name not in current:
                self.get_logger().warn(f"Missing joint in /joint_states: {joint_name}")
                return None

            joint_rad = current[joint_name]
            joint_deg = math.degrees(joint_rad)

            cal = self.calibration[servo_key]
            servo_deg = cal["offset"] + cal["scale"] * joint_deg

            low, high = self.safe_range[servo_key]
            servo_deg = self.clamp(servo_deg, low, high)

            angles[servo_key] = round(servo_deg, 2)

        if self.send_gripper:
            low, high = self.safe_range["gripper"]
            angles["gripper"] = round(self.clamp(self.gripper_angle, low, high), 2)

        return angles

    def changed_enough(self, angles: Dict[str, float]) -> bool:
        if self.last_sent_angles is None:
            return True

        for key, value in angles.items():
            old = self.last_sent_angles.get(key)

            if old is None:
                return True

            if abs(value - old) >= self.angle_deadband_deg:
                return True

        return False

    def send_angles(self, angles: Dict[str, float]):
        now = time.time()

        if now - self.last_send_time < self.send_period:
            return

        if not self.changed_enough(angles):
            return

        if self.packet_format == "new":
            packet = {
                "duration": self.step_duration,
                "angles": angles,
            }
        elif self.packet_format == "old":
            packet = angles
        else:
            self.get_logger().error("packet_format must be 'new' or 'old'")
            return

        if self.dry_run:
            self.get_logger().info(f"[DRY RUN] Would send: {packet}")
        else:
            self.sock.sendto(
                json.dumps(packet).encode("utf-8"),
                (self.pi_ip, self.pi_port),
            )
            self.get_logger().info(f"✅ Sent live execute pose: {packet}")

        self.last_sent_angles = dict(angles)
        self.last_send_time = now

    def joint_state_callback(self, msg: JointState):
        current = self.joint_state_to_dict(msg)

        moving = self.is_real_motion(current)

        self.previous_positions = current

        if not moving:
            return

        angles = self.convert_to_servo_angles(current)

        if angles is None:
            return

        self.send_angles(angles)


def main(args=None):
    rclpy.init(args=args)

    node = MoveItExecuteLiveToRealArm()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()