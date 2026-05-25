#!/usr/bin/env python3

import json
import math
import threading
from typing import Optional, Dict

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from pymoveit2 import MoveIt2


class GoToGreenCubeIKJoints(Node):
    def __init__(self):
        super().__init__("go_to_green_cube_ik_joints")

        self.declare_parameter("green_cube_json_topic", "/digital_twin/green_cube_json")

        self.declare_parameter("move_group_name", "arm_group")
        self.declare_parameter("base_link_name", "base_link")
        self.declare_parameter("end_effector_name", "link_gripper_base")

        # Arm geometry from your URDF.
        self.declare_parameter("shoulder_height", 0.141)
        self.declare_parameter("upper_arm_length", 0.105)
        self.declare_parameter("forearm_length", 0.220)

        # Target offsets.
        self.declare_parameter("target_height_above_cube", 0.10)
        self.declare_parameter("extra_x_offset", 0.00)
        self.declare_parameter("extra_y_offset", 0.00)

        # Keep target inside safe reachable ring.
        self.declare_parameter("min_radius", 0.17)
        self.declare_parameter("max_radius", 0.30)

        # If joints move opposite direction, flip these signs.
        self.declare_parameter("shoulder_sign", 1.0)
        self.declare_parameter("elbow_sign", 1.0)
        self.declare_parameter("wrist_sign", 1.0)

        # Wrist roll.
        self.declare_parameter("wrist_roll", 0.0)

        self.declare_parameter("auto_go", True)

        self.green_cube_json_topic = self.get_parameter("green_cube_json_topic").value

        self.move_group_name = self.get_parameter("move_group_name").value
        self.base_link_name = self.get_parameter("base_link_name").value
        self.end_effector_name = self.get_parameter("end_effector_name").value

        self.shoulder_height = float(self.get_parameter("shoulder_height").value)
        self.upper_arm_length = float(self.get_parameter("upper_arm_length").value)
        self.forearm_length = float(self.get_parameter("forearm_length").value)

        self.target_height_above_cube = float(
            self.get_parameter("target_height_above_cube").value
        )
        self.extra_x_offset = float(self.get_parameter("extra_x_offset").value)
        self.extra_y_offset = float(self.get_parameter("extra_y_offset").value)

        self.min_radius = float(self.get_parameter("min_radius").value)
        self.max_radius = float(self.get_parameter("max_radius").value)

        self.shoulder_sign = float(self.get_parameter("shoulder_sign").value)
        self.elbow_sign = float(self.get_parameter("elbow_sign").value)
        self.wrist_sign = float(self.get_parameter("wrist_sign").value)

        self.wrist_roll = float(self.get_parameter("wrist_roll").value)
        self.auto_go = bool(self.get_parameter("auto_go").value)

        self.joint_names = [
            "joint_6_base",
            "joint_5_shoulder",
            "joint_4_elbow",
            "joint_3_wrist_pitch",
            "joint_2_wrist_roll",
        ]

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_link_name,
            end_effector_name=self.end_effector_name,
            group_name=self.move_group_name,
        )

        self.moveit2.max_velocity = 0.25
        self.moveit2.max_acceleration = 0.25

        self.latest_cube: Optional[Dict] = None
        self.has_moved = False
        self.is_moving = False

        self.sub = self.create_subscription(
            String,
            self.green_cube_json_topic,
            self.green_cube_callback,
            10,
        )

        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info("✅ Green cube joint-IK mover started")
        self.get_logger().info(f"Listening: {self.green_cube_json_topic}")
        self.get_logger().info("This uses calculated joint angles, not pose IK.")

    def green_cube_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            if not data.get("found", False):
                self.latest_cube = None
                return

            self.latest_cube = data

        except Exception as e:
            self.get_logger().warn(f"Failed to parse green cube JSON: {e}")

    def timer_callback(self):
        if not self.auto_go:
            return

        if self.has_moved or self.is_moving:
            return

        if self.latest_cube is None:
            return

        self.is_moving = True
        thread = threading.Thread(target=self.move_to_cube, daemon=True)
        thread.start()

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def calculate_ik(self, x, y, z):
        """
        Simple 2-link planar IK after base yaw.

        Uses:
        joint_6_base = atan2(y, x)
        joint_5_shoulder = shoulder angle
        joint_4_elbow = elbow angle
        joint_3_wrist_pitch = keeps wrist roughly level
        joint_2_wrist_roll = fixed / later object yaw
        """

        base_yaw = math.atan2(y, x)

        radius = math.sqrt(x * x + y * y)

        if radius < self.min_radius:
            self.get_logger().warn(
                f"Target radius {radius:.3f} is too close. Clamping to {self.min_radius:.3f}"
            )
            radius = self.min_radius

        if radius > self.max_radius:
            self.get_logger().warn(
                f"Target radius {radius:.3f} is too far. Clamping to {self.max_radius:.3f}"
            )
            radius = self.max_radius

        z_rel = z - self.shoulder_height

        l1 = self.upper_arm_length
        l2 = self.forearm_length

        d = math.sqrt(radius * radius + z_rel * z_rel)

        max_reach = l1 + l2 - 0.005
        min_reach = abs(l1 - l2) + 0.005

        if d > max_reach:
            scale = max_reach / d
            radius *= scale
            z_rel *= scale
            d = max_reach
            self.get_logger().warn("Target outside max reach. Clamped inward.")

        if d < min_reach:
            d = min_reach
            self.get_logger().warn("Target inside minimum reach. Clamped outward.")

        # Angle from vertical direction.
        theta = math.atan2(radius, z_rel)

        cos_beta = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
        cos_beta = self.clamp(cos_beta, -1.0, 1.0)
        beta = math.acos(cos_beta)

        cos_elbow = (l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)
        cos_elbow = self.clamp(cos_elbow, -1.0, 1.0)

        elbow_internal = math.acos(cos_elbow)

        # Convert to your joint convention.
        shoulder = theta - beta
        elbow = math.pi - elbow_internal

        # Rough wrist compensation so gripper does not curl too much.
        wrist_pitch = -(shoulder + elbow)

        shoulder *= self.shoulder_sign
        elbow *= self.elbow_sign
        wrist_pitch *= self.wrist_sign

        # Apply URDF limits.
        base_yaw = self.clamp(base_yaw, -1.5708, 1.5708)
        shoulder = self.clamp(shoulder, -1.2217, 1.2217)
        elbow = self.clamp(elbow, -1.3963, 1.3963)
        wrist_pitch = self.clamp(wrist_pitch, -1.5708, 1.5708)
        wrist_roll = self.clamp(self.wrist_roll, -1.3963, 1.3963)

        return [
            base_yaw,
            shoulder,
            elbow,
            wrist_pitch,
            wrist_roll,
        ]

    def move_to_cube(self):
        cube = self.latest_cube

        if cube is None:
            self.is_moving = False
            return

        x = float(cube["x"]) + self.extra_x_offset
        y = float(cube["y"]) + self.extra_y_offset
        z = float(cube["z"]) + self.target_height_above_cube

        self.get_logger().info(
            f"🟢 Cube target for joint IK: x={x:.3f}, y={y:.3f}, z={z:.3f}"
        )

        target_joints = self.calculate_ik(x, y, z)

        self.get_logger().info(
            "Joint target: "
            f"base={target_joints[0]:+.3f}, "
            f"shoulder={target_joints[1]:+.3f}, "
            f"elbow={target_joints[2]:+.3f}, "
            f"wrist_pitch={target_joints[3]:+.3f}, "
            f"wrist_roll={target_joints[4]:+.3f}"
        )

        try:
            self.moveit2.move_to_configuration(target_joints)
            self.moveit2.wait_until_executed()

            self.get_logger().info("✅ Sent joint-space motion to cube area")
            self.has_moved = True

        except Exception as e:
            self.get_logger().error(f"Joint-space move failed: {e}")

        self.is_moving = False


def main():
    rclpy.init()

    node = GoToGreenCubeIKJoints()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()