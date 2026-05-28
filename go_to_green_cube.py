#!/usr/bin/env python3

import json
import math
import threading
import time
from typing import Dict, Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GoToGreenCubeDirectIK(Node):
    def __init__(self):
        super().__init__("go_to_green_cube_ik_joints")

        # ---------------------------------------------------------
        # Input
        # ---------------------------------------------------------
        self.declare_parameter("green_cube_json_topic", "/digital_twin/green_cube_json")

        # Manual test mode
        self.declare_parameter("use_manual_target", False)
        self.declare_parameter("manual_x", 0.25)
        self.declare_parameter("manual_y", 0.0)
        self.declare_parameter("manual_z", 0.14)

        # ---------------------------------------------------------
        # Direct controller topic
        # ---------------------------------------------------------
        self.declare_parameter(
            "controller_topic",
            "/arm_group_controller/joint_trajectory",
        )
        self.declare_parameter("move_time", 1.5)

        # ---------------------------------------------------------
        # Robot geometry from your URDF
        # ---------------------------------------------------------
        # base_link 0.126 + shoulder link 0.015 = 0.141
        self.declare_parameter("shoulder_height", 0.141)

        # link_bicep length
        self.declare_parameter("upper_arm_length", 0.105)

        # link_forearm length
        self.declare_parameter("forearm_length", 0.145)

        # For tool0:
        # wrist link 0.075 + tool0 fixed offset 0.10 = 0.175
        self.declare_parameter("wrist_length", 0.175)

        # ---------------------------------------------------------
        # Target offsets
        # ---------------------------------------------------------
        # Stable pre-grasp defaults from your testing
        self.declare_parameter("target_height_above_cube", 0.12)
        self.declare_parameter("extra_x_offset", 0.02)
        self.declare_parameter("extra_y_offset", 0.0)
        self.declare_parameter("extra_z_offset", 0.0)

        # ---------------------------------------------------------
        # Joint signs
        # ---------------------------------------------------------
        self.declare_parameter("base_sign", 1.0)
        self.declare_parameter("shoulder_sign", 1.0)
        self.declare_parameter("elbow_sign", 1.0)
        self.declare_parameter("wrist_sign", 1.0)

        self.declare_parameter("wrist_roll", 0.0)

        # ---------------------------------------------------------
        # IK search settings
        # ---------------------------------------------------------
        self.declare_parameter("tool_angle_min", -2.4)
        self.declare_parameter("tool_angle_max", 0.8)
        self.declare_parameter("tool_angle_steps", 160)

        self.declare_parameter("min_radius", 0.08)
        self.declare_parameter("max_radius", 0.42)

        self.declare_parameter("max_ik_error", 0.05)
        self.declare_parameter("allow_approximate_ik", False)

        # ---------------------------------------------------------
        # Execution behavior
        # ---------------------------------------------------------
        self.declare_parameter("auto_go", True)
        self.declare_parameter("move_once", True)

        # Safety: reject IK if FK radial result goes behind the robot.
        self.declare_parameter("reject_opposite_side", True)

        # Wait after publishing trajectory so TF/debugger updates.
        self.declare_parameter("settle_time", 0.5)

        # ---------------------------------------------------------
        # Read parameters
        # ---------------------------------------------------------
        self.green_cube_json_topic = self.get_parameter("green_cube_json_topic").value

        self.use_manual_target = bool(self.get_parameter("use_manual_target").value)
        self.manual_x = float(self.get_parameter("manual_x").value)
        self.manual_y = float(self.get_parameter("manual_y").value)
        self.manual_z = float(self.get_parameter("manual_z").value)

        self.controller_topic = self.get_parameter("controller_topic").value
        self.move_time = float(self.get_parameter("move_time").value)

        self.shoulder_height = float(self.get_parameter("shoulder_height").value)
        self.upper_arm_length = float(self.get_parameter("upper_arm_length").value)
        self.forearm_length = float(self.get_parameter("forearm_length").value)
        self.wrist_length = float(self.get_parameter("wrist_length").value)

        self.target_height_above_cube = float(
            self.get_parameter("target_height_above_cube").value
        )
        self.extra_x_offset = float(self.get_parameter("extra_x_offset").value)
        self.extra_y_offset = float(self.get_parameter("extra_y_offset").value)
        self.extra_z_offset = float(self.get_parameter("extra_z_offset").value)

        self.base_sign = float(self.get_parameter("base_sign").value)
        self.shoulder_sign = float(self.get_parameter("shoulder_sign").value)
        self.elbow_sign = float(self.get_parameter("elbow_sign").value)
        self.wrist_sign = float(self.get_parameter("wrist_sign").value)
        self.wrist_roll = float(self.get_parameter("wrist_roll").value)

        self.tool_angle_min = float(self.get_parameter("tool_angle_min").value)
        self.tool_angle_max = float(self.get_parameter("tool_angle_max").value)
        self.tool_angle_steps = int(self.get_parameter("tool_angle_steps").value)

        self.min_radius = float(self.get_parameter("min_radius").value)
        self.max_radius = float(self.get_parameter("max_radius").value)

        self.max_ik_error = float(self.get_parameter("max_ik_error").value)
        self.allow_approximate_ik = bool(
            self.get_parameter("allow_approximate_ik").value
        )

        self.auto_go = bool(self.get_parameter("auto_go").value)
        self.move_once = bool(self.get_parameter("move_once").value)
        self.reject_opposite_side = bool(
            self.get_parameter("reject_opposite_side").value
        )
        self.settle_time = float(self.get_parameter("settle_time").value)

        # ---------------------------------------------------------
        # Joint order for arm_group_controller
        # ---------------------------------------------------------
        self.joint_names = [
            "joint_6_base",
            "joint_5_shoulder",
            "joint_4_elbow",
            "joint_3_wrist_pitch",
            "joint_2_wrist_roll",
        ]

        # ---------------------------------------------------------
        # ROS
        # ---------------------------------------------------------
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10,
        )

        self.sub = self.create_subscription(
            String,
            self.green_cube_json_topic,
            self.green_cube_callback,
            10,
        )

        self.timer = self.create_timer(0.5, self.timer_callback)

        self.latest_cube: Optional[Dict] = None
        self.is_moving = False
        self.has_moved = False

        self.get_logger().info("✅ Green cube Direct IK mover started")
        self.get_logger().info(f"Listening: {self.green_cube_json_topic}")
        self.get_logger().info(f"Publishing trajectory to: {self.controller_topic}")
        self.get_logger().info(
            f"Geometry: shoulder_height={self.shoulder_height:.3f}, "
            f"upper={self.upper_arm_length:.3f}, "
            f"forearm={self.forearm_length:.3f}, "
            f"wrist/tool={self.wrist_length:.3f}"
        )
        self.get_logger().info(
            f"Offsets: height={self.target_height_above_cube:.3f}, "
            f"x={self.extra_x_offset:.3f}, "
            f"y={self.extra_y_offset:.3f}, "
            f"z={self.extra_z_offset:.3f}"
        )
        self.get_logger().info(
            f"Signs: base={self.base_sign}, shoulder={self.shoulder_sign}, "
            f"elbow={self.elbow_sign}, wrist={self.wrist_sign}"
        )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

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

        if self.is_moving:
            return

        if self.move_once and self.has_moved:
            return

        if not self.use_manual_target and self.latest_cube is None:
            return

        self.is_moving = True
        thread = threading.Thread(target=self.move_to_target, daemon=True)
        thread.start()

    # ---------------------------------------------------------
    # Math helpers
    # ---------------------------------------------------------

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def in_joint_limits(self, joints: List[float]) -> bool:
        base, shoulder, elbow, wrist_pitch, wrist_roll = joints

        return (
            -1.5708 <= base <= 1.5708
            and -1.2217 <= shoulder <= 1.2217
            and -1.3963 <= elbow <= 1.3963
            and -1.5708 <= wrist_pitch <= 1.5708
            and -1.3963 <= wrist_roll <= 1.3963
        )

    def apply_joint_limits(self, joints: List[float]) -> List[float]:
        base, shoulder, elbow, wrist_pitch, wrist_roll = joints

        return [
            self.clamp(base, -1.5708, 1.5708),
            self.clamp(shoulder, -1.2217, 1.2217),
            self.clamp(elbow, -1.3963, 1.3963),
            self.clamp(wrist_pitch, -1.5708, 1.5708),
            self.clamp(wrist_roll, -1.3963, 1.3963),
        ]

    # ---------------------------------------------------------
    # FK and IK
    # ---------------------------------------------------------

    def forward_kinematics_planar(
        self,
        shoulder: float,
        elbow: float,
        wrist_pitch: float,
    ) -> Tuple[float, float]:
        """
        Returns radial distance and z height for tool0.

        Internal convention:
          segment vector = [-sin(c), cos(c)]

        c1 = shoulder
        c2 = shoulder + elbow
        c3 = shoulder + elbow - wrist_pitch
        """

        l1 = self.upper_arm_length
        l2 = self.forearm_length
        l3 = self.wrist_length

        q_s = shoulder / self.shoulder_sign
        q_e = elbow / self.elbow_sign
        q_w = wrist_pitch / self.wrist_sign

        c1 = q_s
        c2 = q_s + q_e
        c3 = q_s + q_e - q_w

        r = (
            l1 * (-math.sin(c1))
            + l2 * (-math.sin(c2))
            + l3 * (-math.sin(c3))
        )

        z = (
            self.shoulder_height
            + l1 * math.cos(c1)
            + l2 * math.cos(c2)
            + l3 * math.cos(c3)
        )

        return r, z

    def solve_planar_ik_for_tool_angle(
        self,
        radius: float,
        z: float,
        tool_angle: float,
        elbow_solution_sign: float,
    ) -> Optional[List[float]]:
        """
        Solve shoulder/elbow/wrist for desired final tool angle.
        """

        l1 = self.upper_arm_length
        l2 = self.forearm_length
        l3 = self.wrist_length

        z_rel = z - self.shoulder_height

        # Remove final tool/wrist segment from target.
        wrist_r = radius - l3 * (-math.sin(tool_angle))
        wrist_z = z_rel - l3 * math.cos(tool_angle)

        d2 = wrist_r * wrist_r + wrist_z * wrist_z
        d = math.sqrt(d2)

        if d < 1e-6:
            return None

        if d > (l1 + l2 + 1e-6):
            return None

        if d < (abs(l1 - l2) - 1e-6):
            return None

        cos_elbow = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_elbow = self.clamp(cos_elbow, -1.0, 1.0)

        elbow_raw = elbow_solution_sign * math.acos(cos_elbow)

        k1 = l1 + l2 * math.cos(elbow_raw)
        k2 = l2 * math.sin(elbow_raw)

        alpha = math.atan2(wrist_z, wrist_r) - math.atan2(k2, k1)

        # Convert standard 2D angle to internal convention.
        shoulder_raw = alpha - (math.pi / 2.0)

        cumulative_after_elbow = shoulder_raw + elbow_raw

        # c3 = shoulder_raw + elbow_raw - wrist_raw
        wrist_raw = cumulative_after_elbow - tool_angle

        shoulder_cmd = self.shoulder_sign * shoulder_raw
        elbow_cmd = self.elbow_sign * elbow_raw
        wrist_cmd = self.wrist_sign * wrist_raw

        return [shoulder_cmd, elbow_cmd, wrist_cmd]

    def calculate_best_ik(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
    ) -> Optional[Tuple[List[float], float]]:
        radius = math.sqrt(target_x * target_x + target_y * target_y)
        original_radius = radius

        radius = self.clamp(radius, self.min_radius, self.max_radius)

        base_yaw = self.base_sign * math.atan2(target_y, target_x)
        base_yaw = self.clamp(base_yaw, -1.5708, 1.5708)

        best_joints = None
        best_score = 999.0
        best_true_error = 999.0
        best_debug = None

        steps = max(1, self.tool_angle_steps)

        for i in range(steps):
            if steps == 1:
                tool_angle = self.tool_angle_min
            else:
                t = i / float(steps - 1)
                tool_angle = self.tool_angle_min + t * (
                    self.tool_angle_max - self.tool_angle_min
                )

            for elbow_solution_sign in [1.0, -1.0]:
                planar = self.solve_planar_ik_for_tool_angle(
                    radius=radius,
                    z=target_z,
                    tool_angle=tool_angle,
                    elbow_solution_sign=elbow_solution_sign,
                )

                if planar is None:
                    continue

                shoulder, elbow, wrist_pitch = planar

                joints = [
                    base_yaw,
                    shoulder,
                    elbow,
                    wrist_pitch,
                    self.wrist_roll,
                ]

                if not self.in_joint_limits(joints):
                    continue

                fk_r, fk_z = self.forward_kinematics_planar(
                    shoulder,
                    elbow,
                    wrist_pitch,
                )

                true_error = math.sqrt((fk_r - radius) ** 2 + (fk_z - target_z) ** 2)

                if self.reject_opposite_side:
                    if target_x > 0.0 and fk_r < 0.0:
                        continue

                # Prefer exact FK match, but avoid very extreme wrist pitch a little.
                score = true_error + 0.003 * abs(wrist_pitch)

                if score < best_score:
                    best_score = score
                    best_true_error = true_error
                    best_joints = joints
                    best_debug = (
                        tool_angle,
                        elbow_solution_sign,
                        fk_r,
                        fk_z,
                    )

        if best_joints is None:
            if self.allow_approximate_ik:
                self.get_logger().warn("No exact IK. Using approximate fallback.")
                return self.calculate_approximate_ik(target_x, target_y, target_z)

            self.get_logger().error("No IK solution inside joint limits.")
            return None

        tool_angle, elbow_solution_sign, fk_r, fk_z = best_debug

        self.get_logger().info(
            f"IK best: requested_radius={original_radius:.3f}, "
            f"used_radius={radius:.3f}, target_z={target_z:.3f}, "
            f"fk_radius={fk_r:.3f}, fk_z={fk_z:.3f}, "
            f"error={best_true_error:.4f}, "
            f"tool_angle={tool_angle:+.3f}, "
            f"elbow_solution={elbow_solution_sign:+.0f}"
        )

        if best_true_error > self.max_ik_error and not self.allow_approximate_ik:
            self.get_logger().error(
                f"IK error {best_true_error:.3f}m > max_ik_error "
                f"{self.max_ik_error:.3f}m."
            )
            return None

        return best_joints, best_true_error

    def calculate_approximate_ik(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
    ) -> Optional[Tuple[List[float], float]]:
        radius = math.sqrt(target_x * target_x + target_y * target_y)
        radius = self.clamp(radius, self.min_radius, self.max_radius)

        base_yaw = self.base_sign * math.atan2(target_y, target_x)
        base_yaw = self.clamp(base_yaw, -1.5708, 1.5708)

        candidate_sets = [
            [base_yaw, 0.30, 0.80, -0.80, self.wrist_roll],
            [base_yaw, 0.45, 0.70, -0.90, self.wrist_roll],
            [base_yaw, 0.20, 1.00, -1.00, self.wrist_roll],
            [base_yaw, 0.60, 0.50, -0.90, self.wrist_roll],
        ]

        best = None
        best_error = 999.0

        for joints in candidate_sets:
            joints = self.apply_joint_limits(joints)

            fk_r, fk_z = self.forward_kinematics_planar(
                joints[1],
                joints[2],
                joints[3],
            )

            error = math.sqrt((fk_r - radius) ** 2 + (fk_z - target_z) ** 2)

            if error < best_error:
                best_error = error
                best = joints

        if best is None:
            return None

        self.get_logger().warn(f"Using approximate IK. error={best_error:.3f}m")
        return best, best_error

    # ---------------------------------------------------------
    # Target building
    # ---------------------------------------------------------

    def get_target(self) -> Optional[Tuple[float, float, float]]:
        if self.use_manual_target:
            self.get_logger().info(
                f"Using manual target: x={self.manual_x:.3f}, "
                f"y={self.manual_y:.3f}, z={self.manual_z:.3f}"
            )
            return self.manual_x, self.manual_y, self.manual_z

        if self.latest_cube is None:
            return None

        cube_x = float(self.latest_cube["x"])
        cube_y = float(self.latest_cube["y"])
        cube_z = float(self.latest_cube["z"])

        target_x = cube_x + self.extra_x_offset
        target_y = cube_y + self.extra_y_offset
        target_z = cube_z + self.target_height_above_cube + self.extra_z_offset

        self.get_logger().info(
            f"🟢 Cube JSON: x={cube_x:.3f}, y={cube_y:.3f}, z={cube_z:.3f}"
        )
        self.get_logger().info(
            f"🎯 Target tool0: x={target_x:.3f}, "
            f"y={target_y:.3f}, z={target_z:.3f}"
        )

        return target_x, target_y, target_z

    # ---------------------------------------------------------
    # Direct topic execution
    # ---------------------------------------------------------

    def send_joint_goal_direct(self, joints: List[float]) -> bool:
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joints]
        point.velocities = [0.0] * len(joints)

        point.time_from_start.sec = int(self.move_time)
        point.time_from_start.nanosec = int((self.move_time % 1.0) * 1e9)

        msg.points.append(point)

        self.trajectory_pub.publish(msg)

        self.get_logger().info("✅ Published joint trajectory to controller topic")

        return True

    # ---------------------------------------------------------
    # Main motion
    # ---------------------------------------------------------

    def move_to_target(self):
        target = self.get_target()

        if target is None:
            self.is_moving = False
            return

        target_x, target_y, target_z = target

        result = self.calculate_best_ik(target_x, target_y, target_z)

        if result is None:
            self.get_logger().error("No usable IK solution. Not moving.")
            self.is_moving = False
            return

        target_joints, ik_error = result

        self.get_logger().info(
            "Joint target rad: "
            f"base={target_joints[0]:+.3f}, "
            f"shoulder={target_joints[1]:+.3f}, "
            f"elbow={target_joints[2]:+.3f}, "
            f"wrist_pitch={target_joints[3]:+.3f}, "
            f"wrist_roll={target_joints[4]:+.3f}"
        )

        self.get_logger().info(
            "Joint target deg: "
            f"base={math.degrees(target_joints[0]):+.1f}, "
            f"shoulder={math.degrees(target_joints[1]):+.1f}, "
            f"elbow={math.degrees(target_joints[2]):+.1f}, "
            f"wrist_pitch={math.degrees(target_joints[3]):+.1f}, "
            f"wrist_roll={math.degrees(target_joints[4]):+.1f}, "
            f"ik_error={ik_error:.3f}m"
        )

        ok = self.send_joint_goal_direct(target_joints)

        if ok:
            self.has_moved = True
        else:
            self.get_logger().error("❌ Failed to publish joint trajectory")

        time.sleep(self.settle_time)

        self.is_moving = False


def main():
    rclpy.init()

    node = GoToGreenCubeDirectIK()

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