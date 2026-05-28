#!/usr/bin/env python3

import json
import math
import time
from typing import Dict, Optional, List, Tuple

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GoToGreenCubeAuto(Node):
    def __init__(self):
        super().__init__("go_to_green_cube_auto")

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------
        self.declare_parameter("green_cube_json_topic", "/digital_twin/green_cube_json")
        self.declare_parameter("error_json_topic", "/digital_twin/arm_cube_error_json")
        self.declare_parameter("controller_topic", "/arm_group_controller/joint_trajectory")

        # ---------------------------------------------------------
        # Main user parameter
        # ---------------------------------------------------------
        # tool0.z = cube.z + desired_z_above_cube
        self.declare_parameter("desired_z_above_cube", 0.04)

        # ---------------------------------------------------------
        # Coarse IK settings
        # ---------------------------------------------------------
        self.declare_parameter("coarse_height_above_cube", 0.12)

        # Your current tool0 offset is 0.08, wrist is 0.075:
        # 0.075 + 0.08 = 0.155
        self.declare_parameter("wrist_length", 0.155)

        self.declare_parameter("shoulder_height", 0.141)
        self.declare_parameter("upper_arm_length", 0.105)
        self.declare_parameter("forearm_length", 0.145)

        self.declare_parameter("extra_x_offset", 0.014)
        self.declare_parameter("extra_y_offset", 0.001)
        self.declare_parameter("extra_z_offset", 0.0)

        self.declare_parameter("base_sign", 1.0)
        self.declare_parameter("shoulder_sign", 1.0)
        self.declare_parameter("elbow_sign", 1.0)
        self.declare_parameter("wrist_sign", 1.0)
        self.declare_parameter("wrist_roll", 0.0)

        self.declare_parameter("tool_angle_min", -2.4)
        self.declare_parameter("tool_angle_max", 0.8)
        self.declare_parameter("tool_angle_steps", 160)

        self.declare_parameter("min_radius", 0.08)
        self.declare_parameter("max_radius", 0.42)
        self.declare_parameter("max_ik_error", 0.05)

        # ---------------------------------------------------------
        # Final correction settings
        # ---------------------------------------------------------
        self.declare_parameter("xy_tolerance", 0.004)
        self.declare_parameter("z_tolerance", 0.008)

        self.declare_parameter("max_iterations", 12)
        self.declare_parameter("initial_step", 0.035)
        self.declare_parameter("min_step", 0.004)

        self.declare_parameter("move_time", 0.65)
        self.declare_parameter("settle_time", 0.25)

        self.declare_parameter("xy_weight", 1.0)
        self.declare_parameter("z_weight", 1.0)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.green_cube_json_topic = self.get_parameter("green_cube_json_topic").value
        self.error_json_topic = self.get_parameter("error_json_topic").value
        self.controller_topic = self.get_parameter("controller_topic").value

        self.desired_z_above_cube = float(self.get_parameter("desired_z_above_cube").value)
        self.coarse_height_above_cube = float(
            self.get_parameter("coarse_height_above_cube").value
        )

        self.wrist_length = float(self.get_parameter("wrist_length").value)
        self.shoulder_height = float(self.get_parameter("shoulder_height").value)
        self.upper_arm_length = float(self.get_parameter("upper_arm_length").value)
        self.forearm_length = float(self.get_parameter("forearm_length").value)

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

        self.xy_tolerance = float(self.get_parameter("xy_tolerance").value)
        self.z_tolerance = float(self.get_parameter("z_tolerance").value)

        self.max_iterations = int(self.get_parameter("max_iterations").value)
        self.initial_step = float(self.get_parameter("initial_step").value)
        self.min_step = float(self.get_parameter("min_step").value)

        self.move_time = float(self.get_parameter("move_time").value)
        self.settle_time = float(self.get_parameter("settle_time").value)

        self.xy_weight = float(self.get_parameter("xy_weight").value)
        self.z_weight = float(self.get_parameter("z_weight").value)

        # ---------------------------------------------------------
        # Joint setup
        # ---------------------------------------------------------
        self.joint_names = [
            "joint_6_base",
            "joint_5_shoulder",
            "joint_4_elbow",
            "joint_3_wrist_pitch",
            "joint_2_wrist_roll",
        ]

        self.joint_limits = {
            "joint_6_base": (-1.5708, 1.5708),
            "joint_5_shoulder": (-1.2217, 1.2217),
            "joint_4_elbow": (-1.3963, 1.3963),
            "joint_3_wrist_pitch": (-1.5708, 1.5708),
            "joint_2_wrist_roll": (-1.3963, 1.3963),
        }

        # ---------------------------------------------------------
        # ROS state
        # ---------------------------------------------------------
        self.latest_cube: Optional[Dict] = None
        self.latest_error: Optional[Dict] = None
        self.latest_joint_state: Optional[JointState] = None

        self.create_subscription(
            String,
            self.green_cube_json_topic,
            self.cube_callback,
            10,
        )

        self.create_subscription(
            String,
            self.error_json_topic,
            self.error_callback,
            10,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10,
        )

        self.get_logger().info("✅ Green cube AUTO mover started")
        self.get_logger().info(f"Goal: tool0.z = cube.z + {self.desired_z_above_cube:.3f} m")
        self.get_logger().info(f"Controller topic: {self.controller_topic}")

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    def cube_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            if data.get("found", False):
                self.latest_cube = data
            else:
                self.latest_cube = None

        except Exception as e:
            self.get_logger().warn(f"Could not parse cube JSON: {e}")

    def error_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            if data.get("found", False):
                self.latest_error = data
            else:
                self.latest_error = None

        except Exception as e:
            self.get_logger().warn(f"Could not parse error JSON: {e}")

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def spin_for(self, seconds: float):
        start = time.time()

        while rclpy.ok() and time.time() - start < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_cube(self, timeout_sec: float = 8.0) -> bool:
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_cube is not None:
                return True

        return False

    def wait_for_error_and_joints(self, timeout_sec: float = 8.0) -> bool:
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_error is not None and self.latest_joint_state is not None:
                return True

        return False

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def clamp_joints(self, joints: List[float]) -> List[float]:
        result = []

        for name, value in zip(self.joint_names, joints):
            low, high = self.joint_limits[name]
            result.append(self.clamp(value, low, high))

        return result

    def get_current_joints(self) -> Optional[List[float]]:
        if self.latest_joint_state is None:
            return None

        values = {}

        for name, position in zip(self.latest_joint_state.name, self.latest_joint_state.position):
            values[name] = float(position)

        missing = [name for name in self.joint_names if name not in values]

        if missing:
            self.get_logger().error(f"Missing joints in /joint_states: {missing}")
            return None

        return [values[name] for name in self.joint_names]

    def publish_joints(self, joints: List[float]):
        joints = self.clamp_joints(joints)

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joints]
        point.velocities = [0.0] * len(joints)

        point.time_from_start.sec = int(self.move_time)
        point.time_from_start.nanosec = int((self.move_time % 1.0) * 1e9)

        msg.points.append(point)
        self.trajectory_pub.publish(msg)

        time.sleep(self.settle_time)
        self.spin_for(0.25)

    # ---------------------------------------------------------
    # IK
    # ---------------------------------------------------------

    def forward_kinematics_planar(
        self,
        shoulder: float,
        elbow: float,
        wrist_pitch: float,
    ) -> Tuple[float, float]:
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

        l1 = self.upper_arm_length
        l2 = self.forearm_length
        l3 = self.wrist_length

        z_rel = z - self.shoulder_height

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

        shoulder_raw = alpha - (math.pi / 2.0)

        cumulative_after_elbow = shoulder_raw + elbow_raw
        wrist_raw = cumulative_after_elbow - tool_angle

        shoulder_cmd = self.shoulder_sign * shoulder_raw
        elbow_cmd = self.elbow_sign * elbow_raw
        wrist_cmd = self.wrist_sign * wrist_raw

        return [shoulder_cmd, elbow_cmd, wrist_cmd]

    def in_joint_limits(self, joints: List[float]) -> bool:
        for name, value in zip(self.joint_names, joints):
            low, high = self.joint_limits[name]
            if value < low or value > high:
                return False

        return True

    def calculate_best_ik(self, target_x: float, target_y: float, target_z: float):
        radius = math.sqrt(target_x * target_x + target_y * target_y)
        radius = self.clamp(radius, self.min_radius, self.max_radius)

        base_yaw = self.base_sign * math.atan2(target_y, target_x)
        base_yaw = self.clamp(base_yaw, -1.5708, 1.5708)

        best_joints = None
        best_score = 999.0
        best_error = 999.0
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
                    radius,
                    target_z,
                    tool_angle,
                    elbow_solution_sign,
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

                score = true_error + 0.003 * abs(wrist_pitch)

                if score < best_score:
                    best_score = score
                    best_error = true_error
                    best_joints = joints
                    best_debug = (tool_angle, elbow_solution_sign, fk_r, fk_z)

        if best_joints is None:
            return None

        tool_angle, elbow_solution_sign, fk_r, fk_z = best_debug

        self.get_logger().info(
            f"IK: target_r={radius:.3f}, target_z={target_z:.3f}, "
            f"fk_r={fk_r:.3f}, fk_z={fk_z:.3f}, "
            f"err={best_error:.4f}, tool_angle={tool_angle:+.3f}, "
            f"elbow_solution={elbow_solution_sign:+.0f}"
        )

        if best_error > self.max_ik_error:
            self.get_logger().warn(
                f"IK error {best_error:.3f} > max_ik_error {self.max_ik_error:.3f}"
            )

        return best_joints

    # ---------------------------------------------------------
    # Error scoring for final servo
    # ---------------------------------------------------------

    def get_servo_error(self) -> Optional[Dict]:
        if self.latest_error is None:
            return None

        err = self.latest_error["error"]

        dx = float(err["dx_cube_minus_arm"])
        dy = float(err["dy_cube_minus_arm"])
        dz_cube_minus_arm = float(err["dz_cube_minus_arm"])

        desired_dz = -self.desired_z_above_cube
        z_error = dz_cube_minus_arm - desired_dz

        xy_error = math.sqrt(dx * dx + dy * dy)

        score = math.sqrt(dx * dx + dy * dy + z_error * z_error)

        return {
            "dx": dx,
            "dy": dy,
            "dz_cube_minus_arm": dz_cube_minus_arm,
            "desired_dz": desired_dz,
            "z_error": z_error,
            "xy_error": xy_error,
            "score": score,
        }

    def print_error(self, prefix: str, err: Dict):
        self.get_logger().info(
            f"{prefix}: "
            f"dx={err['dx']:+.4f}, "
            f"dy={err['dy']:+.4f}, "
            f"dz={err['dz_cube_minus_arm']:+.4f}, "
            f"target_dz={err['desired_dz']:+.4f}, "
            f"z_error={err['z_error']:+.4f}, "
            f"xy={err['xy_error']:.4f}, "
            f"score={err['score']:.4f}"
        )

    def build_servo_candidates(self, current_joints: List[float], step: float):
        s = abs(step)

        return [
            ("base +", [current_joints[0] + s, current_joints[1], current_joints[2], current_joints[3], current_joints[4]]),
            ("base -", [current_joints[0] - s, current_joints[1], current_joints[2], current_joints[3], current_joints[4]]),

            ("shoulder +", [current_joints[0], current_joints[1] + s, current_joints[2], current_joints[3], current_joints[4]]),
            ("shoulder -", [current_joints[0], current_joints[1] - s, current_joints[2], current_joints[3], current_joints[4]]),

            ("elbow +", [current_joints[0], current_joints[1], current_joints[2] + s, current_joints[3], current_joints[4]]),
            ("elbow -", [current_joints[0], current_joints[1], current_joints[2] - s, current_joints[3], current_joints[4]]),

            ("wrist +", [current_joints[0], current_joints[1], current_joints[2], current_joints[3] + s, current_joints[4]]),
            ("wrist -", [current_joints[0], current_joints[1], current_joints[2], current_joints[3] - s, current_joints[4]]),

            ("shoulder+ elbow-", [current_joints[0], current_joints[1] + s, current_joints[2] - s, current_joints[3], current_joints[4]]),
            ("shoulder- elbow+", [current_joints[0], current_joints[1] - s, current_joints[2] + s, current_joints[3], current_joints[4]]),

            ("elbow+ wrist-", [current_joints[0], current_joints[1], current_joints[2] + s, current_joints[3] - s, current_joints[4]]),
            ("elbow- wrist+", [current_joints[0], current_joints[1], current_joints[2] - s, current_joints[3] + s, current_joints[4]]),

            ("shoulder+ elbow- wrist-", [current_joints[0], current_joints[1] + s, current_joints[2] - s, current_joints[3] - s, current_joints[4]]),
            ("shoulder- elbow+ wrist+", [current_joints[0], current_joints[1] - s, current_joints[2] + s, current_joints[3] + s, current_joints[4]]),
        ]

    # ---------------------------------------------------------
    # Main sequence
    # ---------------------------------------------------------

    def coarse_move(self) -> bool:
        self.get_logger().info("Waiting for green cube...")

        if not self.wait_for_cube():
            self.get_logger().error("No green cube found.")
            return False

        cube = self.latest_cube

        cube_x = float(cube["x"])
        cube_y = float(cube["y"])
        cube_z = float(cube["z"])

        target_x = cube_x + self.extra_x_offset
        target_y = cube_y + self.extra_y_offset
        target_z = cube_z + self.coarse_height_above_cube + self.extra_z_offset

        self.get_logger().info(
            f"Cube: x={cube_x:.3f}, y={cube_y:.3f}, z={cube_z:.3f}"
        )
        self.get_logger().info(
            f"Coarse target: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}"
        )

        joints = self.calculate_best_ik(target_x, target_y, target_z)

        if joints is None:
            self.get_logger().error("Coarse IK failed.")
            return False

        joints = self.clamp_joints(joints)

        self.get_logger().info(
            "Coarse joints deg: "
            f"base={math.degrees(joints[0]):+.1f}, "
            f"shoulder={math.degrees(joints[1]):+.1f}, "
            f"elbow={math.degrees(joints[2]):+.1f}, "
            f"wrist={math.degrees(joints[3]):+.1f}, "
            f"roll={math.degrees(joints[4]):+.1f}"
        )

        self.publish_joints(joints)
        self.spin_for(1.0)

        return True

    def final_servo(self):
        self.get_logger().info("Starting final XYZ correction...")

        if not self.wait_for_error_and_joints():
            self.get_logger().error("No error/joint data for final correction.")
            return

        step = self.initial_step

        for iteration in range(self.max_iterations):
            self.spin_for(0.2)

            current_joints = self.get_current_joints()
            current_error = self.get_servo_error()

            if current_joints is None or current_error is None:
                self.get_logger().error("Missing current joints/error.")
                return

            self.print_error(f"ITER {iteration}", current_error)

            if (
                current_error["xy_error"] <= self.xy_tolerance
                and abs(current_error["z_error"]) <= self.z_tolerance
            ):
                self.get_logger().info("✅ Final target reached.")
                return

            candidates = self.build_servo_candidates(current_joints, step)

            best_name = "none"
            best_joints = current_joints[:]
            best_error = current_error
            best_score = current_error["score"]

            for name, candidate_joints in candidates:
                candidate_joints = self.clamp_joints(candidate_joints)

                self.publish_joints(candidate_joints)
                candidate_error = self.get_servo_error()

                self.publish_joints(current_joints)

                if candidate_error is None:
                    continue

                if candidate_error["score"] < best_score:
                    best_name = name
                    best_joints = candidate_joints[:]
                    best_error = candidate_error
                    best_score = candidate_error["score"]

            if best_name == "none":
                step *= 0.5
                self.get_logger().warn(f"No improvement. Reducing step to {step:.4f}")

                if step < self.min_step:
                    self.get_logger().warn("Step too small. Stopping.")
                    return

                continue

            self.get_logger().info(
                f"BEST {best_name}: "
                f"dx={best_error['dx']:+.4f}, "
                f"dy={best_error['dy']:+.4f}, "
                f"dz={best_error['dz_cube_minus_arm']:+.4f}, "
                f"z_error={best_error['z_error']:+.4f}, "
                f"xy={best_error['xy_error']:.4f}, "
                f"score={best_error['score']:.4f}"
            )

            self.publish_joints(best_joints)

        self.get_logger().warn("Reached max_iterations.")

    def run(self):
        ok = self.coarse_move()

        if not ok:
            return

        self.final_servo()

        final_error = self.get_servo_error()

        if final_error is not None:
            self.print_error("FINAL", final_error)


def main():
    rclpy.init()

    node = GoToGreenCubeAuto()

    try:
        node.run()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()