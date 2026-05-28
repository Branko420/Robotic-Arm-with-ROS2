#!/usr/bin/env python3

import json
import math
import time
from typing import Dict, Optional, List, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GoToCubeFinalServo(Node):
    def __init__(self):
        super().__init__("go_to_cube_final_servo")

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------
        self.declare_parameter("error_json_topic", "/digital_twin/arm_cube_error_json")
        self.declare_parameter("controller_topic", "/arm_group_controller/joint_trajectory")

        # ---------------------------------------------------------
        # Desired final pose
        # ---------------------------------------------------------
        # tool0 should be this much above the cube point.
        self.declare_parameter("desired_z_above_cube", 0.04)

        # Final XY and Z tolerance.
        self.declare_parameter("xy_tolerance", 0.003)
        self.declare_parameter("z_tolerance", 0.006)

        # Wrist roll orientation. This rotates gripper around its own axis.
        # It should not move tool0 position much.
        self.declare_parameter("use_wrist_roll_orientation", True)
        self.declare_parameter("desired_wrist_roll", 0.0)

        # Optional wrist pitch preference.
        # Keep this OFF unless you really need it, because wrist pitch changes XYZ.
        self.declare_parameter("use_wrist_pitch_preference", False)
        self.declare_parameter("desired_wrist_pitch", 0.0)
        self.declare_parameter("wrist_pitch_weight", 0.03)

        # ---------------------------------------------------------
        # Search behavior
        # ---------------------------------------------------------
        # Not 100 gazillion loops. Default: max 15 improvement iterations.
        self.declare_parameter("max_iterations", 15)

        # Joint step starts here and shrinks if no improvement.
        self.declare_parameter("initial_step", 0.035)
        self.declare_parameter("min_step", 0.003)

        # Movement timing.
        self.declare_parameter("move_time", 0.55)
        self.declare_parameter("settle_time", 0.20)

        # Weights for score.
        self.declare_parameter("xy_weight", 1.0)
        self.declare_parameter("z_weight", 1.0)

        # Safety.
        self.declare_parameter("max_xy_allowed", 0.08)
        self.declare_parameter("max_z_error_allowed", 0.15)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.error_json_topic = self.get_parameter("error_json_topic").value
        self.controller_topic = self.get_parameter("controller_topic").value

        self.desired_z_above_cube = float(self.get_parameter("desired_z_above_cube").value)
        self.xy_tolerance = float(self.get_parameter("xy_tolerance").value)
        self.z_tolerance = float(self.get_parameter("z_tolerance").value)

        self.use_wrist_roll_orientation = bool(
            self.get_parameter("use_wrist_roll_orientation").value
        )
        self.desired_wrist_roll = float(self.get_parameter("desired_wrist_roll").value)

        self.use_wrist_pitch_preference = bool(
            self.get_parameter("use_wrist_pitch_preference").value
        )
        self.desired_wrist_pitch = float(self.get_parameter("desired_wrist_pitch").value)
        self.wrist_pitch_weight = float(self.get_parameter("wrist_pitch_weight").value)

        self.max_iterations = int(self.get_parameter("max_iterations").value)
        self.initial_step = float(self.get_parameter("initial_step").value)
        self.min_step = float(self.get_parameter("min_step").value)

        self.move_time = float(self.get_parameter("move_time").value)
        self.settle_time = float(self.get_parameter("settle_time").value)

        self.xy_weight = float(self.get_parameter("xy_weight").value)
        self.z_weight = float(self.get_parameter("z_weight").value)

        self.max_xy_allowed = float(self.get_parameter("max_xy_allowed").value)
        self.max_z_error_allowed = float(self.get_parameter("max_z_error_allowed").value)

        # ---------------------------------------------------------
        # Arm joints
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
        # ROS
        # ---------------------------------------------------------
        self.latest_joint_state: Optional[JointState] = None
        self.latest_error: Optional[Dict] = None

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.create_subscription(
            String,
            self.error_json_topic,
            self.error_callback,
            10,
        )

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            self.controller_topic,
            10,
        )

        self.get_logger().info("✅ Final cube servo started")
        self.get_logger().info(f"Error topic: {self.error_json_topic}")
        self.get_logger().info(f"Controller topic: {self.controller_topic}")
        self.get_logger().info(
            f"Goal: XY≈0, tool0.z = cube.z + {self.desired_z_above_cube:.3f} m"
        )
        self.get_logger().info(
            f"Orientation: wrist_roll={self.desired_wrist_roll:+.3f}, "
            f"enabled={self.use_wrist_roll_orientation}"
        )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def error_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            if not data.get("found", False):
                self.latest_error = None
                return

            self.latest_error = data

        except Exception as e:
            self.get_logger().warn(f"Could not parse error JSON: {e}")

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def spin_for(self, seconds: float):
        start = time.time()

        while rclpy.ok() and time.time() - start < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_data(self, timeout_sec: float = 6.0) -> bool:
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_joint_state is not None and self.latest_error is not None:
                return True

        return False

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def clamp_joints(self, joints: List[float]) -> List[float]:
        output = []

        for name, value in zip(self.joint_names, joints):
            low, high = self.joint_limits[name]
            output.append(self.clamp(value, low, high))

        return output

    def get_current_joints(self) -> Optional[List[float]]:
        if self.latest_joint_state is None:
            return None

        values = {}

        for name, position in zip(self.latest_joint_state.name, self.latest_joint_state.position):
            values[name] = float(position)

        missing = [joint for joint in self.joint_names if joint not in values]

        if missing:
            self.get_logger().error(f"Missing joints from /joint_states: {missing}")
            return None

        return [values[joint] for joint in self.joint_names]

    def get_error(self, joints: Optional[List[float]] = None) -> Optional[Dict]:
        if self.latest_error is None:
            return None

        err = self.latest_error["error"]

        dx = float(err["dx_cube_minus_arm"])
        dy = float(err["dy_cube_minus_arm"])
        dz_cube_minus_arm = float(err["dz_cube_minus_arm"])

        # desired:
        # tool0.z = cube.z + desired_z_above_cube
        # therefore:
        # cube.z - tool0.z = -desired_z_above_cube
        desired_dz = -self.desired_z_above_cube
        z_error = dz_cube_minus_arm - desired_dz

        xy_error = math.sqrt(dx * dx + dy * dy)

        score = math.sqrt(
            self.xy_weight * (dx * dx + dy * dy)
            + self.z_weight * (z_error * z_error)
        )

        if joints is not None and self.use_wrist_pitch_preference:
            wrist_pitch = joints[3]
            pitch_error = wrist_pitch - self.desired_wrist_pitch
            score += self.wrist_pitch_weight * abs(pitch_error)

        return {
            "dx": dx,
            "dy": dy,
            "dz_cube_minus_arm": dz_cube_minus_arm,
            "desired_dz": desired_dz,
            "z_error": z_error,
            "xy_error": xy_error,
            "score": score,
        }

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

    def is_goal_reached(self, err: Dict) -> bool:
        return (
            err["xy_error"] <= self.xy_tolerance
            and abs(err["z_error"]) <= self.z_tolerance
        )

    def build_candidates(self, current_joints: List[float], step: float) -> List[Tuple[str, List[float]]]:
        s = abs(step)
        candidates: List[Tuple[str, List[float]]] = []

        # Joint order:
        # 0 base
        # 1 shoulder
        # 2 elbow
        # 3 wrist_pitch
        # 4 wrist_roll

        base_moves = [
            ("base +", [+s, 0.0, 0.0, 0.0, 0.0]),
            ("base -", [-s, 0.0, 0.0, 0.0, 0.0]),

            ("shoulder +", [0.0, +s, 0.0, 0.0, 0.0]),
            ("shoulder -", [0.0, -s, 0.0, 0.0, 0.0]),

            ("elbow +", [0.0, 0.0, +s, 0.0, 0.0]),
            ("elbow -", [0.0, 0.0, -s, 0.0, 0.0]),

            ("wrist_pitch +", [0.0, 0.0, 0.0, +s, 0.0]),
            ("wrist_pitch -", [0.0, 0.0, 0.0, -s, 0.0]),

            ("shoulder+ elbow-", [0.0, +s, -s, 0.0, 0.0]),
            ("shoulder- elbow+", [0.0, -s, +s, 0.0, 0.0]),

            ("elbow+ wrist-", [0.0, 0.0, +s, -s, 0.0]),
            ("elbow- wrist+", [0.0, 0.0, -s, +s, 0.0]),

            ("shoulder+ elbow- wrist-", [0.0, +s, -s, -s, 0.0]),
            ("shoulder- elbow+ wrist+", [0.0, -s, +s, +s, 0.0]),

            ("shoulder+ elbow+ wrist-", [0.0, +s, +s, -s, 0.0]),
            ("shoulder- elbow- wrist+", [0.0, -s, -s, +s, 0.0]),

            ("base+ shoulder-", [+s, -s, 0.0, 0.0, 0.0]),
            ("base- shoulder+", [-s, +s, 0.0, 0.0, 0.0]),
        ]

        for name, delta in base_moves:
            target = [a + b for a, b in zip(current_joints, delta)]

            if self.use_wrist_roll_orientation:
                target[4] = self.desired_wrist_roll

            candidates.append((name, self.clamp_joints(target)))

        # Also test only setting wrist roll orientation.
        if self.use_wrist_roll_orientation:
            target = current_joints[:]
            target[4] = self.desired_wrist_roll
            candidates.append(("set wrist_roll", self.clamp_joints(target)))

        return candidates

    # ---------------------------------------------------------
    # Main algorithm
    # ---------------------------------------------------------

    def run(self):
        if not self.wait_for_data():
            self.get_logger().error("No /joint_states or arm_cube_error_json data.")
            return

        step = self.initial_step

        for iteration in range(self.max_iterations):
            self.spin_for(0.2)

            current_joints = self.get_current_joints()
            current_error = self.get_error(current_joints)

            if current_joints is None or current_error is None:
                self.get_logger().error("Missing current joints/error.")
                return

            self.print_error(f"ITER {iteration}", current_error)

            if self.is_goal_reached(current_error):
                self.get_logger().info("✅ Final XYZ position reached.")
                return

            if current_error["xy_error"] > self.max_xy_allowed:
                self.get_logger().warn(
                    f"XY error {current_error['xy_error']:.3f} is large. "
                    "The script will still try, but start closer if it fails."
                )

            if abs(current_error["z_error"]) > self.max_z_error_allowed:
                self.get_logger().warn(
                    f"Z error {current_error['z_error']:.3f} is large. "
                    "The script will still try, but start closer if it fails."
                )

            candidates = self.build_candidates(current_joints, step)

            best_name = "none"
            best_joints = current_joints[:]
            best_error = current_error
            best_score = current_error["score"]

            for name, candidate_joints in candidates:
                self.publish_joints(candidate_joints)

                candidate_error = self.get_error(candidate_joints)

                if candidate_error is None:
                    self.publish_joints(current_joints)
                    continue

                # Return to current pose before trying another candidate.
                self.publish_joints(current_joints)

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

        final_joints = self.get_current_joints()
        final_error = self.get_error(final_joints)

        if final_error:
            self.print_error("FINAL", final_error)

        self.get_logger().warn("Reached max_iterations.")


def main():
    rclpy.init()

    node = GoToCubeFinalServo()

    try:
        node.run()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()