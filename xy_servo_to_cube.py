#!/usr/bin/env python3

import json
import math
import time
from typing import Dict, Optional, List

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class XYServoToCube(Node):
    def __init__(self):
        super().__init__("xy_servo_to_cube")

        self.declare_parameter("error_json_topic", "/digital_twin/arm_cube_error_json")
        self.declare_parameter("controller_topic", "/arm_group_controller/joint_trajectory")

        self.declare_parameter("xy_tolerance", 0.002)  # 2 mm
        self.declare_parameter("max_iterations", 20)

        # Small joint step. Lower = safer/slower, higher = faster.
        self.declare_parameter("step", 0.015)

        self.declare_parameter("move_time", 0.8)
        self.declare_parameter("settle_time", 0.4)

        self.error_json_topic = self.get_parameter("error_json_topic").value
        self.controller_topic = self.get_parameter("controller_topic").value

        self.xy_tolerance = float(self.get_parameter("xy_tolerance").value)
        self.max_iterations = int(self.get_parameter("max_iterations").value)
        self.step = float(self.get_parameter("step").value)

        self.move_time = float(self.get_parameter("move_time").value)
        self.settle_time = float(self.get_parameter("settle_time").value)

        self.joint_names = [
            "joint_6_base",
            "joint_5_shoulder",
            "joint_4_elbow",
            "joint_3_wrist_pitch",
            "joint_2_wrist_roll",
        ]

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

        self.get_logger().info("✅ XY servo-to-cube node started")
        self.get_logger().info(f"Goal XY tolerance: {self.xy_tolerance:.4f} m")

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def error_callback(self, msg: String):
        try:
            self.latest_error = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Could not parse error JSON: {e}")

    def wait_for_data(self, timeout_sec=5.0) -> bool:
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_joint_state is not None and self.latest_error is not None:
                if self.latest_error.get("found", False):
                    return True

        return False

    def spin_update(self, seconds=0.5):
        start = time.time()

        while rclpy.ok() and time.time() - start < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)

    def get_current_joints(self) -> Optional[List[float]]:
        if self.latest_joint_state is None:
            return None

        values = {}

        for name, pos in zip(self.latest_joint_state.name, self.latest_joint_state.position):
            values[name] = float(pos)

        missing = [j for j in self.joint_names if j not in values]

        if missing:
            self.get_logger().error(f"Missing joints: {missing}")
            return None

        return [values[j] for j in self.joint_names]

    def get_error(self) -> Optional[Dict]:
        if self.latest_error is None:
            return None

        if not self.latest_error.get("found", False):
            return None

        err = self.latest_error["error"]

        return {
            "dx": float(err["dx_cube_minus_arm"]),
            "dy": float(err["dy_cube_minus_arm"]),
            "dz": float(err["dz_cube_minus_arm"]),
            "xy": float(err["distance_xy"]),
        }

    def send_joints(self, joints: List[float]):
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
        self.spin_update(0.4)

    def score_error(self, err: Dict) -> float:
        return err["xy"]

    def run_servo(self):
        if not self.wait_for_data():
            self.get_logger().error("No joint/error data.")
            return

        for iteration in range(self.max_iterations):
            self.spin_update(0.2)

            current_joints = self.get_current_joints()
            current_error = self.get_error()

            if current_joints is None or current_error is None:
                self.get_logger().error("Missing current joints/error.")
                return

            dx = current_error["dx"]
            dy = current_error["dy"]
            xy = current_error["xy"]

            self.get_logger().info(
                f"ITER {iteration}: dx={dx:+.4f}, dy={dy:+.4f}, xy={xy:.4f}"
            )

            if xy <= self.xy_tolerance:
                self.get_logger().info("✅ XY error is inside tolerance.")
                return

            s = self.step

            # Test small candidate moves.
            # Joint order:
            # 0 base
            # 1 shoulder
            # 2 elbow
            # 3 wrist pitch
            # 4 wrist roll
            candidates = [
                ("base +",      [+s, 0.0, 0.0, 0.0, 0.0]),
                ("base -",      [-s, 0.0, 0.0, 0.0, 0.0]),

                ("shoulder +",  [0.0, +s, 0.0, 0.0, 0.0]),
                ("shoulder -",  [0.0, -s, 0.0, 0.0, 0.0]),

                ("elbow +",     [0.0, 0.0, +s, 0.0, 0.0]),
                ("elbow -",     [0.0, 0.0, -s, 0.0, 0.0]),

                ("wrist +",     [0.0, 0.0, 0.0, +s, 0.0]),
                ("wrist -",     [0.0, 0.0, 0.0, -s, 0.0]),

                ("base+ shoulder-", [+s, -s, 0.0, 0.0, 0.0]),
                ("base- shoulder+", [-s, +s, 0.0, 0.0, 0.0]),

                ("shoulder+ elbow-", [0.0, +s, -s, 0.0, 0.0]),
                ("shoulder- elbow+", [0.0, -s, +s, 0.0, 0.0]),
            ]

            best_name = "none"
            best_joints = current_joints[:]
            best_error = current_error
            best_score = self.score_error(current_error)

            for name, delta in candidates:
                target = [a + b for a, b in zip(current_joints, delta)]

                self.send_joints(target)
                err = self.get_error()

                if err is None:
                    continue

                score = self.score_error(err)

                self.get_logger().info(
                    f"  test {name}: dx={err['dx']:+.4f}, dy={err['dy']:+.4f}, xy={err['xy']:.4f}"
                )

                # Return to original pose before testing next candidate.
                self.send_joints(current_joints)

                if score < best_score:
                    best_score = score
                    best_name = name
                    best_joints = target[:]
                    best_error = err

            if best_name == "none":
                self.get_logger().warn("No candidate improved XY. Try smaller step.")
                return

            self.get_logger().info(
                f"BEST {best_name}: dx={best_error['dx']:+.4f}, "
                f"dy={best_error['dy']:+.4f}, xy={best_error['xy']:.4f}"
            )

            self.send_joints(best_joints)

        self.get_logger().warn("Reached max iterations.")


def main():
    rclpy.init()

    node = XYServoToCube()

    try:
        node.run_servo()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()