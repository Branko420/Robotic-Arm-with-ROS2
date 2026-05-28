#!/usr/bin/env python3

import json
import time
from typing import Dict, Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from std_msgs.msg import String

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class NudgeTool0Down(Node):
    def __init__(self):
        super().__init__("nudge_tool0_down")

        self.declare_parameter("error_json_topic", "/digital_twin/arm_cube_error_json")
        self.declare_parameter(
            "controller_action",
            "/arm_group_controller/follow_joint_trajectory",
        )

        self.declare_parameter("desired_dz", -0.04)
        self.declare_parameter("max_xy_error", 0.035)
        self.declare_parameter("step", 0.04)
        self.declare_parameter("move_time", 1.2)
        self.declare_parameter("settle_time", 0.7)

        self.error_json_topic = self.get_parameter("error_json_topic").value
        self.controller_action = self.get_parameter("controller_action").value

        self.desired_dz = float(self.get_parameter("desired_dz").value)
        self.max_xy_error = float(self.get_parameter("max_xy_error").value)
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

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.controller_action,
        )

        self.get_logger().info("✅ Tool0 nudge-down tester started")
        self.get_logger().info(f"Controller action: {self.controller_action}")
        self.get_logger().info(f"Target desired dz={self.desired_dz:.3f} m")
        self.get_logger().info("Run this AFTER the arm is already at the good pre-grasp pose.")

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def error_callback(self, msg: String):
        try:
            self.latest_error = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Could not parse error JSON: {e}")

    def wait_for_data(self, timeout_sec: float = 5.0) -> bool:
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_joint_state is not None and self.latest_error is not None:
                if self.latest_error.get("found", False):
                    return True

        return False

    def wait_for_error_update(self, timeout_sec: float = 2.0):
        start = time.time()

        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

    def get_current_arm_joints(self) -> Optional[List[float]]:
        js = self.latest_joint_state

        if js is None:
            return None

        values = {}

        for name, position in zip(js.name, js.position):
            values[name] = float(position)

        missing = [name for name in self.joint_names if name not in values]

        if missing:
            self.get_logger().error(f"Missing joints in /joint_states: {missing}")
            return None

        return [values[name] for name in self.joint_names]

    def get_error_score(self) -> Optional[Dict]:
        if self.latest_error is None:
            return None

        if not self.latest_error.get("found", False):
            return None

        err = self.latest_error["error"]

        dx = float(err["dx_cube_minus_arm"])
        dy = float(err["dy_cube_minus_arm"])
        dz = float(err["dz_cube_minus_arm"])
        xy = float(err["distance_xy"])

        # Lower score is better.
        # desired_dz=-0.04 means tool0 should be about 4 cm above cube.
        score = abs(dz - self.desired_dz) + 2.0 * xy

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "xy": xy,
            "score": score,
        }

    def send_joint_goal(self, joints: List[float]) -> bool:
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f"Controller action not available: {self.controller_action}"
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = joints
        point.velocities = [0.0] * len(joints)
        point.time_from_start.sec = int(self.move_time)
        point.time_from_start.nanosec = int((self.move_time % 1.0) * 1e9)

        goal.trajectory.points.append(point)

        send_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error("Controller goal returned None")
            return False

        if not goal_handle.accepted:
            self.get_logger().error("Controller goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result is None:
            self.get_logger().error("Controller result returned None")
            return False

        time.sleep(self.settle_time)
        self.wait_for_error_update()

        return True

    def print_error(self, label: str, err: Dict):
        self.get_logger().info(
            f"{label}: dx={err['dx']:+.3f}, "
            f"dy={err['dy']:+.3f}, dz={err['dz']:+.3f}, "
            f"xy={err['xy']:.3f}, score={err['score']:.3f}"
        )

    def run_search(self):
        if not self.wait_for_data():
            self.get_logger().error("Did not receive /joint_states and error JSON.")
            return

        base_joints = self.get_current_arm_joints()
        start_error = self.get_error_score()

        if base_joints is None or start_error is None:
            self.get_logger().error("Could not read starting joints/error.")
            return

        self.print_error("START", start_error)

        s = self.step

        # Joint order:
        # 0 base
        # 1 shoulder
        # 2 elbow
        # 3 wrist_pitch
        # 4 wrist_roll
        #
        # We avoid changing base and wrist_roll here.
        candidates: List[Tuple[str, List[float]]] = [
            ("shoulder +", [0.0, +s, 0.0, 0.0, 0.0]),
            ("shoulder -", [0.0, -s, 0.0, 0.0, 0.0]),

            ("elbow +", [0.0, 0.0, +s, 0.0, 0.0]),
            ("elbow -", [0.0, 0.0, -s, 0.0, 0.0]),

            ("wrist +", [0.0, 0.0, 0.0, +s, 0.0]),
            ("wrist -", [0.0, 0.0, 0.0, -s, 0.0]),

            ("shoulder+ elbow-", [0.0, +s, -s, 0.0, 0.0]),
            ("shoulder- elbow+", [0.0, -s, +s, 0.0, 0.0]),

            ("elbow+ wrist-", [0.0, 0.0, +s, -s, 0.0]),
            ("elbow- wrist+", [0.0, 0.0, -s, +s, 0.0]),

            ("shoulder+ elbow- wrist-", [0.0, +s, -s, -s, 0.0]),
            ("shoulder- elbow+ wrist+", [0.0, -s, +s, +s, 0.0]),

            ("shoulder+ elbow+ wrist-", [0.0, +s, +s, -s, 0.0]),
            ("shoulder- elbow- wrist+", [0.0, -s, -s, +s, 0.0]),
        ]

        best_name = "start"
        best_joints = base_joints[:]
        best_error = start_error

        for name, delta in candidates:
            target = [a + b for a, b in zip(base_joints, delta)]

            self.get_logger().info(f"Testing nudge: {name}")

            ok = self.send_joint_goal(target)

            if not ok:
                self.get_logger().warn(f"Move failed for {name}")
                continue

            err = self.get_error_score()

            if err is None:
                self.get_logger().warn(f"No error data after {name}")
                continue

            self.print_error(name, err)

            # Go back to base pose before testing the next candidate.
            self.get_logger().info("Returning to start pose...")
            self.send_joint_goal(base_joints)

            if err["xy"] > self.max_xy_error:
                self.get_logger().info(
                    f"Reject {name}: xy={err['xy']:.3f} > {self.max_xy_error:.3f}"
                )
                continue

            if err["score"] < best_error["score"]:
                best_name = name
                best_joints = target[:]
                best_error = err

        self.get_logger().info(
            f"BEST: {best_name} | "
            f"dx={best_error['dx']:+.3f}, "
            f"dy={best_error['dy']:+.3f}, "
            f"dz={best_error['dz']:+.3f}, "
            f"xy={best_error['xy']:.3f}, "
            f"score={best_error['score']:.3f}"
        )

        self.get_logger().info("Moving to best nudge...")
        self.send_joint_goal(best_joints)

        final_error = self.get_error_score()

        if final_error is not None:
            self.print_error("FINAL", final_error)


def main():
    rclpy.init()

    node = NudgeTool0Down()

    try:
        node.run_search()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()