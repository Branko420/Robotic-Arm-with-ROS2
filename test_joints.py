#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from rclpy.executors import MultiThreadedExecutor
import threading


class TestJointMove(Node):
    def __init__(self):
        super().__init__("test_joint_move")

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
            base_link_name="base_link",
            end_effector_name="link_gripper_base",
            group_name="arm_group",
        )

        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3

        self.timer = self.create_timer(1.0, self.go_once)
        self.done = False

    def go_once(self):
        if self.done:
            return

        self.done = True
        thread = threading.Thread(target=self.move, daemon=True)
        thread.start()

    def move(self):
        self.get_logger().info("Testing joint-space motion...")

        target = [
            0.0,    # base
            0.45,   # shoulder
            -0.55,  # elbow
            0.35,   # wrist pitch
            0.0,    # wrist roll
        ]

        self.moveit2.move_to_configuration(target)
        self.moveit2.wait_until_executed()

        self.get_logger().info("Done joint-space test")


def main():
    rclpy.init()

    node = TestJointMove()
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