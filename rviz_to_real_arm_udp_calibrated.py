#!/usr/bin/env python3

import json
import math
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


SAFE_RANGE = {
    "base": (0.0, 180.0),
    "shoulder": (0.0, 180.0),
    "elbow": (0.0, 180.0),
    "wrist": (0.0, 180.0),
    "wrist_rotation": (0.0, 180.0),
    "gripper": (70.0, 130.0),
}


class PiecewiseCalibrator:
    def __init__(self, servo_name: str, cfg: Dict):
        self.servo_name = servo_name
        self.mode = cfg.get("mode", "piecewise")
        self.default_offset = float(cfg.get("default_offset", 90.0))
        self.default_scale = float(cfg.get("default_scale", 1.0))

        points = cfg.get("points", [])
        self.points: List[Tuple[float, float]] = [
            (float(x), float(y)) for x, y in points
        ]
        self.points.sort(key=lambda p: p[0])

        self.linear_scale = self.default_scale
        self.linear_offset = self.default_offset

        if len(self.points) >= 2:
            self.linear_scale, self.linear_offset = self.fit_line(self.points)

    def fit_line(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        n = len(points)

        mx = sum(xs) / n
        my = sum(ys) / n

        den = sum((x - mx) ** 2 for x in xs)
        if abs(den) < 1e-9:
            return self.default_scale, self.default_offset

        num = sum((x - mx) * (y - my) for x, y in points)
        scale = num / den
        offset = my - scale * mx
        return scale, offset

    def linear(self, rviz_deg: float) -> float:
        return self.linear_offset + self.linear_scale * rviz_deg

    def default_linear(self, rviz_deg: float) -> float:
        return self.default_offset + self.default_scale * rviz_deg

    def piecewise(self, rviz_deg: float) -> float:
        if len(self.points) == 0:
            return self.default_linear(rviz_deg)

        if len(self.points) == 1:
            x0, y0 = self.points[0]

            if abs(x0) < 1e-6:
                return y0

            scale = (y0 - self.default_offset) / x0
            return self.default_offset + scale * rviz_deg

        # Before first point: extrapolate using first segment.
        if rviz_deg <= self.points[0][0]:
            x0, y0 = self.points[0]
            x1, y1 = self.points[1]
            return self.interpolate(rviz_deg, x0, y0, x1, y1)

        # After last point: extrapolate using last segment.
        if rviz_deg >= self.points[-1][0]:
            x0, y0 = self.points[-2]
            x1, y1 = self.points[-1]
            return self.interpolate(rviz_deg, x0, y0, x1, y1)

        # Inside range: interpolate between surrounding points.
        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]

            if x0 <= rviz_deg <= x1:
                return self.interpolate(rviz_deg, x0, y0, x1, y1)

        return self.linear(rviz_deg)

    def interpolate(self, x: float, x0: float, y0: float, x1: float, y1: float) -> float:
        if abs(x1 - x0) < 1e-9:
            return y0

        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def convert(self, rviz_deg: float) -> float:
        if self.mode == "linear":
            return self.linear(rviz_deg)

        if self.mode == "piecewise":
            return self.piecewise(rviz_deg)

        if self.mode == "default":
            return self.default_linear(rviz_deg)

        return self.piecewise(rviz_deg)


class RVizToRealArmUDPCalibrated(Node):
    def __init__(self):
        super().__init__("rviz_to_real_arm_udp_calibrated")

        self.declare_parameter("pi_ip", "192.168.188.144")
        self.declare_parameter("pi_port", 9090)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("calibration_file", "/workspace/arm_calibration_piecewise.json")
        self.declare_parameter("duration", 10.0)
        self.declare_parameter("gripper", 80.0)
        self.declare_parameter("send_gripper", True)
        self.declare_parameter("dry_run", False)

        self.pi_ip = self.get_parameter("pi_ip").value
        self.pi_port = int(self.get_parameter("pi_port").value)
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.calibration_file = Path(self.get_parameter("calibration_file").value)
        self.duration = float(self.get_parameter("duration").value)
        self.gripper = float(self.get_parameter("gripper").value)
        self.send_gripper = bool(self.get_parameter("send_gripper").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)

        self.joint_to_servo = {
            "joint_6_base": "base",
            "joint_5_shoulder": "shoulder",
            "joint_4_elbow": "elbow",
            "joint_3_wrist_pitch": "wrist",
            "joint_2_wrist_roll": "wrist_rotation",
            "joint_1_gripper": "gripper",
        }

        self.latest_joint_state: Optional[JointState] = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.calibrators = self.load_calibration()

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            10,
        )

        self.get_logger().info("✅ Dynamic piecewise RViz → real arm UDP sender")
        self.get_logger().info(f"Calibration: {self.calibration_file}")
        self.get_logger().info(f"Pi: {self.pi_ip}:{self.pi_port}")
        self.get_logger().info(f"Dry run: {self.dry_run}")
        self.print_calibration_summary()

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def load_calibration(self) -> Dict[str, PiecewiseCalibrator]:
        if not self.calibration_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_file}")

        data = json.loads(self.calibration_file.read_text())

        calibrators = {}
        for servo in ["base", "shoulder", "elbow", "wrist", "wrist_rotation", "gripper"]:
            cfg = data.get(servo, {})
            calibrators[servo] = PiecewiseCalibrator(servo, cfg)

        return calibrators

    def print_calibration_summary(self):
        for servo, cal in self.calibrators.items():
            self.get_logger().info(
                f"{servo:15s} mode={cal.mode:9s} "
                f"points={len(cal.points)} "
                f"linear_scale={cal.linear_scale:+.4f} "
                f"linear_offset={cal.linear_offset:+.4f}"
            )

    def wait_for_joint_state(self, timeout: float = 5.0) -> bool:
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_joint_state is not None:
                return True
        return False

    def current_joint_degrees(self) -> Dict[str, float]:
        if self.latest_joint_state is None:
            return {}

        return {
            name: math.degrees(float(pos))
            for name, pos in zip(self.latest_joint_state.name, self.latest_joint_state.position)
        }

    def convert_current_pose(self) -> Dict[str, float]:
        joint_degs = self.current_joint_degrees()
        angles = {}

        for joint_name, servo_name in self.joint_to_servo.items():
            if joint_name not in joint_degs:
                continue

            if servo_name == "gripper":
                if not self.send_gripper:
                    continue
                servo_deg = self.gripper
            else:
                rviz_deg = joint_degs[joint_name]
                servo_deg = self.calibrators[servo_name].convert(rviz_deg)

            low, high = SAFE_RANGE[servo_name]
            servo_deg = self.clamp(servo_deg, low, high)
            angles[servo_name] = round(servo_deg, 2)

        return angles

    def print_table(self, angles: Dict[str, float]):
        joint_degs = self.current_joint_degrees()

        self.get_logger().info("Conversion table:")
        for joint_name, servo_name in self.joint_to_servo.items():
            if joint_name not in joint_degs:
                continue

            rviz_deg = joint_degs[joint_name]
            if servo_name not in angles:
                continue

            self.get_logger().info(
                f"  {joint_name:22s} {rviz_deg:+8.2f}°  ->  "
                f"{servo_name:15s} {angles[servo_name]:+8.2f}°"
            )

    def send_once(self):
        if not self.wait_for_joint_state():
            self.get_logger().error("No /joint_states received.")
            return

        angles = self.convert_current_pose()
        self.print_table(angles)

        packet = {
            "duration": self.duration,
            "angles": angles,
        }

        if self.dry_run:
            self.get_logger().info(f"[DRY RUN] Would send: {packet}")
        else:
            self.sock.sendto(json.dumps(packet).encode("utf-8"), (self.pi_ip, self.pi_port))
            self.get_logger().info(f"Sent to real arm: {packet}")


def main(args=None):
    rclpy.init(args=args)

    node = RVizToRealArmUDPCalibrated()

    try:
        node.send_once()
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()