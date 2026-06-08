#!/usr/bin/env python3

import json
import math
import socket
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


SAFE_RANGE = {
    "base": (0.0, 180.0),
    "shoulder": (0.0, 180.0),
    "elbow": (0.0, 180.0),
    "wrist": (0.0, 180.0),
    "wrist_rotation": (0.0, 180.0),
    "gripper": (70.0, 130.0),
}


class PiecewiseCalibrator:
    def __init__(self, cfg: Dict):
        self.mode = cfg.get("mode", "piecewise")
        self.default_offset = float(cfg.get("default_offset", 90.0))
        self.default_scale = float(cfg.get("default_scale", 1.0))

        self.points = [(float(a), float(b)) for a, b in cfg.get("points", [])]
        self.points.sort(key=lambda p: p[0])

    def convert(self, rviz_deg: float) -> float:
        if self.mode == "manual":
            return self.default_offset

        if len(self.points) == 0:
            return self.default_offset + self.default_scale * rviz_deg

        if len(self.points) == 1:
            return self.default_offset + self.default_scale * rviz_deg

        if rviz_deg <= self.points[0][0]:
            return self.interpolate(rviz_deg, self.points[0], self.points[1])

        if rviz_deg >= self.points[-1][0]:
            return self.interpolate(rviz_deg, self.points[-2], self.points[-1])

        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]
            if p0[0] <= rviz_deg <= p1[0]:
                return self.interpolate(rviz_deg, p0, p1)

        return self.default_offset + self.default_scale * rviz_deg

    def interpolate(self, x: float, p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        x0, y0 = p0
        x1, y1 = p1

        if abs(x1 - x0) < 1e-9:
            return y0

        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


class GoToPointGeometricIKUDP(Node):
    def __init__(self):
        super().__init__("go_to_point_geometric_ik_udp")

        self.declare_parameter("cube_topic", "/digital_twin/green_cube_json")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("controller_topic", "/arm_group_controller/joint_trajectory")

        self.declare_parameter("use_cube", True)

        self.declare_parameter("offset_x", 0.0)
        self.declare_parameter("offset_y", 0.0)
        self.declare_parameter("offset_z", 0.06)

        self.declare_parameter("target_x", 0.30)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 0.05)

        # Your measured geometry
        self.declare_parameter("shoulder_height", 0.1162)
        self.declare_parameter("upper_arm_length", 0.1041)
        self.declare_parameter("forearm_length", 0.1463)
        self.declare_parameter("wrist_length", 0.1779)

        # Tool/wrist search
        self.declare_parameter("phi_min_deg", -120.0)
        self.declare_parameter("phi_max_deg", 160.0)
        self.declare_parameter("phi_step_deg", 1.0)

        # Joint signs for your URDF convention
        self.declare_parameter("base_sign", 1.0)
        self.declare_parameter("shoulder_sign", -1.0)
        self.declare_parameter("elbow_sign", -1.0)
        self.declare_parameter("wrist_sign", 1.0)

        # RViz joint limits in degrees
        self.declare_parameter("base_min_deg", -90.0)
        self.declare_parameter("base_max_deg", 90.0)
        self.declare_parameter("shoulder_min_deg", -90.0)
        self.declare_parameter("shoulder_max_deg", 90.0)
        self.declare_parameter("elbow_min_deg", -90.0)
        self.declare_parameter("elbow_max_deg", 90.0)
        self.declare_parameter("wrist_min_deg", -90.0)
        self.declare_parameter("wrist_max_deg", 90.0)

        # Best-effort behavior
        self.declare_parameter("max_position_error", 0.025)
        self.declare_parameter("allow_best_effort", True)
        self.declare_parameter("prefer_wrist_usage", True)

        self.declare_parameter("sim_duration", 1.0)
        self.declare_parameter("real_duration", 10.0)

        self.declare_parameter("send_to_sim", True)
        self.declare_parameter("send_to_real", True)
        self.declare_parameter("dry_run", False)

        self.declare_parameter("pi_ip", "192.168.188.144")
        self.declare_parameter("pi_port", 9090)

        self.declare_parameter("calibration_file", "/workspace/arm_calibration_piecewise.json")
        self.declare_parameter("gripper", 80.0)
        self.declare_parameter("send_gripper", True)

        self.cube_topic = self.get_parameter("cube_topic").value
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.controller_topic = self.get_parameter("controller_topic").value

        self.use_cube = bool(self.get_parameter("use_cube").value)

        self.offset_x = float(self.get_parameter("offset_x").value)
        self.offset_y = float(self.get_parameter("offset_y").value)
        self.offset_z = float(self.get_parameter("offset_z").value)

        self.target_x = float(self.get_parameter("target_x").value)
        self.target_y = float(self.get_parameter("target_y").value)
        self.target_z = float(self.get_parameter("target_z").value)

        self.shoulder_height = float(self.get_parameter("shoulder_height").value)
        self.L1 = float(self.get_parameter("upper_arm_length").value)
        self.L2 = float(self.get_parameter("forearm_length").value)
        self.L3 = float(self.get_parameter("wrist_length").value)

        self.phi_min_deg = float(self.get_parameter("phi_min_deg").value)
        self.phi_max_deg = float(self.get_parameter("phi_max_deg").value)
        self.phi_step_deg = float(self.get_parameter("phi_step_deg").value)

        self.base_sign = float(self.get_parameter("base_sign").value)
        self.shoulder_sign = float(self.get_parameter("shoulder_sign").value)
        self.elbow_sign = float(self.get_parameter("elbow_sign").value)
        self.wrist_sign = float(self.get_parameter("wrist_sign").value)

        self.limits_deg = {
            "joint_6_base": (
                float(self.get_parameter("base_min_deg").value),
                float(self.get_parameter("base_max_deg").value),
            ),
            "joint_5_shoulder": (
                float(self.get_parameter("shoulder_min_deg").value),
                float(self.get_parameter("shoulder_max_deg").value),
            ),
            "joint_4_elbow": (
                float(self.get_parameter("elbow_min_deg").value),
                float(self.get_parameter("elbow_max_deg").value),
            ),
            "joint_3_wrist_pitch": (
                float(self.get_parameter("wrist_min_deg").value),
                float(self.get_parameter("wrist_max_deg").value),
            ),
        }

        self.max_position_error = float(self.get_parameter("max_position_error").value)
        self.allow_best_effort = bool(self.get_parameter("allow_best_effort").value)
        self.prefer_wrist_usage = bool(self.get_parameter("prefer_wrist_usage").value)

        self.sim_duration = float(self.get_parameter("sim_duration").value)
        self.real_duration = float(self.get_parameter("real_duration").value)

        self.send_to_sim = bool(self.get_parameter("send_to_sim").value)
        self.send_to_real = bool(self.get_parameter("send_to_real").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)

        self.pi_ip = self.get_parameter("pi_ip").value
        self.pi_port = int(self.get_parameter("pi_port").value)

        self.calibration_file = Path(self.get_parameter("calibration_file").value)
        self.gripper = float(self.get_parameter("gripper").value)
        self.send_gripper = bool(self.get_parameter("send_gripper").value)

        self.latest_cube: Optional[Dict] = None
        self.latest_joint_state: Optional[JointState] = None

        self.joint_order = [
            "joint_6_base",
            "joint_5_shoulder",
            "joint_4_elbow",
            "joint_3_wrist_pitch",
            "joint_2_wrist_roll",
        ]

        self.joint_to_servo = {
            "joint_6_base": "base",
            "joint_5_shoulder": "shoulder",
            "joint_4_elbow": "elbow",
            "joint_3_wrist_pitch": "wrist",
            "joint_2_wrist_roll": "wrist_rotation",
        }

        self.calibrators = self.load_calibration()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.create_subscription(String, self.cube_topic, self.cube_callback, 10)
        self.create_subscription(JointState, self.joint_state_topic, self.joint_state_callback, 10)

        self.traj_pub = self.create_publisher(JointTrajectory, self.controller_topic, 10)

        self.get_logger().info("✅ Improved geometric IK direct point → RViz + UDP started")
        self.get_logger().info("Wrist pitch is used as a main reach joint.")
        self.get_logger().info(f"use_cube={self.use_cube}")
        self.get_logger().info(f"offset xyz = {self.offset_x}, {self.offset_y}, {self.offset_z}")
        self.get_logger().info(f"max_position_error={self.max_position_error}, allow_best_effort={self.allow_best_effort}")
        self.get_logger().info(f"send_to_sim={self.send_to_sim}, send_to_real={self.send_to_real}, dry_run={self.dry_run}")

    def cube_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            if data.get("found", False):
                self.latest_cube = data
        except Exception:
            pass

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def load_calibration(self):
        calibrators = {}

        if not self.calibration_file.exists():
            self.get_logger().warn(f"Calibration file not found: {self.calibration_file}")
            return calibrators

        data = json.loads(self.calibration_file.read_text())

        for servo in ["base", "shoulder", "elbow", "wrist", "wrist_rotation", "gripper"]:
            calibrators[servo] = PiecewiseCalibrator(data.get(servo, {}))

        return calibrators

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def wait_for_inputs(self):
        start = time.time()

        while rclpy.ok() and time.time() - start < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_joint_state is None:
                continue

            if self.use_cube and self.latest_cube is None:
                continue

            return True

        return False

    def current_joint_deg(self):
        if self.latest_joint_state is None:
            return {}

        return {
            name: math.degrees(float(pos))
            for name, pos in zip(self.latest_joint_state.name, self.latest_joint_state.position)
        }

    def get_target(self):
        if self.use_cube:
            cube = self.latest_cube

            raw_x = float(cube["x"])
            raw_y = float(cube["y"])
            raw_z = float(cube["z"])

            x = raw_x + self.offset_x
            y = raw_y + self.offset_y
            z = raw_z + self.offset_z

            self.get_logger().info(
                f"Cube raw: x={raw_x:+.4f}, y={raw_y:+.4f}, z={raw_z:+.4f}"
            )
            self.get_logger().info(
                f"Target:   x={x:+.4f}, y={y:+.4f}, z={z:+.4f}"
            )

            return x, y, z

        self.get_logger().info(
            f"Target direct: x={self.target_x:+.4f}, y={self.target_y:+.4f}, z={self.target_z:+.4f}"
        )

        return self.target_x, self.target_y, self.target_z

    def within_limit(self, joint, deg):
        low, high = self.limits_deg[joint]
        return low <= deg <= high

    def fk_tool(self, joint_rad: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Forward kinematics for our simplified 5DOF reach model.
        Returns tool0 xyz in base_link.
        """

        q_base = joint_rad["joint_6_base"]
        q_shoulder = joint_rad["joint_5_shoulder"]
        q_elbow = joint_rad["joint_4_elbow"]
        q_wrist = joint_rad["joint_3_wrist_pitch"]

        # Undo signs to get internal link angles from vertical.
        a1 = q_shoulder / self.shoulder_sign
        a2 = a1 + (q_elbow / self.elbow_sign)
        a3 = a2 + (q_wrist / self.wrist_sign)

        # In vertical-plane model:
        # r grows with sin(angle from vertical)
        # z grows with cos(angle from vertical)
        r = (
            self.L1 * math.sin(a1)
            + self.L2 * math.sin(a2)
            + self.L3 * math.sin(a3)
        )

        z = (
            self.shoulder_height
            + self.L1 * math.cos(a1)
            + self.L2 * math.cos(a2)
            + self.L3 * math.cos(a3)
        )

        x = r * math.cos(q_base)
        y = r * math.sin(q_base)

        return x, y, z

    def position_error(self, reached, target):
        dx = reached[0] - target[0]
        dy = reached[1] - target[1]
        dz = reached[2] - target[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz), dx, dy, dz

    def desired_wrist_abs_for_radial(self, radial):
        if radial < 0.25:
            return 80.0
        if radial < 0.35:
            return 65.0
        if radial < 0.45:
            return 50.0
        return 35.0

    def solve_geometric_ik(self, x, y, z):
        target = (x, y, z)

        base_rad = self.base_sign * math.atan2(y, x)
        radial = math.sqrt(x * x + y * y)

        current = self.current_joint_deg()

        best_valid = None
        best_valid_score = 1e18

        best_any = None
        best_any_error = 1e18

        phi = self.phi_min_deg

        while phi <= self.phi_max_deg + 1e-9:
            phi_rad = math.radians(phi)

            # phi is the final tool angle from vertical.
            # Calculate wrist-center position by removing wrist/tool segment.
            wrist_center_r = radial - self.L3 * math.sin(phi_rad)
            wrist_center_z = (z - self.shoulder_height) - self.L3 * math.cos(phi_rad)

            D = (
                wrist_center_r * wrist_center_r
                + wrist_center_z * wrist_center_z
                - self.L1 * self.L1
                - self.L2 * self.L2
            ) / (2.0 * self.L1 * self.L2)

            if -1.0 <= D <= 1.0:
                for elbow_branch in [1.0, -1.0]:
                    q2_std = elbow_branch * math.acos(D)

                    theta1 = math.atan2(wrist_center_z, wrist_center_r) - math.atan2(
                        self.L2 * math.sin(q2_std),
                        self.L1 + self.L2 * math.cos(q2_std),
                    )

                    theta2 = theta1 + q2_std

                    # Convert standard plane angles to angles-from-vertical
                    a1 = math.pi / 2.0 - theta1
                    a2 = math.pi / 2.0 - theta2
                    a3 = phi_rad

                    q_base = base_rad
                    q_shoulder = self.shoulder_sign * a1
                    q_elbow = self.elbow_sign * (a2 - a1)
                    q_wrist = self.wrist_sign * (a3 - a2)
                    q_wrist_roll = 0.0

                    candidate = {
                        "joint_6_base": q_base,
                        "joint_5_shoulder": q_shoulder,
                        "joint_4_elbow": q_elbow,
                        "joint_3_wrist_pitch": q_wrist,
                        "joint_2_wrist_roll": q_wrist_roll,
                    }

                    degs = {j: math.degrees(v) for j, v in candidate.items()}

                    reached = self.fk_tool(candidate)
                    err, dx, dy, dz = self.position_error(reached, target)

                    if err < best_any_error:
                        best_any_error = err
                        best_any = {
                            "candidate": candidate,
                            "degs": degs,
                            "reached": reached,
                            "error": err,
                            "dx": dx,
                            "dy": dy,
                            "dz": dz,
                            "phi": phi,
                            "reason": "best_any",
                        }

                    in_limits = (
                        self.within_limit("joint_6_base", degs["joint_6_base"])
                        and self.within_limit("joint_5_shoulder", degs["joint_5_shoulder"])
                        and self.within_limit("joint_4_elbow", degs["joint_4_elbow"])
                        and self.within_limit("joint_3_wrist_pitch", degs["joint_3_wrist_pitch"])
                    )

                    if not in_limits:
                        continue

                    # Main scoring: smallest XYZ error first.
                    score = err * 1000.0

                    base_deg = degs["joint_6_base"]
                    shoulder_deg = degs["joint_5_shoulder"]
                    elbow_deg = degs["joint_4_elbow"]
                    wrist_deg = degs["joint_3_wrist_pitch"]

                    if self.prefer_wrist_usage:
                        desired_wrist_abs = self.desired_wrist_abs_for_radial(radial)

                        # Strongly prefer useful wrist pitch.
                        score += 0.60 * abs(abs(wrist_deg) - desired_wrist_abs)

                        # Do not punish wrist itself.
                        # Lightly avoid shoulder/elbow extremes.
                        score += 0.02 * abs(shoulder_deg)
                        score += 0.02 * abs(elbow_deg)
                        score += 0.005 * abs(base_deg)

                        # Weakly prefer current pose, but do not let it dominate.
                        for j in [
                            "joint_6_base",
                            "joint_5_shoulder",
                            "joint_4_elbow",
                            "joint_3_wrist_pitch",
                        ]:
                            if j in current:
                                score += 0.03 * abs(degs[j] - current[j])
                    else:
                        for j in [
                            "joint_6_base",
                            "joint_5_shoulder",
                            "joint_4_elbow",
                            "joint_3_wrist_pitch",
                        ]:
                            if j in current:
                                score += 0.10 * abs(degs[j] - current[j])
                            score += 0.02 * abs(degs[j])

                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid = {
                            "candidate": candidate,
                            "degs": degs,
                            "reached": reached,
                            "error": err,
                            "dx": dx,
                            "dy": dy,
                            "dz": dz,
                            "phi": phi,
                            "reason": "valid",
                            "score": score,
                        }

            phi += self.phi_step_deg

        if best_valid is None:
            return None, best_any

        if best_valid["error"] <= self.max_position_error:
            return best_valid, best_any

        if self.allow_best_effort:
            return best_valid, best_any

        return None, best_valid

    def send_sim_trajectory(self, joint_rad):
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_order)

        point = JointTrajectoryPoint()
        point.positions = [joint_rad[j] for j in self.joint_order]
        point.time_from_start = Duration(
            sec=int(self.sim_duration),
            nanosec=int((self.sim_duration - int(self.sim_duration)) * 1e9),
        )

        traj.points.append(point)

        if self.dry_run:
            self.get_logger().info("[DRY RUN] Would publish trajectory to RViz/sim.")
        else:
            self.traj_pub.publish(traj)
            self.get_logger().info("✅ Sent one-point trajectory to RViz/sim.")

    def convert_to_servo(self, joint_rad):
        out = {}

        for joint_name, servo_name in self.joint_to_servo.items():
            rviz_deg = math.degrees(joint_rad[joint_name])

            if servo_name in self.calibrators:
                servo_deg = self.calibrators[servo_name].convert(rviz_deg)
            else:
                servo_deg = 90.0 + rviz_deg

            low, high = SAFE_RANGE[servo_name]
            out[servo_name] = round(self.clamp(servo_deg, low, high), 2)

        if self.send_gripper:
            low, high = SAFE_RANGE["gripper"]
            out["gripper"] = round(self.clamp(self.gripper, low, high), 2)

        return out

    def send_udp(self, servo_angles):
        packet = {
            "duration": self.real_duration,
            "angles": servo_angles,
        }

        if self.dry_run:
            self.get_logger().info(f"[DRY RUN] Would send UDP: {packet}")
        else:
            self.sock.sendto(
                json.dumps(packet).encode("utf-8"),
                (self.pi_ip, self.pi_port),
            )
            self.get_logger().info(f"✅ Sent to real arm: {packet}")

    def print_solution(self, solution, target, servo_angles):
        joint_rad = solution["candidate"]
        degs = solution["degs"]
        reached = solution["reached"]

        self.get_logger().info("========== IK SOLUTION ==========")
        self.get_logger().info(
            f"Target xyz:  x={target[0]:+.4f}, y={target[1]:+.4f}, z={target[2]:+.4f}"
        )
        self.get_logger().info(
            f"Reached xyz: x={reached[0]:+.4f}, y={reached[1]:+.4f}, z={reached[2]:+.4f}"
        )
        self.get_logger().info(
            f"Error: {solution['error'] * 1000.0:.1f} mm "
            f"(dx={solution['dx']*1000.0:+.1f}, "
            f"dy={solution['dy']*1000.0:+.1f}, "
            f"dz={solution['dz']*1000.0:+.1f})"
        )
        self.get_logger().info(f"Chosen phi/tool angle: {solution['phi']:+.1f} deg")

        self.get_logger().info("RViz joint solution:")
        for joint in self.joint_order:
            self.get_logger().info(f"  {joint:22s}: {degs[joint]:+8.2f} deg")

        self.get_logger().info("Servo command:")
        for k, v in servo_angles.items():
            self.get_logger().info(f"  {k:15s}: {v:8.2f}")

    def print_failed_debug(self, debug, target):
        self.get_logger().error("Geometric IK failed.")
        self.get_logger().error(
            f"Target xyz: x={target[0]:+.4f}, y={target[1]:+.4f}, z={target[2]:+.4f}"
        )

        if debug is None:
            self.get_logger().error("No candidate found at all.")
            return

        self.get_logger().error("Closest candidate found:")
        self.get_logger().error(
            f"Reached xyz: x={debug['reached'][0]:+.4f}, "
            f"y={debug['reached'][1]:+.4f}, "
            f"z={debug['reached'][2]:+.4f}"
        )
        self.get_logger().error(
            f"Error: {debug['error'] * 1000.0:.1f} mm "
            f"(dx={debug['dx']*1000.0:+.1f}, "
            f"dy={debug['dy']*1000.0:+.1f}, "
            f"dz={debug['dz']*1000.0:+.1f})"
        )
        self.get_logger().error(f"phi={debug['phi']:+.1f} deg")
        for joint, deg in debug["degs"].items():
            self.get_logger().error(f"  {joint:22s}: {deg:+8.2f} deg")

    def run_once(self):
        if not self.wait_for_inputs():
            self.get_logger().error("Missing /joint_states or cube topic.")
            return

        x, y, z = self.get_target()
        target = (x, y, z)

        solution, debug = self.solve_geometric_ik(x, y, z)

        if solution is None:
            self.print_failed_debug(debug, target)
            return

        if solution["error"] > self.max_position_error:
            self.get_logger().warn(
                f"Best effort solution error is {solution['error']*1000.0:.1f} mm, "
                f"above max_position_error={self.max_position_error*1000.0:.1f} mm"
            )

        joint_rad = solution["candidate"]
        servo_angles = self.convert_to_servo(joint_rad)

        self.print_solution(solution, target, servo_angles)

        if self.send_to_sim:
            self.send_sim_trajectory(joint_rad)

        if self.send_to_real:
            self.send_udp(servo_angles)


def main(args=None):
    rclpy.init(args=args)
    node = GoToPointGeometricIKUDP()

    try:
        node.run_once()
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()