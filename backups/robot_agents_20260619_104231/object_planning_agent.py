#!/usr/bin/env python3

import json
import math
import os
import time
import re
from typing import Any, Dict, List, Optional, Tuple

import redis
import rclpy
from dotenv import load_dotenv
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray


load_dotenv("/workspace/.env")

def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


REDIS_URL = env("REDIS_URL", "redis://127.0.0.1:6379/0")
MOTION_POINTS_KEY = env("MOTION_POINTS_KEY", "workspace:motion_points")
PLANNING_REQUEST_STREAM = env("PLANNING_REQUEST_STREAM", "stream:planning_requests")
PLANNING_RESULT_STREAM = env("PLANNING_RESULT_STREAM", "stream:planning_results")
TARGET_OFFSETS_KEY = env("TARGET_OFFSETS_KEY", "workspace:target_offsets")
LAST_TARGET_KEY = env("LAST_TARGET_KEY", "workspace:last_planning_target")


class ObjectPlanningAgent(Node):
    def __init__(self) -> None:
        super().__init__("object_planning_agent")

        self.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.last_request_id = self.stream_tail_id(PLANNING_REQUEST_STREAM)

        self.joint_order = [
            "joint_6_base",
            "joint_5_shoulder",
            "joint_4_elbow",
            "joint_3_wrist_pitch",
            "joint_2_wrist_roll",
        ]

        self.limits_deg = {
            "joint_6_base": (
                env_float("PLANNER_BASE_MIN_DEG", -84.5),
                env_float("PLANNER_BASE_MAX_DEG", 86.5),
            ),
            "joint_5_shoulder": (
                env_float("PLANNER_SHOULDER_MIN_DEG", -90.0),
                env_float("PLANNER_SHOULDER_MAX_DEG", 90.0),
            ),
            "joint_4_elbow": (
                env_float("PLANNER_ELBOW_MIN_DEG", -83.9),
                env_float("PLANNER_ELBOW_MAX_DEG", 71.4),
            ),
            "joint_3_wrist_pitch": (
                env_float("PLANNER_WRIST_MIN_DEG", -85.0),
                env_float("PLANNER_WRIST_MAX_DEG", 80.0),
            ),
        }

        self.shoulder_height = env_float("PLANNER_SHOULDER_HEIGHT", 0.1162)
        self.L1 = env_float("PLANNER_UPPER_ARM_LENGTH", 0.1041)
        self.L2 = env_float("PLANNER_FOREARM_LENGTH", 0.1463)
        self.L3 = env_float("PLANNER_WRIST_LENGTH", 0.1779)

        self.base_sign = env_float("PLANNER_BASE_SIGN", 1.0)
        self.shoulder_sign = env_float("PLANNER_SHOULDER_SIGN", -1.0)
        self.elbow_sign = env_float("PLANNER_ELBOW_SIGN", -1.0)
        self.wrist_sign = env_float("PLANNER_WRIST_SIGN", 1.0)

        self.phi_min_deg = env_float("PLANNER_PHI_MIN_DEG", -120.0)
        self.phi_max_deg = env_float("PLANNER_PHI_MAX_DEG", 160.0)
        self.phi_step_deg = env_float("PLANNER_PHI_STEP_DEG", 2.0)
        self.max_position_error = env_float("PLANNER_MAX_POSITION_ERROR", 0.03)
        self.allow_best_effort = env_bool("PLANNER_ALLOW_BEST_EFFORT", True)

        self.move_duration = env_float("PLANNER_MOVE_DURATION_SEC", 0.8)
        self.preview_only_default = env_bool("PLANNER_PREVIEW_ONLY", False)
        self.send_to_real = env_bool("PLANNER_SEND_TO_REAL", False)
        self.place_clearance = env_float("PLANNER_PLACE_CLEARANCE", 0.012)
        self.place_approach_height = env_float("PLANNER_PLACE_APPROACH_HEIGHT", 0.10)
        self.sequence_pause = env_float("PLANNER_SEQUENCE_PAUSE_SEC", 0.15)

        self.home_deg = {
            "joint_6_base": env_float("POINT0_BASE_DEG", 0.0),
            "joint_5_shoulder": env_float("POINT0_SHOULDER_DEG", 0.0),
            "joint_4_elbow": env_float("POINT0_ELBOW_DEG", 0.0),
            "joint_3_wrist_pitch": env_float("POINT0_WRIST_DEG", 0.0),
            "joint_2_wrist_roll": env_float("POINT0_WRIST_ROLL_DEG", 0.0),
        }

        self.latest_joint_state: Optional[JointState] = None
        self.cancel_requested = False

        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 20)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/arm_group_controller/joint_trajectory",
            10,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/digital_twin/planner_markers",
            10,
        )

        self.create_timer(0.05, self.poll_requests)

        self.get_logger().info("✅ object_planning_agent ready")
        self.get_logger().info("Reads workspace:motion_points and controls RViz simulation")
        self.get_logger().info(f"Request stream: {PLANNING_REQUEST_STREAM}")
        self.get_logger().info(f"send_to_real={self.send_to_real}")

    def stream_tail_id(self, stream: str) -> str:
        try:
            rows = self.redis.xrevrange(stream, count=1)
            if rows:
                return decode(rows[0][0])
        except Exception:
            pass
        return "0-0"

    def joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def current_joint_deg(self) -> Dict[str, float]:
        if self.latest_joint_state is None:
            return {}
        return {
            name: math.degrees(float(position))
            for name, position in zip(
                self.latest_joint_state.name,
                self.latest_joint_state.position,
            )
        }

    def poll_requests(self) -> None:
        try:
            messages = self.redis.xread(
                {PLANNING_REQUEST_STREAM: self.last_request_id},
                count=10,
            )
        except Exception as exc:
            self.get_logger().error(f"Redis xread failed: {exc}")
            return

        for _stream, entries in messages:
            for entry_id, fields in entries:
                self.last_request_id = decode(entry_id)
                raw = fields.get("payload", "{}")
                try:
                    request = json.loads(raw)
                except Exception:
                    request = {}

                request_id = request.get("request_id", f"plan-{time.time_ns()}")
                self.get_logger().info(
                    f"📥 planning request: {request_id} action={request.get('action')} target={request.get('target')}"
                )

                try:
                    result = self.handle_request(request_id, request)
                except Exception as exc:
                    self.get_logger().exception(f"planning request crashed: {exc}")
                    result = {
                        "ok": False,
                        "request_id": request_id,
                        "state": "PLANNING_CRASHED",
                        "error": str(exc),
                        "timestamp": time.time(),
                    }

                self.redis.xadd(
                    PLANNING_RESULT_STREAM,
                    {"payload": json.dumps(result, ensure_ascii=False)},
                )

    def handle_request(self, request_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        action = str(request.get("action", "move")).strip().lower()

        if action == "stop":
            self.cancel_requested = True
            self.publish_hold_position()
            return self.result(request_id, True, "STOPPED")

        self.cancel_requested = False

        if action == "home":
            joints = {name: math.radians(value) for name, value in self.home_deg.items()}
            execute = bool(request.get("execute", True))
            self.publish_plan_markers(None, None, "point_0")
            if execute:
                self.publish_trajectory(joints, self.move_duration)
            return self.result(
                request_id,
                True,
                "HOME_SENT" if execute else "HOME_PREVIEWED",
                target="point_0",
                joint_degrees=self.home_deg,
            )

        if action == "correct":
            target = str(request.get("target") or self.redis.get(LAST_TARGET_KEY) or "nearest_object")
            offsets = self.update_offsets(
                target,
                float(request.get("dx", 0.0)),
                float(request.get("dy", 0.0)),
                float(request.get("dz", 0.0)),
            )
            return self.result(
                request_id,
                True,
                "OFFSET_UPDATED",
                target=target,
                offsets=offsets,
            )

        points_doc = self.load_motion_points()
        if points_doc is None:
            return self.result(request_id, False, "NO_WORKSPACE_POINTS")

        points = points_doc.get("motion_points", [])

        if action == "pick_and_place":
            source_name = str(request.get("source") or "nearest_object").strip()
            destination_name = str(request.get("destination") or "nearest_surface").strip()
            source = self.resolve_target(points, source_name)
            destination = self.resolve_target(points, destination_name)

            if source is None:
                return self.result(
                    request_id,
                    False,
                    "SOURCE_NOT_FOUND",
                    source=source_name,
                    available=[p.get("point") for p in points],
                )
            if destination is None:
                return self.result(
                    request_id,
                    False,
                    "DESTINATION_NOT_FOUND",
                    destination=destination_name,
                    available=[p.get("point") for p in points],
                )
            if source.get("safety_blocker") or destination.get("safety_blocker"):
                return self.result(
                    request_id,
                    False,
                    "SAFETY_BLOCKED",
                    source=source.get("point"),
                    destination=destination.get("point"),
                )

            execute = bool(request.get("execute", not self.preview_only_default))
            return self.execute_pick_and_place_sequence(
                request_id,
                source,
                destination,
                execute,
            )

        target_name = str(request.get("target", "nearest_object")).strip()
        point = self.resolve_target(points_doc.get("motion_points", []), target_name)
        if point is None:
            return self.result(
                request_id,
                False,
                "TARGET_NOT_FOUND",
                target=target_name,
                available=[p.get("point") for p in points_doc.get("motion_points", [])],
            )

        if point.get("safety_blocker"):
            return self.result(
                request_id,
                False,
                "SAFETY_BLOCKED",
                target=point.get("point"),
            )

        position_mode = str(
            request.get(
                "target_position",
                "approach" if action in {"approach", "pick"} else ("place" if action == "place" else "center"),
            )
        ).strip().lower()

        target_xyz = self.position_for_mode(point, position_mode)
        offsets = self.load_offsets(point.get("point", target_name))
        target_xyz = (
            target_xyz[0] + offsets["x"] + float(request.get("dx", 0.0)),
            target_xyz[1] + offsets["y"] + float(request.get("dy", 0.0)),
            target_xyz[2] + offsets["z"] + float(request.get("dz", 0.0)),
        )

        solution, debug = self.solve_geometric_ik(*target_xyz)
        if solution is None:
            self.publish_plan_markers(target_xyz, None, f"IK FAILED: {point.get('point')}")
            return self.result(
                request_id,
                False,
                "IK_FAILED",
                target=point.get("point"),
                target_xyz=self.xyz_dict(target_xyz),
                closest_error=None if debug is None else debug.get("error"),
            )

        execute = bool(request.get("execute", not self.preview_only_default))
        if action == "preview":
            execute = False

        self.redis.set(LAST_TARGET_KEY, point.get("point", target_name))
        self.publish_plan_markers(target_xyz, solution["reached"], point.get("point", target_name))

        if action == "pick":
            return self.execute_pick_sequence(request_id, point, execute)

        if execute:
            self.publish_trajectory(solution["candidate"], self.move_duration)
            if action == "place":
                time.sleep(self.move_duration + self.sequence_pause)
                self.publish_gripper(close=False)

        return self.result(
            request_id,
            True,
            "EXECUTED" if execute else "PREVIEWED",
            target=point.get("point"),
            target_position=position_mode,
            target_xyz=self.xyz_dict(target_xyz),
            reached_xyz=self.xyz_dict(solution["reached"]),
            position_error=solution["error"],
            joint_degrees=solution["degs"],
            offsets=offsets,
            executed=execute,
            simulation_only=not self.send_to_real,
        )

    def execute_pick_sequence(
        self,
        request_id: str,
        point: Dict[str, Any],
        execute: bool,
    ) -> Dict[str, Any]:
        sequence = []
        for mode in ["approach", "grasp", "approach"]:
            xyz = self.position_for_mode(point, mode)
            offsets = self.load_offsets(point.get("point", "nearest_object"))
            xyz = (xyz[0] + offsets["x"], xyz[1] + offsets["y"], xyz[2] + offsets["z"])
            solution, _ = self.solve_geometric_ik(*xyz)
            if solution is None:
                return self.result(
                    request_id,
                    False,
                    f"PICK_{mode.upper()}_IK_FAILED",
                    target=point.get("point"),
                )
            sequence.append((mode, xyz, solution))

        if execute:
            for mode, _xyz, solution in sequence:
                if self.cancel_requested:
                    return self.result(request_id, False, "PICK_CANCELLED")
                self.publish_trajectory(solution["candidate"], self.move_duration)
                time.sleep(self.move_duration + 0.15)
                if mode == "grasp":
                    self.publish_gripper(close=True)
                    time.sleep(0.5)

        final = sequence[-1][2]
        return self.result(
            request_id,
            True,
            "PICK_EXECUTED" if execute else "PICK_PREVIEWED",
            target=point.get("point"),
            sequence=[mode for mode, _, _ in sequence],
            position_error=final["error"],
            executed=execute,
            simulation_only=not self.send_to_real,
        )

    def execute_pick_and_place_sequence(
        self,
        request_id: str,
        source: Dict[str, Any],
        destination: Dict[str, Any],
        execute: bool,
    ) -> Dict[str, Any]:
        source_name = str(source.get("point") or source.get("label") or "source")
        destination_name = str(
            destination.get("point") or destination.get("label") or "destination"
        )

        source_offsets = self.load_offsets(source_name)
        destination_offsets = self.load_offsets(destination_name)

        source_approach = self.position_for_mode(source, "approach")
        source_grasp = self.position_for_mode(source, "grasp")
        source_retreat = source_approach

        destination_top = self.position_for_mode(destination, "top")
        source_height = float(
            (source.get("dimensions") or {}).get("z", source.get("height", 0.03))
            or 0.03
        )
        place_z = (
            destination_top[2]
            + max(source_height / 2.0, 0.012)
            + self.place_clearance
        )
        destination_place = (
            destination_top[0],
            destination_top[1],
            place_z,
        )
        destination_approach = (
            destination_place[0],
            destination_place[1],
            destination_place[2] + self.place_approach_height,
        )

        def apply_offsets(
            xyz: Tuple[float, float, float],
            offsets: Dict[str, float],
        ) -> Tuple[float, float, float]:
            return (
                xyz[0] + offsets["x"],
                xyz[1] + offsets["y"],
                xyz[2] + offsets["z"],
            )

        sequence_spec = [
            ("source_approach", apply_offsets(source_approach, source_offsets), None),
            ("source_grasp", apply_offsets(source_grasp, source_offsets), "close"),
            ("source_retreat", apply_offsets(source_retreat, source_offsets), None),
            (
                "destination_approach",
                apply_offsets(destination_approach, destination_offsets),
                None,
            ),
            (
                "destination_place",
                apply_offsets(destination_place, destination_offsets),
                "open",
            ),
            (
                "destination_retreat",
                apply_offsets(destination_approach, destination_offsets),
                None,
            ),
        ]

        solved = []
        for step_name, xyz, gripper_action in sequence_spec:
            solution, debug = self.solve_geometric_ik(*xyz)
            if solution is None:
                self.publish_plan_markers(
                    xyz,
                    None,
                    f"IK FAILED: {step_name}",
                )
                return self.result(
                    request_id,
                    False,
                    f"PICK_PLACE_{step_name.upper()}_IK_FAILED",
                    source=source_name,
                    destination=destination_name,
                    target_xyz=self.xyz_dict(xyz),
                    closest_error=None if debug is None else debug.get("error"),
                )
            solved.append((step_name, xyz, gripper_action, solution))

        self.redis.set(LAST_TARGET_KEY, destination_name)
        self.publish_pick_place_markers(
            source_approach,
            destination_approach,
            source_name,
            destination_name,
        )

        if execute:
            # Start with an open gripper in simulation.
            self.publish_gripper(close=False)
            time.sleep(0.35)

            for step_name, _xyz, gripper_action, solution in solved:
                if self.cancel_requested:
                    return self.result(
                        request_id,
                        False,
                        "PICK_PLACE_CANCELLED",
                        source=source_name,
                        destination=destination_name,
                    )

                self.publish_trajectory(solution["candidate"], self.move_duration)
                time.sleep(self.move_duration + self.sequence_pause)

                if gripper_action == "close":
                    self.publish_gripper(close=True)
                    time.sleep(0.45)
                elif gripper_action == "open":
                    self.publish_gripper(close=False)
                    time.sleep(0.45)

        return self.result(
            request_id,
            True,
            "PICK_PLACE_EXECUTED" if execute else "PICK_PLACE_PREVIEWED",
            source=source_name,
            destination=destination_name,
            source_geometry_quality=source.get("geometry_quality"),
            destination_geometry_quality=destination.get("geometry_quality"),
            sequence=[row[0] for row in solved],
            destination_place_xyz=self.xyz_dict(
                apply_offsets(destination_place, destination_offsets)
            ),
            executed=execute,
            simulation_only=not self.send_to_real,
        )

    def publish_pick_place_markers(
        self,
        source: Tuple[float, float, float],
        destination: Tuple[float, float, float],
        source_name: str,
        destination_name: str,
    ) -> None:
        arr = MarkerArray()
        delete = Marker()
        delete.header.frame_id = "base_link"
        delete.header.stamp = self.get_clock().now().to_msg()
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)

        line = Marker()
        line.header.frame_id = "base_link"
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = "pick_place_path"
        line.id = 10
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.012
        line.color.r = 0.2
        line.color.g = 1.0
        line.color.b = 0.3
        line.color.a = 1.0
        line.points = [
            Point(x=source[0], y=source[1], z=source[2]),
            Point(x=destination[0], y=destination[1], z=destination[2]),
        ]
        arr.markers.append(line)

        for marker_id, xyz, name, red, green, blue in (
            (11, source, source_name, 1.0, 0.3, 0.1),
            (12, destination, destination_name, 0.1, 1.0, 0.3),
        ):
            marker = Marker()
            marker.header = line.header
            marker.ns = "pick_place_targets"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = xyz[0]
            marker.pose.position.y = xyz[1]
            marker.pose.position.z = xyz[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.04
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = 1.0
            arr.markers.append(marker)

            text = Marker()
            text.header = line.header
            text.ns = "pick_place_text"
            text.id = marker_id + 10
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = xyz[0]
            text.pose.position.y = xyz[1]
            text.pose.position.z = xyz[2] + 0.06
            text.pose.orientation.w = 1.0
            text.scale.z = 0.03
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = name
            arr.markers.append(text)

        self.marker_pub.publish(arr)

    def publish_gripper(self, close: bool) -> None:
        publisher = getattr(self, "gripper_pub", None)
        if publisher is None:
            self.gripper_pub = self.create_publisher(
                JointTrajectory,
                "/gripper_group_controller/joint_trajectory",
                10,
            )
            publisher = self.gripper_pub

        traj = JointTrajectory()
        traj.joint_names = ["joint_1_gripper"]
        p = JointTrajectoryPoint()
        p.positions = [math.radians(35.0 if close else -10.0)]
        p.time_from_start = Duration(sec=0, nanosec=500_000_000)
        traj.points = [p]
        publisher.publish(traj)

    def load_motion_points(self) -> Optional[Dict[str, Any]]:
        raw = self.redis.get(MOTION_POINTS_KEY)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    @staticmethod
    def normalize_query(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^0-9a-zа-яё_\- ]+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[\s\-]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    def resolve_target(
        self,
        points: List[Dict[str, Any]],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        if not points:
            return None

        original_query = str(query or "").strip().lower()
        q = self.normalize_query(original_query)

        color_words = {
            "green": {"green", "зелен", "зеления", "зелено"},
            "blue": {"blue", "син", "синия", "синьо"},
            "yellow": {"yellow", "жълт", "жълтия", "жълто"},
            "red": {"red", "червен", "червения", "червено"},
            "white": {"white", "бял", "белия", "бяла", "бяло"},
            "black": {"black", "черен", "черния", "черна", "черно"},
        }
        requested_colors = {
            canonical
            for canonical, variants in color_words.items()
            if any(word in original_query for word in variants)
        }

        exact: List[Dict[str, Any]] = []
        fuzzy: List[Tuple[int, Dict[str, Any]]] = []

        for point in points:
            names = {
                self.normalize_query(point.get("point")),
                self.normalize_query(point.get("label")),
                self.normalize_query(point.get("safe_label")),
                self.normalize_query(point.get("display_name_bg")),
                self.normalize_query(point.get("role")),
                *[
                    self.normalize_query(alias)
                    for alias in point.get("aliases", [])
                ],
            }
            names.discard("")
            colors = {
                self.normalize_query(color)
                for color in point.get("colors", [])
            }

            if q in names:
                exact.append(point)
                continue

            score = 0
            for name in names:
                if name and (name in q or q in name):
                    score = max(score, min(len(name), len(q)))
                else:
                    q_tokens = set(q.split("_"))
                    name_tokens = set(name.split("_"))
                    score = max(score, len(q_tokens & name_tokens) * 3)

            if requested_colors:
                if colors & requested_colors:
                    score += 20
                else:
                    score -= 20

            if score > 0:
                fuzzy.append((score, point))

        candidates = exact
        if not candidates and fuzzy:
            best_score = max(score for score, _point in fuzzy)
            candidates = [point for score, point in fuzzy if score == best_score]

        if not candidates and q in {
            "nearest_object",
            "nearest",
            "object",
            "предмет",
            "обект",
        }:
            candidates = [p for p in points if not p.get("safety_blocker")]

        if not candidates and q in {
            "nearest_surface",
            "nearest_destination",
            "placement_surface",
            "повърхност",
            "място_за_поставяне",
        }:
            candidates = [
                p
                for p in points
                if p.get("role")
                in {"placement_surface", "reference_surface", "support_surface"}
            ]

        if requested_colors and candidates:
            color_filtered = [
                point
                for point in candidates
                if {
                    self.normalize_query(color)
                    for color in point.get("colors", [])
                }
                & requested_colors
            ]
            if color_filtered:
                candidates = color_filtered

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda point: math.hypot(
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
            ),
        )

    @staticmethod
    def position_for_mode(
        point: Dict[str, Any],
        mode: str,
    ) -> Tuple[float, float, float]:
        key_map = {
            "center": "center",
            "object_center": "center",
            "top": "top_position",
            "grasp": "grasp_position",
            "approach": "approach_position",
            "place": "place_position",
            "destination": "destination_position",
        }
        key = key_map.get(str(mode).strip().lower(), "center")
        position = point.get(key) or point.get("position") or point
        return (
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
        )

    def load_offsets(self, target: str) -> Dict[str, float]:
        try:
            all_offsets = json.loads(self.redis.get(TARGET_OFFSETS_KEY) or "{}")
        except Exception:
            all_offsets = {}
        row = all_offsets.get(target, {})
        return {
            "x": float(row.get("x", 0.0)),
            "y": float(row.get("y", 0.0)),
            "z": float(row.get("z", 0.0)),
        }

    def update_offsets(self, target: str, dx: float, dy: float, dz: float) -> Dict[str, float]:
        try:
            all_offsets = json.loads(self.redis.get(TARGET_OFFSETS_KEY) or "{}")
        except Exception:
            all_offsets = {}
        current = all_offsets.get(target, {"x": 0.0, "y": 0.0, "z": 0.0})
        current = {
            "x": float(current.get("x", 0.0)) + dx,
            "y": float(current.get("y", 0.0)) + dy,
            "z": float(current.get("z", 0.0)) + dz,
        }
        all_offsets[target] = current
        self.redis.set(TARGET_OFFSETS_KEY, json.dumps(all_offsets))
        return current

    def within_limit(self, joint: str, deg: float) -> bool:
        low, high = self.limits_deg[joint]
        return low <= deg <= high

    def fk_tool(self, joint_rad: Dict[str, float]) -> Tuple[float, float, float]:
        q_base = joint_rad["joint_6_base"]
        a1 = joint_rad["joint_5_shoulder"] / self.shoulder_sign
        a2 = a1 + joint_rad["joint_4_elbow"] / self.elbow_sign
        a3 = a2 + joint_rad["joint_3_wrist_pitch"] / self.wrist_sign

        radial = self.L1 * math.sin(a1) + self.L2 * math.sin(a2) + self.L3 * math.sin(a3)
        z = self.shoulder_height + self.L1 * math.cos(a1) + self.L2 * math.cos(a2) + self.L3 * math.cos(a3)
        return radial * math.cos(q_base), radial * math.sin(q_base), z

    def solve_geometric_ik(self, x: float, y: float, z: float):
        target = (x, y, z)
        base_rad = self.base_sign * math.atan2(y, x)
        radial = math.hypot(x, y)
        current = self.current_joint_deg()

        best_valid = None
        best_score = float("inf")
        best_any = None
        best_any_error = float("inf")

        phi = self.phi_min_deg
        while phi <= self.phi_max_deg + 1e-9:
            phi_rad = math.radians(phi)
            wr = radial - self.L3 * math.sin(phi_rad)
            wz = (z - self.shoulder_height) - self.L3 * math.cos(phi_rad)
            D = (wr * wr + wz * wz - self.L1 * self.L1 - self.L2 * self.L2) / (2 * self.L1 * self.L2)

            if -1.0 <= D <= 1.0:
                for branch in (1.0, -1.0):
                    q2 = branch * math.acos(D)
                    theta1 = math.atan2(wz, wr) - math.atan2(
                        self.L2 * math.sin(q2),
                        self.L1 + self.L2 * math.cos(q2),
                    )
                    theta2 = theta1 + q2
                    a1 = math.pi / 2.0 - theta1
                    a2 = math.pi / 2.0 - theta2
                    a3 = phi_rad

                    candidate = {
                        "joint_6_base": base_rad,
                        "joint_5_shoulder": self.shoulder_sign * a1,
                        "joint_4_elbow": self.elbow_sign * (a2 - a1),
                        "joint_3_wrist_pitch": self.wrist_sign * (a3 - a2),
                        "joint_2_wrist_roll": 0.0,
                    }
                    degs = {name: math.degrees(value) for name, value in candidate.items()}
                    reached = self.fk_tool(candidate)
                    dx, dy, dz = reached[0] - x, reached[1] - y, reached[2] - z
                    error = math.sqrt(dx * dx + dy * dy + dz * dz)
                    row = {
                        "candidate": candidate,
                        "degs": degs,
                        "reached": reached,
                        "error": error,
                        "phi": phi,
                    }

                    if error < best_any_error:
                        best_any_error = error
                        best_any = row

                    in_limits = all(
                        self.within_limit(name, degs[name])
                        for name in self.limits_deg
                    )
                    if not in_limits:
                        continue

                    score = error * 1000.0
                    for name in self.limits_deg:
                        if name in current:
                            score += 0.02 * abs(degs[name] - current[name])
                    score += 0.01 * abs(degs["joint_3_wrist_pitch"])

                    if score < best_score:
                        best_score = score
                        best_valid = row

            phi += self.phi_step_deg

        if best_valid is None:
            return None, best_any
        if best_valid["error"] <= self.max_position_error or self.allow_best_effort:
            return best_valid, best_any
        return None, best_valid

    def publish_trajectory(self, joints: Dict[str, float], duration: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_order)
        p = JointTrajectoryPoint()
        p.positions = [joints[name] for name in self.joint_order]
        sec = int(duration)
        p.time_from_start = Duration(sec=sec, nanosec=int((duration - sec) * 1e9))
        traj.points = [p]
        self.traj_pub.publish(traj)
        self.get_logger().info("✅ trajectory sent to RViz controller")

    def publish_hold_position(self) -> None:
        current = self.current_joint_deg()
        if not current:
            return
        joints = {
            name: math.radians(current.get(name, 0.0))
            for name in self.joint_order
        }
        self.publish_trajectory(joints, 0.2)

    def publish_plan_markers(
        self,
        target: Optional[Tuple[float, float, float]],
        reached: Optional[Tuple[float, float, float]],
        label: str,
    ) -> None:
        arr = MarkerArray()
        delete = Marker()
        delete.header.frame_id = "base_link"
        delete.header.stamp = self.get_clock().now().to_msg()
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)

        if target is not None:
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "planner_target"
            m.id = 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = target
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.035
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 1.0
            arr.markers.append(m)

            text = Marker()
            text.header = m.header
            text.ns = "planner_text"
            text.id = 2
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = target[0]
            text.pose.position.y = target[1]
            text.pose.position.z = target[2] + 0.06
            text.pose.orientation.w = 1.0
            text.scale.z = 0.035
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"PLAN: {label}\n({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
            arr.markers.append(text)

        if target is not None and reached is not None:
            line = Marker()
            line.header.frame_id = "base_link"
            line.header.stamp = self.get_clock().now().to_msg()
            line.ns = "planner_error"
            line.id = 3
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.008
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 1.0, 0.0, 1.0
            line.points = [Point(x=target[0], y=target[1], z=target[2]), Point(x=reached[0], y=reached[1], z=reached[2])]
            arr.markers.append(line)

        self.marker_pub.publish(arr)

    @staticmethod
    def xyz_dict(xyz: Tuple[float, float, float]) -> Dict[str, float]:
        return {"x": xyz[0], "y": xyz[1], "z": xyz[2]}

    @staticmethod
    def result(request_id: str, ok: bool, state: str, **extra: Any) -> Dict[str, Any]:
        out = {
            "ok": ok,
            "request_id": request_id,
            "state": state,
            "timestamp": time.time(),
        }
        out.update(extra)
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectPlanningAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
