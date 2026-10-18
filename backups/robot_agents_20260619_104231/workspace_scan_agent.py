#!/usr/bin/env python3

"""
Workspace scan agent.

Pipeline:
    RGB image -> AI classification/bboxes -> dense /aurora/points2 geometry
    -> calibrated camera transform -> point for every classified entity
    -> Redis + RViz markers.

Important behavior:
- Every AI detection receives a point.
- Movable objects use a foreground 3D cluster when possible.
- Flat destinations such as white paper, a mat, tray, plate or lid use the
  surface points under their image bbox.
- When the cloud is sparse, the agent falls back to nearest projected points.
- As a final fallback, it intersects the bbox-center camera ray with the table
  plane and marks the geometry quality as low instead of silently losing the
  classified entity.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from dotenv import load_dotenv
from geometry_msgs.msg import Point
from openai import OpenAI
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import sensor_msgs_py.point_cloud2 as pc2


ENV_FILE = "/workspace/.env"
load_dotenv(ENV_FILE)


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def decode_redis(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def canonical_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^0-9a-zа-яё_\- ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def safe_label(value: Any) -> str:
    text = canonical_text(value)
    # Keep ROS/Redis names ASCII where possible, but do not return empty.
    ascii_text = re.sub(r"[^0-9a-z_]+", "_", text)
    ascii_text = re.sub(r"_+", "_", ascii_text).strip("_")
    return ascii_text or "object"


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError("Vision model response was not a JSON object")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL = env("REDIS_URL", "redis://127.0.0.1:6379/0")

try:
    from robot_system.storage.redis_store import store

    REDIS = store.client
except Exception:
    import redis

    REDIS = redis.Redis.from_url(REDIS_URL, decode_responses=True)

OPENAI_VISION_MODEL = env("OPENAI_VISION_MODEL", "gpt-4.1-mini")

RGB_IMAGE_TOPIC = env("RGB_IMAGE_TOPIC", "/aurora/rgb/image_raw")
CAMERA_INFO_TOPIC = env(
    "GEOMETRY_CAMERA_INFO_TOPIC",
    "/aurora/rgb/camera_info",
)
POINT_CLOUD_TOPIC = env("POINT_CLOUD_TOPIC", "/aurora/points2")

OBJECTS_2D_JSON_TOPIC = env(
    "OBJECTS_2D_JSON_TOPIC",
    "/digital_twin/objects_2d_json",
)
OBJECTS_JSON_TOPIC = env(
    "OBJECTS_JSON_TOPIC",
    "/digital_twin/objects_json",
)
MOTION_POINTS_JSON_TOPIC = env(
    "MOTION_POINTS_JSON_TOPIC",
    "/digital_twin/motion_points_json",
)
MOTION_POINT_MARKERS_TOPIC = env(
    "MOTION_POINT_MARKERS_TOPIC",
    "/digital_twin/motion_point_markers",
)
AI_DEBUG_IMAGE_TOPIC = env(
    "AI_DEBUG_IMAGE_TOPIC",
    "/digital_twin/debug_ai_input_image",
)
SELECTED_CLUSTER_TOPIC = env(
    "SELECTED_CLUSTER_TOPIC",
    "/digital_twin/selected_object_clusters",
)

WORKSPACE_SCAN_REQUEST_STREAM = env(
    "WORKSPACE_SCAN_REQUEST_STREAM",
    "stream:workspace_scan_requests",
)
WORKSPACE_SCAN_RESULT_STREAM = env(
    "WORKSPACE_SCAN_RESULT_STREAM",
    "stream:workspace_scan_results",
)
MOTION_POINTS_KEY = env("MOTION_POINTS_KEY", "workspace:motion_points")
WORKSPACE_STATE_KEY = env("WORKSPACE_STATE_KEY", "workspace:state")

TARGET_FRAME = env("GEOMETRY_TARGET_FRAME", "base_link")

CAMERA_X = env_float("CAMERA_X", -0.219)
CAMERA_Y = env_float("CAMERA_Y", 0.01)
CAMERA_Z = env_float("CAMERA_Z", 0.439)
CAMERA_TILT_DEGREES = env_float("CAMERA_TILT_DEGREES", 47.2)
CAMERA_YAW_DEGREES = env_float("CAMERA_YAW_DEGREES", -2.0)
CAMERA_ROLL_DEGREES = env_float("CAMERA_ROLL_DEGREES", 10.0)
IMAGE_RIGHT_IS_NEGATIVE_Y = env_bool("IMAGE_RIGHT_IS_NEGATIVE_Y", True)
PROJECTION_U_OFFSET = env_float("PROJECTION_U_OFFSET", 0.0)
PROJECTION_V_OFFSET = env_float("PROJECTION_V_OFFSET", 0.0)
FLIP_PROJECTION_X = env_bool("FLIP_PROJECTION_X", False)
FLIP_PROJECTION_Y = env_bool("FLIP_PROJECTION_Y", False)

VISION_AI_MAX_WIDTH = env_int("VISION_AI_MAX_WIDTH", 768)
VISION_AI_JPEG_QUALITY = env_int("VISION_AI_JPEG_QUALITY", 88)
VISION_MAX_OBJECTS = env_int("VISION_MAX_OBJECTS", 30)
VISION_MIN_CONFIDENCE = env_float("VISION_MIN_CONFIDENCE", 0.15)
DETECTION_NMS_IOU = env_float("DETECTION_NMS_IOU", 0.72)
SAME_LABEL_NMS_IOU = env_float("SAME_LABEL_NMS_IOU", 0.48)

CAMERA_STALE_TIMEOUT_SEC = env_float("CAMERA_STALE_TIMEOUT_SEC", 10.0)
POINT_CLOUD_STALE_TIMEOUT_SEC = env_float(
    "POINT_CLOUD_STALE_TIMEOUT_SEC",
    10.0,
)

MIN_CAMERA_DEPTH = env_float("GEOMETRY_MIN_DEPTH_M", 0.05)
MAX_CAMERA_DEPTH = env_float("GEOMETRY_MAX_DEPTH_M", 3.0)
BBOX_PADDING_PIXELS = env_int("GEOMETRY_BBOX_PADDING_PIXELS", 7)
MIN_OBJECT_POINTS = env_int("GEOMETRY_MIN_OBJECT_POINTS", 8)
MIN_SURFACE_POINTS = env_int("GEOMETRY_MIN_SURFACE_POINTS", 6)
NEAREST_POINT_COUNT = env_int("GEOMETRY_NEAREST_POINT_COUNT", 100)
MAX_NEAREST_PIXEL_RADIUS = env_float(
    "GEOMETRY_MAX_NEAREST_PIXEL_RADIUS",
    100.0,
)
DEPTH_CLUSTER_GAP_M = env_float("GEOMETRY_DEPTH_CLUSTER_GAP_M", 0.018)
XY_CLUSTER_RESOLUTION = env_float(
    "GEOMETRY_CLUSTER_GRID_RESOLUTION",
    0.018,
)
MAX_OBJECT_DIMENSION_M = env_float(
    "GEOMETRY_MAX_OBJECT_DIMENSION_M",
    0.60,
)
BBOX_LOW_PERCENTILE = env_float("GEOMETRY_BBOX_LOW_PERCENTILE", 4.0)
BBOX_HIGH_PERCENTILE = env_float("GEOMETRY_BBOX_HIGH_PERCENTILE", 96.0)
MIN_MARKER_SIZE = env_float("GEOMETRY_MIN_MARKER_SIZE", 0.018)
MIN_SURFACE_THICKNESS = env_float("GEOMETRY_MIN_SURFACE_THICKNESS", 0.006)

WORKSPACE_X_MIN = env_float("GEOMETRY_WORKSPACE_X_MIN", -0.80)
WORKSPACE_X_MAX = env_float("GEOMETRY_WORKSPACE_X_MAX", 1.50)
WORKSPACE_Y_MIN = env_float("GEOMETRY_WORKSPACE_Y_MIN", -1.00)
WORKSPACE_Y_MAX = env_float("GEOMETRY_WORKSPACE_Y_MAX", 1.00)
WORKSPACE_Z_MIN = env_float("GEOMETRY_WORKSPACE_Z_MIN", -0.30)
WORKSPACE_Z_MAX = env_float("GEOMETRY_WORKSPACE_Z_MAX", 1.30)

TABLE_Z_DEFAULT = env_float("GEOMETRY_TABLE_Z", 0.0)
TABLE_Z_ESTIMATE_MIN = env_float("GEOMETRY_TABLE_Z_ESTIMATE_MIN", -0.12)
TABLE_Z_ESTIMATE_MAX = env_float("GEOMETRY_TABLE_Z_ESTIMATE_MAX", 0.20)
TABLE_Z_BIN_SIZE = env_float("GEOMETRY_TABLE_Z_BIN_SIZE", 0.004)
SURFACE_PLANE_BAND_M = env_float("GEOMETRY_SURFACE_PLANE_BAND_M", 0.012)
DEFAULT_RAY_DEPTH_M = env_float("GEOMETRY_DEFAULT_RAY_DEPTH_M", 0.60)

POINT_APPROACH_HEIGHT = env_float("POINT_APPROACH_HEIGHT", 0.10)
POINT_GRASP_HEIGHT_OFFSET = env_float("POINT_GRASP_HEIGHT_OFFSET", 0.018)
POINT_PLACE_HEIGHT_OFFSET = env_float("POINT_PLACE_HEIGHT_OFFSET", 0.025)


SURFACE_ROLES = {
    "placement_surface",
    "reference_surface",
    "support_surface",
}

SURFACE_LABEL_WORDS = {
    "paper",
    "sheet",
    "white_paper",
    "paper_sheet",
    "mat",
    "pad",
    "tray",
    "plate",
    "lid",
    "board",
    "cardboard",
    "cloth",
    "towel",
    "book",
    "notebook",
    "лист",
    "хартия",
    "подложка",
    "поднос",
    "чиния",
    "капак",
    "дъска",
}

SAFETY_LABEL_WORDS = {
    "hand",
    "person",
    "human",
    "finger",
    "arm",
    "robot_arm",
    "gripper",
    "ръка",
    "човек",
    "пръст",
}

LABEL_SYNONYMS: Dict[str, List[str]] = {
    "cube": ["cube", "block", "куб", "кубче"],
    "paper": ["paper", "sheet", "paper_sheet", "лист", "хартия"],
    "white_paper": [
        "white_paper",
        "white_sheet",
        "бял_лист",
        "белия_лист",
        "бяла_хартия",
    ],
    "tape_measure": ["tape_measure", "measuring_tape", "рулетка"],
    "screwdriver": ["screwdriver", "отвертка"],
    "phone": ["phone", "smartphone", "mobile_phone", "телефон"],
    "smartphone": ["smartphone", "phone", "телефон"],
    "box": ["box", "container", "кутия", "контейнер"],
    "tray": ["tray", "поднос"],
    "plate": ["plate", "чиния"],
    "lid": ["lid", "капак"],
    "mat": ["mat", "pad", "подложка"],
}

COLOR_SYNONYMS: Dict[str, List[str]] = {
    "green": ["green", "зелен", "зеления", "зелено"],
    "blue": ["blue", "син", "синия", "синьо"],
    "yellow": ["yellow", "жълт", "жълтия", "жълто"],
    "red": ["red", "червен", "червения", "червено"],
    "white": ["white", "бял", "белия", "бяла", "бяло"],
    "black": ["black", "черен", "черния", "черна", "черно"],
    "orange": ["orange", "оранжев", "оранжевия"],
    "gray": ["gray", "grey", "сив", "сивия"],
}


class WorkspaceScanAgent(Node):
    def __init__(self) -> None:
        super().__init__("workspace_scan_agent")

        self.bridge = CvBridge()
        self.openai = OpenAI()

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_time = 0.0
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_camera_info_time = 0.0
        self.latest_cloud: Optional[PointCloud2] = None
        self.latest_cloud_time = 0.0

        self.last_warn_times: Dict[str, float] = {}
        self.last_request_id = self.stream_tail_id(WORKSPACE_SCAN_REQUEST_STREAM)

        self.rotation_base_from_optical = self.make_camera_rotation_matrix()
        self.translation_base = np.array(
            [CAMERA_X, CAMERA_Y, CAMERA_Z],
            dtype=np.float32,
        )

        self.create_subscription(
            Image,
            RGB_IMAGE_TOPIC,
            self.rgb_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            POINT_CLOUD_TOPIC,
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        marker_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.objects_2d_pub = self.create_publisher(
            String,
            OBJECTS_2D_JSON_TOPIC,
            10,
        )
        self.objects_pub = self.create_publisher(
            String,
            OBJECTS_JSON_TOPIC,
            10,
        )
        self.motion_points_pub = self.create_publisher(
            String,
            MOTION_POINTS_JSON_TOPIC,
            10,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            AI_DEBUG_IMAGE_TOPIC,
            10,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            MOTION_POINT_MARKERS_TOPIC,
            marker_qos,
        )
        self.selected_cluster_pub = self.create_publisher(
            PointCloud2,
            SELECTED_CLUSTER_TOPIC,
            2,
        )

        self.create_timer(0.05, self.poll_scan_requests)

        self.get_logger().info("✅ workspace_scan_agent ready")
        self.get_logger().info(
            "Every classified entity receives a center/place point"
        )
        self.get_logger().info(f"RGB: {RGB_IMAGE_TOPIC}")
        self.get_logger().info(f"Point cloud: {POINT_CLOUD_TOPIC}")
        self.get_logger().info(
            "Camera calibration: "
            f"xyz=({CAMERA_X:.3f},{CAMERA_Y:.3f},{CAMERA_Z:.3f}) "
            f"tilt={CAMERA_TILT_DEGREES:.1f} "
            f"yaw={CAMERA_YAW_DEGREES:.1f} "
            f"roll={CAMERA_ROLL_DEGREES:.1f}"
        )

    # ------------------------------------------------------------------
    # ROS callbacks and utility logging
    # ------------------------------------------------------------------

    def warn_throttled(
        self,
        key: str,
        message: str,
        period_sec: float = 3.0,
    ) -> None:
        now = time.time()
        if now - self.last_warn_times.get(key, 0.0) >= period_sec:
            self.get_logger().warn(message)
            self.last_warn_times[key] = now

    def rgb_callback(self, msg: Image) -> None:
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_rgb_time = time.time()
        except Exception as exc:
            self.warn_throttled("rgb", f"RGB conversion failed: {exc}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg
        self.latest_camera_info_time = time.time()

    def cloud_callback(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg
        self.latest_cloud_time = time.time()

    # ------------------------------------------------------------------
    # Redis request loop
    # ------------------------------------------------------------------

    def stream_tail_id(self, stream_name: str) -> str:
        try:
            rows = REDIS.xrevrange(stream_name, count=1)
            if rows:
                return decode_redis(rows[0][0])
        except Exception:
            pass
        return "0-0"

    def poll_scan_requests(self) -> None:
        try:
            rows = REDIS.xread(
                {WORKSPACE_SCAN_REQUEST_STREAM: self.last_request_id},
                count=10,
            )
        except Exception as exc:
            self.warn_throttled("redis", f"Redis xread failed: {exc}")
            return

        for _stream_name, entries in rows:
            for entry_id, fields in entries:
                self.last_request_id = decode_redis(entry_id)
                decoded_fields = {
                    decode_redis(k): decode_redis(v)
                    for k, v in fields.items()
                }

                try:
                    payload = json.loads(decoded_fields.get("payload", "{}"))
                except Exception:
                    payload = {}

                request_id = str(
                    payload.get("request_id")
                    or f"workspace-scan-{time.time_ns()}"
                )

                lock_key = f"lock:workspace_scan_request:{request_id}"
                try:
                    got_lock = bool(REDIS.set(lock_key, "1", nx=True, ex=120))
                except Exception:
                    got_lock = True

                if not got_lock:
                    self.get_logger().warn(
                        f"⏭ duplicate scan request ignored: {request_id}"
                    )
                    continue

                self.get_logger().info(
                    f"📥 scan request received: {request_id} "
                    f"redis_id={self.last_request_id}"
                )

                try:
                    result = self.scan_workspace(request_id)
                except Exception as exc:
                    self.get_logger().exception(
                        f"workspace scan crashed: {exc}"
                    )
                    result = {
                        "ok": False,
                        "request_id": request_id,
                        "scan_state": "SCAN_CRASHED",
                        "error": str(exc),
                        "objects": [],
                        "motion_points": [],
                        "count": 0,
                        "timestamp": time.time(),
                    }

                try:
                    REDIS.xadd(
                        WORKSPACE_SCAN_RESULT_STREAM,
                        {"payload": json.dumps(result, ensure_ascii=False)},
                    )
                except Exception as exc:
                    self.get_logger().error(
                        f"Could not publish scan result to Redis: {exc}"
                    )

    # ------------------------------------------------------------------
    # Complete scan
    # ------------------------------------------------------------------

    def scan_workspace(self, request_id: str) -> Dict[str, Any]:
        scan_start = time.time()

        if self.latest_rgb is None:
            return self.fail_result(request_id, "NO_RGB")
        if self.latest_camera_info is None:
            return self.fail_result(request_id, "NO_CAMERA_INFO")
        if self.latest_cloud is None:
            return self.fail_result(request_id, "NO_POINT_CLOUD")

        if scan_start - self.latest_rgb_time > CAMERA_STALE_TIMEOUT_SEC:
            return self.fail_result(request_id, "STALE_RGB")
        if scan_start - self.latest_cloud_time > POINT_CLOUD_STALE_TIMEOUT_SEC:
            return self.fail_result(request_id, "STALE_POINT_CLOUD")

        # Snapshot data before the AI request blocks the single-threaded node.
        rgb = self.latest_rgb.copy()
        info = self.latest_camera_info
        cloud = self.latest_cloud

        detections_raw = self.call_vision_model(rgb)
        detections = self.normalize_detections(
            detections_raw,
            rgb.shape[1],
            rgb.shape[0],
        )
        detections = self.remove_duplicate_detections(detections)
        detections = detections[:VISION_MAX_OBJECTS]

        debug_image = rgb.copy()

        objects_2d_document = {
            "source": "workspace_scan_agent",
            "scan_state": "OK" if detections else "NO_DETECTIONS",
            "image_width": int(rgb.shape[1]),
            "image_height": int(rgb.shape[0]),
            "objects": detections,
            "timestamp": time.time(),
        }
        self.objects_2d_pub.publish(
            String(data=json.dumps(objects_2d_document, ensure_ascii=False))
        )

        if not detections:
            self.publish_debug_image(debug_image)
            return self.fail_result(request_id, "NO_DETECTIONS")

        points_cam = self.read_xyz_points(cloud)
        if points_cam is None or points_cam.shape[0] == 0:
            self.publish_debug_image(debug_image)
            return self.fail_result(request_id, "EMPTY_POINT_CLOUD")

        projected = self.project_cloud(
            points_cam,
            info,
            rgb.shape[1],
            rgb.shape[0],
        )
        if projected is None or projected.shape[0] == 0:
            self.publish_debug_image(debug_image)
            return self.fail_result(request_id, "POINT_CLOUD_PROJECTION_EMPTY")

        table_z = self.estimate_table_z(projected[:, 2:5])
        objects: List[Dict[str, Any]] = []
        all_selected_points: List[np.ndarray] = []

        for det in detections:
            obj, selected_points = self.geometry_for_detection(
                det,
                projected,
                info,
                rgb.shape[1],
                rgb.shape[0],
                table_z,
            )

            # geometry_for_detection always returns an object. Keep the guard to
            # avoid a scan crash if future code is changed.
            if obj is None:
                obj = self.make_emergency_object(det, table_z)
                selected_points = np.empty((0, 3), dtype=np.float32)

            objects.append(obj)
            if selected_points is not None and selected_points.shape[0] > 0:
                all_selected_points.append(selected_points)

            if obj.get("safety_blocker"):
                color = (0, 0, 255)
            elif obj.get("geometry_quality") == "low":
                color = (0, 165, 255)
            elif obj.get("role") in SURFACE_ROLES:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)

            self.draw_detection(debug_image, det, obj, color)

        objects = self.assign_names_and_aliases(objects)
        objects_document, motion_document = self.publish_workspace(objects)
        self.publish_debug_image(debug_image)

        if all_selected_points:
            self.publish_selected_clusters(
                np.concatenate(all_selected_points, axis=0)
            )

        return {
            "ok": True,
            "request_id": request_id,
            "scan_state": "OK",
            "objects": objects_document["objects"],
            "motion_points": motion_document["motion_points"],
            "count": len(motion_document["motion_points"]),
            "table_z": table_z,
            "timestamp": time.time(),
        }

    def fail_result(
        self,
        request_id: str,
        reason: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        self.warn_throttled(
            f"scan_fail_{reason}",
            f"workspace scan failed: {reason}",
            1.0,
        )
        result: Dict[str, Any] = {
            "ok": False,
            "request_id": request_id,
            "scan_state": reason,
            "objects": [],
            "motion_points": [],
            "count": 0,
            "timestamp": time.time(),
        }
        result.update(extra)
        return result

    # ------------------------------------------------------------------
    # Vision classification
    # ------------------------------------------------------------------

    def call_vision_model(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        height, width = image_bgr.shape[:2]
        scale = 1.0
        image_to_send = image_bgr

        if width > VISION_AI_MAX_WIDTH:
            scale = VISION_AI_MAX_WIDTH / float(width)
            resized_height = max(1, int(round(height * scale)))
            image_to_send = cv2.resize(
                image_bgr,
                (VISION_AI_MAX_WIDTH, resized_height),
                interpolation=cv2.INTER_AREA,
            )

        ok, jpg = cv2.imencode(
            ".jpg",
            image_to_send,
            [int(cv2.IMWRITE_JPEG_QUALITY), VISION_AI_JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("Could not encode the RGB image as JPEG")

        encoded = base64.b64encode(jpg.tobytes()).decode("ascii")

        prompt = """
Return ONLY one JSON object with an `objects` array.

Inspect the complete robotic-arm workspace and classify EVERY distinct visible
entity that a user may refer to in a command. Do not restrict the answer to
simple graspable objects.

Include:
- movable objects: cubes, tools, phones, tape measures, boxes, bottles, parts;
- placement destinations and flat reference surfaces: white paper, sheets,
  cards, books, mats, pads, trays, plates, lids, boards and cloth;
- containers and supports that are visually distinct from the main table.

Do not include:
- the main table itself;
- the robot arm or gripper;
- shadows, reflections, texture patches or empty background;
- cables unless they are clearly the object the user could request;
- a duplicate detection of the same physical entity.

For every physical entity return:
- `label`: concise English snake_case identifier;
- `display_name_bg`: short natural Bulgarian name;
- `role`: one of movable_object, placement_surface, reference_surface,
  container, tool, safety_blocker;
- `confidence`: number from 0 to 1;
- `colors`: list of simple English color names;
- `bbox`: pixel coordinates x1, y1, x2, y2 in the supplied image;
- `safety_blocker`: true only for a human hand/person or another immediate
  safety obstacle.

Be especially careful to include a white sheet of paper even though it is flat
and almost coplanar with the table.

Schema:
{
  "objects": [
    {
      "label": "white_paper",
      "display_name_bg": "белият лист",
      "role": "placement_surface",
      "confidence": 0.93,
      "colors": ["white"],
      "bbox": {"x1": 100, "y1": 120, "x2": 250, "y2": 260},
      "safety_blocker": false
    }
  ]
}
"""

        response = self.openai.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            temperature=0.0,
            max_tokens=2200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
        )

        data = extract_json(response.choices[0].message.content or "{}")
        objects = data.get("objects", [])
        if not isinstance(objects, list):
            return []

        if scale != 1.0:
            inverse = 1.0 / scale
            for obj in objects:
                bbox = obj.get("bbox") if isinstance(obj, dict) else None
                if not isinstance(bbox, dict):
                    continue
                for key in ("x1", "x2", "y1", "y2"):
                    if key in bbox:
                        try:
                            bbox[key] = float(bbox[key]) * inverse
                        except (TypeError, ValueError):
                            pass

        return [obj for obj in objects if isinstance(obj, dict)]

    def normalize_detections(
        self,
        detections: Sequence[Dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for raw_index, detection in enumerate(detections):
            bbox = (
                detection.get("bbox")
                or detection.get("box")
                or detection.get("bounding_box")
            )
            if not isinstance(bbox, dict):
                continue

            try:
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
            except (KeyError, TypeError, ValueError):
                continue

            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                x1 *= image_width
                x2 *= image_width
                y1 *= image_height
                y2 *= image_height

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            x1 = max(0.0, min(float(image_width - 1), x1))
            y1 = max(0.0, min(float(image_height - 1), y1))
            x2 = max(1.0, min(float(image_width), x2))
            y2 = max(1.0, min(float(image_height), y2))

            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue

            confidence = float(detection.get("confidence", 0.0) or 0.0)
            if confidence < VISION_MIN_CONFIDENCE:
                continue

            label = str(
                detection.get("label")
                or detection.get("name")
                or f"object_{raw_index}"
            ).strip()
            display_name_bg = str(
                detection.get("display_name_bg")
                or detection.get("name_bg")
                or label
            ).strip()
            role = canonical_text(detection.get("role") or "movable_object")
            normalized_label = canonical_text(label)

            if role not in {
                "movable_object",
                "placement_surface",
                "reference_surface",
                "support_surface",
                "container",
                "tool",
                "safety_blocker",
            }:
                role = "movable_object"

            if normalized_label in SURFACE_LABEL_WORDS:
                role = "placement_surface"

            safety_blocker = bool(detection.get("safety_blocker", False))
            if normalized_label in SAFETY_LABEL_WORDS or role == "safety_blocker":
                safety_blocker = True
                role = "safety_blocker"

            colors = detection.get("colors", [])
            if not isinstance(colors, list):
                colors = [colors]
            colors = [canonical_text(color) for color in colors if str(color).strip()]

            pixel_bbox = [
                max(0, int(math.floor(x1 - BBOX_PADDING_PIXELS))),
                max(0, int(math.floor(y1 - BBOX_PADDING_PIXELS))),
                min(image_width, int(math.ceil(x2 + BBOX_PADDING_PIXELS))),
                min(image_height, int(math.ceil(y2 + BBOX_PADDING_PIXELS))),
            ]

            normalized.append(
                {
                    "raw_index": raw_index,
                    "label": label,
                    "safe_label": safe_label(label),
                    "display_name_bg": display_name_bg,
                    "role": role,
                    "confidence": confidence,
                    "colors": colors,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "bbox_pixels_rgb": pixel_bbox,
                    "safety_blocker": safety_blocker,
                }
            )

        return normalized

    @staticmethod
    def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
        area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
        return intersection / max(1.0, area_a + area_b - intersection)

    def remove_duplicate_detections(
        self,
        detections: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            detections,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        kept: List[Dict[str, Any]] = []

        for candidate in ordered:
            candidate_bbox = candidate["bbox_pixels_rgb"]
            candidate_label = candidate.get("safe_label")
            duplicate = False

            for existing in kept:
                iou = self.bbox_iou(
                    candidate_bbox,
                    existing["bbox_pixels_rgb"],
                )
                same_label = candidate_label == existing.get("safe_label")
                if iou >= DETECTION_NMS_IOU or (
                    same_label and iou >= SAME_LABEL_NMS_IOU
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(candidate)

        return kept

    # ------------------------------------------------------------------
    # Camera transform and cloud projection
    # ------------------------------------------------------------------

    def make_camera_rotation_matrix(self) -> np.ndarray:
        """Match the manual transform used by colorized_pointcloud.py."""
        tilt = math.radians(CAMERA_TILT_DEGREES)
        yaw = math.radians(CAMERA_YAW_DEGREES)
        roll = math.radians(CAMERA_ROLL_DEGREES)

        optical_z = np.array(
            [math.cos(tilt), 0.0, -math.sin(tilt)],
            dtype=np.float64,
        )
        optical_x = np.array(
            [0.0, -1.0 if IMAGE_RIGHT_IS_NEGATIVE_Y else 1.0, 0.0],
            dtype=np.float64,
        )
        optical_y = np.cross(optical_z, optical_x)

        optical_x /= max(np.linalg.norm(optical_x), 1e-12)
        optical_y /= max(np.linalg.norm(optical_y), 1e-12)
        optical_z /= max(np.linalg.norm(optical_z), 1e-12)

        # Positive roll is a right-handed rotation around the camera optical Z.
        rolled_x = math.cos(roll) * optical_x + math.sin(roll) * optical_y
        rolled_y = -math.sin(roll) * optical_x + math.cos(roll) * optical_y

        rotation_no_yaw = np.column_stack((rolled_x, rolled_y, optical_z))

        yaw_rotation = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return (yaw_rotation @ rotation_no_yaw).astype(np.float32)

    def read_xyz_points(self, cloud: PointCloud2) -> Optional[np.ndarray]:
        try:
            result = pc2.read_points(
                cloud,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )

            if isinstance(result, np.ndarray):
                if result.dtype.fields:
                    points = np.stack(
                        [result["x"], result["y"], result["z"]],
                        axis=1,
                    ).astype(np.float32)
                else:
                    points = np.asarray(result, dtype=np.float32)[:, :3]
            else:
                points = np.asarray(
                    [[row[0], row[1], row[2]] for row in result],
                    dtype=np.float32,
                )

            if points.size == 0:
                return None

            valid = (
                np.isfinite(points).all(axis=1)
                & (points[:, 2] >= MIN_CAMERA_DEPTH)
                & (points[:, 2] <= MAX_CAMERA_DEPTH)
            )
            points = points[valid]
            return points if points.shape[0] else None

        except Exception as exc:
            self.warn_throttled(
                "cloud_read",
                f"Could not read PointCloud2 xyz: {exc}",
            )
            return None

    def project_cloud(
        self,
        points_cam: np.ndarray,
        info: CameraInfo,
        image_width: int,
        image_height: int,
    ) -> Optional[np.ndarray]:
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None

        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        u_info = fx * x_cam / z_cam + cx + PROJECTION_U_OFFSET
        v_info = fy * y_cam / z_cam + cy + PROJECTION_V_OFFSET

        info_width = float(info.width or image_width)
        info_height = float(info.height or image_height)
        u = u_info * (float(image_width) / info_width)
        v = v_info * (float(image_height) / info_height)

        if FLIP_PROJECTION_X:
            u = float(image_width - 1) - u
        if FLIP_PROJECTION_Y:
            v = float(image_height - 1) - v

        points_base = (
            points_cam @ self.rotation_base_from_optical.T
            + self.translation_base
        ).astype(np.float32)

        valid = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(points_base).all(axis=1)
            & (u >= 0.0)
            & (u < image_width)
            & (v >= 0.0)
            & (v < image_height)
            & (points_base[:, 0] >= WORKSPACE_X_MIN)
            & (points_base[:, 0] <= WORKSPACE_X_MAX)
            & (points_base[:, 1] >= WORKSPACE_Y_MIN)
            & (points_base[:, 1] <= WORKSPACE_Y_MAX)
            & (points_base[:, 2] >= WORKSPACE_Z_MIN)
            & (points_base[:, 2] <= WORKSPACE_Z_MAX)
        )

        if not np.any(valid):
            return None

        # Columns: u, v, x_base, y_base, z_base, camera_depth.
        return np.column_stack(
            [
                u[valid].astype(np.float32),
                v[valid].astype(np.float32),
                points_base[valid],
                z_cam[valid].astype(np.float32),
            ]
        )

    # ------------------------------------------------------------------
    # Geometry for every classified entity
    # ------------------------------------------------------------------

    def estimate_table_z(self, points_base: np.ndarray) -> float:
        z = points_base[:, 2]
        z = z[
            np.isfinite(z)
            & (z >= TABLE_Z_ESTIMATE_MIN)
            & (z <= TABLE_Z_ESTIMATE_MAX)
        ]
        if z.size < 20:
            return TABLE_Z_DEFAULT

        bin_size = max(TABLE_Z_BIN_SIZE, 0.001)
        low = float(np.min(z))
        high = float(np.max(z))
        count = max(2, int(math.ceil((high - low) / bin_size)) + 1)
        histogram, edges = np.histogram(z, bins=count, range=(low, high))
        index = int(np.argmax(histogram))
        center = float((edges[index] + edges[index + 1]) / 2.0)
        local = z[np.abs(z - center) <= bin_size * 1.5]
        if local.size:
            center = float(np.median(local))
        return center

    def geometry_for_detection(
        self,
        detection: Dict[str, Any],
        projected: np.ndarray,
        info: CameraInfo,
        image_width: int,
        image_height: int,
        table_z: float,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        x1, y1, x2, y2 = detection["bbox_pixels_rgb"]
        inside = (
            (projected[:, 0] >= x1)
            & (projected[:, 0] <= x2)
            & (projected[:, 1] >= y1)
            & (projected[:, 1] <= y2)
        )
        candidates = projected[inside]
        role = detection.get("role", "movable_object")

        if role in SURFACE_ROLES:
            selected = self.select_surface_points(candidates)
            if selected is not None:
                return (
                    self.object_from_points(
                        selected,
                        detection,
                        geometry_source="bbox_surface_points",
                        geometry_quality="high",
                        table_z=table_z,
                    ),
                    selected[:, 2:5],
                )
        else:
            selected = self.select_object_cluster(candidates, detection)
            if selected is not None:
                return (
                    self.object_from_points(
                        selected,
                        detection,
                        geometry_source="bbox_foreground_cluster",
                        geometry_quality="high",
                        table_z=table_z,
                    ),
                    selected[:, 2:5],
                )

        # Medium-quality fallback: use real projected points nearest to the bbox
        # centre. This is particularly useful for thin tools and sparse clouds.
        nearest = self.select_nearest_projected_points(
            projected,
            detection,
        )
        if nearest is not None:
            if role in SURFACE_ROLES:
                nearest_surface = self.select_surface_points(nearest)
                if nearest_surface is not None:
                    nearest = nearest_surface
            else:
                nearest_cluster = self.select_object_cluster(nearest, detection)
                if nearest_cluster is not None:
                    nearest = nearest_cluster

            return (
                self.object_from_points(
                    nearest,
                    detection,
                    geometry_source="nearest_projected_points",
                    geometry_quality="medium",
                    table_z=table_z,
                ),
                nearest[:, 2:5],
            )

        # Guaranteed final point for every classified entity.
        fallback = self.object_from_bbox_ray(
            detection,
            info,
            image_width,
            image_height,
            table_z,
        )
        return fallback, np.empty((0, 3), dtype=np.float32)

    def select_surface_points(
        self,
        candidates: np.ndarray,
    ) -> Optional[np.ndarray]:
        if candidates is None or candidates.shape[0] < MIN_SURFACE_POINTS:
            return None

        candidates = candidates[np.isfinite(candidates).all(axis=1)]
        if candidates.shape[0] < MIN_SURFACE_POINTS:
            return None

        # Select the dominant horizontal z band. For a sheet of paper this is
        # intentionally the table/paper plane under the visual bbox.
        z = candidates[:, 4]
        bin_size = max(TABLE_Z_BIN_SIZE, 0.002)
        low = float(np.min(z))
        high = float(np.max(z))
        if high - low < bin_size:
            selected = candidates
        else:
            count = max(2, int(math.ceil((high - low) / bin_size)) + 1)
            histogram, edges = np.histogram(z, bins=count, range=(low, high))
            index = int(np.argmax(histogram))
            z_mode = float((edges[index] + edges[index + 1]) / 2.0)
            selected = candidates[np.abs(z - z_mode) <= SURFACE_PLANE_BAND_M]
            if selected.shape[0] < MIN_SURFACE_POINTS:
                selected = candidates

        return self.robust_trim(selected, minimum=MIN_SURFACE_POINTS)

    def select_object_cluster(
        self,
        candidates: np.ndarray,
        detection: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        if candidates is None or candidates.shape[0] < MIN_OBJECT_POINTS:
            return None

        candidates = candidates[np.isfinite(candidates).all(axis=1)]
        if candidates.shape[0] < MIN_OBJECT_POINTS:
            return None

        depth_bands = self.split_depth_bands(candidates)
        clusters: List[np.ndarray] = []

        for band in depth_bands:
            xy_clusters = self.cluster_candidates_xy(band)
            if xy_clusters:
                clusters.extend(xy_clusters)
            elif band.shape[0] >= MIN_OBJECT_POINTS:
                clusters.append(band)

        if not clusters:
            return None

        bbox = detection["bbox_pixels_rgb"]
        bbox_cx = (bbox[0] + bbox[2]) / 2.0
        bbox_cy = (bbox[1] + bbox[3]) / 2.0
        bbox_diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        minimum_depth = float(np.min(candidates[:, 5]))

        best: Optional[np.ndarray] = None
        best_score = -float("inf")

        for cluster in clusters:
            if cluster.shape[0] < MIN_OBJECT_POINTS:
                continue

            trimmed = self.robust_trim(cluster, minimum=MIN_OBJECT_POINTS)
            if trimmed is None:
                continue

            xyz = trimmed[:, 2:5]
            low = np.percentile(xyz, BBOX_LOW_PERCENTILE, axis=0)
            high = np.percentile(xyz, BBOX_HIGH_PERCENTILE, axis=0)
            dims = high - low

            if np.any(~np.isfinite(dims)):
                continue

            pixel_center_u = float(np.median(trimmed[:, 0]))
            pixel_center_v = float(np.median(trimmed[:, 1]))
            pixel_distance = math.hypot(
                pixel_center_u - bbox_cx,
                pixel_center_v - bbox_cy,
            ) / bbox_diag

            depth_median = float(np.median(trimmed[:, 5]))
            foreground_delta = max(0.0, depth_median - minimum_depth)
            max_dim = float(np.max(dims))
            flatness = float(dims[2])

            score = 0.0
            score += min(trimmed.shape[0], 500) * 0.015
            score -= pixel_distance * 6.0
            score -= foreground_delta * 30.0
            score -= float(np.std(trimmed[:, 5])) * 20.0

            if max_dim > MAX_OBJECT_DIMENSION_M:
                score -= 100.0
            if flatness < 0.003 and max(float(dims[0]), float(dims[1])) > 0.20:
                score -= 50.0

            if score > best_score:
                best_score = score
                best = trimmed

        return best

    def split_depth_bands(self, candidates: np.ndarray) -> List[np.ndarray]:
        if candidates.shape[0] < MIN_OBJECT_POINTS:
            return []

        order = np.argsort(candidates[:, 5])
        sorted_candidates = candidates[order]
        depths = sorted_candidates[:, 5]
        gaps = np.diff(depths)
        split_indices = np.where(gaps > DEPTH_CLUSTER_GAP_M)[0] + 1
        bands = np.split(sorted_candidates, split_indices)
        return [band for band in bands if band.shape[0] >= MIN_OBJECT_POINTS]

    def cluster_candidates_xy(self, candidates: np.ndarray) -> List[np.ndarray]:
        if candidates.shape[0] < MIN_OBJECT_POINTS:
            return []

        resolution = max(XY_CLUSTER_RESOLUTION, 0.004)
        xyz = candidates[:, 2:5]
        gx = np.floor((xyz[:, 0] - np.min(xyz[:, 0])) / resolution).astype(np.int32)
        gy = np.floor((xyz[:, 1] - np.min(xyz[:, 1])) / resolution).astype(np.int32)

        cells: Dict[Tuple[int, int], List[int]] = {}
        for index, cell in enumerate(zip(gx.tolist(), gy.tolist())):
            cells.setdefault(cell, []).append(index)

        visited: set[Tuple[int, int]] = set()
        clusters: List[np.ndarray] = []

        for start in cells:
            if start in visited:
                continue

            queue: deque[Tuple[int, int]] = deque([start])
            visited.add(start)
            indices: List[int] = []

            while queue:
                cell = queue.popleft()
                indices.extend(cells.get(cell, []))
                cx, cy = cell
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbor = (cx + dx, cy + dy)
                        if neighbor in cells and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            if len(indices) >= MIN_OBJECT_POINTS:
                clusters.append(candidates[np.asarray(indices, dtype=np.int32)])

        return clusters

    def select_nearest_projected_points(
        self,
        projected: np.ndarray,
        detection: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        bbox = detection["bbox_pixels_rgb"]
        center_u = (bbox[0] + bbox[2]) / 2.0
        center_v = (bbox[1] + bbox[3]) / 2.0

        distances = np.hypot(
            projected[:, 0] - center_u,
            projected[:, 1] - center_v,
        )
        order = np.argsort(distances)
        count = min(NEAREST_POINT_COUNT, projected.shape[0])
        selected = projected[order[:count]]
        selected_distances = distances[order[:count]]
        within_radius = selected_distances <= MAX_NEAREST_PIXEL_RADIUS
        selected = selected[within_radius]

        minimum = (
            MIN_SURFACE_POINTS
            if detection.get("role") in SURFACE_ROLES
            else MIN_OBJECT_POINTS
        )
        if selected.shape[0] < minimum:
            return None

        return selected

    @staticmethod
    def robust_trim(
        candidates: np.ndarray,
        minimum: int,
    ) -> Optional[np.ndarray]:
        if candidates.shape[0] < minimum:
            return None
        if candidates.shape[0] < 12:
            return candidates

        xyz = candidates[:, 2:5]
        low = np.percentile(xyz, 2.0, axis=0)
        high = np.percentile(xyz, 98.0, axis=0)
        keep = np.all((xyz >= low) & (xyz <= high), axis=1)
        trimmed = candidates[keep]
        return trimmed if trimmed.shape[0] >= minimum else candidates

    def object_from_points(
        self,
        selected: np.ndarray,
        detection: Dict[str, Any],
        geometry_source: str,
        geometry_quality: str,
        table_z: float,
    ) -> Dict[str, Any]:
        xyz = selected[:, 2:5]
        low = np.percentile(xyz, BBOX_LOW_PERCENTILE, axis=0)
        high = np.percentile(xyz, BBOX_HIGH_PERCENTILE, axis=0)
        median = np.median(xyz, axis=0)

        role = detection.get("role", "movable_object")
        raw_dimensions = np.maximum(high - low, 0.0)

        if role in SURFACE_ROLES:
            center = median.astype(np.float64)
            center[2] = float(np.median(xyz[:, 2]))
            dimensions = np.array(
                [
                    max(float(raw_dimensions[0]), MIN_MARKER_SIZE),
                    max(float(raw_dimensions[1]), MIN_MARKER_SIZE),
                    max(float(raw_dimensions[2]), MIN_SURFACE_THICKNESS),
                ],
                dtype=np.float64,
            )
            z_min = center[2] - dimensions[2] / 2.0
            z_max = center[2] + dimensions[2] / 2.0
        else:
            center = median.astype(np.float64)
            dimensions = np.maximum(raw_dimensions, MIN_MARKER_SIZE)
            z_min = float(low[2])
            z_max = float(high[2])

        return self.finish_object(
            detection=detection,
            center=(float(center[0]), float(center[1]), float(center[2])),
            dimensions=(
                float(dimensions[0]),
                float(dimensions[1]),
                float(dimensions[2]),
            ),
            raw_dimensions=(
                float(raw_dimensions[0]),
                float(raw_dimensions[1]),
                float(raw_dimensions[2]),
            ),
            bbox_3d={
                "x_min": float(low[0]),
                "x_max": float(high[0]),
                "y_min": float(low[1]),
                "y_max": float(high[1]),
                "z_min": z_min,
                "z_max": z_max,
            },
            geometry_source=geometry_source,
            geometry_quality=geometry_quality,
            point_count=int(selected.shape[0]),
            table_z=table_z,
        )

    def object_from_bbox_ray(
        self,
        detection: Dict[str, Any],
        info: CameraInfo,
        image_width: int,
        image_height: int,
        table_z: float,
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = detection["bbox_pixels_rgb"]
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0

        center = self.pixel_ray_to_plane(
            center_u,
            center_v,
            info,
            image_width,
            image_height,
            table_z,
        )

        corner_points: List[np.ndarray] = []
        for u, v in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            point = self.pixel_ray_to_plane(
                float(u),
                float(v),
                info,
                image_width,
                image_height,
                table_z,
            )
            if point is not None:
                corner_points.append(point)

        if center is None:
            # Last deterministic fallback. It remains visibly marked as low
            # quality and therefore is not presented as precise geometry.
            optical_ray = np.array([0.0, 0.0, DEFAULT_RAY_DEPTH_M], dtype=np.float32)
            center = self.translation_base + self.rotation_base_from_optical @ optical_ray

        if corner_points:
            corners = np.asarray(corner_points, dtype=np.float64)
            low = np.min(corners, axis=0)
            high = np.max(corners, axis=0)
            raw_x = max(float(high[0] - low[0]), MIN_MARKER_SIZE)
            raw_y = max(float(high[1] - low[1]), MIN_MARKER_SIZE)
        else:
            raw_x = 0.04
            raw_y = 0.04

        role = detection.get("role", "movable_object")
        raw_z = MIN_SURFACE_THICKNESS if role in SURFACE_ROLES else MIN_MARKER_SIZE
        dimensions = (raw_x, raw_y, raw_z)

        return self.finish_object(
            detection=detection,
            center=(float(center[0]), float(center[1]), float(center[2])),
            dimensions=dimensions,
            raw_dimensions=dimensions,
            bbox_3d={
                "x_min": float(center[0] - raw_x / 2.0),
                "x_max": float(center[0] + raw_x / 2.0),
                "y_min": float(center[1] - raw_y / 2.0),
                "y_max": float(center[1] + raw_y / 2.0),
                "z_min": float(center[2] - raw_z / 2.0),
                "z_max": float(center[2] + raw_z / 2.0),
            },
            geometry_source="bbox_ray_table_fallback",
            geometry_quality="low",
            point_count=0,
            table_z=table_z,
        )

    def pixel_ray_to_plane(
        self,
        u_image: float,
        v_image: float,
        info: CameraInfo,
        image_width: int,
        image_height: int,
        plane_z: float,
    ) -> Optional[np.ndarray]:
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None

        info_width = float(info.width or image_width)
        info_height = float(info.height or image_height)
        u = u_image * info_width / float(image_width)
        v = v_image * info_height / float(image_height)

        if FLIP_PROJECTION_X:
            u = info_width - 1.0 - u
        if FLIP_PROJECTION_Y:
            v = info_height - 1.0 - v

        u -= PROJECTION_U_OFFSET
        v -= PROJECTION_V_OFFSET

        ray_optical = np.array(
            [(u - cx) / fx, (v - cy) / fy, 1.0],
            dtype=np.float64,
        )
        ray_optical /= max(np.linalg.norm(ray_optical), 1e-12)
        ray_base = self.rotation_base_from_optical.astype(np.float64) @ ray_optical
        origin = self.translation_base.astype(np.float64)

        if abs(ray_base[2]) < 1e-7:
            return None

        distance = (plane_z - origin[2]) / ray_base[2]
        if distance <= 0.0:
            return None

        point = origin + distance * ray_base
        return point.astype(np.float64)

    def make_emergency_object(
        self,
        detection: Dict[str, Any],
        table_z: float,
    ) -> Dict[str, Any]:
        return self.finish_object(
            detection=detection,
            center=(0.25, 0.0, table_z),
            dimensions=(0.04, 0.04, MIN_MARKER_SIZE),
            raw_dimensions=(0.04, 0.04, MIN_MARKER_SIZE),
            bbox_3d={
                "x_min": 0.23,
                "x_max": 0.27,
                "y_min": -0.02,
                "y_max": 0.02,
                "z_min": table_z,
                "z_max": table_z + MIN_MARKER_SIZE,
            },
            geometry_source="emergency_fallback",
            geometry_quality="low",
            point_count=0,
            table_z=table_z,
        )

    def finish_object(
        self,
        detection: Dict[str, Any],
        center: Tuple[float, float, float],
        dimensions: Tuple[float, float, float],
        raw_dimensions: Tuple[float, float, float],
        bbox_3d: Dict[str, float],
        geometry_source: str,
        geometry_quality: str,
        point_count: int,
        table_z: float,
    ) -> Dict[str, Any]:
        x, y, z = center
        dx, dy, dz = dimensions
        role = detection.get("role", "movable_object")

        if role in SURFACE_ROLES:
            top_z = z
            grasp_z = z + POINT_GRASP_HEIGHT_OFFSET
            place_z = z + POINT_PLACE_HEIGHT_OFFSET
        else:
            top_z = max(float(bbox_3d["z_max"]), z + dz / 2.0)
            grasp_z = top_z + POINT_GRASP_HEIGHT_OFFSET
            place_z = top_z + POINT_PLACE_HEIGHT_OFFSET

        approach_z = top_z + POINT_APPROACH_HEIGHT

        return {
            "label": detection["label"],
            "safe_label": detection["safe_label"],
            "display_name_bg": detection.get("display_name_bg", detection["label"]),
            "role": role,
            "confidence": detection.get("confidence", 0.0),
            "colors": detection.get("colors", []),
            "safety_blocker": bool(detection.get("safety_blocker", False)),
            "frame": TARGET_FRAME,
            "x": x,
            "y": y,
            "z": z,
            "center": {"x": x, "y": y, "z": z},
            "object_position": {"x": x, "y": y, "z": z},
            "top_position": {"x": x, "y": y, "z": top_z},
            "grasp_position": {"x": x, "y": y, "z": grasp_z},
            "approach_position": {"x": x, "y": y, "z": approach_z},
            "place_position": {"x": x, "y": y, "z": place_z},
            "destination_position": {"x": x, "y": y, "z": place_z},
            "dimensions": {"x": dx, "y": dy, "z": dz},
            "raw_dimensions": {
                "x": raw_dimensions[0],
                "y": raw_dimensions[1],
                "z": raw_dimensions[2],
            },
            "width": dy,
            "depth": dx,
            "height": dz,
            "bbox_3d": bbox_3d,
            "bbox_pixels_rgb": detection.get("bbox_pixels_rgb"),
            "points": point_count,
            "geometry_source": geometry_source,
            "geometry_quality": geometry_quality,
            "table_z": table_z,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Names, aliases and output documents
    # ------------------------------------------------------------------

    def assign_names_and_aliases(
        self,
        objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        objects.sort(
            key=lambda obj: math.hypot(
                float(obj.get("x", 0.0)),
                float(obj.get("y", 0.0)),
            )
        )
        counts: Dict[str, int] = {}

        for global_index, obj in enumerate(objects):
            label = str(obj["safe_label"])
            instance_index = counts.get(label, 0)
            counts[label] = instance_index + 1

            obj["global_index"] = global_index
            obj["instance_index"] = instance_index
            obj["object_id"] = f"object_{global_index}"
            obj["point_label"] = f"{label}_{instance_index}_center"

            aliases = {
                obj["object_id"],
                f"object_{global_index}_center",
                label,
                f"{label}_{instance_index}",
                f"{label}_{instance_index}_center",
                canonical_text(obj.get("display_name_bg")),
                canonical_text(obj.get("label")),
                canonical_text(obj.get("role")),
            }

            for synonym in LABEL_SYNONYMS.get(label, []):
                aliases.add(canonical_text(synonym))

            for color in obj.get("colors", []):
                canonical_color = canonical_text(color)
                aliases.add(f"{canonical_color}_{label}")
                for color_word in COLOR_SYNONYMS.get(canonical_color, []):
                    aliases.add(canonical_text(f"{color_word}_{obj.get('label')}"))
                    aliases.add(canonical_text(f"{color_word}_{obj.get('display_name_bg')}"))

            if obj.get("role") in SURFACE_ROLES:
                aliases.update(
                    {
                        "placement_surface",
                        "destination_surface",
                        "място_за_поставяне",
                        "повърхност_за_поставяне",
                    }
                )

            obj["aliases"] = sorted(alias for alias in aliases if alias)

        if objects:
            objects[0]["aliases"].append("nearest_object")
            max(objects, key=lambda obj: float(obj["y"]))["aliases"].append(
                "left_object"
            )
            min(objects, key=lambda obj: float(obj["y"]))["aliases"].append(
                "right_object"
            )

        labels = sorted({str(obj["safe_label"]) for obj in objects})
        for label in labels:
            same = [obj for obj in objects if obj["safe_label"] == label]
            same[0]["aliases"].append(f"nearest_{label}")
            max(same, key=lambda obj: float(obj["y"]))["aliases"].append(
                f"left_{label}"
            )
            min(same, key=lambda obj: float(obj["y"]))["aliases"].append(
                f"right_{label}"
            )

        surfaces = [obj for obj in objects if obj.get("role") in SURFACE_ROLES]
        if surfaces:
            surfaces[0]["aliases"].extend(
                ["nearest_surface", "nearest_destination"]
            )

        for obj in objects:
            obj["aliases"] = sorted(set(obj["aliases"]))

        return objects

    def build_motion_points(
        self,
        objects: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        points: List[Dict[str, Any]] = []

        for obj in objects:
            points.append(
                {
                    "point": obj["point_label"],
                    "point_type": (
                        "placement_surface"
                        if obj.get("role") in SURFACE_ROLES
                        else "object_center"
                    ),
                    "object_id": obj["object_id"],
                    "label": obj["label"],
                    "safe_label": obj["safe_label"],
                    "display_name_bg": obj.get("display_name_bg"),
                    "role": obj.get("role"),
                    "colors": obj.get("colors", []),
                    "instance_index": obj["instance_index"],
                    "global_index": obj["global_index"],
                    "aliases": obj["aliases"],
                    "frame": TARGET_FRAME,
                    "x": obj["x"],
                    "y": obj["y"],
                    "z": obj["z"],
                    "position": obj["center"],
                    "center": obj["center"],
                    "object_position": obj["object_position"],
                    "top_position": obj["top_position"],
                    "grasp_position": obj["grasp_position"],
                    "approach_position": obj["approach_position"],
                    "place_position": obj["place_position"],
                    "destination_position": obj["destination_position"],
                    "dimensions": obj["dimensions"],
                    "raw_dimensions": obj["raw_dimensions"],
                    "width": obj["width"],
                    "height": obj["height"],
                    "depth": obj["depth"],
                    "bbox_3d": obj["bbox_3d"],
                    "bbox_pixels_rgb": obj.get("bbox_pixels_rgb"),
                    "points": obj["points"],
                    "geometry_source": obj["geometry_source"],
                    "geometry_quality": obj["geometry_quality"],
                    "safety_blocker": obj["safety_blocker"],
                    "source": "workspace_scan_agent",
                    "timestamp": obj["timestamp"],
                }
            )

        return {
            "source": "workspace_scan_agent",
            "scan_state": "OK",
            "frame": TARGET_FRAME,
            "motion_points": points,
            "timestamp": time.time(),
        }

    def publish_workspace(
        self,
        objects: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        objects_document = {
            "source": "workspace_scan_agent",
            "scan_state": "OK",
            "frame": TARGET_FRAME,
            "objects": objects,
            "timestamp": time.time(),
        }
        motion_document = self.build_motion_points(objects)

        objects_json = json.dumps(objects_document, ensure_ascii=False)
        motion_json = json.dumps(motion_document, ensure_ascii=False)

        self.objects_pub.publish(String(data=objects_json))
        self.motion_points_pub.publish(String(data=motion_json))
        REDIS.set(WORKSPACE_STATE_KEY, objects_json)
        REDIS.set(MOTION_POINTS_KEY, motion_json)

        self.publish_markers(objects)

        summary = ", ".join(
            f"{obj['point_label']}[{obj['geometry_quality']}:"
            f"{obj['geometry_source']}]"
            for obj in objects
        )
        self.get_logger().info(f"✅ workspace points saved: {summary}")
        return objects_document, motion_document

    # ------------------------------------------------------------------
    # RViz output
    # ------------------------------------------------------------------

    def publish_markers(self, objects: Sequence[Dict[str, Any]]) -> None:
        array = MarkerArray()
        delete = Marker()
        delete.header.frame_id = TARGET_FRAME
        delete.header.stamp = self.get_clock().now().to_msg()
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        marker_id = 1
        for obj in objects:
            array.markers.append(self.make_box_marker(marker_id, obj))
            marker_id += 1
            array.markers.append(self.make_center_marker(marker_id, obj))
            marker_id += 1
            array.markers.append(self.make_text_marker(marker_id, obj))
            marker_id += 1

        self.marker_pub.publish(array)

    def marker_color(self, obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
        if obj.get("safety_blocker"):
            return 1.0, 0.0, 0.0, 0.55
        if obj.get("geometry_quality") == "low":
            return 1.0, 0.45, 0.0, 0.50
        if obj.get("role") in SURFACE_ROLES:
            return 0.1, 1.0, 0.25, 0.38
        return 0.0, 0.85, 1.0, 0.38

    def make_box_marker(self, marker_id: int, obj: Dict[str, Any]) -> Marker:
        marker = Marker()
        marker.header.frame_id = TARGET_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "workspace_object_boxes"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(obj["x"])
        marker.pose.position.y = float(obj["y"])
        marker.pose.position.z = float(obj["z"])
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(obj["dimensions"]["x"])
        marker.scale.y = float(obj["dimensions"]["y"])
        marker.scale.z = float(obj["dimensions"]["z"])
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.marker_color(obj)
        return marker

    def make_center_marker(self, marker_id: int, obj: Dict[str, Any]) -> Marker:
        marker = Marker()
        marker.header.frame_id = TARGET_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "workspace_object_centers"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(obj["x"])
        marker.pose.position.y = float(obj["y"])
        marker.pose.position.z = float(obj["z"])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.024
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        return marker

    def make_text_marker(self, marker_id: int, obj: Dict[str, Any]) -> Marker:
        marker = Marker()
        marker.header.frame_id = TARGET_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "workspace_object_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(obj["x"])
        marker.pose.position.y = float(obj["y"])
        marker.pose.position.z = float(obj["z"]) + max(
            0.055,
            float(obj["dimensions"]["z"]) * 0.8,
        )
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.028
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
        marker.text = (
            f"{obj['display_name_bg']}\n"
            f"{obj['point_label']} | {obj['role']}\n"
            f"{obj['geometry_quality']}:{obj['geometry_source']}\n"
            f"({obj['x']:.3f}, {obj['y']:.3f}, {obj['z']:.3f})\n"
            f"dim=({obj['dimensions']['x']:.3f},"
            f"{obj['dimensions']['y']:.3f},"
            f"{obj['dimensions']['z']:.3f})"
        )
        return marker

    def publish_debug_image(self, image: np.ndarray) -> None:
        try:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_image_pub.publish(msg)
        except Exception as exc:
            self.warn_throttled("debug_image", f"Debug image failed: {exc}")

    def draw_detection(
        self,
        image: np.ndarray,
        detection: Dict[str, Any],
        obj: Dict[str, Any],
        color: Tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = detection["bbox_pixels_rgb"]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = (
            f"{obj['display_name_bg']} "
            f"[{obj['geometry_quality']}:{obj['points']}]"
        )
        cv2.putText(
            image,
            text[:70],
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )

    def publish_selected_clusters(self, points: np.ndarray) -> None:
        if points.shape[0] == 0:
            return
        try:
            header = self.get_clock().now().to_msg()
            from std_msgs.msg import Header

            cloud_header = Header()
            cloud_header.stamp = header
            cloud_header.frame_id = TARGET_FRAME
            msg = pc2.create_cloud_xyz32(cloud_header, points.tolist())
            self.selected_cluster_pub.publish(msg)
        except Exception as exc:
            self.warn_throttled(
                "selected_cloud",
                f"Could not publish selected clusters: {exc}",
            )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = WorkspaceScanAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
