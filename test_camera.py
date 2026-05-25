#!/usr/bin/env python3

import json
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray

from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

from tf2_ros import StaticTransformBroadcaster

import sensor_msgs_py.point_cloud2 as pc2


class PointCloudDigitalTwin3D(Node):
    def __init__(self):
        super().__init__("pointcloud_digital_twin_3d")

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------
        self.declare_parameter("pointcloud_topic", "/aurora/points2")

        # Full live 3D cloud from camera
        self.declare_parameter("live_cloud_topic", "/digital_twin/live_cloud")

        # Object-filtered cloud
        self.declare_parameter("filtered_cloud_topic", "/digital_twin/filtered_cloud")

        self.declare_parameter("marker_topic", "/digital_twin/objects")
        self.declare_parameter("scene_json_topic", "/digital_twin/objects_json")
        self.declare_parameter("planning_scene_topic", "/planning_scene")

        # ---------------------------------------------------------
        # Frames
        # ---------------------------------------------------------
        self.declare_parameter("stand_frame", "stand_base")
        self.declare_parameter("camera_frame", "depth_camera_link")

        self.declare_parameter("camera_x", 0.0)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 0.48)
        self.declare_parameter("camera_tilt_degrees", 45.0)

        self.declare_parameter("image_right_is_negative_y", True)
        self.declare_parameter("publish_camera_static_tf", True)

        # ---------------------------------------------------------
        # Workspace filtering in stand_base
        # ---------------------------------------------------------
        self.declare_parameter("table_z", 0.0)

        # Object filter height.
        self.declare_parameter("min_object_height", 0.008)
        self.declare_parameter("max_object_height", 0.35)

        # Main useful workspace.
        self.declare_parameter("workspace_x_min", -0.10)
        self.declare_parameter("workspace_x_max", 1.00)
        self.declare_parameter("workspace_y_min", -0.60)
        self.declare_parameter("workspace_y_max", 0.60)

        # Live cloud Z range.
        # This is bigger than object filter, so it shows more of the environment.
        self.declare_parameter("live_z_min", -0.20)
        self.declare_parameter("live_z_max", 1.20)

        # Raw camera depth filter.
        self.declare_parameter("min_camera_depth", 0.05)
        self.declare_parameter("max_camera_depth", 2.00)

        # ---------------------------------------------------------
        # 3D ignore box for robot arm/base
        # ---------------------------------------------------------
        self.declare_parameter("use_arm_ignore_box", True)

        self.declare_parameter("arm_ignore_x_min", -0.05)
        self.declare_parameter("arm_ignore_x_max", 0.45)
        self.declare_parameter("arm_ignore_y_min", -0.18)
        self.declare_parameter("arm_ignore_y_max", 0.18)
        self.declare_parameter("arm_ignore_z_min", 0.00)
        self.declare_parameter("arm_ignore_z_max", 0.45)

        # ---------------------------------------------------------
        # Clustering / realtime performance
        # ---------------------------------------------------------
        self.declare_parameter("voxel_size", 0.015)
        self.declare_parameter("cluster_grid_resolution", 0.035)

        self.declare_parameter("min_cluster_points", 30)
        self.declare_parameter("max_cluster_points", 50000)

        self.declare_parameter("min_box_size", 0.025)
        self.declare_parameter("default_box_height", 0.035)

        self.declare_parameter("max_object_size_x", 0.45)
        self.declare_parameter("max_object_size_y", 0.45)
        self.declare_parameter("max_object_size_z", 0.35)

        # ---------------------------------------------------------
        # Publishing
        # ---------------------------------------------------------
        self.declare_parameter("publish_live_cloud", True)
        self.declare_parameter("publish_filtered_cloud", True)

        self.declare_parameter("max_live_points_publish", 50000)
        self.declare_parameter("max_filtered_points_publish", 30000)

        # Keep False while testing realtime.
        self.declare_parameter("publish_moveit_collision", False)

        # 1 = every frame, 2 = every second frame.
        self.declare_parameter("process_every_n_frames", 1)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.live_cloud_topic = self.get_parameter("live_cloud_topic").value
        self.filtered_cloud_topic = self.get_parameter("filtered_cloud_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.scene_json_topic = self.get_parameter("scene_json_topic").value
        self.planning_scene_topic = self.get_parameter("planning_scene_topic").value

        self.stand_frame = self.get_parameter("stand_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.camera_x = float(self.get_parameter("camera_x").value)
        self.camera_y = float(self.get_parameter("camera_y").value)
        self.camera_z = float(self.get_parameter("camera_z").value)
        self.camera_tilt_degrees = float(
            self.get_parameter("camera_tilt_degrees").value
        )

        self.image_right_is_negative_y = bool(
            self.get_parameter("image_right_is_negative_y").value
        )

        self.publish_camera_static_tf_enabled = bool(
            self.get_parameter("publish_camera_static_tf").value
        )

        self.table_z = float(self.get_parameter("table_z").value)
        self.min_object_height = float(self.get_parameter("min_object_height").value)
        self.max_object_height = float(self.get_parameter("max_object_height").value)

        self.workspace_x_min = float(self.get_parameter("workspace_x_min").value)
        self.workspace_x_max = float(self.get_parameter("workspace_x_max").value)
        self.workspace_y_min = float(self.get_parameter("workspace_y_min").value)
        self.workspace_y_max = float(self.get_parameter("workspace_y_max").value)

        self.live_z_min = float(self.get_parameter("live_z_min").value)
        self.live_z_max = float(self.get_parameter("live_z_max").value)

        self.min_camera_depth = float(self.get_parameter("min_camera_depth").value)
        self.max_camera_depth = float(self.get_parameter("max_camera_depth").value)

        self.use_arm_ignore_box = bool(self.get_parameter("use_arm_ignore_box").value)

        self.arm_ignore_x_min = float(self.get_parameter("arm_ignore_x_min").value)
        self.arm_ignore_x_max = float(self.get_parameter("arm_ignore_x_max").value)
        self.arm_ignore_y_min = float(self.get_parameter("arm_ignore_y_min").value)
        self.arm_ignore_y_max = float(self.get_parameter("arm_ignore_y_max").value)
        self.arm_ignore_z_min = float(self.get_parameter("arm_ignore_z_min").value)
        self.arm_ignore_z_max = float(self.get_parameter("arm_ignore_z_max").value)

        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.cluster_grid_resolution = float(
            self.get_parameter("cluster_grid_resolution").value
        )

        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.max_cluster_points = int(self.get_parameter("max_cluster_points").value)

        self.min_box_size = float(self.get_parameter("min_box_size").value)
        self.default_box_height = float(
            self.get_parameter("default_box_height").value
        )

        self.max_object_size_x = float(self.get_parameter("max_object_size_x").value)
        self.max_object_size_y = float(self.get_parameter("max_object_size_y").value)
        self.max_object_size_z = float(self.get_parameter("max_object_size_z").value)

        self.publish_live_cloud_enabled = bool(
            self.get_parameter("publish_live_cloud").value
        )
        self.publish_filtered_cloud_enabled = bool(
            self.get_parameter("publish_filtered_cloud").value
        )

        self.max_live_points_publish = int(
            self.get_parameter("max_live_points_publish").value
        )
        self.max_filtered_points_publish = int(
            self.get_parameter("max_filtered_points_publish").value
        )

        self.publish_moveit_collision = bool(
            self.get_parameter("publish_moveit_collision").value
        )

        self.process_every_n_frames = max(
            1,
            int(self.get_parameter("process_every_n_frames").value),
        )

        # ---------------------------------------------------------
        # ROS setup
        # ---------------------------------------------------------
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        self.live_cloud_pub = self.create_publisher(
            PointCloud2,
            self.live_cloud_topic,
            5,
        )

        self.filtered_cloud_pub = self.create_publisher(
            PointCloud2,
            self.filtered_cloud_topic,
            5,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            10,
        )

        self.scene_json_pub = self.create_publisher(
            String,
            self.scene_json_topic,
            10,
        )

        self.planning_scene_pub = self.create_publisher(
            PlanningScene,
            self.planning_scene_topic,
            10,
        )

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.scene_objects: Dict[str, Dict] = {}
        self.last_warn_times: Dict[str, float] = {}

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processed_frames = 0

        self.last_planning_scene_publish_time = 0.0
        self.planning_scene_publish_period = 0.5

        self.camera_rotation_base_from_optical = self.make_camera_rotation_matrix()
        self.camera_translation_base = np.array(
            [self.camera_x, self.camera_y, self.camera_z],
            dtype=np.float32,
        )

        self.static_tf_published_for_frame: Optional[str] = None

        if self.publish_camera_static_tf_enabled:
            self.publish_camera_static_tf(force=True)

        self.get_logger().info("✅ Realtime Unorganized PointCloud Digital Twin started")
        self.get_logger().info(f"Subscribing to: {self.pointcloud_topic}")
        self.get_logger().info(f"Publishing live cloud: {self.live_cloud_topic}")
        self.get_logger().info(f"Publishing filtered cloud: {self.filtered_cloud_topic}")
        self.get_logger().info(f"Publishing markers: {self.marker_topic}")
        self.get_logger().info(f"RViz Fixed Frame: {self.stand_frame}")
        self.get_logger().info(
            f"Camera mount x={self.camera_x:.3f}, y={self.camera_y:.3f}, "
            f"z={self.camera_z:.3f}, tilt={self.camera_tilt_degrees:.1f}"
        )
        self.get_logger().info(
            f"Realtime settings: voxel_size={self.voxel_size}, "
            f"cluster_grid_resolution={self.cluster_grid_resolution}, "
            f"process_every_n_frames={self.process_every_n_frames}, "
            f"MoveIt collision={self.publish_moveit_collision}"
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def warn_throttled(self, key: str, message: str, period_sec: float = 3.0):
        now = time.time()
        last = self.last_warn_times.get(key, 0.0)

        if now - last >= period_sec:
            self.get_logger().warn(message)
            self.last_warn_times[key] = now

    def log_fps(self):
        self.processed_frames += 1
        now = time.time()

        if now - self.last_fps_time >= 2.0:
            fps = self.processed_frames / (now - self.last_fps_time)
            self.get_logger().info(
                f"Realtime processing: {fps:.1f} FPS, objects={len(self.scene_objects)}"
            )
            self.last_fps_time = now
            self.processed_frames = 0

    # -------------------------------------------------------------------------
    # Camera transform
    # -------------------------------------------------------------------------

    def make_camera_rotation_matrix(self) -> np.ndarray:
        """
        stand_base:
          +X forward
          +Y left
          +Z up

        camera optical:
          +X image right
          +Y image down
          +Z depth forward
        """
        tilt = math.radians(self.camera_tilt_degrees)

        optical_z = np.array(
            [math.cos(tilt), 0.0, -math.sin(tilt)],
            dtype=np.float32,
        )

        if self.image_right_is_negative_y:
            optical_x = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            optical_x = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        optical_y = np.cross(optical_z, optical_x)
        norm_y = np.linalg.norm(optical_y)

        if norm_y < 1e-6:
            optical_y = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            optical_y = optical_y / norm_y

        optical_x = optical_x / np.linalg.norm(optical_x)
        optical_z = optical_z / np.linalg.norm(optical_z)

        rotation = np.column_stack((optical_x, optical_y, optical_z))
        return rotation.astype(np.float32)

    def rotation_matrix_to_quaternion(
        self,
        rotation: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        m = rotation
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

        return float(qx), float(qy), float(qz), float(qw)

    def publish_camera_static_tf(self, force: bool = False):
        if not force and self.static_tf_published_for_frame == self.camera_frame:
            return

        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(
            self.camera_rotation_base_from_optical
        )

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.stand_frame
        tf_msg.child_frame_id = self.camera_frame

        tf_msg.transform.translation.x = float(self.camera_x)
        tf_msg.transform.translation.y = float(self.camera_y)
        tf_msg.transform.translation.z = float(self.camera_z)

        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.static_tf_broadcaster.sendTransform(tf_msg)
        self.static_tf_published_for_frame = self.camera_frame

    # -------------------------------------------------------------------------
    # Point cloud callback
    # -------------------------------------------------------------------------

    def pointcloud_callback(self, msg: PointCloud2):
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0:
            return

        if msg.header.frame_id and msg.header.frame_id != self.camera_frame:
            self.camera_frame = msg.header.frame_id

            if self.publish_camera_static_tf_enabled:
                self.publish_camera_static_tf(force=True)

        points_cam = self.read_xyz_points(msg)

        if points_cam is None or points_cam.shape[0] == 0:
            self.scene_objects = {}

            if self.publish_live_cloud_enabled:
                self.publish_empty_cloud_to_topic(self.live_cloud_pub)

            if self.publish_filtered_cloud_enabled:
                self.publish_empty_cloud_to_topic(self.filtered_cloud_pub)

            self.publish_markers()
            self.publish_scene_json()

            self.warn_throttled(
                "no_points",
                "No valid points read from point cloud.",
                3.0,
            )
            return

        points_base = self.transform_points_to_base(points_cam)

        # ---------------------------------------------------------
        # LIVE CLOUD: everything camera sees in useful workspace
        # ---------------------------------------------------------
        live_points = self.filter_live_workspace(points_base)
        live_points = self.fast_voxel_downsample(live_points, self.voxel_size)

        if self.publish_live_cloud_enabled:
            self.publish_cloud_to_topic(
                live_points,
                self.live_cloud_pub,
                self.max_live_points_publish,
            )

        # ---------------------------------------------------------
        # FILTERED CLOUD: only object/pickable points
        # ---------------------------------------------------------
        filtered_points = self.filter_object_workspace(points_base)

        if filtered_points.shape[0] == 0:
            self.scene_objects = {}

            if self.publish_filtered_cloud_enabled:
                self.publish_empty_cloud_to_topic(self.filtered_cloud_pub)

            self.publish_markers()
            self.publish_scene_json()
            self.log_fps()
            return

        filtered_points = self.fast_voxel_downsample(filtered_points, self.voxel_size)

        objects = self.cluster_points_xy(filtered_points)

        self.update_scene_objects(objects)

        if self.publish_filtered_cloud_enabled:
            self.publish_cloud_to_topic(
                filtered_points,
                self.filtered_cloud_pub,
                self.max_filtered_points_publish,
            )

        self.publish_markers()
        self.publish_scene_json()

        if self.publish_moveit_collision:
            self.publish_planning_scene_throttled()

        self.log_fps()

    # -------------------------------------------------------------------------
    # Point cloud reading
    # -------------------------------------------------------------------------

    def read_xyz_points(self, msg: PointCloud2) -> Optional[np.ndarray]:
        """
        Reads Aurora /aurora/points2.

        Your cloud:
          height: 1
          width: ~169838
          fields: x y z
          datatype: FLOAT32
          point_step: 16
        """
        if msg.width == 0 or msg.height == 0:
            return None

        field_map = {field.name: field for field in msg.fields}

        if "x" not in field_map or "y" not in field_map or "z" not in field_map:
            self.warn_throttled(
                "missing_xyz",
                f"Cloud fields missing x/y/z: {[f.name for f in msg.fields]}",
                3.0,
            )
            return None

        x_field = field_map["x"]
        y_field = field_map["y"]
        z_field = field_map["z"]

        if (
            x_field.datatype != PointField.FLOAT32
            or y_field.datatype != PointField.FLOAT32
            or z_field.datatype != PointField.FLOAT32
        ):
            self.warn_throttled(
                "bad_xyz_type",
                "PointCloud2 x/y/z fields are not FLOAT32.",
                3.0,
            )
            return None

        endian = ">" if msg.is_bigendian else "<"

        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [endian + "f4", endian + "f4", endian + "f4"],
            "offsets": [
                x_field.offset,
                y_field.offset,
                z_field.offset,
            ],
            "itemsize": msg.point_step,
        })

        try:
            raw = np.frombuffer(msg.data, dtype=dtype)
            expected = msg.width * msg.height

            if raw.shape[0] < expected:
                self.warn_throttled(
                    "cloud_short",
                    f"Cloud shorter than expected: got {raw.shape[0]}, expected {expected}",
                    3.0,
                )

            raw = raw[:expected]

            points = np.empty((raw.shape[0], 3), dtype=np.float32)
            points[:, 0] = raw["x"]
            points[:, 1] = raw["y"]
            points[:, 2] = raw["z"]

            valid = (
                np.isfinite(points[:, 0])
                & np.isfinite(points[:, 1])
                & np.isfinite(points[:, 2])
                & (points[:, 2] > self.min_camera_depth)
                & (points[:, 2] < self.max_camera_depth)
            )

            return points[valid]

        except Exception as e:
            self.warn_throttled(
                "read_failed",
                f"Manual point cloud read failed: {e}",
                3.0,
            )
            return None

    # -------------------------------------------------------------------------
    # Geometry processing
    # -------------------------------------------------------------------------

    def transform_points_to_base(self, points_cam: np.ndarray) -> np.ndarray:
        points_base = points_cam @ self.camera_rotation_base_from_optical.T
        points_base = points_base + self.camera_translation_base
        return points_base.astype(np.float32)

    def filter_live_workspace(self, points_base: np.ndarray) -> np.ndarray:
        """
        Full live camera cloud for RViz.
        This is NOT object detection.
        It shows everything the camera sees inside the useful workspace.
        """
        x = points_base[:, 0]
        y = points_base[:, 1]
        z = points_base[:, 2]

        mask = (
            (x >= self.workspace_x_min)
            & (x <= self.workspace_x_max)
            & (y >= self.workspace_y_min)
            & (y <= self.workspace_y_max)
            & (z >= self.live_z_min)
            & (z <= self.live_z_max)
        )

        return points_base[mask]

    def filter_object_workspace(self, points_base: np.ndarray) -> np.ndarray:
        """
        Object cloud for detection.
        This removes table/background by height and optionally removes robot arm/base.
        """
        x = points_base[:, 0]
        y = points_base[:, 1]
        z = points_base[:, 2]

        height_min = self.table_z + self.min_object_height
        height_max = self.table_z + self.max_object_height

        mask = (
            (z >= height_min)
            & (z <= height_max)
            & (x >= self.workspace_x_min)
            & (x <= self.workspace_x_max)
            & (y >= self.workspace_y_min)
            & (y <= self.workspace_y_max)
        )

        if self.use_arm_ignore_box:
            arm_mask = (
                (x >= self.arm_ignore_x_min)
                & (x <= self.arm_ignore_x_max)
                & (y >= self.arm_ignore_y_min)
                & (y <= self.arm_ignore_y_max)
                & (z >= self.arm_ignore_z_min)
                & (z <= self.arm_ignore_z_max)
            )

            mask = mask & (~arm_mask)

        return points_base[mask]

    def fast_voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Keeps the first point per voxel.
        """
        if points.shape[0] == 0 or voxel_size <= 0:
            return points

        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        ix = voxel_indices[:, 0].astype(np.int64) + 100000
        iy = voxel_indices[:, 1].astype(np.int64) + 100000
        iz = voxel_indices[:, 2].astype(np.int64) + 100000

        keys = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791

        _, unique_indices = np.unique(keys, return_index=True)

        return points[unique_indices]

    def cluster_points_xy(self, points: np.ndarray) -> List[Dict]:
        if points.shape[0] == 0:
            return []

        resolution = self.cluster_grid_resolution

        x_min = float(np.min(points[:, 0]))
        y_min = float(np.min(points[:, 1]))

        grid_x = np.floor((points[:, 0] - x_min) / resolution).astype(np.int32)
        grid_y = np.floor((points[:, 1] - y_min) / resolution).astype(np.int32)

        width = int(np.max(grid_x)) + 1
        height = int(np.max(grid_y)) + 1

        if width <= 0 or height <= 0 or width > 1000 or height > 1000:
            self.warn_throttled(
                "grid_size_bad",
                f"Bad cluster grid size: {width}x{height}",
                3.0,
            )
            return []

        occupancy = np.zeros((height, width), dtype=np.uint8)
        occupancy[grid_y, grid_x] = 255

        kernel = np.ones((3, 3), dtype=np.uint8)
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)

        num_labels, labels = cv2.connectedComponents(
            occupancy,
            connectivity=8,
        )

        point_labels = labels[grid_y, grid_x]

        objects = []

        for label_id in range(1, num_labels):
            cluster_points = points[point_labels == label_id]

            if cluster_points.shape[0] < self.min_cluster_points:
                continue

            if cluster_points.shape[0] > self.max_cluster_points:
                continue

            obj = self.points_to_object(len(objects), cluster_points)

            if obj is not None:
                objects.append(obj)

        return objects

    def points_to_object(
        self,
        object_index: int,
        points: np.ndarray,
    ) -> Optional[Dict]:
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        obj_x = float(np.median(xs))
        obj_y = float(np.median(ys))
        obj_z = float(np.median(zs))

        size_x = float(np.percentile(xs, 95) - np.percentile(xs, 5))
        size_y = float(np.percentile(ys, 95) - np.percentile(ys, 5))
        size_z = float(np.percentile(zs, 95) - np.percentile(zs, 5))

        size_x = max(size_x, self.min_box_size)
        size_y = max(size_y, self.min_box_size)
        size_z = max(size_z, self.default_box_height)

        if size_x > self.max_object_size_x:
            return None

        if size_y > self.max_object_size_y:
            return None

        if size_z > self.max_object_size_z:
            return None

        name, category, colors = self.fallback_classify(size_x, size_y, size_z)

        return {
            "id": f"object_{object_index}",
            "name": name,
            "category": category,
            "colors": colors,
            "graspable": True,
            "confidence": 0.50,
            "frame": self.stand_frame,
            "x": obj_x,
            "y": obj_y,
            "z": obj_z,
            "width": size_x,
            "height": size_y,
            "depth": size_z,
            "points": int(points.shape[0]),
            "last_seen": time.time(),
        }

    def fallback_classify(
        self,
        size_x: float,
        size_y: float,
        size_z: float,
    ) -> Tuple[str, str, List[str]]:
        xy_ratio = max(size_x, size_y) / max(min(size_x, size_y), 0.001)

        if size_x < 0.09 and size_y < 0.09 and size_z > 0.020 and xy_ratio < 2.0:
            return "object cube", "cube", []

        if size_x > 0.12 or size_y > 0.12:
            return "long object", "tool_or_long_object", []

        return "unknown object", "object", []

    # -------------------------------------------------------------------------
    # Scene + publishing
    # -------------------------------------------------------------------------

    def update_scene_objects(self, detected_objects: List[Dict]):
        self.scene_objects = {}

        for obj in detected_objects:
            self.scene_objects[obj["id"]] = obj

    def publish_empty_cloud_to_topic(self, publisher):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.stand_frame

        msg = pc2.create_cloud_xyz32(header, [])
        publisher.publish(msg)

    def publish_cloud_to_topic(self, points: np.ndarray, publisher, max_points: int):
        if points.shape[0] == 0:
            self.publish_empty_cloud_to_topic(publisher)
            return

        publish_points = points

        if publish_points.shape[0] > max_points:
            indices = np.linspace(
                0,
                publish_points.shape[0] - 1,
                max_points,
                dtype=np.int32,
            )
            publish_points = publish_points[indices]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.stand_frame

        msg = pc2.create_cloud_xyz32(header, publish_points.tolist())
        publisher.publish(msg)

    def publish_markers(self):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.stand_frame
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for obj in self.scene_objects.values():
            marker_id = self.safe_marker_id(obj["id"])

            marker = Marker()
            marker.header.frame_id = obj["frame"]
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "digital_twin_objects"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = obj["x"]
            marker.pose.position.y = obj["y"]
            marker.pose.position.z = obj["z"]
            marker.pose.orientation.w = 1.0

            marker.scale.x = max(float(obj["width"]), self.min_box_size)
            marker.scale.y = max(float(obj["height"]), self.min_box_size)
            marker.scale.z = max(float(obj["depth"]), self.min_box_size)

            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker.color.a = 0.75

            marker.lifetime.sec = 0
            marker_array.markers.append(marker)

            text = Marker()
            text.header.frame_id = obj["frame"]
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "digital_twin_labels"
            text.id = marker_id + 100000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = obj["x"]
            text.pose.position.y = obj["y"]
            text.pose.position.z = obj["z"] + 0.08
            text.pose.orientation.w = 1.0

            text.scale.z = 0.04
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = obj["name"]
            text.lifetime.sec = 0

            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    def publish_scene_json(self):
        objects_json = []

        for obj in self.scene_objects.values():
            objects_json.append({
                "id": obj["id"],
                "name": obj["name"],
                "category": obj["category"],
                "colors": obj["colors"],
                "graspable": obj["graspable"],
                "confidence": obj["confidence"],
                "frame": obj["frame"],
                "x": obj["x"],
                "y": obj["y"],
                "z": obj["z"],
                "width": obj["width"],
                "height": obj["height"],
                "depth": obj["depth"],
                "points": obj["points"],
                "last_seen": obj["last_seen"],
            })

        msg = String()
        msg.data = json.dumps({
            "frame": self.stand_frame,
            "objects": objects_json,
            "timestamp": time.time(),
        })

        self.scene_json_pub.publish(msg)

    def publish_planning_scene_throttled(self):
        now = time.time()

        if now - self.last_planning_scene_publish_time < self.planning_scene_publish_period:
            return

        self.last_planning_scene_publish_time = now
        self.publish_planning_scene()

    def publish_planning_scene(self):
        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        for obj in self.scene_objects.values():
            col_obj = CollisionObject()
            col_obj.header.frame_id = obj["frame"]
            col_obj.header.stamp = self.get_clock().now().to_msg()
            col_obj.id = obj["id"]

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [
                max(float(obj["width"]), self.min_box_size),
                max(float(obj["height"]), self.min_box_size),
                max(float(obj["depth"]), self.min_box_size),
            ]

            pose = Pose()
            pose.position.x = obj["x"]
            pose.position.y = obj["y"]
            pose.position.z = obj["z"]
            pose.orientation.w = 1.0

            col_obj.primitives.append(primitive)
            col_obj.primitive_poses.append(pose)
            col_obj.operation = CollisionObject.ADD

            planning_scene.world.collision_objects.append(col_obj)

        self.planning_scene_pub.publish(planning_scene)

    def safe_marker_id(self, text: str) -> int:
        return abs(hash(text)) % 90000


def main(args=None):
    rclpy.init(args=args)

    node = PointCloudDigitalTwin3D()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()