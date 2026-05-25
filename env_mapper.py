#!/usr/bin/env python3

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped

from tf2_ros import StaticTransformBroadcaster

import sensor_msgs_py.point_cloud2 as pc2


class EnvironmentMapper(Node):
    def __init__(self):
        super().__init__("environment_mapper")

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------
        self.declare_parameter("pointcloud_topic", "/aurora/points2")
        self.declare_parameter("live_cloud_topic", "/digital_twin/environment_cloud")
        self.declare_parameter("map_cloud_topic", "/digital_twin/environment_map")

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
        # What part of world to keep
        # ---------------------------------------------------------
        self.declare_parameter("min_camera_depth", 0.05)
        self.declare_parameter("max_camera_depth", 2.50)

        # Big workspace, includes table + objects + arm area.
        self.declare_parameter("workspace_x_min", -0.50)
        self.declare_parameter("workspace_x_max", 1.50)
        self.declare_parameter("workspace_y_min", -1.00)
        self.declare_parameter("workspace_y_max", 1.00)
        self.declare_parameter("workspace_z_min", -0.20)
        self.declare_parameter("workspace_z_max", 1.20)

        # Downsampling.
        # Smaller = more detailed but heavier.
        self.declare_parameter("live_voxel_size", 0.008)
        self.declare_parameter("map_voxel_size", 0.012)

        # Limit publishing size.
        self.declare_parameter("max_live_points", 60000)
        self.declare_parameter("max_map_points", 120000)

        # Accumulated map memory.
        self.declare_parameter("enable_accumulated_map", True)
        self.declare_parameter("map_publish_period", 0.5)

        # If True, clears old map every few seconds. Useful while testing.
        self.declare_parameter("auto_clear_map", False)
        self.declare_parameter("auto_clear_period", 10.0)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.live_cloud_topic = self.get_parameter("live_cloud_topic").value
        self.map_cloud_topic = self.get_parameter("map_cloud_topic").value

        self.stand_frame = self.get_parameter("stand_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.camera_x = float(self.get_parameter("camera_x").value)
        self.camera_y = float(self.get_parameter("camera_y").value)
        self.camera_z = float(self.get_parameter("camera_z").value)
        self.camera_tilt_degrees = float(self.get_parameter("camera_tilt_degrees").value)

        self.image_right_is_negative_y = bool(
            self.get_parameter("image_right_is_negative_y").value
        )
        self.publish_camera_static_tf_enabled = bool(
            self.get_parameter("publish_camera_static_tf").value
        )

        self.min_camera_depth = float(self.get_parameter("min_camera_depth").value)
        self.max_camera_depth = float(self.get_parameter("max_camera_depth").value)

        self.workspace_x_min = float(self.get_parameter("workspace_x_min").value)
        self.workspace_x_max = float(self.get_parameter("workspace_x_max").value)
        self.workspace_y_min = float(self.get_parameter("workspace_y_min").value)
        self.workspace_y_max = float(self.get_parameter("workspace_y_max").value)
        self.workspace_z_min = float(self.get_parameter("workspace_z_min").value)
        self.workspace_z_max = float(self.get_parameter("workspace_z_max").value)

        self.live_voxel_size = float(self.get_parameter("live_voxel_size").value)
        self.map_voxel_size = float(self.get_parameter("map_voxel_size").value)

        self.max_live_points = int(self.get_parameter("max_live_points").value)
        self.max_map_points = int(self.get_parameter("max_map_points").value)

        self.enable_accumulated_map = bool(
            self.get_parameter("enable_accumulated_map").value
        )
        self.map_publish_period = float(self.get_parameter("map_publish_period").value)

        self.auto_clear_map = bool(self.get_parameter("auto_clear_map").value)
        self.auto_clear_period = float(self.get_parameter("auto_clear_period").value)

        # ---------------------------------------------------------
        # ROS
        # ---------------------------------------------------------
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            5,
        )

        self.live_cloud_pub = self.create_publisher(
            PointCloud2,
            self.live_cloud_topic,
            5,
        )

        self.map_cloud_pub = self.create_publisher(
            PointCloud2,
            self.map_cloud_topic,
            5,
        )

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.last_warn_times: Dict[str, float] = {}
        self.last_map_publish_time = 0.0
        self.last_map_clear_time = time.time()

        self.map_voxels = {}

        self.camera_rotation_base_from_optical = self.make_camera_rotation_matrix()
        self.camera_translation_base = np.array(
            [self.camera_x, self.camera_y, self.camera_z],
            dtype=np.float32,
        )

        self.static_tf_published_for_frame: Optional[str] = None

        if self.publish_camera_static_tf_enabled:
            self.publish_camera_static_tf(force=True)

        self.get_logger().info("✅ Environment mapper started")
        self.get_logger().info(f"Input cloud: {self.pointcloud_topic}")
        self.get_logger().info(f"Live cloud:  {self.live_cloud_topic}")
        self.get_logger().info(f"Map cloud:   {self.map_cloud_topic}")
        self.get_logger().info(f"RViz Fixed Frame: {self.stand_frame}")

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def warn_throttled(self, key: str, message: str, period_sec: float = 3.0):
        now = time.time()
        last = self.last_warn_times.get(key, 0.0)

        if now - last >= period_sec:
            self.get_logger().warn(message)
            self.last_warn_times[key] = now

    # -------------------------------------------------------------------------
    # Camera transform
    # -------------------------------------------------------------------------

    def make_camera_rotation_matrix(self) -> np.ndarray:
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
    # Main cloud callback
    # -------------------------------------------------------------------------

    def pointcloud_callback(self, msg: PointCloud2):
        if msg.header.frame_id and msg.header.frame_id != self.camera_frame:
            self.camera_frame = msg.header.frame_id

            if self.publish_camera_static_tf_enabled:
                self.publish_camera_static_tf(force=True)

        points_cam = self.read_xyz_points(msg)

        if points_cam is None or points_cam.shape[0] == 0:
            self.warn_throttled("no_points", "No points read from camera cloud.")
            return

        points_base = self.transform_points_to_base(points_cam)
        points_base = self.filter_workspace(points_cam, points_base)

        if points_base.shape[0] == 0:
            return

        live_points = self.voxel_downsample(points_base, self.live_voxel_size)

        self.publish_cloud(
            live_points,
            self.live_cloud_pub,
            self.live_cloud_topic,
            self.max_live_points,
        )

        if self.enable_accumulated_map:
            self.update_accumulated_map(live_points)

            now = time.time()

            if self.auto_clear_map and now - self.last_map_clear_time > self.auto_clear_period:
                self.map_voxels = {}
                self.last_map_clear_time = now
                self.get_logger().info("🧹 Auto-cleared environment map")

            if now - self.last_map_publish_time >= self.map_publish_period:
                self.last_map_publish_time = now
                map_points = self.get_map_points()

                self.publish_cloud(
                    map_points,
                    self.map_cloud_pub,
                    self.map_cloud_topic,
                    self.max_map_points,
                )

    # -------------------------------------------------------------------------
    # Read PointCloud2
    # -------------------------------------------------------------------------

    def read_xyz_points(self, msg: PointCloud2) -> Optional[np.ndarray]:
        if msg.width == 0 or msg.height == 0:
            return None

        field_map = {field.name: field for field in msg.fields}

        if "x" not in field_map or "y" not in field_map or "z" not in field_map:
            self.warn_throttled(
                "missing_xyz",
                f"Cloud fields missing x/y/z: {[f.name for f in msg.fields]}",
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
                "Cloud x/y/z fields are not FLOAT32.",
            )
            return None

        endian = ">" if msg.is_bigendian else "<"

        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [endian + "f4", endian + "f4", endian + "f4"],
            "offsets": [x_field.offset, y_field.offset, z_field.offset],
            "itemsize": msg.point_step,
        })

        try:
            raw = np.frombuffer(msg.data, dtype=dtype)
            expected = msg.width * msg.height
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
            self.warn_throttled("read_failed", f"Point cloud read failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    def transform_points_to_base(self, points_cam: np.ndarray) -> np.ndarray:
        points_base = points_cam @ self.camera_rotation_base_from_optical.T
        points_base = points_base + self.camera_translation_base
        return points_base.astype(np.float32)

    def filter_workspace(
        self,
        points_cam: np.ndarray,
        points_base: np.ndarray,
    ) -> np.ndarray:
        x = points_base[:, 0]
        y = points_base[:, 1]
        z = points_base[:, 2]

        mask = (
            (x >= self.workspace_x_min)
            & (x <= self.workspace_x_max)
            & (y >= self.workspace_y_min)
            & (y <= self.workspace_y_max)
            & (z >= self.workspace_z_min)
            & (z <= self.workspace_z_max)
        )

        return points_base[mask]

    def voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        if points.shape[0] == 0 or voxel_size <= 0:
            return points

        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        _, unique_indices = np.unique(
            voxel_indices,
            axis=0,
            return_index=True,
        )

        return points[unique_indices]

    # -------------------------------------------------------------------------
    # Accumulated map
    # -------------------------------------------------------------------------

    def update_accumulated_map(self, points: np.ndarray):
        if points.shape[0] == 0:
            return

        voxel_indices = np.floor(points / self.map_voxel_size).astype(np.int32)

        for idx, point in zip(voxel_indices, points):
            key = (int(idx[0]), int(idx[1]), int(idx[2]))
            self.map_voxels[key] = point

        if len(self.map_voxels) > self.max_map_points * 2:
            # Keep a bounded map size.
            keys = list(self.map_voxels.keys())
            keep_keys = keys[-self.max_map_points:]

            self.map_voxels = {
                key: self.map_voxels[key]
                for key in keep_keys
            }

    def get_map_points(self) -> np.ndarray:
        if not self.map_voxels:
            return np.empty((0, 3), dtype=np.float32)

        points = np.array(list(self.map_voxels.values()), dtype=np.float32)

        if points.shape[0] > self.max_map_points:
            indices = np.linspace(
                0,
                points.shape[0] - 1,
                self.max_map_points,
                dtype=np.int32,
            )
            points = points[indices]

        return points

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def publish_cloud(
        self,
        points: np.ndarray,
        publisher,
        topic_name: str,
        max_points: int,
    ):
        if points.shape[0] == 0:
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


def main(args=None):
    rclpy.init(args=args)

    node = EnvironmentMapper()

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