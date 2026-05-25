#!/usr/bin/env python3

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from geometry_msgs.msg import TransformStamped

from tf2_ros import StaticTransformBroadcaster

import sensor_msgs_py.point_cloud2 as pc2


class ColorizedPointCloud(Node):
    def __init__(self):
        super().__init__("colorized_pointcloud")

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------
        self.declare_parameter("pointcloud_topic", "/aurora/points2")
        self.declare_parameter("rgb_topic", "/aurora/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/aurora/rgb/camera_info")

        self.declare_parameter("colorized_cloud_topic", "/digital_twin/live_cloud_rgb")

        # ---------------------------------------------------------
        # Frames
        # ---------------------------------------------------------
        self.declare_parameter("stand_frame", "base_link")
        self.declare_parameter("camera_frame", "depth_camera_link")

        # Camera position relative to stand_frame/base_link.
        self.declare_parameter("camera_x", -0.188)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 0.48)

        # Camera angle.
        self.declare_parameter("camera_tilt_degrees", 45.0)

        # NEW: rotate whole map around Z axis of base_link/world.
        # Try 0, 90, -90, 180 until mapping lines up with robot.
        self.declare_parameter("camera_yaw_degrees", 0.0)

        self.declare_parameter("image_right_is_negative_y", True)
        self.declare_parameter("publish_camera_static_tf", True)

        # ---------------------------------------------------------
        # Projection tuning
        # ---------------------------------------------------------
        self.declare_parameter("projection_u_offset", 0.0)
        self.declare_parameter("projection_v_offset", 0.0)

        self.declare_parameter("flip_projection_x", False)
        self.declare_parameter("flip_projection_y", False)

        # ---------------------------------------------------------
        # Filtering
        # ---------------------------------------------------------
        self.declare_parameter("min_camera_depth", 0.05)
        self.declare_parameter("max_camera_depth", 2.50)

        self.declare_parameter("workspace_x_min", -0.80)
        self.declare_parameter("workspace_x_max", 1.50)
        self.declare_parameter("workspace_y_min", -1.00)
        self.declare_parameter("workspace_y_max", 1.00)
        self.declare_parameter("workspace_z_min", -0.30)
        self.declare_parameter("workspace_z_max", 1.30)

        # Bigger = faster, less detail.
        self.declare_parameter("voxel_size", 0.008)
        self.declare_parameter("max_points_publish", 80000)

        self.declare_parameter("process_every_n_frames", 1)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.colorized_cloud_topic = self.get_parameter("colorized_cloud_topic").value

        self.stand_frame = self.get_parameter("stand_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.camera_x = float(self.get_parameter("camera_x").value)
        self.camera_y = float(self.get_parameter("camera_y").value)
        self.camera_z = float(self.get_parameter("camera_z").value)

        self.camera_tilt_degrees = float(
            self.get_parameter("camera_tilt_degrees").value
        )
        self.camera_yaw_degrees = float(
            self.get_parameter("camera_yaw_degrees").value
        )

        self.image_right_is_negative_y = bool(
            self.get_parameter("image_right_is_negative_y").value
        )
        self.publish_camera_static_tf_enabled = bool(
            self.get_parameter("publish_camera_static_tf").value
        )

        self.projection_u_offset = float(
            self.get_parameter("projection_u_offset").value
        )
        self.projection_v_offset = float(
            self.get_parameter("projection_v_offset").value
        )
        self.flip_projection_x = bool(self.get_parameter("flip_projection_x").value)
        self.flip_projection_y = bool(self.get_parameter("flip_projection_y").value)

        self.min_camera_depth = float(self.get_parameter("min_camera_depth").value)
        self.max_camera_depth = float(self.get_parameter("max_camera_depth").value)

        self.workspace_x_min = float(self.get_parameter("workspace_x_min").value)
        self.workspace_x_max = float(self.get_parameter("workspace_x_max").value)
        self.workspace_y_min = float(self.get_parameter("workspace_y_min").value)
        self.workspace_y_max = float(self.get_parameter("workspace_y_max").value)
        self.workspace_z_min = float(self.get_parameter("workspace_z_min").value)
        self.workspace_z_max = float(self.get_parameter("workspace_z_max").value)

        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.max_points_publish = int(self.get_parameter("max_points_publish").value)
        self.process_every_n_frames = max(
            1,
            int(self.get_parameter("process_every_n_frames").value),
        )

        # ---------------------------------------------------------
        # ROS
        # ---------------------------------------------------------
        self.bridge = CvBridge()

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.colorized_cloud_topic,
            5,
        )

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.latest_rgb: Optional[np.ndarray] = None

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        self.frame_count = 0
        self.last_warn_times: Dict[str, float] = {}

        self.last_fps_time = time.time()
        self.processed_frames = 0

        self.camera_rotation_base_from_optical = self.make_camera_rotation_matrix()
        self.camera_translation_base = np.array(
            [self.camera_x, self.camera_y, self.camera_z],
            dtype=np.float32,
        )

        self.static_tf_published_for_frame: Optional[str] = None

        if self.publish_camera_static_tf_enabled:
            self.publish_camera_static_tf(force=True)

        self.get_logger().info("✅ Colorized PointCloud node started")
        self.get_logger().info(f"PointCloud input: {self.pointcloud_topic}")
        self.get_logger().info(f"RGB input:        {self.rgb_topic}")
        self.get_logger().info(f"CameraInfo input: {self.camera_info_topic}")
        self.get_logger().info(f"Output cloud:     {self.colorized_cloud_topic}")
        self.get_logger().info(f"RViz Fixed Frame: {self.stand_frame}")
        self.get_logger().info(
            f"Camera transform: x={self.camera_x:.3f}, y={self.camera_y:.3f}, "
            f"z={self.camera_z:.3f}, tilt={self.camera_tilt_degrees:.1f}, "
            f"yaw={self.camera_yaw_degrees:.1f}"
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
            self.get_logger().info(f"Colorized cloud FPS: {fps:.1f}")
            self.last_fps_time = now
            self.processed_frames = 0

    # -------------------------------------------------------------------------
    # Camera transform
    # -------------------------------------------------------------------------

    def make_camera_rotation_matrix(self) -> np.ndarray:
        """
        stand_frame/base_link:
          +X forward from robot
          +Y left from robot
          +Z up

        camera optical:
          +X image right
          +Y image down
          +Z depth forward

        camera_yaw_degrees rotates the whole mapping around stand_frame Z.
        """

        tilt = math.radians(self.camera_tilt_degrees)
        yaw = math.radians(self.camera_yaw_degrees)

        # Camera forward/depth direction before yaw.
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

        rotation_no_yaw = np.column_stack((optical_x, optical_y, optical_z))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        yaw_rotation = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw,  cos_yaw, 0.0],
                [0.0,      0.0,     1.0],
            ],
            dtype=np.float32,
        )

        rotation = yaw_rotation @ rotation_no_yaw

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
    # ROS callbacks
    # -------------------------------------------------------------------------

    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.warn_throttled("rgb_error", f"RGB conversion failed: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def pointcloud_callback(self, msg: PointCloud2):
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames != 0:
            return

        if self.latest_rgb is None:
            self.warn_throttled("no_rgb", "Waiting for RGB image...")
            return

        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.warn_throttled("no_camera_info", "Waiting for RGB CameraInfo...")
            return

        if msg.header.frame_id and msg.header.frame_id != self.camera_frame:
            self.camera_frame = msg.header.frame_id

            if self.publish_camera_static_tf_enabled:
                self.publish_camera_static_tf(force=True)

        points_cam = self.read_xyz_points(msg)

        if points_cam is None or points_cam.shape[0] == 0:
            self.publish_empty_cloud()
            return

        points_cam_colored, colors_rgb = self.colorize_points_from_rgb(points_cam)

        if points_cam_colored.shape[0] == 0:
            self.publish_empty_cloud()
            return

        points_base = self.transform_points_to_base(points_cam_colored)

        points_base, colors_rgb = self.filter_workspace(points_base, colors_rgb)

        if points_base.shape[0] == 0:
            self.publish_empty_cloud()
            return

        points_base, colors_rgb = self.fast_voxel_downsample_with_colors(
            points_base,
            colors_rgb,
            self.voxel_size,
        )

        self.publish_colorized_cloud(points_base, colors_rgb)
        self.log_fps()

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
                "PointCloud2 x/y/z fields are not FLOAT32.",
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
            self.warn_throttled("read_failed", f"PointCloud read failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Colorization
    # -------------------------------------------------------------------------

    def colorize_points_from_rgb(
        self,
        points_cam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rgb = self.latest_rgb

        img_h, img_w = rgb.shape[:2]

        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]

        u = (self.fx * x / z) + self.cx + self.projection_u_offset
        v = (self.fy * y / z) + self.cy + self.projection_v_offset

        if self.flip_projection_x:
            u = (img_w - 1) - u

        if self.flip_projection_y:
            v = (img_h - 1) - v

        u_i = np.round(u).astype(np.int32)
        v_i = np.round(v).astype(np.int32)

        valid = (
            (u_i >= 0)
            & (u_i < img_w)
            & (v_i >= 0)
            & (v_i < img_h)
            & np.isfinite(u)
            & np.isfinite(v)
        )

        if not np.any(valid):
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )

        points_valid = points_cam[valid]
        u_valid = u_i[valid]
        v_valid = v_i[valid]

        # OpenCV image is BGR. Convert to RGB.
        bgr = rgb[v_valid, u_valid]
        colors_rgb = bgr[:, ::-1].astype(np.uint8)

        return points_valid, colors_rgb

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    def transform_points_to_base(self, points_cam: np.ndarray) -> np.ndarray:
        points_base = points_cam @ self.camera_rotation_base_from_optical.T
        points_base = points_base + self.camera_translation_base
        return points_base.astype(np.float32)

    def filter_workspace(
        self,
        points_base: np.ndarray,
        colors_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        return points_base[mask], colors_rgb[mask]

    def fast_voxel_downsample_with_colors(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if points.shape[0] == 0 or voxel_size <= 0:
            return points, colors

        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        ix = voxel_indices[:, 0].astype(np.int64) + 100000
        iy = voxel_indices[:, 1].astype(np.int64) + 100000
        iz = voxel_indices[:, 2].astype(np.int64) + 100000

        keys = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791

        _, unique_indices = np.unique(keys, return_index=True)

        points_ds = points[unique_indices]
        colors_ds = colors[unique_indices]

        if points_ds.shape[0] > self.max_points_publish:
            indices = np.linspace(
                0,
                points_ds.shape[0] - 1,
                self.max_points_publish,
                dtype=np.int32,
            )
            points_ds = points_ds[indices]
            colors_ds = colors_ds[indices]

        return points_ds, colors_ds

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def pack_rgb_uint32(self, colors_rgb: np.ndarray) -> np.ndarray:
        r = colors_rgb[:, 0].astype(np.uint32)
        g = colors_rgb[:, 1].astype(np.uint32)
        b = colors_rgb[:, 2].astype(np.uint32)

        rgb_uint32 = (r << 16) | (g << 8) | b
        return rgb_uint32

    def publish_colorized_cloud(
        self,
        points_base: np.ndarray,
        colors_rgb: np.ndarray,
    ):
        if points_base.shape[0] == 0:
            self.publish_empty_cloud()
            return

        rgb_uint32 = self.pack_rgb_uint32(colors_rgb)

        cloud_points = np.zeros(
            points_base.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ],
        )

        cloud_points["x"] = points_base[:, 0]
        cloud_points["y"] = points_base[:, 1]
        cloud_points["z"] = points_base[:, 2]
        cloud_points["rgb"] = rgb_uint32

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.stand_frame

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        msg = pc2.create_cloud(header, fields, cloud_points)
        self.cloud_pub.publish(msg)

    def publish_empty_cloud(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.stand_frame

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        empty = np.zeros(
            0,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ],
        )

        msg = pc2.create_cloud(header, fields, empty)
        self.cloud_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ColorizedPointCloud()

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