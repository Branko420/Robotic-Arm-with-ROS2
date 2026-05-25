#!/usr/bin/env python3

import json
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

import sensor_msgs_py.point_cloud2 as pc2


class GreenCubeFinder(Node):
    def __init__(self):
        super().__init__("green_cube_finder")

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------
        self.declare_parameter("colorized_cloud_topic", "/digital_twin/live_cloud_rgb")
        self.declare_parameter("marker_topic", "/digital_twin/green_cube_marker")
        self.declare_parameter("position_topic", "/digital_twin/green_cube_position")
        self.declare_parameter("json_topic", "/digital_twin/green_cube_json")

        self.declare_parameter("frame_id", "stand_base")

        # ---------------------------------------------------------
        # Green color thresholds
        # These are RGB thresholds from the colorized point cloud.
        # Tune if needed.
        # ---------------------------------------------------------
        self.declare_parameter("green_r_max", 120)
        self.declare_parameter("green_g_min", 90)
        self.declare_parameter("green_b_max", 140)

        # Extra rule: green must be stronger than red/blue.
        self.declare_parameter("green_margin", 25)

        # ---------------------------------------------------------
        # 3D workspace filter
        # ---------------------------------------------------------
        self.declare_parameter("workspace_x_min", -0.10)
        self.declare_parameter("workspace_x_max", 1.00)
        self.declare_parameter("workspace_y_min", -0.60)
        self.declare_parameter("workspace_y_max", 0.60)
        self.declare_parameter("workspace_z_min", -0.05)
        self.declare_parameter("workspace_z_max", 0.50)

        # ---------------------------------------------------------
        # Clustering
        # ---------------------------------------------------------
        self.declare_parameter("cluster_grid_resolution", 0.025)
        self.declare_parameter("min_cluster_points", 20)
        self.declare_parameter("max_cluster_points", 20000)

        # Cube size filter in meters.
        self.declare_parameter("min_cube_size", 0.015)
        self.declare_parameter("max_cube_size", 0.12)

        # RViz marker size fallback.
        self.declare_parameter("min_marker_size", 0.025)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.colorized_cloud_topic = self.get_parameter("colorized_cloud_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.position_topic = self.get_parameter("position_topic").value
        self.json_topic = self.get_parameter("json_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.green_r_max = int(self.get_parameter("green_r_max").value)
        self.green_g_min = int(self.get_parameter("green_g_min").value)
        self.green_b_max = int(self.get_parameter("green_b_max").value)
        self.green_margin = int(self.get_parameter("green_margin").value)

        self.workspace_x_min = float(self.get_parameter("workspace_x_min").value)
        self.workspace_x_max = float(self.get_parameter("workspace_x_max").value)
        self.workspace_y_min = float(self.get_parameter("workspace_y_min").value)
        self.workspace_y_max = float(self.get_parameter("workspace_y_max").value)
        self.workspace_z_min = float(self.get_parameter("workspace_z_min").value)
        self.workspace_z_max = float(self.get_parameter("workspace_z_max").value)

        self.cluster_grid_resolution = float(
            self.get_parameter("cluster_grid_resolution").value
        )
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.max_cluster_points = int(self.get_parameter("max_cluster_points").value)

        self.min_cube_size = float(self.get_parameter("min_cube_size").value)
        self.max_cube_size = float(self.get_parameter("max_cube_size").value)
        self.min_marker_size = float(self.get_parameter("min_marker_size").value)

        # ---------------------------------------------------------
        # ROS setup
        # ---------------------------------------------------------
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.colorized_cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            10,
        )

        self.position_pub = self.create_publisher(
            PointStamped,
            self.position_topic,
            10,
        )

        self.json_pub = self.create_publisher(
            String,
            self.json_topic,
            10,
        )

        self.last_warn_times: Dict[str, float] = {}
        self.last_log_time = 0.0

        self.get_logger().info("✅ Green cube finder started")
        self.get_logger().info(f"Input cloud: {self.colorized_cloud_topic}")
        self.get_logger().info(f"Marker topic: {self.marker_topic}")
        self.get_logger().info(f"Position topic: {self.position_topic}")

    # -------------------------------------------------------------------------
    # Logging helper
    # -------------------------------------------------------------------------

    def warn_throttled(self, key: str, message: str, period_sec: float = 3.0):
        now = time.time()
        last = self.last_warn_times.get(key, 0.0)

        if now - last >= period_sec:
            self.get_logger().warn(message)
            self.last_warn_times[key] = now

    # -------------------------------------------------------------------------
    # Main callback
    # -------------------------------------------------------------------------

    def cloud_callback(self, msg: PointCloud2):
        points, colors = self.read_xyz_rgb(msg)

        if points is None or points.shape[0] == 0:
            self.publish_no_cube()
            return

        points, colors = self.filter_workspace(points, colors)

        if points.shape[0] == 0:
            self.publish_no_cube()
            return

        green_points = self.extract_green_points(points, colors)

        if green_points.shape[0] == 0:
            self.publish_no_cube()
            return

        clusters = self.cluster_points_xy(green_points)

        if not clusters:
            self.publish_no_cube()
            return

        best = self.select_best_cube_cluster(clusters)

        if best is None:
            self.publish_no_cube()
            return

        cube = self.points_to_cube(best)

        self.publish_cube(cube)

    # -------------------------------------------------------------------------
    # Read XYZRGB cloud
    # -------------------------------------------------------------------------

    def read_xyz_rgb(
        self,
        msg: PointCloud2,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        field_map = {field.name: field for field in msg.fields}

        if "x" not in field_map or "y" not in field_map or "z" not in field_map:
            self.warn_throttled("missing_xyz", "Cloud missing x/y/z fields.")
            return None, None

        rgb_field_name = None

        if "rgb" in field_map:
            rgb_field_name = "rgb"
        elif "rgba" in field_map:
            rgb_field_name = "rgba"

        if rgb_field_name is None:
            self.warn_throttled(
                "missing_rgb",
                "Cloud missing rgb/rgba field. Use /digital_twin/live_cloud_rgb.",
            )
            return None, None

        x_field = field_map["x"]
        y_field = field_map["y"]
        z_field = field_map["z"]
        rgb_field = field_map[rgb_field_name]

        endian = ">" if msg.is_bigendian else "<"

        if rgb_field.datatype == PointField.FLOAT32:
            rgb_format = endian + "f4"
        elif rgb_field.datatype == PointField.UINT32:
            rgb_format = endian + "u4"
        else:
            self.warn_throttled(
                "bad_rgb_type",
                f"Unsupported rgb datatype: {rgb_field.datatype}",
            )
            return None, None

        dtype = np.dtype({
            "names": ["x", "y", "z", "rgb"],
            "formats": [endian + "f4", endian + "f4", endian + "f4", rgb_format],
            "offsets": [
                x_field.offset,
                y_field.offset,
                z_field.offset,
                rgb_field.offset,
            ],
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

            if rgb_field.datatype == PointField.FLOAT32:
                rgb_uint32 = raw["rgb"].view(np.uint32)
            else:
                rgb_uint32 = raw["rgb"].astype(np.uint32)

            r = ((rgb_uint32 >> 16) & 255).astype(np.uint8)
            g = ((rgb_uint32 >> 8) & 255).astype(np.uint8)
            b = (rgb_uint32 & 255).astype(np.uint8)

            colors = np.stack([r, g, b], axis=1)

            valid = (
                np.isfinite(points[:, 0])
                & np.isfinite(points[:, 1])
                & np.isfinite(points[:, 2])
            )

            return points[valid], colors[valid]

        except Exception as e:
            self.warn_throttled("read_failed", f"Failed reading XYZRGB cloud: {e}")
            return None, None

    # -------------------------------------------------------------------------
    # Filtering / detection
    # -------------------------------------------------------------------------

    def filter_workspace(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        mask = (
            (x >= self.workspace_x_min)
            & (x <= self.workspace_x_max)
            & (y >= self.workspace_y_min)
            & (y <= self.workspace_y_max)
            & (z >= self.workspace_z_min)
            & (z <= self.workspace_z_max)
        )

        return points[mask], colors[mask]

    def extract_green_points(self, points: np.ndarray, colors: np.ndarray) -> np.ndarray:
        r = colors[:, 0].astype(np.int16)
        g = colors[:, 1].astype(np.int16)
        b = colors[:, 2].astype(np.int16)

        green_mask = (
            (g >= self.green_g_min)
            & (r <= self.green_r_max)
            & (b <= self.green_b_max)
            & ((g - r) >= self.green_margin)
            & ((g - b) >= self.green_margin)
        )

        return points[green_mask]

    def cluster_points_xy(self, points: np.ndarray) -> List[np.ndarray]:
        if points.shape[0] == 0:
            return []

        resolution = self.cluster_grid_resolution

        x_min = float(np.min(points[:, 0]))
        y_min = float(np.min(points[:, 1]))

        grid_x = np.floor((points[:, 0] - x_min) / resolution).astype(np.int32)
        grid_y = np.floor((points[:, 1] - y_min) / resolution).astype(np.int32)

        width = int(np.max(grid_x)) + 1
        height = int(np.max(grid_y)) + 1

        if width <= 0 or height <= 0 or width > 1200 or height > 1200:
            self.warn_throttled(
                "bad_grid",
                f"Bad green cluster grid size: {width}x{height}",
            )
            return []

        occupancy = np.zeros((height, width), dtype=np.uint8)
        occupancy[grid_y, grid_x] = 255

        kernel = np.ones((3, 3), dtype=np.uint8)
        occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)

        num_labels, labels = cv2.connectedComponents(occupancy, connectivity=8)
        point_labels = labels[grid_y, grid_x]

        clusters = []

        for label_id in range(1, num_labels):
            cluster = points[point_labels == label_id]

            if cluster.shape[0] < self.min_cluster_points:
                continue

            if cluster.shape[0] > self.max_cluster_points:
                continue

            clusters.append(cluster)

        return clusters

    def select_best_cube_cluster(self, clusters: List[np.ndarray]) -> Optional[np.ndarray]:
        candidates = []

        for cluster in clusters:
            xs = cluster[:, 0]
            ys = cluster[:, 1]
            zs = cluster[:, 2]

            size_x = float(np.percentile(xs, 95) - np.percentile(xs, 5))
            size_y = float(np.percentile(ys, 95) - np.percentile(ys, 5))
            size_z = float(np.percentile(zs, 95) - np.percentile(zs, 5))

            size_x = max(size_x, self.min_marker_size)
            size_y = max(size_y, self.min_marker_size)
            size_z = max(size_z, self.min_marker_size)

            max_size = max(size_x, size_y, size_z)
            min_size = max(min(size_x, size_y, size_z), 0.001)
            ratio = max_size / min_size

            cube_like = (
                (self.min_cube_size <= size_x <= self.max_cube_size)
                and (self.min_cube_size <= size_y <= self.max_cube_size)
                and (self.min_cube_size <= size_z <= self.max_cube_size)
                and ratio < 3.0
            )

            if cube_like:
                candidates.append((cluster.shape[0], cluster))

        if not candidates:
            # Fallback: choose biggest green cluster.
            return max(clusters, key=lambda c: c.shape[0])

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def points_to_cube(self, points: np.ndarray) -> Dict:
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        x = float(np.median(xs))
        y = float(np.median(ys))
        z = float(np.median(zs))

        width = float(np.percentile(xs, 95) - np.percentile(xs, 5))
        height = float(np.percentile(ys, 95) - np.percentile(ys, 5))
        depth = float(np.percentile(zs, 95) - np.percentile(zs, 5))

        width = max(width, self.min_marker_size)
        height = max(height, self.min_marker_size)
        depth = max(depth, self.min_marker_size)

        return {
            "name": "green cube",
            "category": "cube",
            "colors": ["green"],
            "frame": self.frame_id,
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "height": height,
            "depth": depth,
            "points": int(points.shape[0]),
            "timestamp": time.time(),
        }

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def publish_no_cube(self):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)

        msg = String()
        msg.data = json.dumps({
            "found": False,
            "name": "green cube",
            "timestamp": time.time(),
        })
        self.json_pub.publish(msg)

    def publish_cube(self, cube: Dict):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = cube["frame"]
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker = Marker()
        marker.header.frame_id = cube["frame"]
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "green_cube"
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = cube["x"]
        marker.pose.position.y = cube["y"]
        marker.pose.position.z = cube["z"]
        marker.pose.orientation.w = 1.0

        marker.scale.x = cube["width"]
        marker.scale.y = cube["height"]
        marker.scale.z = cube["depth"]

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.85

        marker_array.markers.append(marker)

        sphere = Marker()
        sphere.header.frame_id = cube["frame"]
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = "green_cube_center"
        sphere.id = 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD

        sphere.pose.position.x = cube["x"]
        sphere.pose.position.y = cube["y"]
        sphere.pose.position.z = cube["z"] + 0.03
        sphere.pose.orientation.w = 1.0

        sphere.scale.x = 0.035
        sphere.scale.y = 0.035
        sphere.scale.z = 0.035

        sphere.color.r = 1.0
        sphere.color.g = 1.0
        sphere.color.b = 0.0
        sphere.color.a = 1.0

        marker_array.markers.append(sphere)

        arrow = Marker()
        arrow.header.frame_id = cube["frame"]
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = "green_cube_arrow"
        arrow.id = 3
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        arrow.pose.position.x = cube["x"]
        arrow.pose.position.y = cube["y"]
        arrow.pose.position.z = cube["z"] + 0.20
        arrow.pose.orientation.x = 0.0
        arrow.pose.orientation.y = 1.0
        arrow.pose.orientation.z = 0.0
        arrow.pose.orientation.w = 0.0

        arrow.scale.x = 0.16
        arrow.scale.y = 0.025
        arrow.scale.z = 0.025

        arrow.color.r = 1.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0

        marker_array.markers.append(arrow)

        text = Marker()
        text.header.frame_id = cube["frame"]
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = "green_cube_label"
        text.id = 4
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD

        text.pose.position.x = cube["x"]
        text.pose.position.y = cube["y"]
        text.pose.position.z = cube["z"] + 0.12
        text.pose.orientation.w = 1.0

        text.scale.z = 0.05
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = (
            f"GREEN CUBE\n"
            f"x={cube['x']:.3f}, y={cube['y']:.3f}, z={cube['z']:.3f}"
        )

        marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

        point_msg = PointStamped()
        point_msg.header.frame_id = cube["frame"]
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x = cube["x"]
        point_msg.point.y = cube["y"]
        point_msg.point.z = cube["z"]

        self.position_pub.publish(point_msg)

        json_msg = String()
        json_msg.data = json.dumps({
            "found": True,
            **cube,
        })
        self.json_pub.publish(json_msg)

        now = time.time()
        if now - self.last_log_time > 1.0:
            self.last_log_time = now
            self.get_logger().info(
                f"🟢 Green cube at "
                f"x={cube['x']:.3f}, y={cube['y']:.3f}, z={cube['z']:.3f}, "
                f"points={cube['points']}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = GreenCubeFinder()

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