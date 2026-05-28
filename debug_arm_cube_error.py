#!/usr/bin/env python3

import json
import math
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener


class ArmCubeErrorDebugger(Node):
    def __init__(self):
        super().__init__("arm_cube_error_debugger")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.declare_parameter("cube_json_topic", "/digital_twin/green_cube_json")
        self.declare_parameter("error_json_topic", "/digital_twin/arm_cube_error_json")
        self.declare_parameter("marker_topic", "/digital_twin/arm_cube_error_markers")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("end_effector_frame", "link_gripper_base")

        # If you later add tool0, use:
        # -p end_effector_frame:=tool0

        self.declare_parameter("timer_period", 0.2)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------
        self.cube_json_topic = self.get_parameter("cube_json_topic").value
        self.error_json_topic = self.get_parameter("error_json_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value

        self.base_frame = self.get_parameter("base_frame").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value

        self.timer_period = float(self.get_parameter("timer_period").value)

        # ---------------------------------------------------------
        # TF
        # ---------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------------------------------------
        # ROS
        # ---------------------------------------------------------
        self.cube_sub = self.create_subscription(
            String,
            self.cube_json_topic,
            self.cube_callback,
            10,
        )

        self.error_pub = self.create_publisher(
            String,
            self.error_json_topic,
            10,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            10,
        )

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.latest_cube: Optional[Dict] = None
        self.last_log_time = 0.0

        self.get_logger().info("✅ Arm/cube error debugger started")
        self.get_logger().info(f"Cube topic: {self.cube_json_topic}")
        self.get_logger().info(f"Error JSON: {self.error_json_topic}")
        self.get_logger().info(f"Markers: {self.marker_topic}")
        self.get_logger().info(f"Base frame: {self.base_frame}")
        self.get_logger().info(f"End-effector frame: {self.end_effector_frame}")

    # ---------------------------------------------------------
    # Cube input
    # ---------------------------------------------------------

    def cube_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            if not data.get("found", False):
                self.latest_cube = None
                return

            self.latest_cube = data

        except Exception as e:
            self.get_logger().warn(f"Could not parse cube JSON: {e}")

    # ---------------------------------------------------------
    # TF helper
    # ---------------------------------------------------------

    def get_arm_point(self) -> Optional[Dict]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                rclpy.time.Time(),
            )

            return {
                "frame": self.base_frame,
                "x": float(tf.transform.translation.x),
                "y": float(tf.transform.translation.y),
                "z": float(tf.transform.translation.z),
            }

        except Exception as e:
            now = time.time()
            if now - self.last_log_time > 2.0:
                self.get_logger().warn(
                    f"Could not get TF {self.base_frame} -> {self.end_effector_frame}: {e}"
                )
                self.last_log_time = now

            return None

    # ---------------------------------------------------------
    # Main timer
    # ---------------------------------------------------------

    def timer_callback(self):
        arm = self.get_arm_point()

        if arm is None:
            return

        cube = self.latest_cube

        if cube is None:
            self.publish_no_cube(arm)
            return

        cube_x = float(cube["x"])
        cube_y = float(cube["y"])
        cube_z = float(cube["z"])

        arm_x = arm["x"]
        arm_y = arm["y"]
        arm_z = arm["z"]

        dx = cube_x - arm_x
        dy = cube_y - arm_y
        dz = cube_z - arm_z

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        xy_distance = math.sqrt(dx * dx + dy * dy)

        result = {
            "found": True,
            "base_frame": self.base_frame,
            "end_effector_frame": self.end_effector_frame,
            "arm": {
                "x": arm_x,
                "y": arm_y,
                "z": arm_z,
            },
            "cube": {
                "x": cube_x,
                "y": cube_y,
                "z": cube_z,
            },
            "error": {
                "dx_cube_minus_arm": dx,
                "dy_cube_minus_arm": dy,
                "dz_cube_minus_arm": dz,
                "distance_3d": distance,
                "distance_xy": xy_distance,
            },
            "timestamp": time.time(),
        }

        msg = String()
        msg.data = json.dumps(result)
        self.error_pub.publish(msg)

        self.publish_markers(arm, cube, result)

        now = time.time()
        if now - self.last_log_time > 1.0:
            self.last_log_time = now
            self.get_logger().info(
                f"ARM  x={arm_x:+.3f} y={arm_y:+.3f} z={arm_z:+.3f} | "
                f"CUBE x={cube_x:+.3f} y={cube_y:+.3f} z={cube_z:+.3f} | "
                f"ERR  dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} "
                f"dist={distance:.3f}"
            )

    # ---------------------------------------------------------
    # Publishing
    # ---------------------------------------------------------

    def publish_no_cube(self, arm: Dict):
        result = {
            "found": False,
            "base_frame": self.base_frame,
            "end_effector_frame": self.end_effector_frame,
            "arm": arm,
            "message": "No green cube currently detected.",
            "timestamp": time.time(),
        }

        msg = String()
        msg.data = json.dumps(result)
        self.error_pub.publish(msg)

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.base_frame
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        self.marker_pub.publish(marker_array)

    def publish_markers(self, arm: Dict, cube: Dict, result: Dict):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.base_frame
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # Arm point marker
        arm_marker = Marker()
        arm_marker.header.frame_id = self.base_frame
        arm_marker.header.stamp = self.get_clock().now().to_msg()
        arm_marker.ns = "arm_point"
        arm_marker.id = 1
        arm_marker.type = Marker.SPHERE
        arm_marker.action = Marker.ADD
        arm_marker.pose.position.x = arm["x"]
        arm_marker.pose.position.y = arm["y"]
        arm_marker.pose.position.z = arm["z"]
        arm_marker.pose.orientation.w = 1.0
        arm_marker.scale.x = 0.035
        arm_marker.scale.y = 0.035
        arm_marker.scale.z = 0.035
        arm_marker.color.r = 0.0
        arm_marker.color.g = 0.3
        arm_marker.color.b = 1.0
        arm_marker.color.a = 1.0
        marker_array.markers.append(arm_marker)

        # Cube marker
        cube_marker = Marker()
        cube_marker.header.frame_id = self.base_frame
        cube_marker.header.stamp = self.get_clock().now().to_msg()
        cube_marker.ns = "cube_point"
        cube_marker.id = 2
        cube_marker.type = Marker.SPHERE
        cube_marker.action = Marker.ADD
        cube_marker.pose.position.x = float(cube["x"])
        cube_marker.pose.position.y = float(cube["y"])
        cube_marker.pose.position.z = float(cube["z"])
        cube_marker.pose.orientation.w = 1.0
        cube_marker.scale.x = 0.04
        cube_marker.scale.y = 0.04
        cube_marker.scale.z = 0.04
        cube_marker.color.r = 0.0
        cube_marker.color.g = 1.0
        cube_marker.color.b = 0.0
        cube_marker.color.a = 1.0
        marker_array.markers.append(cube_marker)

        # Arrow from arm to cube
        arrow = Marker()
        arrow.header.frame_id = self.base_frame
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = "error_arrow"
        arrow.id = 3
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        arrow.points = []

        from geometry_msgs.msg import Point

        p1 = Point()
        p1.x = arm["x"]
        p1.y = arm["y"]
        p1.z = arm["z"]

        p2 = Point()
        p2.x = float(cube["x"])
        p2.y = float(cube["y"])
        p2.z = float(cube["z"])

        arrow.points.append(p1)
        arrow.points.append(p2)

        arrow.scale.x = 0.015
        arrow.scale.y = 0.035
        arrow.scale.z = 0.035

        arrow.color.r = 1.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0
        marker_array.markers.append(arrow)

        # Text label
        err = result["error"]

        text = Marker()
        text.header.frame_id = self.base_frame
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = "error_text"
        text.id = 4
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(cube["x"])
        text.pose.position.y = float(cube["y"])
        text.pose.position.z = float(cube["z"]) + 0.12
        text.pose.orientation.w = 1.0
        text.scale.z = 0.04
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = (
            f"ARM → CUBE ERROR\n"
            f"dx={err['dx_cube_minus_arm']:+.3f}\n"
            f"dy={err['dy_cube_minus_arm']:+.3f}\n"
            f"dz={err['dz_cube_minus_arm']:+.3f}\n"
            f"dist={err['distance_3d']:.3f}"
        )
        marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()

    node = ArmCubeErrorDebugger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()