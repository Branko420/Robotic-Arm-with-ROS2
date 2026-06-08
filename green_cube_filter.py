#!/usr/bin/env python3

import json
import time
import statistics
from collections import deque
from typing import Optional, Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GreenCubeFilter(Node):
    def __init__(self):
        super().__init__("green_cube_filter")

        self.declare_parameter("input_topic", "/digital_twin/green_cube_json")
        self.declare_parameter("output_topic", "/digital_twin/green_cube_filtered_json")

        # Number of good cube detections used for median smoothing
        self.declare_parameter("window_size", 12)

        # Reject sudden jumps larger than this in meters
        self.declare_parameter("max_jump", 0.035)

        # Publish last good filtered cube even if current frame loses cube
        self.declare_parameter("hold_last_when_lost", True)

        # If True, after first stable value, freeze it until restart
        self.declare_parameter("lock_after_stable", False)

        # How many samples needed before lock
        self.declare_parameter("stable_samples_to_lock", 12)

        self.declare_parameter("publish_rate_hz", 10.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.window_size = int(self.get_parameter("window_size").value)
        self.max_jump = float(self.get_parameter("max_jump").value)
        self.hold_last_when_lost = bool(self.get_parameter("hold_last_when_lost").value)
        self.lock_after_stable = bool(self.get_parameter("lock_after_stable").value)
        self.stable_samples_to_lock = int(self.get_parameter("stable_samples_to_lock").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.history = deque(maxlen=self.window_size)
        self.last_filtered: Optional[Dict] = None
        self.locked = False

        self.sub = self.create_subscription(
            String,
            self.input_topic,
            self.raw_cube_callback,
            10,
        )

        self.pub = self.create_publisher(
            String,
            self.output_topic,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_filtered,
        )

        self.get_logger().info("✅ Green cube filter started")
        self.get_logger().info(f"Input:  {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")
        self.get_logger().info(f"window_size={self.window_size}, max_jump={self.max_jump:.3f}m")
        self.get_logger().info(f"lock_after_stable={self.lock_after_stable}")

    def distance_3d(self, a, b):
        dx = float(a["x"]) - float(b["x"])
        dy = float(a["y"]) - float(b["y"])
        dz = float(a["z"]) - float(b["z"])
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def raw_cube_callback(self, msg: String):
        if self.locked:
            return

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Could not parse cube JSON: {e}")
            return

        if not data.get("found", False):
            return

        sample = {
            "x": float(data["x"]),
            "y": float(data["y"]),
            "z": float(data["z"]),
            "width": float(data.get("width", 0.0)),
            "height": float(data.get("height", 0.0)),
            "depth": float(data.get("depth", 0.0)),
            "points": int(data.get("points", 0)),
            "frame": data.get("frame", "base_link"),
            "base_frame": data.get("base_frame", data.get("frame", "base_link")),
        }

        # Reject big sudden jumps from last filtered value
        if self.last_filtered is not None:
            jump = self.distance_3d(sample, self.last_filtered)
            if jump > self.max_jump:
                self.get_logger().warn(
                    f"Rejected cube jump {jump:.3f}m: "
                    f"x={sample['x']:.3f}, y={sample['y']:.3f}, z={sample['z']:.3f}"
                )
                return

        self.history.append(sample)

        if len(self.history) >= 3:
            self.update_filtered()

        if self.lock_after_stable and len(self.history) >= self.stable_samples_to_lock:
            self.locked = True
            self.get_logger().info(
                f"🔒 Locked filtered cube: "
                f"x={self.last_filtered['x']:.4f}, "
                f"y={self.last_filtered['y']:.4f}, "
                f"z={self.last_filtered['z']:.4f}"
            )

    def median_of(self, key):
        return statistics.median([float(s[key]) for s in self.history])

    def update_filtered(self):
        if not self.history:
            return

        latest = self.history[-1]

        self.last_filtered = {
            "found": True,
            "filtered": True,
            "locked": self.locked,
            "name": "green cube filtered",
            "category": "cube",
            "colors": ["green"],
            "frame": latest.get("frame", "base_link"),
            "base_frame": latest.get("base_frame", "base_link"),

            # Median-smoothed position
            "x": self.median_of("x"),
            "y": self.median_of("y"),
            "z": self.median_of("z"),

            # Size also smoothed
            "width": self.median_of("width"),
            "height": self.median_of("height"),
            "depth": self.median_of("depth"),

            "points": int(latest.get("points", 0)),
            "samples": len(self.history),
            "timestamp": time.time(),
        }

    def publish_filtered(self):
        msg = String()

        if self.last_filtered is None:
            msg.data = json.dumps({
                "found": False,
                "filtered": True,
                "locked": self.locked,
                "samples": len(self.history),
                "timestamp": time.time(),
            })
            self.pub.publish(msg)
            return

        out = dict(self.last_filtered)
        out["locked"] = self.locked
        out["timestamp"] = time.time()

        msg.data = json.dumps(out)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = GreenCubeFilter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()