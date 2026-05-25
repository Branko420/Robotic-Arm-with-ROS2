#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class IgnoreZoneDrawer(Node):
    def __init__(self):
        super().__init__("ignore_zone_drawer")

        self.declare_parameter("rgb_topic", "/aurora/rgb/image_raw")
        self.rgb_topic = self.get_parameter("rgb_topic").value

        self.bridge = CvBridge()
        self.latest_frame = None
        self.points = []

        self.sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10,
        )

        self.window_name = "Draw Arm Ignore Zone"

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.get_logger().info("✅ Ignore zone drawer started")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info("Instructions:")
        self.get_logger().info("LEFT CLICK  = add polygon point")
        self.get_logger().info("RIGHT CLICK = remove last point")
        self.get_logger().info("C           = clear points")
        self.get_logger().info("S           = save/print ROS parameter")
        self.get_logger().info("Q or ESC    = quit")

        self.timer = self.create_timer(0.03, self.update_window)

    def rgb_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if self.latest_frame is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Added point: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"Removed point: {removed}")

    def update_window(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        height, width = frame.shape[:2]

        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            cv2.putText(
                frame,
                str(i + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Draw polygon lines
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(frame, self.points[i], self.points[i + 1], (0, 255, 0), 2)

        # Close polygon preview
        if len(self.points) >= 3:
            cv2.line(frame, self.points[-1], self.points[0], (0, 0, 255), 2)

            overlay = frame.copy()
            polygon = cv2.convexHull(
                cv2.UMat(
                    cv2.UMat.get(cv2.UMat(
                        __import__("numpy").array(self.points, dtype="int32")
                    ))
                )
            )

            # Simpler polygon fill
            import numpy as np
            polygon_np = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon_np], (0, 0, 0))
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Redraw outline after overlay
            cv2.polylines(frame, [polygon_np], True, (0, 0, 255), 2)

        instructions = [
            "LEFT CLICK: add point",
            "RIGHT CLICK: undo",
            "C: clear",
            "S: print ROS param",
            "Q/ESC: quit",
        ]

        y = 25
        for text in instructions:
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y += 25

        cv2.imshow(self.window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            self.points = []
            print("Cleared points")

        elif key == ord("s"):
            self.print_ros_parameter(width, height)

        elif key == ord("q") or key == 27:
            print("Quitting...")
            rclpy.shutdown()

    def print_ros_parameter(self, width, height):
        if len(self.points) < 3:
            print("Need at least 3 points for a polygon.")
            return

        ratio_points = []

        for x, y in self.points:
            x_ratio = x / width
            y_ratio = y / height
            ratio_points.append(f"{x_ratio:.4f},{y_ratio:.4f}")

        polygon_string = ";".join(ratio_points)

        print("\n================ COPY THIS =================")
        print(
            f'python3 test_camera_depth_ai.py --ros-args '
            f'-p use_arm_polygon_ignore:=True '
            f'-p ignore_arm_zone:=False '
            f'-p arm_ignore_polygon:="{polygon_string}"'
        )
        print("============================================\n")

        print("Polygon only:")
        print(polygon_string)
        print()


def main(args=None):
    rclpy.init(args=args)
    node = IgnoreZoneDrawer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    try:
        node.destroy_node()
    except Exception:
        pass

    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()