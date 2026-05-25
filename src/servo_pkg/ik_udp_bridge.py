#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
import socket
import json
import math
import time

class MoveItNinjaBridge(Node):
    def __init__(self):
        super().__init__('moveit_ninja_bridge')
        
        # Subscribing to the orange 'ghost' animation topic
        self.subscription = self.create_subscription(
            DisplayTrajectory,
            '/display_planned_path', 
            self.listener_callback,
            10)
        
        # UDP Configuration for your Raspberry Pi
        self.udp_ip = "192.168.188.144" 
        self.udp_port = 9090
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Mapping URDF joint names to your hardware keys
        self.joint_map = {
            "joint_6_base": "base",
            "joint_5_shoulder": "shoulder",
            "joint_4_elbow": "elbow",
            "joint_3_wrist_pitch": "wrist",
            "joint_2_wrist_roll": "wrist_rotation",
            "joint_1_gripper": "gripper"
        }
        
        self.get_logger().info("NINJA BRIDGE ACTIVE: Moving physical arm based on RViz Ghost!")

    def listener_callback(self, msg):
        if not msg.trajectory:
            return
            
        # Extract the waypoints from the planning animation
        for plan in msg.trajectory:
            joint_names = plan.joint_trajectory.joint_names
            points = plan.joint_trajectory.points
            
            self.get_logger().info(f"Intercepted Plan: Sending {len(points)} waypoints to Pi...")
            
            for point in points:
                pose_dict = {}
                
                for i, name in enumerate(joint_names):
                    if name in self.joint_map:
                        hw_key = self.joint_map[name]
                        # Convert radians to degrees with the 90° offset
                        angle_deg = math.degrees(point.positions[i]) + 90.0
                        pose_dict[hw_key] = round(angle_deg, 1)
                
                if pose_dict:
                    try:
                        payload = json.dumps(pose_dict).encode()
                        self.sock.sendto(payload, (self.udp_ip, self.udp_port))
                    except Exception as e:
                        self.get_logger().error(f"UDP Error: {e}")
                
                # Small delay to prevent network congestion on the Pi
                time.sleep(0.04) 
            
        self.get_logger().info("Sequence Finished.")

def main(args=None):
    rclpy.init(args=args)
    node = MoveItNinjaBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()