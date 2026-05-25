#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class LaptopBrain(Node):
    def __init__(self):
        super().__init__('laptop_brain_node')
        
        # The topic must match the Pi's subscription exactly
        self.publisher_ = self.create_publisher(JointState, '/arm_the_first/joint_states', 10)
        
        # Timer set to 0.5 seconds (2Hz) to keep the connection alive
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.get_logger().info('Laptop Brain is online. Sending target pose to Pi...')

    def deg_to_rad(self, deg):
        """
        The Pi code uses: degree = (rad * 180/pi) + 90
        To reverse this: rad = (degree - 90) * (pi/180)
        """
        return (float(deg) - 90.0) * (math.pi / 180.0)

    def timer_callback(self):
        msg = JointState()
        
        # These keys MUST match the keys in the Pi's self.joint_map exactly
        msg.name = [
            "joint_6_base", 
            "joint_5_shoulder", 
            "joint_4_elbow", 
            "joint_3_wrist_pitch", 
            "joint_2_wrist_roll", 
            "joint_1_gripper"
        ]
        
        # Target Pose: 90/130/10/150/90/90
        # This matches the 'target_pose' dictionary in your Pi script
        msg.position = [
            self.deg_to_rad(90.0),   # joint_6_base
            self.deg_to_rad(130.0),  # joint_5_shoulder
            self.deg_to_rad(10.0),   # joint_4_elbow
            self.deg_to_rad(150.0),  # joint_3_wrist_pitch
            self.deg_to_rad(90.0),   # joint_2_wrist_roll
            self.deg_to_rad(90.0)    # joint_1_gripper
        ]
        
        # Add a timestamp (good practice for JointState)
        msg.header.stamp = self.get_clock().now().to_msg()
        
        self.publisher_.publish(msg)
        self.get_logger().info('Shooting target pose to the Pi!')

def main(args=None):
    rclpy.init(args=args)
    node = LaptopBrain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down brain...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()