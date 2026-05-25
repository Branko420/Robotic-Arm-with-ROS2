#!/usr/bin/env python3 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import JointState
import json 
import asyncio
import threading
import websockets
import math

def smoothstep(p: float)  -> float:
    return p*p*(3.0-2.0*p)

class ArmNode(Node): 
    JOINT_ORDER = ["gripper", "wrist_rotation", "wrist", "elbow", "shoulder", "base"]
    
    def __init__(self):
        super().__init__('arm_node')
        self.get_logger().info("BOSS MODE: Controlling URDF from Python logic.")

        # --- PUBLISHER ---
        # This sends the joint positions TO RViz
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # --- TIMER ---
        # We publish to RViz at 30Hz (every 0.033s) to keep the animation smooth
        self.rviz_timer = self.create_timer(0.033, self.publish_to_rviz)

        # Map internal names to URDF joint names
        self.joint_map = {
            "base": "joint_6_base",
            "shoulder": "joint_5_shoulder",
            "elbow": "joint_4_elbow",
            "wrist": "joint_3_wrist_pitch",
            "wrist_rotation": "joint_2_wrist_roll",
            "gripper": "joint_1_gripper"
        }

        self.servos = {name: True for name in self.JOINT_ORDER}

        self.safe_range = {
            "gripper": (70.0, 130.0),
            "wrist_rotation": (10.0, 170.0),
            "wrist": (0.0, 180.0),
            "elbow": (10.0, 170.0),
            "shoulder": (20.0, 160.0),
            "base": (0.0, 180.0),
        }   

        self.auto_relax = False
        self.default_duration = 1.0
        
        self.state = {}
        for j in self.JOINT_ORDER:
            self.state[j] = {
                "active": False,
                "start": 90.0,
                "target": 90.0,
                "current_val": 90.0, # This tracks the live degree for RViz
                "t0": self.get_clock().now(),
                "duration": 1.0,
                "ease": "smooth",
            }

        # Subscriptions for direct topic control
        for joint in self.JOINT_ORDER:
            self.create_subscription(Float32, f'{joint}_angle', 
                lambda msg, j=joint: self.move(j, msg.data), 10)
            self.create_subscription(Float32MultiArray, f'{joint}_cmd', 
                lambda msg, j=joint: self.plan_move_from_array(j, msg), 10)

        # Main logic update timer (50Hz)
        self.update_timer = self.create_timer(0.02, self._update)
        self.get_logger().info("Virtual Arm Node Ready (Listening to WS and Topics)")

    def publish_to_rviz(self):
        """Broadcasts the current state of the arm to RViz."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for internal_name in self.JOINT_ORDER:
            # Add the URDF name
            msg.name.append(self.joint_map[internal_name])
            
            # MATH: Convert 0-180 degrees back to -1.57 to 1.57 radians
            degree = self.state[internal_name]['current_val']
            radian = (degree - 90.0) * (math.pi / 180.0)
            msg.position.append(radian)

        self.joint_pub.publish(msg)

    def _clamp_joint_angle(self, joint: str, angle: float) -> float:
        lo, hi = self.safe_range.get(joint, (0.0, 180.0))
        return max(lo, min(hi, angle))

    def _current_angle(self, joint: str) -> float:
        return self.state[joint]['current_val']
    
    def relax(self, joint: str | None = None):
        if joint is None: 
            self.get_logger().info("Relaxing all joints")
        else: 
            self.get_logger().info(f"Relaxing joint: {joint}")

    def plan_move_from_array(self, joint:str, msg: Float32MultiArray):
        if len(msg.data) == 0: return
        angle = float(msg.data[0])
        duration = float(msg.data[1]) if len(msg.data) > 1 else None
        self.plan_move(joint, angle, duration)

    def plan_move(self, joint: str, target_angle: float, duration: float | None):
        try: 
            if joint not in self.servos: return
            
            duration = max(0.1, float(duration if duration is not None else self.default_duration))
            target = self._clamp_joint_angle(joint, float(target_angle))
            start = self._current_angle(joint)

            self.state[joint].update({
                "active": True,
                "start": start,
                "target": target,
                "t0": self.get_clock().now(),
                "duration": duration
            })
            self.get_logger().info(f"PLAN: {joint} -> {target:.1f}° over {duration}s")
        except Exception as e:
            self.get_logger().error(f"Plan failed: {e}")

    def move(self, joint: str, angle: float):
        """Direct jump to angle (no animation)"""
        angle = self._clamp_joint_angle(joint, float(angle))
        self.state[joint]['target'] = angle
        self.state[joint]['current_val'] = angle
        self.state[joint]['active'] = False
        self.get_logger().info(f"DIRECT: {joint} -> {angle:.1f}°")

    def _update(self):
        """Internal loop that handles the smooth movement interpolation."""
        now = self.get_clock().now()

        for joint in self.JOINT_ORDER:
            st = self.state[joint]
            if not st["active"]:
                continue 
            
            elapsed = (now - st["t0"]).nanoseconds / 1e9
            p = min(1.0, max(0.0, elapsed / st["duration"]))
            f = smoothstep(p)
            
            # Calculate the in-between angle
            new_angle = st["start"] + (st["target"] - st["start"]) * f 
            st['current_val'] = new_angle

            if p >= 1.0:
                st["active"] = False
                self.get_logger().info(f"REACHED: {joint} at {st['target']:.1f}°")

    def destroy_node(self): 
        self.get_logger().info("Shutting down node...")
        return super().destroy_node()

# --- WEBSOCKET LOGIC ---
async def socket_handler(websocket, arm_node):
    async for message in websocket:
        try:
            data = json.loads(message)
            duration = data.get("duration", 1.0)
            joints_map = data.get("joints")

            if joints_map and isinstance(joints_map, dict):
                for joint_name, angle in joints_map.items():
                    if joint_name in arm_node.servos:
                        arm_node.plan_move(joint_name, float(angle), float(duration))
            elif "joint" in data:
                arm_node.plan_move(data["joint"], float(data["angle"]), float(duration))
        except Exception as e:
            arm_node.get_logger().error(f"WS Error: {e}")

def start_ws_server(node):
    async def run_server():
        async with websockets.serve(lambda ws: socket_handler(ws, node), "0.0.0.0", 8765):
            await asyncio.Future()
    asyncio.run(run_server())

def main(args=None):
    rclpy.init(args=args)
    node = ArmNode()
    threading.Thread(target=start_ws_server, args=(node,), daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()