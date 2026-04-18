#!/usr/bin/env python3 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
import json 
import asyncio
import threading
import websockets

def smoothstep(p: float)  -> float:
    return p*p*(3.0-2.0*p)

class ArmNode(Node): 
    JOINT_ORDER = ["gripper", "wrist_rotation", "wrist", "elbow", "shoulder", "base"]
    
    def __init__(self):
        super().__init__('arm_node')

        # Hardware simulation / Mocking
        self.get_logger().info("I2C/PCA9685 Hardware bypassed for testing.")

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
                "t0": self.get_clock().now(),
                "duration": 0.0, # Renamed from 'dur' to match plan_move usage
                "ease": "smooth",
            }

        # Subscriptions
        for joint in self.JOINT_ORDER:
            self.create_subscription(Float32, f'{joint}_angle', 
                lambda msg, j=joint: self.move(j, msg.data), 10)
            self.create_subscription(Float32MultiArray, f'{joint}_cmd', 
                lambda msg, j=joint: self.plan_move_from_array(j, msg), 10)

        self.update_timer = self.create_timer(0.02, self._update)
        self.get_logger().info("Virtual Arm Node Ready (Listening to WS and Topics)")

    def _clamp_joint_angle(self, joint: str, angle: float) -> float:
        lo, hi = self.safe_range.get(joint, (0.0, 180.0))
        return max(lo, min(hi, angle))

    def _current_angle(self, joint: str) -> float:
        # In mock mode, we just return the last target state
        return self.state[joint]['target']
    
    def relax(self, joint: str | None = None):
        if joint is None: 
            self.get_logger().info("Relaxing all joints (Duty Cycle 0)")
        else: 
            self.get_logger().info(f"Relaxing joint: {joint}")

    def plan_move_from_array(self, joint:str, msg: Float32MultiArray):
        if len(msg.data) == 0: 
            self.get_logger().warn(f"{joint}_cmd: empty data")
            return
        angle = float(msg.data[0])
        duration = float(msg.data[1]) if len(msg.data) > 1 else None
        self.plan_move(joint, angle, duration)

    def plan_move(self, joint: str, target_angle: float, duration: float | None):
        try: 
            if joint not in self.servos:
                self.get_logger().error(f"Unknown joint: {joint}")
                return
            
            if duration is None:
                duration = self.default_duration
            duration = max(0.1, float(duration))

            target = self._clamp_joint_angle(joint, float(target_angle))
            start = self._current_angle(joint)

            self.state[joint].update({
                "active": True,
                "start": start,
                "target": target,
                "t0": self.get_clock().now(),
                "duration": duration,
                "ease": "smooth"
            })

            self.get_logger().info(f"PLAN: {joint} -> {target:.1f}° in {duration:.2f}s")
        except Exception as e:
            self.get_logger().error(f"{joint} plan failed: {e}")

    def move(self, joint: str, angle: float):
        angle = self._clamp_joint_angle(joint, float(angle))
        self.state[joint]['target'] = angle # Update virtual position
        self.get_logger().info(f"DIRECT: {joint} -> {angle:.1f}°")

    def _update(self):
        now = self.get_clock().now()

        for joint in self.JOINT_ORDER:
            st = self.state[joint]
            if not st.get("active", False):
                continue 
            
            elapsed = (now - st["t0"]).nanoseconds / 1e9
            p = min(1.0, max(0.0, elapsed / st["duration"]))

            f = smoothstep(p) if st["ease"] == "smooth" else p
            angle = st["start"] + (st["target"] - st["start"]) * f 
            
            # Here is where hardware write would happen
            # self.get_logger().debug(f"Updating {joint} to {angle}")

            if p >= 1.0:
                st["active"] = False
                self.get_logger().info(f"REACHED: {joint} at {st['target']:.1f}°")
                if self.auto_relax:
                    self.relax(joint)

    def destroy_node(self): 
        self.get_logger().info("Shutting down node...")
        return super().destroy_node()
    
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
                joint = data.get("joint")
                angle = data.get("angle")
                if joint in arm_node.servos:
                    arm_node.plan_move(joint, float(angle), float(duration))
        except Exception as e:
            arm_node.get_logger().error(f"WS processing error: {e}")

def start_ws_server(node):
    async def run_server():
        # Using the newer websockets.serve syntax 
        # (Note: we use serve as an async context manager)
        async with websockets.serve(lambda ws: socket_handler(ws, node), "0.0.0.0", 8765):
            node.get_logger().info("WebSocket Server started on port 8765")
            await asyncio.Future()  # This runs forever

    try:
        asyncio.run(run_server())
    except Exception as e:
        node.get_logger().error(f"WebSocket Server failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArmNode()

    ws_thread = threading.Thread(target=start_ws_server, args=(node,), daemon=True)
    ws_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()