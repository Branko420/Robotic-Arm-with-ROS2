#! usr/bin/env python3 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
import board, busio
import json 
import asyncio
import threading
import websockets
from adafruit_pca9685 import PCA9685 
from adafruit_motor import servo as adafruit_servo
import time

def smoothstep(p: float)  -> float:
    return p*p*(3.0-2.0*p)

class ArmNode(Node): 
    JOINT_ORDER = ["gripper", "wrist_rotation", "wrist", "elbow", "shoulder", "base"]
    def __init__(self):
        super().__init__('arm_node')

        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c, address=0x40)
            self.pca.frequency = 50 
        except Exception as e: 
            self.get_logger().fatal(f"I2C or PCA9586 initialization failed: {e}")

        self.servos = {}

        for index, name in enumerate(self.JOINT_ORDER):
            try:
                self.servos[name] = adafruit_servo.Servo(
                    self.pca.channels[index],
                    min_pulse=600,
                    max_pulse=2400,
                    actuation_range=180,
                )
            except Exception as e:
                self.get_logger().warn(f"Servo init failed for {name} (channel {index}): {e}")
                raise
            

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
        self.default_duration_per_joint = {
            "gripper": 0.4,
            "wrist_rotation": 1.2,
            "wrist": 0.9,
            "elbow": 1.5,
            "shoulder": 1.5,
            "base": 1.0,
        }


        self.state = {}
        for j in self.JOINT_ORDER:
            self.state[j] = {
        "active": False,
        "start": 90.0,
        "target": 90.0,
        "t0": self.get_clock().now(),
        "dur": 0.0,
        "ease": "smooth",
    }

        
        try:    
            self.servos["gripper"].angle = 90.0
            self.servos["wrist_rotation"].angle = 90.0
            self.servos["wrist"].angle = 180.0
            self.servos["elbow"].angle = 170.0
            self.servos["shoulder"].angle = 160.0
            self.servos["base"].angle = 90.0
        except Exception as e:
            self.get_logger().warn(f"Init angle set failed for {j}: {e}")

        self.create_subscription(Float32, 'gripper_angle', lambda msg: self.move('gripper', msg.data), 10)
        self.create_subscription(Float32, 'wrist_rotation_angle', lambda msg: self.move('wrist_rotation', msg.data), 10)
        self.create_subscription(Float32, 'wrist_angle', lambda msg: self.move('wrist', msg.data), 10)
        self.create_subscription(Float32, 'elbow_angle', lambda msg: self.move('elbow', msg.data), 10)
        self.create_subscription(Float32, 'shoulder_angle', lambda msg: self.move('shoulder', msg.data), 10)
        self.create_subscription(Float32, 'base_angle', lambda msg: self.move('base', msg.data), 10)

        self.create_subscription(Float32MultiArray, 'gripper_cmd', lambda msg: self.plan_move_from_array('gripper', msg), 10)
        self.create_subscription(Float32MultiArray, 'wrist_rotation_cmd', lambda msg: self.plan_move_from_array('wrist_rotation', msg), 10)
        self.create_subscription(Float32MultiArray, 'wrist_cmd', lambda msg: self.plan_move_from_array('wrist', msg), 10)
        self.create_subscription(Float32MultiArray, 'elbow_cmd', lambda msg: self.plan_move_from_array('elbow', msg), 10)
        self.create_subscription(Float32MultiArray, 'shoulder_cmd', lambda msg: self.plan_move_from_array('shoulder', msg), 10)
        self.create_subscription(Float32MultiArray, 'base_cmd', lambda msg: self.plan_move_from_array('base', msg), 10)

        self.update_timer = self.create_timer(0.02, self._update)

        self.get_logger().info("arm ready")

    def _clamp_joint_angle(self, joint: str, angle: float) -> float:
        lo, hi = self.safe_range.get(joint, (0.0, 180.0))
        return max(lo, min(hi, angle))

    def _current_angle(self, joint: str) -> float:
        try:
            ang = self.servo[joint].angle
            if ang is None: 
                return self.state[joint]['target']
            return float(ang)
        except Exception:
            return self.state[joint]['target']
    
    def relax(self, joint: str | None = None):
            if joint is None: 
                for index in range(len(self.JOINT_ORDER)):
                    try: 
                        self.pca.channels[index].duty_cycle =0 
                    except Exception as e:
                        self.get_logger().warn(f"relaxed failed ch{index}: {e}")
            else: 
                try: 
                    index = self.JOINT_ORDER.index(joint)
                    self.pca.channels[index].duty_cycle = 0 
                except Exception as e:
                    self.get_logger().warn(f"relaxed failed ch{index}: {e}")

    def plan_move_from_array(self, joint:str, msg: Float32MultiArray):
        if len(msg.data) == 0: 
            self.get_logger().warn(f"{joint}_cmd: empty data")
            return
        angle = float(msg.data[0])
        duration = float(msg.data[1]) if len(msg.data) > 1 else None
        self.plan_move(joint, angle, duration)

    def plan_move( self, joint: str, target_angle: float, duration: float | None):
        try: 
            if joint not in self.servos:
                self.get_logger().error(f"unknow joint: {joint}")
                return
            
            if duration is None:
                duration = self.default_duration
            duration = max(0.5, float(duration))

            target = self._clamp_joint_angle(joint, float(target_angle))
            start = self._current_angle(joint)

            now = self.get_clock().now()

            self.state[joint].update({
                "active": True,
                "start": start,
                "target": target,
                "t0": now,
                "duration": duration,
                "ease": "smooth"
            })

            self.get_logger().info(f"{joint} plan: {start:.1f}° -> {target:.1f}° in {duration:.2f}s")
        except Exception as e:
            self.get_logger().info(f"{joint} plan failed: {e}")

    def move(self, joint: str, angle: float):
        """
        Direct, immediate movement (no smoothing).
        Called when topics send simple Float32 commands.
        """
        try:
            angle = float(angle)
            angle = self._clamp_joint_angle(joint, angle)

            self.servos[joint].angle = angle

            self.get_logger().info(f"{joint}: {angle:.1f}° (direct)")
        except KeyError:
            self.get_logger().error(f"move(): unknown joint '{joint}'")
        except Exception as e:
            self.get_logger().error(f"{joint} move failed: {e}")


    def _update(self):
        now = self.get_clock().now()

        for joint in self.JOINT_ORDER:
            st = self.state[joint]
            if not st or not bool(st.get("active", False)):
                continue 
            elapsed = (now - st["t0"]).nanoseconds /1e9
            p = min(1.0, max(0.0, elapsed / st["duration"]))

            if st["ease"] == "linear":
                f = p 
            else:   
                f = smoothstep(p)
            angle = st["start"] + (st["target"] - st["start"]) * f 
            angle = self._clamp_joint_angle(joint, angle)

            try:
                self.servos[joint].angle = angle
            except Exception as e: 
                self.get_logger().error(f"{joint} set angle failed: {e}")
                st["active"] = False
                continue

            if p >= 1.0:
                st["active"] = False
                self.get_logger().info(f"{joint} reached {angle:.1f}°")
                if self.auto_relax:
                    self.relax(joint)


    def destroy_node(self): 
        try:
            self.relax()
            self.pca.deinit()
        except Exception as e:
            self.get_logger().warn(f"Deini issue: {e}")
        return super().destroy_node()
    
async def socket_handler(websocket, path, arm_node):
    async for message in websocket:
        try:
            data = json.loads(message)
            duration = data.get("duration", 1.0)

            joints_map = data.get("joints")

            if joints_map and isinstance(joints_map, dict):
                for joint_name, angle in joints_map.items():
                    if joint_name in arm_node.servos:
                        arm_node.plan_move(joint_name, float(angle), float(duration))
                arm_node.get_logger().info(f"Multi-joint move: {list(joints_map.keys())}")
            
            elif "joint" in data:
                joint = data.get("joint")
                angle = data.get("angle")
                if joint in arm_node.servos:
                    arm_node.plan_move(joint, float(angle), float(duration))
        except Exception as e:
            print(f"WS processing error: {e}")

def start_ws_server(node):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = websockets.serve(lambda ws, p: socket_handler(ws, p, node), "0.0.0.0", 8765)
    loop.run_until_complete(server)
    loop.run_forever()

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