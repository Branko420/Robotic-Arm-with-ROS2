import socket
import json
import time
import math

# --- ROS 2 IMPORTS ---
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# --- CONFIGURATION ---
PI_IP = "192.168.188.144" 
UDP_PORT = 9090
FPS = 20 

# --- URDF NAME MAPPING ---
# This maps your easy "servo names" to your exact URDF joint names.
URDF_JOINT_NAMES = {
    "base": "joint_6_base", 
    "shoulder": "joint_5_shoulder", 
    "elbow": "joint_4_elbow",
    "wrist": "joint_3_wrist_pitch", 
    "wrist_rotation": "joint_2_wrist_roll", 
    "gripper": "joint_1_gripper"
    # Note: We do NOT need to include "joint_1_gripper_right" because 
    # you used a <mimic> tag in your URDF. RViz will move it automatically!
}

def main(args=None):
    # --- 1. INITIALIZE ROS 2 (For RViz) ---
    rclpy.init(args=args)
    node = rclpy.create_node('laptop_sync_node')
    rviz_pub = node.create_publisher(JointState, '/joint_states', 10)
    
    # --- 2. INITIALIZE UDP SOCKET (For Real Arm) ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Connecting to Arm at {PI_IP}:{UDP_PORT}...")

    # --- THE POSES ---
    resting_pose = {
        "base": 90.0, "shoulder": 130.0, "elbow": 10.0,
        "wrist": 150.0, "wrist_rotation": 90.0, "gripper": 90.0
    }

    reaching_pose = {
        "base": 90.0, "shoulder": 100.0, "elbow": 45.0,
        "wrist": 120.0, "wrist_rotation": 90.0, "gripper": 120.0
    }

    current_pose = resting_pose.copy()

    def send_sync_pose(pose_dict):
        """Fires the UDP packet AND publishes to RViz."""
        # 1. Fire to Raspberry Pi (Degrees)
        try:
            payload = json.dumps(pose_dict).encode()
            sock.sendto(payload, (PI_IP, UDP_PORT))
        except Exception as e:
            print(f"UDP Error: {e}")

        # 2. Fire to RViz (Convert Degrees to Radians)
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = node.get_clock().now().to_msg()
        
        for joint_key, angle_deg in pose_dict.items():
            if joint_key in URDF_JOINT_NAMES:
                urdf_name = URDF_JOINT_NAMES[joint_key]
                
                # Math: Converts Servo Degrees (0-180) to URDF Radians (-1.57 to +1.57)
                # Assumes 90 degrees on the servo is exactly 0.0 radians in URDF.
                angle_rad = math.radians(angle_deg - 90.0) 
                
                msg.name.append(urdf_name)
                msg.position.append(angle_rad)
                
        rviz_pub.publish(msg)

    def move_with_duration(target_pose, duration_seconds):
        """Slices the movement into tiny steps for both RViz and the Real Arm."""
        nonlocal current_pose
        
        if duration_seconds <= 0:
            send_sync_pose(target_pose)
            current_pose = target_pose.copy()
            return

        total_steps = int(duration_seconds * FPS)
        sleep_time = 1.0 / FPS

        step_sizes = {}
        for joint, final_angle in target_pose.items():
            start_angle = current_pose.get(joint, 90.0)
            step_sizes[joint] = (final_angle - start_angle) / total_steps

        # The Animation Loop
        for step in range(1, total_steps + 1):
            frame_pose = {}
            for joint, final_angle in target_pose.items():
                start_angle = current_pose.get(joint, 90.0)
                frame_pose[joint] = start_angle + (step_sizes[joint] * step)
            
            send_sync_pose(frame_pose)
            time.sleep(sleep_time)
            
        current_pose = target_pose.copy()

    # --- THE MAIN SEQUENCE ---
    try:
        print("Syncing arm to RESTING pose (Instant)...")
        move_with_duration(resting_pose, 0)
        time.sleep(1)

        print("Moving to REACHING pose over 3.5 seconds...")
        move_with_duration(reaching_pose, 3.5)
        time.sleep(1)

        print("Going back to RESTING pose FAST over 1 second...")
        move_with_duration(resting_pose, 1.0)
        
        # --- MANUAL CONTROL LOOP ---
        print("\n--- MANUAL JOINT TEST ---")
        print("Type: joint angle duration (e.g., 'base 45 2.0') or 'q' to quit.")
        
        while True:
            command = input("Command: ").strip().lower()
            if command == 'q':
                break
                
            try:
                parts = command.split()
                if len(parts) == 3:
                    joint, angle, duration = parts[0], float(parts[1]), float(parts[2])
                    
                    if joint in current_pose:
                        new_target = current_pose.copy()
                        new_target[joint] = angle
                        print(f"Moving {joint} to {angle}° over {duration}s...")
                        move_with_duration(new_target, duration)
                    else:
                        print(f"Unknown joint! Use: {list(current_pose.keys())}")
                else:
                    print("Invalid format. Use: joint_name angle duration (e.g., base 90 2.5)")
            except ValueError:
                print("Invalid numbers. Try again.")

    except KeyboardInterrupt:
        print("\nStopping test.")
    finally:
        print("Sending safe resting pose over 2 seconds before closing...")
        move_with_duration(resting_pose, 2.0)
        sock.close()
        node.destroy_node()
        rclpy.shutdown()
        print("Systems shut down cleanly.")

if __name__ == '__main__':
    main()