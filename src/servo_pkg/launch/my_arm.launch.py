import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Find the arm_description_pkg folder automatically
    pkg_dir = get_package_share_directory('arm_description_pkg')
    
    # 2. Define the exact paths to your files based on your setup
    urdf_file_path = os.path.join(pkg_dir, 'urdf', 'robotic_arm.urdf')
    
    # Make sure you saved your RViz config as 'my_arm.rviz' inside an 'rviz' folder!
    rviz_config_path = os.path.join(pkg_dir, 'rviz', 'my_arm.rviz') 
    
    # 3. Read the URDF file
    with open(urdf_file_path, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # --- THE SKELETON BUILDER ---
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
        
        # --- THE VISUALIZER (Loading your custom config) ---
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]  
        )
    ])