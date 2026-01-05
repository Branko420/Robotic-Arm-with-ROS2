#!/bin/bash
# Source ROS 2 and workspace
source /opt/ros/jazzy/setup.bash
source /home/pi/arm_ws/install/setup.bash

# Run the servo_node
ros2 run servo_pkg servo_node
