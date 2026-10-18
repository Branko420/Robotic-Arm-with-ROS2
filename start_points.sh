#!/bin/bash

source install/setup.bash
ros2 run robot_system colorized_pointcloud --ros-args   -p pointcloud_topic:=/aurora/points2   -p rgb_topic:=/aurora/rgb/image_raw   -p camera_info_topic:=/aurora/rgb/camera_info   -p colorized_cloud_topic:=/digital_twin/live_cloud_rgb   -p stand_frame:=base_link   -p camera_x:=-0.214   -p camera_y:=-0.015   -p camera_z:=0.454   -p camera_tilt_degrees:=47.2   -p camera_yaw_degrees:=1.0   -p camera_roll_degrees:=-2.0  -p image_right_is_negative_y:=True
