#!/bin/bash

source install/setup.bash
python3 colorized_pointcloud.py --ros-args   -p pointcloud_topic:=/aurora/points2   -p rgb_topic:=/aurora/rgb/image_raw   -p camera_info_topic:=/aurora/rgb/camera_info   -p colorized_cloud_topic:=/digital_twin/live_cloud_rgb   -p stand_frame:=base_link   -p camera_x:=-0.22s8   -p camera_y:=0.0   -p camera_z:=0.48   -p camera_tilt_degrees:=45.0   -p camera_yaw_degrees:=0.0   -p image_right_is_negative_y:=True
