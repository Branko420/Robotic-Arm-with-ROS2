#!/bin/bash
source install/setup.bash

python3 green_cube_finder.py --ros-args \
  -p colorized_cloud_topic:=/digital_twin/live_cloud_rgb \
  -p frame_id:=base_link \
  -p workspace_x_min:=-1.0 \
  -p workspace_x_max:=1.0 \
  -p workspace_y_min:=-1.0 \
  -p workspace_y_max:=1.0 \
  -p workspace_z_min:=-0.30 \
  -p workspace_z_max:=1.00 \
  -p green_r_max:=170 \
  -p green_g_min:=60 \
  -p green_b_max:=180 \
  -p green_margin:=10 \
  -p min_cluster_points:=5