#!/bin/bash

source install/setup.bash
ros2 launch arm_description_pkg display.launch.py \
  use_planner_preview:=false use_gui:=false