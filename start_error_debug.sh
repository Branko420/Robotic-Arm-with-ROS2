#!/bin/bash

source install/setup.bash
python3 debug_arm_cube_error.py --ros-args \
  -p base_frame:=base_link \
  -p end_effector_frame:=tool0