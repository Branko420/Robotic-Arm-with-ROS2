#!/usr/bin/env bash

cd /workspace

source /opt/ros/jazzy/setup.bash

if [ -f /workspace/arm_env/bin/activate ]; then
    source /workspace/arm_env/bin/activate
fi

if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

export PYTHONPATH="/workspace/arm_env/lib/python3.12/site-packages:${PYTHONPATH:-}"
export REDIS_URL="redis://127.0.0.1:6379/0"

echo "Robot env loaded: ROS Jazzy + arm_env + robot_system"
