# arm_description_pkg

Minimal ROS 2 description package for the custom robotic arm.

## What remains

- `urdf/arm.urdf` — the newest measured arm model;
- `launch/display.launch.py` — standalone display without MoveIt 2;
- `rviz/display.rviz` — RobotModel/TF RViz configuration.

The older duplicate `robotic_arm.urdf`, the temporary name `idk_arm.urdf`, the empty
`meshes/` directory and generated package boilerplate were removed.

## Build

```bash
cd ~/arm_ws
source /opt/ros/jazzy/setup.bash
rm -rf build/arm_description_pkg install/arm_description_pkg
colcon build --packages-select arm_description_pkg --symlink-install
source install/setup.bash
```

## Display with joint sliders

```bash
ros2 launch arm_description_pkg display.launch.py
```

## Display without the GUI sliders

```bash
ros2 launch arm_description_pkg display.launch.py use_gui:=false
```

This package does not start or depend on MoveIt 2.

## Planner-driven RViz movement

Default launch mode starts the `robot_system/rviz_trajectory_preview` bridge:

```bash
ros2 launch arm_description_pkg display.launch.py
```

The data flow is:

```text
planning_agent
  -> /arm_group_controller/joint_trajectory
  -> /gripper_group_controller/joint_trajectory
  -> rviz_trajectory_preview
  -> /joint_states
  -> robot_state_publisher
  -> RViz RobotModel
```

For manual sliders instead:

```bash
ros2 launch arm_description_pkg display.launch.py \
  use_planner_preview:=false use_gui:=true
```

When a real `joint_state_broadcaster` is running, disable the preview bridge:

```bash
ros2 launch arm_description_pkg display.launch.py \
  use_planner_preview:=false use_gui:=false
```


## Planner-controlled RViz model

Start this display with `use_gui:=false` (the default). The planning agent
publishes `/joint_states` directly, and `robot_state_publisher` updates the
URDF model in RViz. No preview node or controller is required.

Use `use_gui:=true` only for manual slider testing while the planning agent is
not publishing `/joint_states`.
