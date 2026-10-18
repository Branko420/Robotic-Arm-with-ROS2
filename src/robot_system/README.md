# robot_system

Refactored ROS 2 robotic-arm agent package. The current behavior includes:

- HOME → SCAN_POSE → at least 7 seconds of scanning → HOME;
- no redundant HOME movement when already at HOME;
- gripper opened to 70 physical servo degrees before a pick;
- manual physical-servo angle commands and open-loop angle queries;
- natural same-language realtime responses based on structured action facts;
- held-object movement without a second pick;
- placement of an already-held object without re-grasping;
- height-aware approach, placement and transit planning;
- pick X/Y from `grasp_position` and pick Z exactly from object `center.z`;
- reusable directional corrections against the last Cartesian target.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module responsibilities.

## Install

```bash
cp -r robot_system /workspace/src/robot_system
cd /workspace
source /opt/ros/jazzy/setup.bash
rm -rf build/robot_system install/robot_system
colcon build --packages-select robot_system --symlink-install
source install/setup.bash
```

Copy `.env.example` to `/workspace/.env` and adjust hardware-specific values.

## Entry points

```bash
ros2 run robot_system planning_agent
ros2 run robot_system workspace_scan_agent
ros2 run robot_system realtime_agent
ros2 run robot_system colorized_pointcloud
```

## Static verification

```bash
python3 -m robot_system.tests.test_architecture
python3 -m robot_system.tests.test_behavior_regressions
python3 -m robot_system.tests.test_direct_model_state
```

## Direct RobotModel movement

The planning agent now publishes interpolated `sensor_msgs/JointState` messages
directly on `/joint_states`. `robot_state_publisher` consumes those states and
updates the URDF transforms used by RViz. There is no preview bridge, no
`JointTrajectory` controller topic and no MoveIt dependency in this path.

Disable `PLANNER_PUBLISH_MODEL_JOINT_STATES` only when a real
`joint_state_broadcaster` is already publishing `/joint_states`.


## Position-aware pick, relative motion and saved points

Standalone pick and the pick stage of transfer use: `point + 0.03 m Z -> point -> point + 0.03 m Z`. The object point uses grasp X/Y and exact `center.z`. Gripper commands are 70° open and 130° closed.

The planner continuously stores the current end-effector pose from `/joint_states` through forward kinematics. Relative commands without an object start from that current pose, including HOME.

Saved point voice operations are exposed for save, list, move and delete. Points are persisted in Redis as base_link Cartesian coordinates.

Realtime semantic VAD now enables response interruption. New user speech clears queued assistant audio and, while a tool/motion is active, sends one stop request to the planning stream before the next command is handled.

## Reach filtering

Physical detections farther than `SIMPLE_OBJECT_MAX_BASE_DISTANCE_M` from
`base_link` are excluded before clustering output, motion-point publication,
instance registration, and target resolution. The default is `0.40 m`.

`INSTANCE_MATCH_MAX_DISTANCE_M` defaults to `0.06 m`, so a nearby detection is
not silently reused as the identity of another same-looking object farther
away.

## Fresh object state on every scan

Before processing each scan, the scanner replaces the current workspace and
motion-point documents with empty `SCANNING` documents and deletes the old
object-instance registry and last planning target. Held-object state, object
traceback, learned offsets, and saved Cartesian points are preserved.

## Exact object targeting

Each scan clears stale per-instance target corrections because instance IDs are rebuilt from the new scan. Pick uses the current object-center point and refuses to move when the IK forward-kinematics result differs from the requested target by more than the hard tolerance.
