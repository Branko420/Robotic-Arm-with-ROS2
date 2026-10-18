#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any, Dict

from .settings import (
    LAST_TARGET_KEY,
    PICK_GRIPPER_CLOSE_SERVO_DEG,
    PICK_GRIPPER_OPEN_SERVO_DEG,
    PICK_VERTICAL_CLEARANCE_M,
)
from .common import env_float


class PlanningPickFlowMixin:
    def execute_pick_sequence(
        self,
        request_id: str,
        point: Dict[str, Any],
        execute: bool,
    ) -> Dict[str, Any]:
        stable_key = str(
            point.get("instance_id")
            or point.get("point")
            or "nearest_object"
        )
        stored_offsets = self.targets.load_offsets(stable_key)
        geometry = self.targets.object_geometry(point)
        pick_xyz = self.validate_xyz(
            (
                float(point["x"]),
                float(point["y"]),
                float(point["z"]),
            )
        )
        offsets = {"x": 0.0, "y": 0.0, "z": 0.0}
        execution_tolerance = max(
            0.0005,
            env_float("PLANNER_HARD_MAX_TARGET_ERROR_M", 0.008),
        )
        above_xyz = self.validate_xyz(
            (
                pick_xyz[0],
                pick_xyz[1],
                pick_xyz[2] + PICK_VERTICAL_CLEARANCE_M,
            )
        )

        sequence = []
        current_for_step = self.current_joint_deg()
        for step_name, xyz in (
            ("approach_3cm", above_xyz),
            ("grasp_center", pick_xyz),
            ("retreat_3cm", above_xyz),
        ):
            solution, debug = self.solve_cartesian_target(
                xyz,
                current_joint_deg=current_for_step,
            )
            if solution is None:
                return self.result(
                    request_id,
                    False,
                    f"PICK_{step_name.upper()}_IK_FAILED",
                    target=point.get("point"),
                    target_xyz=self.xyz_dict(xyz),
                    closest_error=(
                        None if debug is None else debug.get("error")
                    ),
                )
            reached_error = math.dist(
                tuple(float(value) for value in solution["reached"]),
                tuple(float(value) for value in xyz),
            )
            if reached_error > execution_tolerance:
                return self.result(
                    request_id,
                    False,
                    f"PICK_{step_name.upper()}_TARGET_MISMATCH",
                    target=point.get("point"),
                    target_xyz=self.xyz_dict(xyz),
                    reached_xyz=self.xyz_dict(solution["reached"]),
                    position_error=reached_error,
                    allowed_error=execution_tolerance,
                )
            self.get_logger().info(
                f"pick:{step_name} target="
                f"({xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f}) "
                f"reached=({solution['reached'][0]:.4f},"
                f"{solution['reached'][1]:.4f},"
                f"{solution['reached'][2]:.4f}) "
                f"error={reached_error:.6f}m "
                f"joints_deg={solution['degs']}"
            )
            sequence.append((step_name, xyz, solution))
            current_for_step = dict(solution["degs"])

        if execute:
            if not self.motion.execute_gripper(
                close=False,
                servo_angle_override=PICK_GRIPPER_OPEN_SERVO_DEG,
            ):
                return self.result(request_id, False, "PICK_CANCELLED")

            for step_name, xyz, solution in sequence:
                if self.cancel_event.is_set():
                    return self.result(request_id, False, "PICK_CANCELLED")
                self.publish_plan_markers(
                    xyz,
                    solution["reached"],
                    f"pick:{step_name}:{point.get('point')}",
                )
                ok, _ = self.motion.execute_joints(
                    solution["candidate"],
                    label=f"pick:{step_name}",
                )
                if not ok:
                    return self.result(
                        request_id,
                        False,
                        "PICK_CANCELLED",
                        failed_step=step_name,
                    )
                self.persist_current_cartesian_pose(
                    solution["reached"],
                    source=f"pick:{step_name}",
                )
                if step_name == "grasp_center":
                    if not self.motion.execute_gripper(
                        close=True,
                        servo_angle_override=PICK_GRIPPER_CLOSE_SERVO_DEG,
                    ):
                        return self.result(
                            request_id,
                            False,
                            "PICK_CANCELLED",
                            failed_step="close_gripper",
                        )
                    self.save_held_object(point)
        else:
            self.motion.publish_rviz_joints(
                sequence[0][2]["candidate"],
                self.motion.timings.rviz_move_sec,
            )

        final = sequence[-1][2]
        if execute:
            self.redis.set(LAST_TARGET_KEY, stable_key)
        return self.result(
            request_id,
            True,
            "PICK_EXECUTED" if execute else "PICK_PREVIEWED",
            target=point.get("point"),
            sequence=[name for name, _, _ in sequence],
            approach_xyz=self.xyz_dict(above_xyz),
            pick_point_xyz=self.xyz_dict(pick_xyz),
            retreat_xyz=self.xyz_dict(above_xyz),
            vertical_clearance_m=PICK_VERTICAL_CLEARANCE_M,
            gripper_open_servo_deg=PICK_GRIPPER_OPEN_SERVO_DEG,
            gripper_close_servo_deg=PICK_GRIPPER_CLOSE_SERVO_DEG,
            position_error=final["error"],
            executed=execute,
            simulation_only=not self.motion.real.enabled,
            timings=self.timing_dict(),
            held_object=(self.load_held_object() if execute else None),
            object_geometry=geometry,
            pick_point_source="scanner_motion_point_xyz",
            pick_point_z_source="point.z",
            applied_target_offsets=offsets,
            ignored_stored_target_offsets=stored_offsets,
            strict_target_tolerance_m=execution_tolerance,
            height_aware=True,
            response_facts={
                "operation": "pick",
                "object": self.point_display_name(point),
                "object_released": False,
                "object_still_held": bool(execute),
            },
        )