#!/usr/bin/env python3

# Позволява използването на отложено оценяване
# на типовите анотации.
from __future__ import annotations

# Типови анотации за общи речници, списъци,
# асоциативни структури, незадължителни стойности и кортежи.
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Конфигурационни стойности за:
# - последно избраната цел;
# - отвореното положение на хващача;
# - записване на симулирани движения в историята.
from .settings import (
    LAST_TARGET_KEY,
    PICK_GRIPPER_OPEN_SERVO_DEG,
    RECORD_SIMULATED_MOVES,
)


class PlanningHeldObjectFlowMixin:
    """
    Съдържа последователността за поставяне на предмет,
    който вече се намира в затворения хващач.

    Операцията включва:
    - изчисляване на безопасна транзитна височина;
    - движение над крайната цел;
    - спускане до позицията за поставяне;
    - отваряне на хващача;
    - оттегляне обратно до безопасната позиция;
    - обновяване на историята на предмета.
    """

    def execute_place_held_sequence(
        self,
        request_id: str,
        held_object: Dict[str, Any],
        destination: Dict[str, Any],
        execute: bool,
        workspace_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Изпълнява последователност за поставяне
        на вече държан предмет върху зададена крайна цел.
        """

        # Получава геометричните параметри
        # на държания предмет.
        held_geometry = self.targets.object_geometry(
            held_object
        )

        # Получава геометричните параметри
        # на крайната цел.
        destination_geometry = (
            self.targets.object_geometry(
                destination
            )
        )

        # Изчислява позициите за:
        # - подход над крайната цел;
        # - поставяне на хващача;
        # - окончателен център на предмета.
        placement = self.targets.placement_targets(
            held_object,
            destination,
            place_clearance=self.place_clearance,
            approach_height=self.place_approach_height,
            minimum_clearance=(
                self.minimum_above_clearance
            ),
        )

        # Създава устойчив идентификатор
        # на крайната цел.
        destination_key = str(
            destination.get("instance_id")
            or destination.get("point")
            or "destination"
        )

        # Зарежда запазените корекции
        # за конкретната крайна цел.
        destination_offsets = (
            self.targets.load_offsets(
                destination_key
            )
        )

        def apply_destination_offsets(
            xyz: Tuple[float, float, float],
        ) -> Tuple[float, float, float]:
            """
            Прилага запазените корекции по X, Y и Z
            към позиция, свързана с крайната цел.
            """

            # Добавя корекциите и валидира
            # получената декартова позиция.
            return self.validate_xyz(
                (
                    xyz[0]
                    + destination_offsets["x"],
                    xyz[1]
                    + destination_offsets["y"],
                    xyz[2]
                    + destination_offsets["z"],
                )
            )

        # Крайна позиция на хващача,
        # при която предметът трябва да бъде освободен.
        gripper_place = apply_destination_offsets(
            placement["gripper_place"]
        )

        # Безопасна позиция за подход
        # над крайната цел.
        gripper_approach = (
            apply_destination_offsets(
                placement["gripper_approach"]
            )
        )

        # Очакван център на предмета
        # след неговото поставяне.
        final_object_center = (
            apply_destination_offsets(
                placement["object_center"]
            )
        )

        # Опитва да използва актуалния
        # проследяван център на държания предмет.
        current_center_row = held_object.get(
            "current_center"
        )

        if isinstance(
            current_center_row,
            Mapping,
        ):
            # Използва центъра, записан след
            # предишно движение на предмета.
            current_object_center = (
                float(current_center_row["x"]),
                float(current_center_row["y"]),
                float(current_center_row["z"]),
            )

        else:
            # При липса на актуален център използва
            # първоначалната геометрична позиция.
            current_object_center = (
                held_geometry["center"]["x"],
                held_geometry["center"]["y"],
                held_geometry["center"]["z"],
            )

        # Получава отместването между центъра
        # на предмета и позицията на хващача.
        grasp_offset = held_geometry[
            "grasp_offset"
        ]

        # Изчислява текущата позиция
        # на хващача спрямо държания предмет.
        start_gripper = (
            current_object_center[0]
            + grasp_offset["x"],
            current_object_center[1]
            + grasp_offset["y"],
            current_object_center[2]
            + grasp_offset["z"],
        )

        # Създава устойчив идентификатор
        # на държания предмет.
        held_id = str(
            held_object.get("instance_id")
            or held_object.get("point")
            or ""
        )

        # Изчислява безопасна транзитна височина,
        # съобразена с височината на предмета
        # и останалите препятствия.
        transit_z, blockers = (
            self.height_aware_transit_z(
                workspace_points,
                held_object,
                start_gripper,
                gripper_approach,
                max(
                    start_gripper[2],
                    gripper_approach[2],
                ),
                ignored_ids={
                    held_id,
                    destination_key,
                },
            )
        )

        # Прекратява операцията, когато не може
        # да бъде намерена достижима безопасна височина.
        if transit_z is None:
            return self.result(
                request_id,
                False,
                "TRANSIT_CLEARANCE_UNREACHABLE",
                blockers=blockers,
                maximum_transit_z=(
                    self.maximum_transit_z
                ),
            )

        # Списък с последователните стъпки.
        #
        # Всеки запис съдържа:
        # - име на стъпката;
        # - декартова позиция;
        # - действие на хващача или None.
        steps: List[
            Tuple[
                str,
                Tuple[float, float, float],
                Optional[str],
            ]
        ] = []

        # Повдига хващача вертикално над
        # началната позиция при необходимост.
        if transit_z > start_gripper[2] + 0.005:
            steps.append(
                (
                    "held_source_transit",
                    (
                        start_gripper[0],
                        start_gripper[1],
                        transit_z,
                    ),
                    None,
                )
            )

        # Премества хващача хоризонтално
        # над крайната цел при транзитната височина.
        if (
            transit_z
            > gripper_approach[2] + 0.005
        ):
            steps.append(
                (
                    "held_destination_transit",
                    (
                        gripper_approach[0],
                        gripper_approach[1],
                        transit_z,
                    ),
                    None,
                )
            )

        # Добавя основната последователност:
        # 1. подход над крайната цел;
        # 2. спускане до мястото за поставяне;
        # 3. отваряне на хващача;
        # 4. оттегляне обратно нагоре.
        steps.extend(
            [
                (
                    "held_destination_approach",
                    gripper_approach,
                    None,
                ),
                (
                    "held_destination_place",
                    gripper_place,
                    "open",
                ),
                (
                    "held_destination_retreat",
                    gripper_approach,
                    None,
                ),
            ]
        )

        # Списък с успешно решените
        # стъпки на обратната кинематика.
        solved = []

        # Начални ставни стойности
        # за първата IK стъпка.
        current_for_step = (
            self.current_joint_deg()
        )

        # Решава последователно всяка стъпка,
        # като използва резултата от предишната
        # като начало за следващата.
        for (
            step_name,
            xyz,
            gripper_action,
        ) in steps:
            # Изчислява решение
            # на обратната кинематика.
            solution, debug = (
                self.solve_cartesian_target(
                    xyz,
                    current_joint_deg=(
                        current_for_step
                    ),
                )
            )

            # При неуспешно IK решение прекратява
            # цялата последователност.
            if solution is None:
                return self.result(
                    request_id,
                    False,
                    "PLACE_HELD_IK_FAILED",
                    failed_step=step_name,
                    target_xyz=self.xyz_dict(xyz),
                    closest_error=(
                        None
                        if debug is None
                        else debug.get("error")
                    ),
                )

            # Запазва решената стъпка.
            solved.append(
                (
                    step_name,
                    xyz,
                    gripper_action,
                    solution,
                )
            )

            # Следващата IK задача започва
            # от текущото получено решение.
            current_for_step = dict(
                solution["degs"]
            )

        # Показва дали предметът действително
        # е бил освободен от хващача.
        released = False

        # При реално изпълнение преминава
        # последователно през всички решени стъпки.
        if execute:
            for (
                step_name,
                _xyz,
                gripper_action,
                solution,
            ) in solved:
                # Изпълнява ставното движение
                # за текущата стъпка.
                ok, _servo = (
                    self.motion.execute_joints(
                        solution["candidate"],
                        label=(
                            f"place_held:{step_name}"
                        ),
                    )
                )

                # Прекратява операцията,
                # когато движението е отменено.
                if not ok:
                    return self.result(
                        request_id,
                        False,
                        "PLACE_HELD_CANCELLED",
                        failed_step=step_name,
                    )

                # Отваря хващача само при достигане
                # на крайната позиция за поставяне.
                if gripper_action == "open":
                    if not self.motion.execute_gripper(
                        close=False,
                        servo_angle_override=(
                            PICK_GRIPPER_OPEN_SERVO_DEG
                        ),
                    ):
                        return self.result(
                            request_id,
                            False,
                            "PLACE_HELD_CANCELLED",
                            failed_step=(
                                "open_gripper"
                            ),
                        )

                    # Предметът е успешно освободен.
                    released = True

                    # Изчиства съхраненото състояние
                    # за държан предмет.
                    self.clear_held_object()

        # При предварителна визуализация публикува
        # първата ставна позиция само в RViz.
        elif solved:
            self.motion.publish_rviz_joints(
                solved[0][3]["candidate"],
                self.motion.timings.rviz_move_sec,
            )

        # Резултат от обновяването
        # на историята на предмета.
        trace_update = None

        # Историята се записва само когато:
        # - има реално изпълнение;
        # - предметът е освободен;
        # - физическата ръка е активна или е разрешено
        #   записване на симулирани движения.
        history_committed = bool(
            execute
            and released
            and (
                self.motion.real.enabled
                or RECORD_SIMULATED_MOVES
            )
        )

        if history_committed:
            # Записва преместването на предмета
            # в неговата история.
            trace_update = (
                self.traceback.record_move(
                    source=held_object,
                    destination=destination,
                    placed_center=(
                        final_object_center
                    ),
                    request_id=request_id,
                )
            )

            # Актуализира регистрирания център
            # на конкретния екземпляр.
            if trace_update.get("object_id"):
                self.instances.update_center(
                    str(
                        trace_update[
                            "object_id"
                        ]
                    ),
                    final_object_center,
                )

            # Обновява маркерите на историята в RViz.
            self.publish_traceback_markers()

        # Получава удобните за потребителя имена
        # на предмета и крайната цел.
        held_name = self.point_display_name(
            held_object
        )

        destination_name = (
            self.point_display_name(
                destination
            )
        )

        # Запазва крайната цел като последно
        # избран предмет или позиция.
        self.redis.set(
            LAST_TARGET_KEY,
            destination_key,
        )

        # Връща подробния резултат
        # от изпълнението или визуализацията.
        return self.result(
            request_id,
            True,
            (
                "HELD_OBJECT_PLACED"
                if execute
                else "HELD_OBJECT_PLACE_PREVIEWED"
            ),

            # Имена на държания предмет
            # и крайната цел.
            held_object_name=held_name,
            destination=destination.get("point"),
            destination_name=destination_name,

            # Последователност на решените
            # и изпълнени стъпки.
            sequence=[
                row[0]
                for row in solved
            ],

            # Очакваният краен център
            # на поставения предмет.
            destination_object_center_xyz=(
                self.xyz_dict(
                    final_object_center
                )
            ),

            # Позицията на хващача
            # при освобождаване на предмета.
            destination_place_xyz=(
                self.xyz_dict(
                    gripper_place
                )
            ),

            # Състояние на предмета и хващача.
            object_released=released,
            object_still_held=not released,

            # Физическа стойност за отваряне
            # на хващача при успешно освобождаване.
            gripper_open_servo_deg=(
                PICK_GRIPPER_OPEN_SERVO_DEG
                if released
                else None
            ),

            executed=execute,
            height_aware=True,

            # Геометрични параметри
            # на двата предмета.
            held_object_geometry=held_geometry,
            destination_geometry=(
                destination_geometry
            ),

            # Данни за безопасното
            # транзитно движение.
            transit_z=transit_z,
            transit_blockers=blockers,

            # Резултат от обновяването на историята.
            trace_update=trace_update,
            history_committed=history_committed,

            # Кратки структурирани факти,
            # използвани от разговорния агент.
            response_facts={
                "operation": (
                    "place_held_object"
                ),
                "object": held_name,
                "destination": destination_name,
                "object_released": released,
                "object_still_held": (
                    not released
                ),
            },
        )