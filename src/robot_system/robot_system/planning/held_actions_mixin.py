#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за математически операции
# и типови анотации.
import math
from typing import Any, Dict, List, Mapping, Tuple

# Помощна функция за четене на числова
# конфигурационна стойност от средата.
from .common import env_float

# Безопасни диапазони на физическите сервомотори.
from .constants import SAFE_SERVO_RANGE

# Конфигурационни ключове и настройки,
# използвани при движенията.
from .settings import (
    LAST_TARGET_KEY,
    MANUAL_JOINT_MOVE_DURATION_SEC,
    POSE_MATCH_TOLERANCE_DEG,
)


class HeldActionsMixin:
    """
    Съдържа действията за преместване на предмет,
    който вече се държи от хващача.

    По време на движението хващачът остава затворен
    и не се изпълнява нова последователност за хващане.
    """

    def handle_move_held_to(
        self,
        request_id: str,
        request: Dict[str, Any],
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Премества вече държан предмет към зададена цел,
        без да го освобождава.

        Движението използва височината на предмета,
        околните препятствия и безопасна транзитна височина.
        """

        # Зарежда текущото състояние
        # на предмета в хващача.
        held_state = self.load_held_object()

        # Прекратява операцията, когато системата
        # няма регистриран държан предмет.
        if held_state is None:
            return self.result(
                request_id,
                False,
                "NO_HELD_OBJECT",
            )

        # Допълва състоянието на държания предмет
        # с актуалните данни от работното пространство.
        held = self.hydrate_held_object(
            held_state,
            points,
        )

        # Получава името на крайната цел.
        #
        # Предпочита полето destination, след това target,
        # а при липса използва най-близката повърхност.
        destination_name = str(
            request.get("destination")
            or request.get("target")
            or "nearest_surface"
        ).strip()

        # Определя конкретната крайна цел
        # от точките в работното пространство.
        destination = self.targets.resolve_target(
            points,
            destination_name,
        )

        # Връща грешка, когато крайната цел
        # не може да бъде намерена.
        if destination is None:
            return self.result(
                request_id,
                False,
                "DESTINATION_NOT_FOUND",
                destination=destination_name,
            )

        # Забранява движение към предмет или зона,
        # маркирани като препятствие за безопасност.
        if destination.get("safety_blocker"):
            return self.result(
                request_id,
                False,
                "SAFETY_BLOCKED",
            )

        # Получава вида на целевата позиция.
        #
        # По подразбиране се използва безопасната
        # позиция за подход над крайната цел.
        mode = str(
            request.get("target_position")
            or "approach"
        ).strip().lower()

        # При режим за движение над предмета
        # изчислява безопасна позиция за подход,
        # съобразена с височината на държания предмет.
        if mode in {
            "approach",
            "above",
            "over",
        }:
            mode = "approach"

            final_target = (
                self.targets.held_object_approach_position(
                    held,
                    destination,
                    clearance=self.above_object_clearance,
                    minimum_clearance=self.minimum_above_clearance,
                )
            )

        else:
            # При друг режим използва съответната
            # позиция на крайната цел.
            final_target = (
                self.targets.position_for_mode(
                    destination,
                    mode,
                )
            )

        # Създава устойчив идентификатор
        # на крайната цел.
        destination_key = str(
            destination.get("instance_id")
            or destination.get("point")
            or destination_name
        )

        # Зарежда натрупаните корекции
        # за конкретната крайна цел.
        offsets = self.targets.load_offsets(
            destination_key
        )

        # Прилага корекциите по X, Y и Z
        # и валидира получената декартова позиция.
        final_target = self.validate_xyz(
            (
                final_target[0] + offsets["x"],
                final_target[1] + offsets["y"],
                final_target[2] + offsets["z"],
            )
        )

        # Получава геометрията на предмета,
        # който се намира в хващача.
        held_geometry = (
            self.targets.object_geometry(held)
        )

        # Опитва да използва актуалния
        # проследяван център на предмета.
        current_center = held.get(
            "current_center"
        )

        if isinstance(
            current_center,
            Mapping,
        ):
            # Използва актуалния център,
            # записан след предишно движение.
            object_center = (
                float(current_center["x"]),
                float(current_center["y"]),
                float(current_center["z"]),
            )

        else:
            # При липса използва геометричния център
            # от данните за предмета.
            center_row = held_geometry["center"]

            object_center = (
                center_row["x"],
                center_row["y"],
                center_row["z"],
            )

        # Получава отместването между центъра
        # на предмета и позицията на хващача.
        grasp_offset = held_geometry[
            "grasp_offset"
        ]

        # Изчислява началната позиция на хващача
        # спрямо актуалния център на предмета.
        start_gripper = (
            object_center[0] + grasp_offset["x"],
            object_center[1] + grasp_offset["y"],
            object_center[2] + grasp_offset["z"],
        )

        # Идентификаторите на държания предмет
        # и крайната цел се изключват от проверката
        # за препятствия по пътя.
        ignored_ids = {
            str(
                held.get("instance_id")
                or held.get("point")
                or ""
            ),
            destination_key,
        }

        # Минималната транзитна височина трябва
        # да бъде поне колкото началната или крайната Z позиция.
        minimum_transit_z = max(
            start_gripper[2],
            final_target[2],
        )

        # Изчислява безопасната височина за движение
        # над препятствията в работното пространство.
        transit_z, blockers = (
            self.height_aware_transit_z(
                points,
                held,
                start_gripper,
                final_target,
                minimum_transit_z,
                ignored_ids=ignored_ids,
            )
        )

        # Прекратява операцията, когато необходимата
        # транзитна височина е извън достижимото пространство.
        if transit_z is None:
            return self.result(
                request_id,
                False,
                "TRANSIT_CLEARANCE_UNREACHABLE",
                blockers=blockers,
                maximum_transit_z=self.maximum_transit_z,
            )

        # Списък с последователните декартови цели.
        step_targets: List[
            Tuple[
                str,
                Tuple[float, float, float],
            ]
        ] = []

        # Първо повдига хващача вертикално над
        # началната позиция, когато това е необходимо.
        if transit_z > start_gripper[2] + 0.005:
            step_targets.append(
                (
                    "held_source_transit",
                    (
                        start_gripper[0],
                        start_gripper[1],
                        transit_z,
                    ),
                )
            )

        # След това премества хващача хоризонтално
        # над крайната позиция при транзитната височина.
        if transit_z > final_target[2] + 0.005:
            step_targets.append(
                (
                    "held_destination_transit",
                    (
                        final_target[0],
                        final_target[1],
                        transit_z,
                    ),
                )
            )

        # Последната стъпка достига
        # точната крайна позиция.
        step_targets.append(
            (
                "held_destination",
                final_target,
            )
        )

        # Списък с успешно решените
        # стъпки на обратната кинематика.
        solved = []

        # Начални ставни стойности
        # за първата IK стъпка.
        current_for_step = (
            self.current_joint_deg()
        )

        # Решава всяка стъпка последователно,
        # като използва предишното решение
        # за начално състояние на следващата.
        for step_name, xyz in step_targets:
            # Валидира декартовата цел.
            xyz = self.validate_xyz(xyz)

            # Изчислява ставното решение
            # чрез обратната кинематика.
            solution, debug = (
                self.solve_cartesian_target(
                    xyz,
                    current_joint_deg=current_for_step,
                )
            )

            # При неуспешно IK решение
            # прекратява цялата последователност.
            if solution is None:
                return self.result(
                    request_id,
                    False,
                    "HELD_MOVE_IK_FAILED",
                    failed_step=step_name,
                    target_xyz=self.xyz_dict(xyz),
                    closest_error=(
                        None
                        if debug is None
                        else debug.get("error")
                    ),
                )

            # Запазва името на стъпката,
            # декартовата цел и IK решението.
            solved.append(
                (
                    step_name,
                    xyz,
                    solution,
                )
            )

            # Следващата IK стъпка започва
            # от решението на текущата.
            current_for_step = dict(
                solution["degs"]
            )

        # Определя дали движението трябва
        # да се изпълни физически.
        execute = bool(
            request.get(
                "execute",
                not self.preview_only_default,
            )
        )

        # Запазва крайната цел като последно
        # избран предмет или позиция.
        self.redis.set(
            LAST_TARGET_KEY,
            destination_key,
        )

        # Публикува визуални маркери
        # за плана и достигнатата позиция.
        self.publish_plan_markers(
            final_target,
            solved[-1][2]["reached"],
            f"held_to:{destination_name}",
        )

        # Последни изчислени физически
        # сервомоторни стойности.
        servo_angles = None

        if execute:
            # Изпълнява всички стъпки
            # последователно върху манипулатора.
            for (
                step_name,
                _xyz,
                solution,
            ) in solved:
                ok, servo_angles = (
                    self.motion.execute_joints(
                        solution["candidate"],
                        label=(
                            f"move_held_to:{step_name}"
                        ),
                    )
                )

                # Прекратява последователността,
                # когато някое движение е отменено.
                if not ok:
                    return self.result(
                        request_id,
                        False,
                        "HELD_MOVE_CANCELLED",
                        failed_step=step_name,
                    )

        else:
            # При предварителна визуализация
            # публикува първата ставна цел само в RViz.
            self.motion.publish_rviz_joints(
                solved[0][2]["candidate"],
                self.motion.timings.rviz_move_sec,
            )

            # Изчислява физическите сервомоторни
            # стойности за последната цел.
            servo_angles = (
                self.motion.real.joints_to_servo(
                    solved[-1][2]["candidate"]
                )
            )

        # По подразбиране запазва предишното
        # състояние на държания предмет.
        updated_held = held_state

        # След реално движение актуализира
        # центъра на държания предмет спрямо
        # крайната позиция на хващача.
        if execute:
            updated_held = (
                self.update_held_object_from_gripper(
                    held,
                    final_target,
                )
            )

        # Получава удобните за потребителя имена
        # на предмета и крайната цел.
        held_name = self.point_display_name(
            held
        )
        destination_display = (
            self.point_display_name(
                destination
            )
        )

        # Връща подробен резултат за изпълненото
        # или визуализираното движение.
        return self.result(
            request_id,
            True,
            (
                "HELD_OBJECT_MOVED"
                if execute
                else "HELD_OBJECT_MOVE_PREVIEWED"
            ),
            held_object=updated_held,
            held_object_name=held_name,
            destination=destination.get("point"),
            destination_name=destination_display,
            target_position=mode,
            target_xyz=self.xyz_dict(
                final_target
            ),
            physical_servo_degrees=servo_angles,

            # Хващачът остава затворен
            # през цялото движение.
            gripper_kept_closed=True,

            # Предметът не се освобождава
            # в края на тази операция.
            object_released=False,

            executed=execute,
            height_aware=True,
            held_object_geometry=held_geometry,
            destination_geometry=(
                self.targets.object_geometry(
                    destination
                )
            ),
            transit_z=transit_z,
            transit_blockers=blockers,

            # Последователност на изпълнените стъпки.
            sequence=[
                row[0]
                for row in solved
            ],

            # Кратки структурирани факти
            # за отговора на разговорния агент.
            response_facts={
                "operation": "move_held_object",
                "object": held_name,
                "destination": destination_display,
                "object_released": False,
                "object_still_held": True,
            },
        )