#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартна библиотека за математически операции
# и типови анотации.
import math
from typing import Any, Dict, List

# Помощна функция за прочитане на числова стойност
# от променливите на средата.
from .common import env_float

# Redis ключ за последно избрания предмет или цел.
from .settings import LAST_TARGET_KEY


class ObjectActionsMixin:
    """
    Съдържа основните действия към отделен предмет
    от работното пространство.

    Поддържаните операции включват:
    - предварителна визуализация;
    - движение до центъра на предмета;
    - движение над предмета;
    - приближаване;
    - хващане;
    - поставяне.
    """

    def handle_object_action(
        self,
        request_id: str,
        request: Dict[str, Any],
        action: str,
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Обработва действие, насочено към конкретен предмет.

        Методът:
        1. определя предмета;
        2. избира подходяща целева позиция;
        3. прилага запазени и временни корекции;
        4. решава обратната кинематика;
        5. проверява точността;
        6. изпълнява движението или го визуализира в RViz.
        """

        # Получава потребителското описание на целта.
        #
        # При липсващо поле target се използва
        # най-близкият предмет.
        target_name = str(
            request.get(
                "target",
                "nearest_object",
            )
        ).strip()

        # Определя конкретния предмет
        # от наличните точки в работното пространство.
        point = self.targets.resolve_target(
            points,
            target_name,
        )

        # Връща грешка, когато не е намерен
        # предмет, отговарящ на описанието.
        if point is None:
            return self.result(
                request_id,
                False,
                "TARGET_NOT_FOUND",
                target=target_name,

                # Връща списък с наличните точки
                # за диагностични цели.
                available=[
                    item.get("point")
                    for item in points
                ],
            )

        # Забранява движение към предмет,
        # маркиран като препятствие за безопасност.
        if point.get("safety_blocker"):
            return self.result(
                request_id,
                False,
                "SAFETY_BLOCKED",
                target=point.get("point"),
            )

        # Създава устойчив идентификатор на целта.
        #
        # Предпочита уникалния instance_id,
        # след това името на точката,
        # а накрая първоначалното описание.
        stable_target_key = str(
            point.get("instance_id")
            or point.get("point")
            or target_name
        )

        # Запомня конкретния предмет като последно
        # успешно определена цел.
        self.targets.remember_target(point)

        # Определя дали движението трябва
        # да бъде реално изпълнено.
        execute = bool(
            request.get(
                "execute",
                not self.preview_only_default,
            )
        )

        # Действието preview винаги се изпълнява
        # само като визуализация в RViz.
        if action == "preview":
            execute = False

        # При действие pick използва отделната
        # пълна последователност за хващане.
        if action == "pick":
            return self.execute_pick_sequence(
                request_id,
                point,
                execute,
            )

        # Определя позицията по подразбиране
        # според поисканото действие.
        default_position = (
            # За приближаване и движение над предмета
            # се използва безопасната позиция approach.
            "approach"
            if action in {
                "approach",
                "above",
                "go_above",
                "move_above",
            }

            # При поставяне се използва позицията place.
            else "place"
            if action == "place"

            # При останалите действия се използва
            # центърът на предмета.
            else "center"
        )

        # Получава изрично зададения режим
        # или използва избраната стойност по подразбиране.
        position_mode = str(
            request.get(
                "target_position",
                default_position,
            )
        ).strip().lower()

        # Нормализира различните имена
        # за позиция над предмета.
        if position_mode in {
            "approach",
            "above",
            "over",
        }:
            position_mode = "approach"

            # Изчислява безопасна позиция над предмета,
            # съобразена с неговата височина
            # и минималния разрешен просвет.
            target_xyz = (
                self.targets.above_object_position(
                    point,
                    clearance=(
                        self.above_object_clearance
                    ),
                    minimum_clearance=(
                        self.minimum_above_clearance
                    ),
                )
            )

        else:
            # При останалите режими извлича
            # съответната позиция на предмета.
            target_xyz = (
                self.targets.position_for_mode(
                    point,
                    position_mode,
                )
            )

        # Зарежда запазените корекции
        # за конкретния предмет.
        offsets = self.targets.load_offsets(
            stable_target_key
        )

        # Прилага:
        # - постоянните корекции на предмета;
        # - временните корекции dx, dy и dz от заявката.
        #
        # След това проверява дали позицията е допустима.
        target_xyz = self.validate_xyz(
            (
                target_xyz[0]
                + offsets["x"]
                + float(
                    request.get(
                        "dx",
                        0.0,
                    )
                ),
                target_xyz[1]
                + offsets["y"]
                + float(
                    request.get(
                        "dy",
                        0.0,
                    )
                ),
                target_xyz[2]
                + offsets["z"]
                + float(
                    request.get(
                        "dz",
                        0.0,
                    )
                ),
            )
        )

        # Максимално допустима разлика между
        # желаната и достигнатата позиция.
        strict_target_tolerance = max(
            0.0005,
            env_float(
                "PLANNER_HARD_MAX_TARGET_ERROR_M",
                0.008,
            ),
        )

        # Решава обратната кинематика
        # за изчислената декартова цел.
        solution, debug = (
            self.solve_cartesian_target(
                target_xyz
            )
        )

        # При липса на валидно IK решение
        # публикува диагностичен маркер и връща грешка.
        if solution is None:
            self.publish_plan_markers(
                target_xyz,
                None,
                (
                    f"IK FAILED: "
                    f"{point.get('point')}"
                ),
            )

            return self.result(
                request_id,
                False,
                "IK_FAILED",
                target=point.get("point"),
                target_xyz=self.xyz_dict(
                    target_xyz
                ),
                closest_error=(
                    None
                    if debug is None
                    else debug.get("error")
                ),
            )

        # Изчислява действителното евклидово разстояние
        # между желаната и достигнатата позиция.
        reached_error = math.dist(
            tuple(
                float(value)
                for value in solution["reached"]
            ),
            tuple(
                float(value)
                for value in target_xyz
            ),
        )

        # Отхвърля решението, когато позиционната грешка
        # е по-голяма от допустимата.
        if reached_error > strict_target_tolerance:
            # Публикува маркери за желаната
            # и действително достигнатата позиция.
            self.publish_plan_markers(
                target_xyz,
                solution["reached"],
                (
                    f"TARGET MISMATCH: "
                    f"{point.get('point')}"
                ),
            )

            return self.result(
                request_id,
                False,
                "TARGET_MISMATCH",
                target=point.get("point"),
                target_xyz=self.xyz_dict(
                    target_xyz
                ),
                reached_xyz=self.xyz_dict(
                    solution["reached"]
                ),
                position_error=reached_error,
                allowed_error=(
                    strict_target_tolerance
                ),
                applied_target_offsets=offsets,
            )

        # Записва устойчивия идентификатор
        # като последно избрана цел в Redis.
        self.redis.set(
            LAST_TARGET_KEY,
            stable_target_key,
        )

        # Публикува маркери за планираната
        # и достигнатата позиция в RViz.
        self.publish_plan_markers(
            target_xyz,
            solution["reached"],
            point.get(
                "point",
                target_name,
            ),
        )

        if execute:
            # Изпълнява ставното решение
            # в RViz и върху физическия манипулатор.
            ok, servo_angles = (
                self.motion.execute_joints(
                    solution["candidate"],
                    label=action,
                )
            )

            # Връща грешка, ако движението
            # е прекратено или отменено.
            if not ok:
                return self.result(
                    request_id,
                    False,
                    "MOVE_CANCELLED",
                    target=point.get("point"),
                )

            # Запазва действително достигнатата
            # декартова позиция като текущо състояние.
            self.persist_current_cartesian_pose(
                solution["reached"],
                source=(
                    f"object_move:{action}"
                ),
            )

            # При действие place отваря хващача
            # след достигане на целевата позиция.
            if action == "place":
                if not self.motion.execute_gripper(
                    close=False
                ):
                    return self.result(
                        request_id,
                        False,
                        "PLACE_GRIPPER_CANCELLED",
                    )

                # Изчиства състоянието за държан предмет,
                # защото предметът вече е освободен.
                self.clear_held_object()

        else:
            # При предварителна визуализация
            # публикува ставното решение само в RViz.
            self.motion.publish_rviz_joints(
                solution["candidate"],
                self.motion.timings.rviz_move_sec,
            )

            # Изчислява физическите сервомоторни стойности,
            # без да ги изпраща към Raspberry Pi.
            servo_angles = (
                self.motion.real.joints_to_servo(
                    solution["candidate"]
                )
            )

        # Връща подробна информация
        # за изпълненото или визуализираното движение.
        return self.result(
            request_id,
            True,
            (
                "EXECUTED"
                if execute
                else "PREVIEWED"
            ),

            # Идентификация на целта.
            target=point.get("point"),

            # Използваният вид позиция.
            target_position=position_mode,

            # Желаната декартова позиция.
            target_xyz=self.xyz_dict(
                target_xyz
            ),

            # Действително достигнатата позиция.
            reached_xyz=self.xyz_dict(
                solution["reached"]
            ),

            # Изчислената позиционна грешка.
            position_error=solution["error"],

            # Изчислените ставни стойности в градуси.
            joint_degrees=solution["degs"],

            # Физическите сервомоторни стойности.
            physical_servo_degrees=servo_angles,

            # Приложените постоянни корекции.
            offsets=offsets,

            # Дали движението е изпълнено реално.
            executed=execute,

            # Показва дали системата работи
            # само като цифров модел в RViz.
            simulation_only=(
                not self.motion.real.enabled
            ),

            # Времетраене на отделните движения.
            timings=self.timing_dict(),
        )