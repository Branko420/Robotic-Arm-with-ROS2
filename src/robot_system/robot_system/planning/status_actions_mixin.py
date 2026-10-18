#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартна библиотека за математически проверки
# и типови анотации.
import math
from typing import Any, Dict, List, Mapping, Tuple

# Помощна функция за прочитане на числова
# конфигурационна стойност от променливите на средата.
from .common import env_float

# Безопасни диапазони на физическите сервомотори.
from .constants import SAFE_SERVO_RANGE

# Конфигурационни стойности за:
# - последно избраната цел;
# - продължителността на ръчно ставно движение;
# - толерантността при именувани позиции.
from .settings import (
    LAST_TARGET_KEY,
    MANUAL_JOINT_MOVE_DURATION_SEC,
    POSE_MATCH_TOLERANCE_DEG,
)


class StatusActionsMixin:
    """
    Съдържа действията за получаване на текущото състояние
    на манипулатора и за директно управление на неговата позиция.

    Поддържаните операции включват:
    - извеждане на текущите ставни и сервомоторни стойности;
    - получаване на текущата декартова позиция;
    - запазване, извеждане и изтриване на точки;
    - движение до запазена точка;
    - относително декартово движение;
    - директно задаване на сервомоторни ъгли.
    """

    def handle_joint_status(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Връща текущото отворено състояние
        на ставите и физическите сервомотори.
        """

        # Получава най-добрите налични сервомоторни стойности
        # и статуса от физическия изпълнител.
        servo_degrees, physical_status = (
            self.current_servo_degrees()
        )

        # Връща както физическите сервомоторни стойности,
        # така и ставните стойности от цифровия модел в RViz.
        return self.result(
            request_id,
            True,
            "JOINT_STATUS",
            servo_degrees=servo_degrees,
            rviz_joint_degrees=self.current_joint_deg(),
            physical_status=physical_status,

            # Показва, че стойностите са командвани
            # и не са измерени чрез физически енкодери.
            state_kind="open_loop_commanded_angles",
            has_encoder_feedback=False,

            # Извлича информация дали физическият
            # изпълнител в момента интерполира движение.
            moving=physical_status.get("moving"),
        )

    def handle_current_pose(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Връща текущата декартова позиция
        на крайния механизъм.
        """

        return self.result(
            request_id,
            True,
            "CURRENT_CARTESIAN_POSE",
            pose=self.current_cartesian_pose(),
        )

    def handle_save_current_point(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Запазва текущата декартова позиция
        под зададено потребителско име.
        """

        # Получава името от полето name,
        # а при липса използва полето target.
        name = str(
            request.get("name")
            or request.get("target")
            or ""
        ).strip()

        # Не допуска запазване на точка без име.
        if not name:
            return self.result(
                request_id,
                False,
                "POINT_NAME_REQUIRED",
            )

        # Запазва текущата позиция в Redis.
        saved = self.save_current_point(name)

        # Връща запазената точка и структурирани факти
        # за разговорния агент.
        return self.result(
            request_id,
            True,
            "CURRENT_POINT_SAVED",
            point=saved,
            response_facts={
                "operation": "save_point",
                "name": saved["name"],
            },
        )

    def handle_list_saved_points(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Връща всички запазени декартови точки.
        """

        # Зарежда речника със запазените точки.
        points = self.load_saved_points()

        # Преобразува стойностите в списък
        # и връща общия им брой.
        return self.result(
            request_id,
            True,
            "SAVED_POINTS",
            points=list(points.values()),
            count=len(points),
        )

    def handle_delete_saved_point(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Изтрива запазена декартова точка по име.
        """

        # Получава името от name или target.
        name = str(
            request.get("name")
            or request.get("target")
            or ""
        ).strip()

        # Извършва изтриването само при непразно име.
        deleted = (
            self.delete_saved_point(name)
            if name
            else False
        )

        # Връща резултат според това
        # дали точката е била намерена.
        return self.result(
            request_id,
            deleted,
            (
                "SAVED_POINT_DELETED"
                if deleted
                else "SAVED_POINT_NOT_FOUND"
            ),
            name=name,
        )

    def handle_move_to_saved_point(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Премества манипулатора до предварително
        запазена декартова точка.
        """

        # Получава името на точката.
        name = str(
            request.get("name")
            or request.get("target")
            or ""
        ).strip()

        # Търси точката само при зададено име.
        saved = (
            self.get_saved_point(name)
            if name
            else None
        )

        # Връща грешка, когато точката не съществува.
        if saved is None:
            return self.result(
                request_id,
                False,
                "SAVED_POINT_NOT_FOUND",
                name=name,
            )

        # Създава и валидира декартовата цел.
        target_xyz = self.validate_xyz(
            (
                saved["x"],
                saved["y"],
                saved["z"],
            )
        )

        # Решава обратната кинематика
        # за запазената точка.
        solution, debug = (
            self.solve_cartesian_target(
                target_xyz
            )
        )

        # Връща диагностична грешка,
        # когато няма допустимо IK решение.
        if solution is None:
            return self.result(
                request_id,
                False,
                "SAVED_POINT_IK_FAILED",
                name=name,
                target_xyz=self.xyz_dict(
                    target_xyz
                ),
                closest_error=(
                    None
                    if debug is None
                    else debug.get("error")
                ),
            )

        # По подразбиране движението се изпълнява.
        execute = bool(
            request.get(
                "execute",
                True,
            )
        )

        if execute:
            # Изпълнява движението в RViz
            # и върху физическия манипулатор.
            ok, servo = (
                self.motion.execute_joints(
                    solution["candidate"],
                    label=f"saved_point:{name}",
                )
            )

            # Прекратява при отменено движение.
            if not ok:
                return self.result(
                    request_id,
                    False,
                    "SAVED_POINT_MOVE_CANCELLED",
                    name=name,
                )

            # Записва достигнатата декартова позиция.
            self.persist_current_cartesian_pose(
                solution["reached"],
                source=f"saved_point:{name}",
            )

        else:
            # При предварителна визуализация
            # публикува движението само към RViz.
            self.motion.publish_rviz_joints(
                solution["candidate"],
                self.motion.timings.rviz_move_sec,
            )

            # Изчислява физическите сервомоторни стойности,
            # без да ги изпраща към Raspberry Pi.
            servo = (
                self.motion.real.joints_to_servo(
                    solution["candidate"]
                )
            )

        # Връща подробен резултат
        # за движението до запазената точка.
        return self.result(
            request_id,
            True,
            (
                "SAVED_POINT_REACHED"
                if execute
                else "SAVED_POINT_PREVIEWED"
            ),
            name=name,
            target_xyz=self.xyz_dict(
                target_xyz
            ),
            reached_xyz=self.xyz_dict(
                solution["reached"]
            ),
            physical_servo_degrees=servo,
            executed=execute,
            response_facts={
                "operation": "move_to_saved_point",
                "name": name,
            },
        )

    def handle_move_relative(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Изпълнява относително декартово движение
        спрямо текущата позиция на крайния механизъм.
        """

        # Получава текущата декартова позиция.
        current = self.current_cartesian_pose()

        # Получава отместванията по X, Y и Z.
        delta = (
            float(
                request.get(
                    "dx",
                    0.0,
                )
            ),
            float(
                request.get(
                    "dy",
                    0.0,
                )
            ),
            float(
                request.get(
                    "dz",
                    0.0,
                )
            ),
        )

        # Добавя отместването към текущата позиция
        # и валидира получената цел.
        target_xyz = self.validate_xyz(
            (
                float(current["x"]) + delta[0],
                float(current["y"]) + delta[1],
                float(current["z"]) + delta[2],
            )
        )

        # Решава обратната кинематика
        # за новата относителна позиция.
        solution, debug = (
            self.solve_cartesian_target(
                target_xyz
            )
        )

        # Връща грешка при липса на решение.
        if solution is None:
            return self.result(
                request_id,
                False,
                "RELATIVE_MOVE_IK_FAILED",
                current_pose=current,
                delta=self.xyz_dict(delta),
                target_xyz=self.xyz_dict(
                    target_xyz
                ),
                closest_error=(
                    None
                    if debug is None
                    else debug.get("error")
                ),
            )

        # По подразбиране движението се изпълнява.
        execute = bool(
            request.get(
                "execute",
                True,
            )
        )

        if execute:
            # Изпълнява ставното движение.
            ok, servo = (
                self.motion.execute_joints(
                    solution["candidate"],
                    label="move_relative",
                )
            )

            # Прекратява при отменено движение.
            if not ok:
                return self.result(
                    request_id,
                    False,
                    "RELATIVE_MOVE_CANCELLED",
                )

            # Записва действително достигнатата позиция.
            updated = (
                self.persist_current_cartesian_pose(
                    solution["reached"],
                    source="move_relative",
                )
            )

        else:
            # При предварителна визуализация
            # публикува решението само към RViz.
            self.motion.publish_rviz_joints(
                solution["candidate"],
                self.motion.timings.rviz_move_sec,
            )

            # Изчислява физическите сервомоторни стойности.
            servo = (
                self.motion.real.joints_to_servo(
                    solution["candidate"]
                )
            )

            # Връща достигнатата позиция
            # без да я записва като реално състояние.
            updated = self.xyz_dict(
                solution["reached"]
            )

        # Връща резултат за относителното движение.
        return self.result(
            request_id,
            True,
            (
                "RELATIVE_MOVE_EXECUTED"
                if execute
                else "RELATIVE_MOVE_PREVIEWED"
            ),
            start_pose=current,
            delta={
                "x": delta[0],
                "y": delta[1],
                "z": delta[2],
            },
            target_xyz=self.xyz_dict(
                target_xyz
            ),
            reached_xyz=updated,
            physical_servo_degrees=servo,
            executed=execute,
            response_facts={
                "operation": "relative_move",
                "delta": {
                    "x": delta[0],
                    "y": delta[1],
                    "z": delta[2],
                },
            },
        )

    def handle_manual_joint_move(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Задава директно физически сервомоторни ъгли.

        В заявката се очаква речник angles, съдържащ
        една или повече стойности за сервомоторите.
        Непосочените сервомотори запазват текущите си стойности.
        """

        # Получава речника с желаните ъгли.
        raw_angles = request.get("angles")

        # Не допуска заявка без речник angles.
        if not isinstance(raw_angles, dict):
            return self.result(
                request_id,
                False,
                "JOINT_ANGLES_REQUIRED",
            )

        # Валидирани и нормализирани
        # сервомоторни стойности.
        requested: Dict[str, float] = {}

        # Проверява всяка подадена стойност.
        for key, raw_value in raw_angles.items():
            # Не допуска непознато име на сервомотор.
            if key not in SAFE_SERVO_RANGE:
                return self.result(
                    request_id,
                    False,
                    "UNKNOWN_JOINT",
                    joint=key,
                    allowed=list(
                        SAFE_SERVO_RANGE
                    ),
                )

            try:
                # Преобразува стойността до float.
                value = float(raw_value)

            except (TypeError, ValueError):
                # Връща грешка при невалиден тип
                # или текстова стойност.
                return self.result(
                    request_id,
                    False,
                    "INVALID_JOINT_ANGLE",
                    joint=key,
                    value=raw_value,
                )

            # Не допуска NaN или безкрайна стойност.
            if not math.isfinite(value):
                return self.result(
                    request_id,
                    False,
                    "INVALID_JOINT_ANGLE",
                    joint=key,
                    value=raw_value,
                )

            # Получава безопасните граници
            # на съответния сервомотор.
            low, high = SAFE_SERVO_RANGE[key]

            # Проверява дали стойността е в диапазона.
            if not low <= value <= high:
                return self.result(
                    request_id,
                    False,
                    "JOINT_ANGLE_OUT_OF_RANGE",
                    joint=key,
                    value=value,
                    minimum=low,
                    maximum=high,
                )

            # Закръгля валидната стойност
            # до три знака след десетичната запетая.
            requested[key] = round(
                value,
                3,
            )

        # Не допуска празен речник след валидирането.
        if not requested:
            return self.result(
                request_id,
                False,
                "JOINT_ANGLES_REQUIRED",
            )

        # Получава текущите командвани
        # сервомоторни стойности и физическия статус.
        current_servo, status_before = (
            self.current_servo_degrees()
        )

        # Започва с текущата пълна сервомоторна позиция.
        target_servo = dict(
            current_servo
        )

        # Заменя само изрично зададените сервомотори.
        target_servo.update(
            requested
        )

        # По подразбиране позицията се изпълнява.
        execute = bool(
            request.get(
                "execute",
                True,
            )
        )

        # Получава времетраенето на движението.
        #
        # Минималната разрешена стойност е 0.2 секунди.
        duration_sec = max(
            0.2,
            float(
                request.get(
                    "duration_sec",
                    MANUAL_JOINT_MOVE_DURATION_SEC,
                )
            ),
        )

        if execute:
            # Изпълнява пълната сервомоторна позиция
            # върху RViz и физическия манипулатор.
            ok, sent = (
                self.motion.execute_servo_pose(
                    target_servo,
                    label="manual_joint_angles",
                    duration_sec=duration_sec,
                    wait=True,
                )
            )

            # Прекратява при отменено движение.
            if not ok:
                return self.result(
                    request_id,
                    False,
                    "MANUAL_JOINT_MOVE_CANCELLED",
                )

            # Преобразува изпратените сервомоторни стойности
            # обратно към ставни стойности за кинематичния модел.
            joints_after = (
                self.motion.real.servo_to_joints(
                    sent
                )
            )

            # Изчислява и записва новата декартова позиция.
            self.persist_current_cartesian_pose(
                self.ik.fk_tool(
                    joints_after
                ),
                source="manual_joint_angles",
            )

        else:
            # При предварителна визуализация
            # преобразува сервомоторите до RViz стави.
            joints = (
                self.motion.real.servo_to_joints(
                    target_servo
                )
            )

            # Публикува основните стави в RViz.
            self.motion.publish_rviz_joints(
                joints,
                self.motion.timings.rviz_move_sec,
            )

            # Публикува отделно положението
            # на хващача в RViz.
            self.motion.publish_rviz_gripper_position(
                self.motion.real.gripper_servo_to_rviz_rad(
                    target_servo["gripper"]
                )
            )

            # При preview връща изчислената цел,
            # без да изпраща физическа команда.
            sent = target_servo

        # Връща подробен резултат за зададените ъгли.
        return self.result(
            request_id,
            True,
            (
                "JOINT_ANGLES_SET"
                if execute
                else "JOINT_ANGLES_PREVIEWED"
            ),

            # Само стойностите, изрично подадени
            # в текущата заявка.
            requested_servo_degrees=requested,

            # Пълната крайна сервомоторна позиция.
            servo_degrees=sent,

            # Сервомоторното състояние преди движението.
            previous_servo_degrees=current_servo,

            # Физическият статус преди изпълнението.
            physical_status_before=status_before,

            # Използваното времетраене на движението.
            duration_sec=duration_sec,

            # Показва дали е изпратена реална команда.
            executed=execute,

            # Системата не получава измерени ъгли
            # от физически енкодери.
            has_encoder_feedback=False,
        )