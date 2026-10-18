#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за измерване на време
# и типови анотации.
import time
from typing import Any, Dict, Mapping, Optional, Tuple

# UDP клиент за управление на физическия манипулатор.
from .arm_client import RealArmUdpClient

# Настройки за времетраенето на движенията,
# ограничаване на стойности и четене на конфигурация.
from .common import MotionTimings, clamp, env_float

# Ред на ставите в цифровия модел
# и безопасни диапазони на сервомоторите.
from .constants import JOINT_ORDER, SAFE_SERVO_RANGE

# Публикува синхронизираното състояние
# на цифровия модел чрез /joint_states.
from .robot_model_state import RobotModelStatePublisher


class MotionBridge:
    """
    Синхронизира решенията на планиращия модул
    с RViz и физическия манипулатор.

    RViz се обновява директно чрез темата /joint_states.
    Физическият манипулатор се управлява отделно чрез UDP.

    В този път за изпълнение не се използват допълнителни
    preview възли или теми за траектории на контролери.
    """

    def __init__(
        self,
        joint_state_publisher: Any,
        clock: Any,
        logger: Any,
        cancel_check: Any,
    ) -> None:
        """
        Инициализира моста между планирането,
        цифровия модел и физическия манипулатор.
        """

        # Логер за диагностични съобщения.
        self.logger = logger

        # Функция, която проверява дали текущото
        # движение трябва да бъде прекратено.
        self.cancel_check = cancel_check

        # Зарежда времената за движение
        # от променливите на средата.
        self.timings = MotionTimings.from_env()

        # Създава UDP клиент
        # за физическия манипулатор.
        self.real = RealArmUdpClient(
            logger
        )

        # Създава издател на състоянието
        # на цифровия модел в RViz.
        self.model = RobotModelStatePublisher(
            joint_state_publisher,
            clock,
            logger,
        )

    def close(self) -> None:
        """
        Освобождава ресурсите на цифровия модел
        и UDP клиента.
        """

        # Спира издателя на състоянието на модела.
        self.model.close()

        # Затваря UDP сокета към Raspberry Pi.
        self.real.close()

    def _sleep_interruptible(
        self,
        seconds: float,
    ) -> bool:
        """
        Изчаква зададено време, като периодично
        проверява дали операцията е отменена.

        Връща True при нормално изчакване
        и False при получена команда за прекратяване.
        """

        # Определя крайния момент за изчакване.
        deadline = (
            time.monotonic()
            + max(
                0.0,
                seconds,
            )
        )

        # Изчаква на кратки интервали,
        # за да може бързо да реагира при отмяна.
        while time.monotonic() < deadline:
            # Прекратява изчакването,
            # когато е получен сигнал за отмяна.
            if self.cancel_check():
                return False

            # Изчаква максимум 50 милисекунди
            # преди следващата проверка.
            time.sleep(
                min(
                    0.05,
                    max(
                        0.0,
                        deadline - time.monotonic(),
                    ),
                )
            )

        # Извършва последна проверка
        # след изтичане на времето.
        return not self.cancel_check()

    def publish_rviz_joints(
        self,
        joints: Mapping[str, float],
        duration_sec: float,
    ) -> None:
        """
        Публикува ставна цел към цифровия модел в RViz.
        """

        # Проверява дали са предоставени
        # всички необходими стави на ръката.
        missing = [
            name
            for name in JOINT_ORDER
            if name not in joints
        ]

        # Не допуска публикуване
        # на непълно състояние.
        if missing:
            raise ValueError(
                "Incomplete robot model arm state; "
                f"missing: {missing}"
            )

        # Задава целевите ставни стойности
        # и времето за интерполация.
        self.model.set_targets(
            {
                name: float(joints[name])
                for name in JOINT_ORDER
            },
            duration_sec,
        )

    def execute_joints(
        self,
        joints: Mapping[str, float],
        label: str,
        servo_override: Optional[
            Mapping[str, float]
        ] = None,
        wait: bool = True,
        *,
        wait_for_physical_completion: bool = False,
    ) -> Tuple[
        bool,
        Dict[str, float],
    ]:
        """
        Изпълнява ставно движение в RViz
        и върху физическия манипулатор.

        servo_override позволява директно задаване
        на физически сервомоторни стойности.

        wait определя дали методът трябва да изчака
        приключването на движението.

        wait_for_physical_completion използва отговора
        от Raspberry Pi за реално завършено движение.
        """

        # Проверява достъпността на физическия манипулатор,
        # когато реалното управление е включено.
        if self.real.enabled:
            self.real.ensure_available()

        # Стартира синхронизираното движение в RViz.
        self.publish_rviz_joints(
            joints,
            self.timings.rviz_move_sec,
        )

        # Преобразува и изпраща ставното движение
        # към физическия манипулатор.
        servo_angles = self.real.send_joints(
            joints,
            self.timings.real_move_sec,
            servo_override=servo_override,
            label=label,
            wait_for_completion=(
                wait_for_physical_completion
                and self.real.enabled
            ),
        )

        # При изключено изчакване
        # връща резултата веднага.
        if not wait:
            return True, servo_angles

        # Когато Raspberry Pi вече е потвърдило
        # завършването на движението, не се използва
        # допълнително времево изчакване.
        if (
            wait_for_physical_completion
            and self.real.enabled
        ):
            return (
                not self.cancel_check(),
                servo_angles,
            )

        # Избира времето за изчакване според това
        # дали физическият манипулатор е активен.
        movement_time = (
            self.timings.real_move_sec
            if self.real.enabled
            else self.timings.rviz_move_sec
        )

        # Записва информация за очакваното
        # време на движението и установяването.
        self.logger.info(
            f"{label}: waiting "
            f"{movement_time:.2f}s movement + "
            f"{self.timings.post_move_delay_sec:.2f}s settle"
        )

        # Изчаква времето за движение
        # и допълнителното време за установяване.
        ok = self._sleep_interruptible(
            movement_time
            + self.timings.post_move_delay_sec
        )

        return ok, servo_angles

    def publish_rviz_gripper_position(
        self,
        position_rad: float,
        duration_sec: Optional[float] = None,
    ) -> None:
        """
        Публикува целевото положение
        на хващача в RViz.
        """

        # Задава цел само за ставата на хващача.
        self.model.set_targets(
            {
                "joint_1_gripper": float(
                    position_rad
                )
            },

            # При липса на изрично зададено време
            # използва стандартното време за хващача.
            (
                self.timings.rviz_gripper_sec
                if duration_sec is None
                else duration_sec
            ),
        )

    def execute_gripper(
        self,
        close: bool,
        wait: bool = True,
        *,
        servo_angle_override: Optional[
            float
        ] = None,
    ) -> bool:
        """
        Отваря или затваря хващача
        в RViz и върху физическия манипулатор.
        """

        # Проверява физическата достъпност,
        # когато реалното управление е включено.
        if self.real.enabled:
            self.real.ensure_available()

        # При липса на директно зададена стойност
        # прочита ъгъла от конфигурацията.
        if servo_angle_override is None:
            servo_deg = env_float(
                (
                    "PLANNER_GRIPPER_CLOSE_SERVO_DEG"
                    if close
                    else "PLANNER_GRIPPER_OPEN_SERVO_DEG"
                ),
                (
                    130.0
                    if close
                    else 70.0
                ),
            )

        else:
            # Използва подадената сервомоторна стойност.
            servo_deg = float(
                servo_angle_override
            )

        # Ограничава стойността
        # до безопасния диапазон на хващача.
        servo_deg = clamp(
            servo_deg,
            *SAFE_SERVO_RANGE["gripper"],
        )

        # Преобразува физическия сервомоторен ъгъл
        # до съответното положение в RViz.
        rviz_position = (
            self.real.gripper_servo_to_rviz_rad(
                servo_deg
            )
        )

        # Стартира движението на хващача в RViz.
        self.publish_rviz_gripper_position(
            rviz_position,
            self.timings.rviz_gripper_sec,
        )

        # Изпраща командата към физическия хващач.
        self.real.send_gripper(
            close,
            self.timings.real_gripper_sec,
            servo_angle_override=servo_deg,
        )

        # При изключено изчакване
        # приключва веднага.
        if not wait:
            return True

        # Избира времето за движение според това
        # дали физическият манипулатор е активен.
        movement_time = (
            self.timings.real_gripper_sec
            if self.real.enabled
            else self.timings.rviz_gripper_sec
        )

        # Изчаква движението и допълнителното
        # време за установяване.
        return self._sleep_interruptible(
            movement_time
            + self.timings.post_gripper_delay_sec
        )

    def execute_servo_pose(
        self,
        servo_angles: Mapping[str, float],
        *,
        label: str,
        duration_sec: float = 3.0,
        wait: bool = True,
    ) -> Tuple[
        bool,
        Dict[str, float],
    ]:
        """
        Изпълнява пълна именувана позиция,
        зададена чрез физически сервомоторни ъгли.

        Позицията се преобразува към ставни стойности
        за синхронно показване в RViz.
        """

        # Филтрира непознатите сервомотори,
        # ограничава стойностите до безопасните диапазони
        # и ги закръгля до три знака.
        clean = {
            key: round(
                clamp(
                    float(value),
                    *SAFE_SERVO_RANGE[key],
                ),
                3,
            )
            for key, value in servo_angles.items()
            if key in SAFE_SERVO_RANGE
        }

        # Проверява дали позицията съдържа
        # стойности за всички използвани сервомотори.
        missing = [
            key
            for key in SAFE_SERVO_RANGE
            if key not in clean
        ]

        # Не допуска изпълнение на непълна позиция.
        if missing:
            raise ValueError(
                "Incomplete servo pose; "
                f"missing: {missing}"
            )

        # Преобразува физическите сервомоторни стойности
        # до ставни стойности в радиани за RViz.
        joints = self.real.servo_to_joints(
            clean
        )

        # Проверява дали всички стави на ръката
        # са успешно преобразувани.
        if any(
            name not in joints
            for name in JOINT_ORDER
        ):
            raise ValueError(
                "Could not convert complete servo pose "
                "to RViz joints"
            )

        # RViz движението остава кратко,
        # дори когато физическото движение е по-бавно.
        model_duration = min(
            max(
                0.1,
                float(duration_sec),
            ),
            self.timings.rviz_move_sec,
        )

        # Публикува ставите на ръката в RViz.
        self.publish_rviz_joints(
            joints,
            model_duration,
        )

        # Публикува отделно положението
        # на хващача в RViz.
        self.publish_rviz_gripper_position(
            self.real.gripper_servo_to_rviz_rad(
                clean["gripper"]
            ),
            self.timings.rviz_gripper_sec,
        )

        # Когато физическото управление е активно,
        # изпраща цялата позиция към Raspberry Pi.
        if self.real.enabled:
            sent = self.real.send_servo_angles(
                clean,
                max(
                    0.1,
                    float(duration_sec),
                ),
                label=label,

                # При wait=True изчаква реално потвърждение
                # за завършено движение.
                wait_for_completion=wait,
            )

            # След потвърдено движение проверява
            # дали не е получена команда за отмяна.
            if wait:
                return (
                    not self.cancel_check(),
                    sent,
                )

            return True, sent

        # При изключен физически манипулатор
        # и wait=False приключва веднага.
        if not wait:
            return True, clean

        # При работа само в RViz изчаква
        # цифровото движение и времето за установяване.
        return (
            self._sleep_interruptible(
                self.timings.rviz_move_sec
                + self.timings.post_move_delay_sec
            ),
            clean,
        )