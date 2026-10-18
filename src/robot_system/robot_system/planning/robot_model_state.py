#!/usr/bin/env python3

"""Direct /joint_states publisher owned by the planning agent.

This is not a bridge and it does not subscribe to controller trajectory topics.
The planning layer gives it final joint targets directly. It interpolates those
targets and publishes the resulting state for robot_state_publisher and RViz.
"""

# Модулът публикува директно състоянието на ставите
# на цифровия модел чрез ROS 2 темата /joint_states.
#
# Планиращият агент му подава крайните ставни цели,
# а модулът интерполира плавно текущите стойности
# до желаните позиции и ги публикува за:
# - robot_state_publisher;
# - визуализацията в RViz.

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Структура за съхраняване на параметрите
# на прехода за всяка отделна става.
from dataclasses import dataclass

# Стандартни библиотеки за математически проверки,
# синхронизация между нишки, време и типови анотации.
import math
import threading
import time
from typing import Any, Dict, Mapping

# ROS 2 съобщение, съдържащо имената и текущите
# позиции на ставите на робота.
from sensor_msgs.msg import JointState

# Помощни функции за прочитане на конфигурационни
# стойности от променливите на средата.
from .common import env_bool, env_float

# Подредба на основните стави на роботизираната ръка.
from .constants import JOINT_ORDER


# Пълен списък със ставите на цифровия модел.
#
# Към петте стави на ръката се добавя отделната
# става за отваряне и затваряне на хващача.
MODEL_JOINTS = [
    *JOINT_ORDER,
    "joint_1_gripper",
]


@dataclass
class JointTransition:
    """
    Съхранява параметрите на плавния преход
    за една става на цифровия модел.
    """

    # Начална ставна стойност в радиани.
    start: float = 0.0

    # Крайна целева стойност в радиани.
    target: float = 0.0

    # Моментът, в който е започнал преходът.
    #
    # Използва се монотонен часовник, който не се влияе
    # от промени в системното календарно време.
    started_at: float = 0.0

    # Продължителност на прехода в секунди.
    duration: float = 0.0

    def value_at(
        self,
        now: float,
    ) -> float:
        """
        Изчислява интерполираната стойност на ставата
        за зададения момент.
        """

        # При нулева или отрицателна продължителност
        # крайната стойност се приема незабавно.
        if self.duration <= 0.0:
            return self.target

        # Изчислява относителния напредък на прехода
        # и го ограничава между 0.0 и 1.0.
        progress = max(
            0.0,
            min(
                1.0,
                (
                    now - self.started_at
                ) / self.duration,
            ),
        )

        # Прилага плавна интерполационна функция
        # smoothstep: 3t² - 2t³.
        #
        # Това осигурява плавно начало и плавен край
        # на движението без рязка промяна на скоростта.
        blend = (
            progress
            * progress
            * (3.0 - 2.0 * progress)
        )

        # Връща текущата интерполирана
        # ставна стойност.
        return (
            self.start
            + (
                self.target - self.start
            ) * blend
        )


class RobotModelStatePublisher:
    """Animate the RViz model directly from planner joint solutions."""

    # Класът управлява състоянието на цифровия модел
    # и публикува текущите ставни стойности директно
    # в ROS 2 темата /joint_states.

    def __init__(
        self,
        publisher: Any,
        clock: Any,
        logger: Any,
    ) -> None:
        """
        Инициализира директния издател
        на ставните състояния.
        """

        # ROS 2 издателят за съобщения JointState.
        self.publisher = publisher

        # ROS 2 часовникът, използван за времевия
        # печат на публикуваните съобщения.
        self.clock = clock

        # Логерът на планиращия агент.
        self.logger = logger

        # Определя дали директното публикуване
        # на /joint_states е разрешено.
        self.enabled = env_bool(
            "PLANNER_PUBLISH_MODEL_JOINT_STATES",
            True,
        )

        # Честота на публикуване на ставните
        # състояния в херци.
        #
        # Минималната позволена честота е 1 Hz.
        self.rate_hz = max(
            1.0,
            env_float(
                "PLANNER_MODEL_JOINT_STATE_RATE_HZ",
                30.0,
            ),
        )

        # Текущ момент, използван като начално време
        # за първоначалните преходи.
        now = time.monotonic()

        # Създава преход за всяка става
        # на цифровия модел.
        #
        # Първоначално всички стави са със стойност 0.0.
        self._transitions: Dict[
            str,
            JointTransition,
        ] = {
            name: JointTransition(
                started_at=now
            )
            for name in MODEL_JOINTS
        }

        # Рекурсивно заключване за безопасен достъп
        # до ставните преходи от различни нишки.
        self._lock = threading.RLock()

        # Събитие за контролирано спиране
        # на публикационната нишка.
        self._stop_event = threading.Event()

        # Отделна фонова нишка, която периодично
        # публикува актуалното състояние.
        self._thread = threading.Thread(
            target=self._publish_loop,
            name="planner-direct-joint-states",
            daemon=True,
        )

        # Стартира публикационната нишка само когато
        # директното публикуване е разрешено.
        if self.enabled:
            self._thread.start()

            # Записва информация за активната честота.
            self.logger.info(
                "Planning agent publishes the robot model directly on "
                f"/joint_states at {self.rate_hz:.1f} Hz"
            )

            # Предупреждава за възможен конфликт,
            # ако друг ROS 2 възел също публикува /joint_states.
            self.logger.warn(
                "Disable PLANNER_PUBLISH_MODEL_JOINT_STATES when a real "
                "joint_state_broadcaster publishes /joint_states."
            )

    def close(self) -> None:
        """
        Спира публикационната нишка
        и освобождава използваните ресурси.
        """

        # Изпраща сигнал за прекратяване.
        self._stop_event.set()

        # Изчаква нишката да приключи,
        # когато директният издател е активен.
        if (
            self.enabled
            and self._thread.is_alive()
        ):
            self._thread.join(
                timeout=2.0
            )

    def set_targets(
        self,
        targets: Mapping[str, float],
        duration_sec: float,
    ) -> None:
        """
        Задава нови целеви стойности
        за една или повече стави.

        Всеки нов преход започва от текущата
        интерполирана позиция на съответната става.
        """

        # При изключено публикуване не се променя
        # състоянието на цифровия модел.
        if not self.enabled:
            return

        # Текущ момент за начало
        # на всички нови преходи.
        now = time.monotonic()

        # Гарантира неотрицателна продължителност.
        duration = max(
            0.0,
            float(duration_sec),
        )

        # Списък с успешно приетите стави.
        accepted = []

        # Заключва структурата с преходите,
        # докато се обновяват целевите стойности.
        with self._lock:
            for name, raw_target in targets.items():
                # Не допуска непознато име на става.
                if name not in self._transitions:
                    raise ValueError(
                        f"Unknown robot model joint: {name}"
                    )

                # Преобразува целевата стойност до float.
                target = float(raw_target)

                # Не допуска NaN или безкрайни стойности.
                if not math.isfinite(target):
                    raise ValueError(
                        f"Non-finite robot model target for {name}"
                    )

                # Получава предишния преход
                # на текущата става.
                previous = self._transitions[
                    name
                ]

                # Изчислява актуалната позиция
                # в момента на задаване на новата цел.
                current = previous.value_at(
                    now
                )

                # Създава нов плавен преход
                # от текущата позиция до новата цел.
                self._transitions[
                    name
                ] = JointTransition(
                    start=current,
                    target=target,
                    started_at=now,
                    duration=duration,
                )

                # Отбелязва ставата като приета.
                accepted.append(name)

        # Записва информация за зададените цели.
        if accepted:
            self.logger.info(
                "RobotModel target set directly for "
                f"{', '.join(accepted)}; "
                f"duration={duration:.2f}s"
            )

    def snapshot(self) -> Dict[str, float]:
        """
        Връща моментна снимка на текущите
        интерполирани ставни стойности.
        """

        # Получава един общ момент за всички стави,
        # така че стойностите да бъдат синхронизирани.
        now = time.monotonic()

        # Заключва преходите по време на прочитането.
        with self._lock:
            return {
                name: transition.value_at(now)
                for name, transition
                in self._transitions.items()
            }

    def _publish_loop(self) -> None:
        """
        Основен цикъл на публикационната нишка.
        """

        # Изчислява периода между две публикувания.
        period = 1.0 / self.rate_hz

        # Продължава до получаване
        # на сигнал за спиране.
        while not self._stop_event.is_set():
            # Запомня началото на текущата итерация.
            started = time.monotonic()

            # Публикува едно съобщение JointState.
            self._publish_once()

            # Изчислява времето, използвано
            # за създаване и публикуване на съобщението.
            elapsed = (
                time.monotonic() - started
            )

            # Изчаква остатъка от периода.
            #
            # Използването на Event.wait() позволява
            # нишката да бъде прекратена незабавно.
            self._stop_event.wait(
                max(
                    0.0,
                    period - elapsed,
                )
            )

    def _publish_once(self) -> None:
        """
        Създава и публикува едно ROS 2
        съобщение от тип JointState.
        """

        # Получава актуалните интерполирани
        # стойности на всички стави.
        values = self.snapshot()

        # Създава ново съобщение JointState.
        message = JointState()

        # Добавя текущ ROS 2 времеви печат.
        message.header.stamp = (
            self.clock.now().to_msg()
        )

        # Задава имената на ставите
        # в постоянната подредба на модела.
        message.name = list(
            MODEL_JOINTS
        )

        # Задава текущите позиции на ставите
        # в радиани в същата подредба.
        message.position = [
            values[name]
            for name in MODEL_JOINTS
        ]

        # Публикува съобщението в /joint_states.
        self.publisher.publish(message)