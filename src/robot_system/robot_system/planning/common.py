#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за работа с променливи на средата,
# структури от данни и общи типове.
import os
from dataclasses import dataclass
from typing import Any

# Зарежда променливи на средата от .env файл.
from dotenv import load_dotenv


# Проверява последователно възможните местоположения
# на конфигурационния .env файл.
for _env_path in (
    # Потребителски зададен път до конфигурационния файл.
    os.getenv("ROBOT_SYSTEM_ENV_FILE"),

    # Основен път в работната среда на роботизираната система.
    "/arm_ws/.env",

    # Резервен път при изпълнение в друга Docker конфигурация.
    "/workspace/.env",
):
    # Зарежда файла само когато пътят е зададен
    # и съответният файл съществува.
    if _env_path and os.path.exists(_env_path):
        # override=False запазва вече зададените
        # променливи на средата.
        load_dotenv(
            _env_path,
            override=False,
        )


def env(
    name: str,
    default: str,
) -> str:
    """
    Връща текстова променлива на средата.

    При липса на променливата се използва
    подадената стойност по подразбиране.
    """

    return os.getenv(
        name,
        default,
    )


def env_float(
    name: str,
    default: float,
) -> float:
    """
    Връща променлива на средата като число
    с плаваща запетая.
    """

    try:
        # Прочита стойността и я преобразува до float.
        return float(
            os.getenv(
                name,
                str(default),
            )
        )

    except (TypeError, ValueError):
        # При липсваща или невалидна стойност
        # връща стойността по подразбиране.
        return default


def env_bool(
    name: str,
    default: bool,
) -> bool:
    """
    Връща променлива на средата като булева стойност.
    """

    # Получава суровата текстова стойност.
    raw = os.getenv(name)

    # При липсваща променлива използва
    # стойността по подразбиране.
    if raw is None:
        return default

    # Приема няколко стандартни текстови
    # представяния на логическата стойност True.
    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def decode(value: Any) -> str:
    """
    Преобразува стойност до текст.

    Последователностите от байтове се декодират
    чрез UTF-8, а останалите стойности се
    преобразуват чрез str().
    """

    # Redis може да върне стойността като bytes.
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    """
    Ограничава числова стойност между
    зададена минимална и максимална граница.
    """

    return max(
        low,
        min(
            high,
            float(value),
        ),
    )


@dataclass(frozen=True)
class MotionTimings:
    """
    Съдържа времената за изпълнение на движенията
    в RViz и върху физическия манипулатор.

    frozen=True предотвратява промяната на стойностите
    след създаване на обекта.
    """

    # Продължителност на движение в RViz.
    rviz_move_sec: float

    # Продължителност на движение
    # върху физическия манипулатор.
    real_move_sec: float

    # Допълнително изчакване след движение.
    post_move_delay_sec: float

    # Продължителност на движението
    # на хващача в RViz.
    rviz_gripper_sec: float

    # Продължителност на физическото
    # движение на хващача.
    real_gripper_sec: float

    # Допълнително изчакване след движение
    # на хващача.
    post_gripper_delay_sec: float

    @classmethod
    def from_env(
        cls,
    ) -> "MotionTimings":
        """
        Създава конфигурацията за времената на движение
        от променливите на средата.
        """

        return cls(
            # Бързо движение на цифровия модел в RViz.
            rviz_move_sec=env_float(
                "PLANNER_RVIZ_MOVE_DURATION_SEC",
                0.8,
            ),

            # По-бавно движение на физическия манипулатор.
            real_move_sec=env_float(
                "PLANNER_REAL_MOVE_DURATION_SEC",
                10.0,
            ),

            # Изчакване след завършване
            # на основно движение.
            post_move_delay_sec=env_float(
                "PLANNER_POST_MOVE_DELAY_SEC",
                1.0,
            ),

            # Продължителност на отварянето или затварянето
            # на хващача в RViz.
            rviz_gripper_sec=env_float(
                "PLANNER_RVIZ_GRIPPER_DURATION_SEC",
                0.45,
            ),

            # Продължителност на физическото движение
            # на хващача.
            real_gripper_sec=env_float(
                "PLANNER_REAL_GRIPPER_DURATION_SEC",
                2.0,
            ),

            # Изчакване след физическото движение
            # на хващача.
            post_gripper_delay_sec=env_float(
                "PLANNER_POST_GRIPPER_DELAY_SEC",
                0.5,
            ),
        )