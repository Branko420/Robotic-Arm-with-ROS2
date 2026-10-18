#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване,
# създаване на уникални идентификатори и типови анотации.
import json
import uuid
from typing import Optional

# Декоратор от OpenAI Agents SDK, чрез който обикновените
# Python функции се предоставят като инструменти на разговорния агент.
from agents import function_tool

# Импортира имената на Redis потоците, времето за изчакване
# и помощните функции за комуникация с планиращия агент
# и физическия манипулатор.
from .runtime import (
    PLANNING_REQUEST_STREAM,
    PLANNING_RESULT_STREAM,
    PLANNING_TOOL_TIMEOUT_SEC,
    _physical_arm_guard,
    _ping_physical_arm,
    _request,
)


@function_tool
def check_physical_arm_availability() -> str:
    """
    Ping the Raspberry Pi attached to the physical arm and return whether it is
    reachable. Use this when the user explicitly asks whether the physical arm
    is online, reachable or available.
    """

    # Изпраща принудителна проверка към Raspberry Pi,
    # независимо от евентуално кеширан предишен резултат.
    result = _ping_physical_arm(force=True)

    # Преобразува резултата в JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )


@function_tool
def get_robot_joint_angles() -> str:
    """
    Return the current open-loop commanded physical servo angles for base,
    shoulder, elbow, wrist, wrist rotation and gripper. These values come from
    the Raspberry Pi executor when available; they are not encoder readings.
    """

    # Изпраща заявка към планиращия агент за получаване
    # на последните командвани сервомоторни стойности.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Създава уникален идентификатор на заявката.
            "request_id": (
                f"joint-status-{uuid.uuid4().hex[:10]}"
            ),

            # Действие за прочитане на текущите ставни стойности.
            "action": "joint_status",

            # Заявката не предизвиква физическо движение.
            "execute": False,
        },

        # Максимално време за изчакване на отговор.
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща резултата като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )


@function_tool
def set_robot_joint_angles(
    base_deg: Optional[float] = None,
    shoulder_deg: Optional[float] = None,
    elbow_deg: Optional[float] = None,
    wrist_deg: Optional[float] = None,
    wrist_rotation_deg: Optional[float] = None,
    gripper_deg: Optional[float] = None,
    duration_sec: float = 3.0,
    execute: bool = True,
) -> str:
    """
    Set one or more physical robot servo joints in degrees while preserving all
    unspecified joints at their current commanded angles.

    Valid ranges:
      base, shoulder, elbow, wrist, wrist_rotation: 0..180 degrees
      gripper: 70..130 degrees
    """

    # Събира всички подадени ставни стойности в общ речник.
    #
    # Стойност None означава, че съответната става
    # не трябва да бъде променяна.
    values = {
        "base": base_deg,
        "shoulder": shoulder_deg,
        "elbow": elbow_deg,
        "wrist": wrist_deg,
        "wrist_rotation": wrist_rotation_deg,
        "gripper": gripper_deg,
    }

    # Премахва ставите, за които не е подадена нова стойност.
    #
    # Така планиращият агент ще запази последните командвани
    # стойности на непосочените стави.
    angles = {
        key: value
        for key, value in values.items()
        if value is not None
    }

    # Прекратява заявката, когато потребителят
    # не е подал нито една ставна стойност.
    if not angles:
        return json.dumps(
            {
                "ok": False,
                "state": "JOINT_ANGLES_REQUIRED",
            },
            ensure_ascii=False,
        )

    # Проверява дали физическият манипулатор е достъпен,
    # когато е поискано реално изпълнение.
    unavailable = _physical_arm_guard(
        execute=execute,
    )

    # При недостъпна физическа ръка движението се изпраща
    # само като предварителна визуализация към RViz.
    if unavailable is not None:
        preview = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Уникален идентификатор за заявката за визуализация.
                "request_id": (
                    f"joint-preview-{uuid.uuid4().hex[:10]}"
                ),

                # Действие за задаване на сервомоторни стойности.
                "action": "set_servo_angles",

                # Само ставите, за които са подадени нови стойности.
                "angles": angles,

                # Продължителност на интерполираното движение.
                "duration_sec": duration_sec,

                # False указва, че не трябва да се изпълнява
                # движение върху физическия манипулатор.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        )

        # Комбинира информацията за недостъпната физическа ръка
        # с резултата от визуализацията в RViz.
        return json.dumps(
            {
                **unavailable,
                "rviz_preview": preview,
            },
            ensure_ascii=False,
        )

    # Изпраща заявката за реално или само цифрово движение,
    # когато проверката за физическа достъпност е успешна
    # или execute е зададено като False.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Създава уникален идентификатор на движението.
            "request_id": (
                f"joint-move-{uuid.uuid4().hex[:10]}"
            ),

            # Действие за задаване на директни сервомоторни стойности.
            "action": "set_servo_angles",

            # Подадените стойности на избраните стави.
            "angles": angles,

            # Продължителност на движението в секунди.
            "duration_sec": duration_sec,

            # Определя дали движението трябва да се изпрати
            # и към физическия манипулатор.
            "execute": execute,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща резултата от планиращия агент като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )