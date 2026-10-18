#!/usr/bin/env python3

# Позволява използването на отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за работа с JSON данни
# и създаване на уникални идентификатори на заявките.
import json
import uuid

# Декоратор от OpenAI Agents SDK, чрез който функциите
# се предоставят като инструменти на разговорния агент.
from agents import function_tool

# Импортира настройките за Redis потоците, времето за изчакване
# и помощните функции за комуникация с планиращия агент.
from .runtime import (
    PLANNING_REQUEST_STREAM,
    PLANNING_RESULT_STREAM,
    PLANNING_TOOL_TIMEOUT_SEC,
    _physical_arm_guard,
    _request,
)


@function_tool
def get_current_robot_position() -> str:
    """Return the planner's continuously tracked current end-effector X/Y/Z."""

    # Изпраща заявка към планиращия агент за получаване
    # на текущата декартова позиция на крайния механизъм.
    return json.dumps(
        _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Създава уникален идентификатор на заявката.
                "request_id": (
                    f"current-pose-{uuid.uuid4().hex[:10]}"
                ),

                # Действие за прочитане на текущата позиция.
                "action": "current_pose",

                # Заявката не предизвиква физическо движение.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        ),
        ensure_ascii=False,
    )


@function_tool
def save_current_robot_point(name: str) -> str:
    """Save the robot's current end-effector X/Y/Z under a reusable name."""

    # Изпраща заявка за запазване на текущата декартова позиция
    # под зададено от потребителя име.
    return json.dumps(
        _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Създава уникален идентификатор на заявката.
                "request_id": (
                    f"save-point-{uuid.uuid4().hex[:10]}"
                ),

                # Действие за запазване на текущата точка.
                "action": "save_current_point",

                # Име, под което точката ще бъде записана.
                "name": name,

                # Запазването не предизвиква движение.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        ),
        ensure_ascii=False,
    )


@function_tool
def list_saved_robot_points() -> str:
    """List all user-saved Cartesian robot points."""

    # Изпраща заявка за получаване на всички
    # запазени от потребителя декартови точки.
    return json.dumps(
        _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Създава уникален идентификатор на заявката.
                "request_id": (
                    f"list-points-{uuid.uuid4().hex[:10]}"
                ),

                # Действие за извеждане на запазените точки.
                "action": "list_saved_points",

                # Заявката е само за прочитане.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        ),
        ensure_ascii=False,
    )


@function_tool
def move_to_saved_robot_point(
    name: str,
    execute: bool = True,
) -> str:
    """Move the end effector to a previously saved Cartesian point."""

    # Проверява дали физическият манипулатор е достъпен,
    # когато е поискано реално изпълнение.
    unavailable = _physical_arm_guard(
        execute=execute
    )

    # При недостъпна физическа ръка прекратява заявката
    # и връща информация за причината.
    if unavailable is not None:
        return json.dumps(
            unavailable,
            ensure_ascii=False,
        )

    # Изпраща заявка за движение до предварително
    # запазената декартова точка.
    return json.dumps(
        _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Създава уникален идентификатор на движението.
                "request_id": (
                    f"move-point-{uuid.uuid4().hex[:10]}"
                ),

                # Действие за движение до запазена точка.
                "action": "move_to_saved_point",

                # Име на точката, към която трябва да се придвижи ръката.
                "name": name,

                # Определя дали движението трябва да бъде
                # изпълнено и от физическия манипулатор.
                "execute": execute,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        ),
        ensure_ascii=False,
    )


@function_tool
def delete_saved_robot_point(name: str) -> str:
    """Delete one saved Cartesian point by name."""

    # Изпраща заявка за изтриване на една
    # запазена декартова точка по нейното име.
    return json.dumps(
        _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Създава уникален идентификатор на заявката.
                "request_id": (
                    f"delete-point-{uuid.uuid4().hex[:10]}"
                ),

                # Действие за изтриване на запазена точка.
                "action": "delete_saved_point",

                # Име на точката, която трябва да бъде изтрита.
                "name": name,

                # Изтриването не предизвиква движение.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        ),
        ensure_ascii=False,
    )