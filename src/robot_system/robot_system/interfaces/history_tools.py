#!/usr/bin/env python3

# Позволява използването на отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване,
# създаване на уникални идентификатори и типови анотации.
import json
import uuid
from typing import Optional

# Декоратор от OpenAI Agents SDK, чрез който функциите
# се предоставят като инструменти на разговорния агент.
from agents import function_tool

# Импортира Redis потоците, времената за изчакване
# и помощните функции за комуникация с планиращия агент.
from .runtime import (
    PLANNING_REQUEST_STREAM,
    PLANNING_RESULT_STREAM,
    PLANNING_TOOL_TIMEOUT_SEC,
    TRACEBACK_RETURN_TIMEOUT_SEC,
    _physical_arm_guard,
    _request,
    _resolve_workspace_target,
    _resolved_identifier,
)


@function_tool
def return_object_to_previous_position(
    object_name: str,
    all_the_way_to_origin: bool = False,
    execute: bool = True,
) -> str:
    """
    Return exactly one resolved object one step backwards, or all the way to
    its original workspace-scan position.
    """

    # Определя кой конкретен предмет съответства
    # на подаденото от потребителя име или описание.
    resolution = _resolve_workspace_target(
        object_name
    )

    # Ако предметът не може да бъде определен еднозначно,
    # връща резултата без да изпраща команда за движение.
    if not resolution.get("ok"):
        return json.dumps(
            resolution,
            ensure_ascii=False,
        )

    # Проверява достъпността на физическия манипулатор,
    # когато е поискано реално изпълнение.
    unavailable = _physical_arm_guard(
        execute=execute
    )

    # Извлича устойчивия идентификатор на намерения предмет.
    # Той позволява правилно разграничаване на няколко
    # сходни или еднакво описани предмета.
    resolved_object = _resolved_identifier(
        resolution,
        object_name,
    )

    # Ако физическият манипулатор не е достъпен,
    # изпълнява операцията само като визуализация в RViz.
    if unavailable is not None:
        rviz_preview = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Създава уникален идентификатор
                # за заявката за предварителна визуализация.
                "request_id": (
                    f"offline-return-preview-"
                    f"{uuid.uuid4().hex[:10]}"
                ),

                # Избира действието според това дали предметът
                # трябва да бъде върнат само една стъпка назад
                # или изцяло до началната си позиция.
                "action": (
                    "return_object_to_origin"
                    if all_the_way_to_origin
                    else "return_object"
                ),

                # Конкретният идентификатор на предмета.
                "target": resolved_object,

                # Указва дали трябва да бъде използвана
                # първоначалната позиция от сканирането.
                "all_the_way_to_origin": all_the_way_to_origin,

                # Забранява изпращането към физическия манипулатор.
                "execute": False,
            },

            # Връщането на предмет може да включва
            # продължителна последователност от движения.
            TRACEBACK_RETURN_TIMEOUT_SEC,
        )

        # Връща информацията за недостъпната физическа ръка
        # заедно с резултата от визуализацията в RViz.
        return json.dumps(
            {
                **unavailable,
                "rviz_preview": rviz_preview,
            },
            ensure_ascii=False,
        )

    # Изпраща заявката към планиращия агент,
    # когато физическото изпълнение е разрешено и достъпно.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Създава уникален идентификатор на заявката.
            "request_id": (
                f"return-object-{uuid.uuid4().hex[:10]}"
            ),

            # Избира между връщане с една стъпка
            # и връщане до първоначалната позиция.
            "action": (
                "return_object_to_origin"
                if all_the_way_to_origin
                else "return_object"
            ),

            # Предметът, който трябва да бъде върнат.
            "target": resolved_object,

            # Предава избрания режим и като отделно поле.
            "all_the_way_to_origin": (
                all_the_way_to_origin
            ),

            # Определя дали движението трябва да се изпълни
            # и върху физическия манипулатор.
            "execute": execute,
        },
        TRACEBACK_RETURN_TIMEOUT_SEC,
    )

    # Връща резултата като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )


@function_tool
def list_object_movement_history(
    object_name: Optional[str] = None,
) -> str:
    """
    Return the saved origin, current location and previous positions. If a
    target name is supplied, it is first resolved to a unique instance.
    """

    # По подразбиране използва директно подаденото име.
    # При липса на име ще бъде върната историята
    # за всички съхранени предмети.
    resolved_name = object_name

    # Ако е посочен конкретен предмет, първо го определя
    # еднозначно спрямо актуалното работно пространство.
    if object_name:
        resolution = _resolve_workspace_target(
            object_name
        )

        # При неуспешно или двусмислено определяне
        # връща резултата без заявка към историята.
        if not resolution.get("ok"):
            return json.dumps(
                resolution,
                ensure_ascii=False,
            )

        # Извлича устойчивия идентификатор
        # на конкретния предмет.
        resolved_name = _resolved_identifier(
            resolution,
            object_name,
        )

    # Изпраща заявка към планиращия агент
    # за прочитане на съхранената история.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"object-history-{uuid.uuid4().hex[:10]}"
            ),

            # Действие за получаване на историята
            # на преместванията.
            "action": "traceback_history",

            # Конкретен предмет или None за всички предмети.
            "target": resolved_name,

            # Заявката е само за прочитане
            # и не трябва да предизвиква движение.
            "execute": False,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща историята като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )