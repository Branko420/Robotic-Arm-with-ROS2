#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване,
# уникални идентификатори и типови анотации.
import json
import uuid
from typing import Optional

# Декоратор от OpenAI Agents SDK, чрез който функциите
# се предоставят като инструменти на разговорния агент.
from agents import function_tool

# Помощни функции и настройки за:
# - комуникация с planning_agent чрез Redis потоци;
# - проверка на физическия манипулатор;
# - разрешаване на потребителско описание до конкретен предмет.
from .runtime import (
    PLANNING_REQUEST_STREAM,
    PLANNING_RESULT_STREAM,
    PLANNING_TOOL_TIMEOUT_SEC,
    _physical_arm_guard,
    _request,
    _resolve_workspace_target,
    _resolved_identifier,
    _with_target_role,
)


@function_tool
def resolve_workspace_target(
    target: str,
) -> str:
    """
    Resolve a user description to exactly one workspace instance without
    moving the arm. A color with exactly one reachable object resolves
    immediately. Pronouns such as "го", "него", "it", and "that" resolve to
    the last successfully selected object. AMBIGUOUS_TARGET is returned only
    for multiple real matching objects.
    """

    # Определя кой конкретен предмет съответства
    # на подаденото описание, без да изпълнява движение.
    resolution = _resolve_workspace_target(target)

    # Връща резултата като JSON текст.
    return json.dumps(
        resolution,
        ensure_ascii=False,
    )


@function_tool
def control_robot_in_rviz(
    action: str,
    target: str = "nearest_object",
    target_position: str = "auto",
    execute: bool = True,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> str:
    """
    Preview or control the synchronized RViz and physical robotic arm.

    Object-targeted actions are resolved to one unique instance before any
    movement. Executed physical movements require a successful Raspberry Pi
    ping. Preview and stop do not require availability.
    """

    # Нормализира името на действието до малки букви
    # и премахва излишните интервали.
    normalized_action = str(action).strip().lower()

    # Нормализира вида на целевата позиция.
    normalized_target_position = (
        str(target_position).strip().lower()
    )

    # Когато не е зададен конкретен вид позиция,
    # избира подходящата стойност според действието.
    if normalized_target_position in {
        "",
        "auto",
        "default",
    }:
        # При подход, движение над предмет или хващане
        # се използва точката за подход.
        if normalized_action in {
            "approach",
            "above",
            "go_above",
            "move_above",
            "pick",
        }:
            normalized_target_position = "approach"

        # При поставяне се използва позицията за поставяне.
        elif normalized_action == "place":
            normalized_target_position = "place"

        # При останалите действия се използва центърът.
        else:
            normalized_target_position = "center"

    # Действия, които изискват предварително определяне
    # на конкретен предмет от работното пространство.
    object_actions = {
        "preview",
        "move",
        "approach",
        "above",
        "go_above",
        "move_above",
        "grasp",
        "pick",
        "place",
    }

    # Определя еднозначно предмета преди изпълнение
    # на действие, насочено към обект.
    if normalized_action in object_actions:
        resolution = _resolve_workspace_target(target)

        # При неуспешно или двусмислено определяне
        # не се изпраща команда за движение.
        if not resolution.get("ok"):
            return json.dumps(
                resolution,
                ensure_ascii=False,
            )

        # Заменя потребителското описание с устойчивия
        # идентификатор на конкретния предмет.
        target = _resolved_identifier(
            resolution,
            target,
        )

    # Физическият манипулатор е необходим само когато:
    # - execute е True;
    # - действието не е preview;
    # - действието не е stop.
    requires_arm = (
        execute
        and normalized_action
        not in {"preview", "stop"}
    )

    # Проверява достъпността на Raspberry Pi,
    # когато действието изисква физическо изпълнение.
    unavailable = _physical_arm_guard(
        execute=requires_arm
    )

    # Ако физическият манипулатор не е достъпен,
    # изпълнява движението само като визуализация в RViz.
    if unavailable is not None:
        rviz_preview = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Уникален идентификатор на заявката.
                "request_id": (
                    f"offline-preview-"
                    f"{uuid.uuid4().hex[:10]}"
                ),

                # HOME и point_0 се преобразуват до home.
                # Останалите действия се визуализират
                # чрез общото действие preview.
                "action": (
                    "preview"
                    if normalized_action
                    not in {"home", "point_0"}
                    else "home"
                ),

                # Конкретната цел на движението.
                "target": target,

                # Избраният вид целева позиция.
                "target_position": (
                    normalized_target_position
                ),

                # Забранява изпращането към физическия манипулатор.
                "execute": False,

                # Допълнителни декартови отмествания.
                "dx": dx,
                "dy": dy,
                "dz": dz,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        )

        # Връща информацията за недостъпната физическа ръка
        # заедно с резултата от RViz визуализацията.
        return json.dumps(
            {
                **unavailable,
                "rviz_preview": rviz_preview,
            },
            ensure_ascii=False,
        )

    # Изпраща нормализираната заявка към planning_agent.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"plan-{uuid.uuid4().hex[:10]}"
            ),

            # Нормализираното действие.
            "action": normalized_action,

            # Конкретната цел.
            "target": target,

            # Избраният вид целева позиция.
            "target_position": (
                normalized_target_position
            ),

            # Определя дали движението трябва да бъде
            # изпратено и към физическия манипулатор.
            "execute": execute,

            # Допълнителни декартови отмествания.
            "dx": dx,
            "dy": dy,
            "dz": dz,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща резултата като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )


@function_tool
def pick_and_place_in_rviz(
    source: str,
    destination: str,
    execute: bool = True,
) -> str:
    """
    Pick one uniquely resolved physical object and place it on one uniquely
    resolved destination. Nothing moves if either description is ambiguous.
    """

    # Определя еднозначно предмета, който трябва да бъде хванат.
    source_resolution = _resolve_workspace_target(
        source
    )

    # При неуспешно определяне маркира грешката
    # като свързана с изходния предмет.
    if not source_resolution.get("ok"):
        return json.dumps(
            _with_target_role(
                source_resolution,
                "source",
            ),
            ensure_ascii=False,
        )

    # Определя еднозначно предмета или точката,
    # използвани като крайна цел.
    destination_resolution = (
        _resolve_workspace_target(destination)
    )

    # При неуспешно определяне маркира грешката
    # като свързана с крайната цел.
    if not destination_resolution.get("ok"):
        return json.dumps(
            _with_target_role(
                destination_resolution,
                "destination",
            ),
            ensure_ascii=False,
        )

    # Проверява физическата достъпност,
    # когато execute е True.
    unavailable = _physical_arm_guard(
        execute=execute
    )

    # Извлича устойчивия идентификатор
    # на предмета за хващане.
    resolved_source = _resolved_identifier(
        source_resolution,
        source,
    )

    # Извлича устойчивия идентификатор
    # на крайната цел.
    resolved_destination = _resolved_identifier(
        destination_resolution,
        destination,
    )

    # При недостъпен физически манипулатор
    # изпълнява последователността само в RViz.
    if unavailable is not None:
        rviz_preview = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Уникален идентификатор на preview заявката.
                "request_id": (
                    "offline-pick-place-preview-"
                    f"{uuid.uuid4().hex[:10]}"
                ),

                # Действие за хващане и поставяне.
                "action": "pick_and_place",

                # Конкретният изходен предмет.
                "source": resolved_source,

                # Конкретната крайна цел.
                "destination": resolved_destination,

                # Забранява физическото изпълнение.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        )

        # Връща състоянието на физическата ръка
        # и резултата от RViz визуализацията.
        return json.dumps(
            {
                **unavailable,
                "rviz_preview": rviz_preview,
            },
            ensure_ascii=False,
        )

    # Изпраща заявката за пълна pick-and-place последователност.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"pick-place-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Действие за хващане и поставяне.
            "action": "pick_and_place",

            # Изходният предмет.
            "source": resolved_source,

            # Крайната цел.
            "destination": resolved_destination,

            # Определя дали да има физическо изпълнение.
            "execute": execute,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща резултата като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )


@function_tool
def place_held_object_at_target(
    destination: str,
    execute: bool = True,
) -> str:
    """
    Place the already-held object on the measured top surface of a destination.

    This never performs another pick. It keeps the gripper closed during
    transit, uses both object heights for clearance and placement, opens the
    gripper only at the final place pose, and then retreats.
    """

    # Определя еднозначно крайната цел.
    resolution = _resolve_workspace_target(
        destination
    )

    # При неуспешно определяне връща грешка,
    # означена като свързана с destination.
    if not resolution.get("ok"):
        return json.dumps(
            _with_target_role(
                resolution,
                "destination",
            ),
            ensure_ascii=False,
        )

    # Проверява физическата достъпност.
    unavailable = _physical_arm_guard(
        execute=execute
    )

    # Извлича устойчивия идентификатор
    # на крайната цел.
    resolved_destination = _resolved_identifier(
        resolution,
        destination,
    )

    # При недостъпна физическа ръка
    # изпълнява операцията само в RViz.
    if unavailable is not None:
        preview = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Уникален идентификатор
                # на заявката за визуализация.
                "request_id": (
                    f"held-place-preview-"
                    f"{uuid.uuid4().hex[:10]}"
                ),

                # Поставяне на вече държан предмет.
                "action": "place_held_at",

                # Крайната цел.
                "destination": resolved_destination,

                # Забранява физическото изпълнение.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        )

        # Комбинира състоянието на физическата ръка
        # с резултата от RViz.
        return json.dumps(
            {
                **unavailable,
                "rviz_preview": preview,
            },
            ensure_ascii=False,
        )

    # Изпраща заявка за поставяне на предмета,
    # който вече се намира в хващача.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"held-place-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Действие за поставяне
            # на вече държан предмет.
            "action": "place_held_at",

            # Крайната цел.
            "destination": resolved_destination,

            # Определя дали да има физическо изпълнение.
            "execute": execute,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща резултата като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )


@function_tool
def move_held_object_to_target(
    destination: str,
    execute: bool = True,
    target_position: str = "approach",
) -> str:
    """
    Move the arm to a destination while keeping the currently held object in
    the closed gripper. This never opens the gripper and never performs another
    pick sequence.
    """

    # Определя еднозначно крайната цел.
    resolution = _resolve_workspace_target(
        destination
    )

    # При неуспешно определяне връща грешка,
    # означена като свързана с destination.
    if not resolution.get("ok"):
        return json.dumps(
            _with_target_role(
                resolution,
                "destination",
            ),
            ensure_ascii=False,
        )

    # Проверява физическата достъпност.
    unavailable = _physical_arm_guard(
        execute=execute
    )

    # Извлича устойчивия идентификатор
    # на крайната цел.
    resolved_destination = _resolved_identifier(
        resolution,
        destination,
    )

    # При недостъпен физически манипулатор
    # изпълнява само RViz визуализация.
    if unavailable is not None:
        preview = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Уникален идентификатор
                # на заявката за визуализация.
                "request_id": (
                    f"held-preview-"
                    f"{uuid.uuid4().hex[:10]}"
                ),

                # Действие за преместване
                # на вече държан предмет.
                "action": "move_held_to",

                # Крайната цел.
                "destination": resolved_destination,

                # Видът на целевата позиция.
                "target_position": target_position,

                # Забранява физическото изпълнение.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        )

        # Връща резултата от preview
        # и състоянието на физическия манипулатор.
        return json.dumps(
            {
                **unavailable,
                "rviz_preview": preview,
            },
            ensure_ascii=False,
        )

    # Изпраща заявка за движение,
    # като хващачът остава затворен.
    result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"held-move-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Действие за движение
            # с вече държан предмет.
            "action": "move_held_to",

            # Крайната цел.
            "destination": resolved_destination,

            # Избраният вид целева позиция.
            "target_position": target_position,

            # Определя дали движението трябва
            # да бъде физически изпълнено.
            "execute": execute,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща резултата като JSON текст.
    return json.dumps(
        result,
        ensure_ascii=False,
    )


@function_tool
def correct_last_robot_target(
    forward_m: float = 0.0,
    backward_m: float = 0.0,
    left_m: float = 0.0,
    right_m: float = 0.0,
    up_m: float = 0.0,
    down_m: float = 0.0,
    target: Optional[str] = None,
    move_now: bool = True,
    target_position: str = "approach",
) -> str:
    """Move relative to the robot's actual current Cartesian pose.

    With no explicit target, the delta is applied directly to the current
    end-effector X/Y/Z, including when the arm is at HOME. With an explicit
    object target, the delta remains a persistent learned object offset.
    """

    # Изчислява крайното отместване по оста X.
    # Напред е положителната посока,
    # а назад е отрицателната.
    dx = (
        float(forward_m)
        - float(backward_m)
    )

    # Изчислява крайното отместване по оста Y.
    # Наляво е положителната посока,
    # а надясно е отрицателната.
    dy = (
        float(left_m)
        - float(right_m)
    )

    # Изчислява крайното отместване по оста Z.
    # Нагоре е положителната посока,
    # а надолу е отрицателната.
    dz = (
        float(up_m)
        - float(down_m)
    )

    # Отказва операцията, когато не е зададено
    # реално отместване по нито една ос.
    if abs(dx) + abs(dy) + abs(dz) < 1e-9:
        return json.dumps(
            {
                "ok": False,
                "state": "ZERO_CORRECTION",
            },
            ensure_ascii=False,
        )

    # Когато няма изрично зададен предмет,
    # движението се прилага спрямо текущата
    # декартова позиция на крайния механизъм.
    if not target:
        # Проверява физическата ръка само когато
        # движението трябва да се изпълни веднага.
        unavailable = _physical_arm_guard(
            execute=move_now
        )

        # При недостъпен физически манипулатор
        # не изпълнява относителното движение.
        if unavailable is not None and move_now:
            return json.dumps(
                unavailable,
                ensure_ascii=False,
            )

        # Изпраща относителното декартово движение.
        result = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Уникален идентификатор на заявката.
                "request_id": (
                    f"relative-"
                    f"{uuid.uuid4().hex[:10]}"
                ),

                # Действие за движение спрямо
                # текущата декартова позиция.
                "action": "move_relative",

                # Изчислените отмествания.
                "dx": dx,
                "dy": dy,
                "dz": dz,

                # Определя дали движението
                # трябва да бъде изпълнено веднага.
                "execute": move_now,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        )

        # Връща резултата от движението.
        return json.dumps(
            result,
            ensure_ascii=False,
        )

    # Когато е зададен предмет, първо го определя еднозначно.
    resolution = _resolve_workspace_target(
        target
    )

    # При неуспешно определяне не записва корекция.
    if not resolution.get("ok"):
        return json.dumps(
            resolution,
            ensure_ascii=False,
        )

    # Извлича устойчивия идентификатор
    # на конкретния предмет.
    resolved_target = _resolved_identifier(
        resolution,
        target,
    )

    # Записва корекцията като постоянно отместване
    # за конкретния предмет.
    correction_result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"correct-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Действие за записване на корекция.
            "action": "correct",

            # Конкретният предмет.
            "target": resolved_target,

            # Изчислените отмествания.
            "dx": dx,
            "dy": dy,
            "dz": dz,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Ако записването на корекцията е неуспешно
    # или не се иска незабавно движение,
    # връща само резултата от корекцията.
    if (
        not correction_result.get("ok")
        or not move_now
    ):
        return json.dumps(
            correction_result,
            ensure_ascii=False,
        )

    # Проверява достъпността на физическия манипулатор
    # преди движението към коригираната цел.
    unavailable = _physical_arm_guard(
        execute=True
    )

    # При недостъпна физическа ръка връща
    # и успешно записаната корекция.
    if unavailable is not None:
        return json.dumps(
            {
                **unavailable,
                "correction": correction_result,
            },
            ensure_ascii=False,
        )

    # Премества манипулатора към коригираната
    # целева позиция на предмета.
    move_result = _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на движението.
            "request_id": (
                f"corrected-move-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Използва движение към точката за подход.
            "action": "approach",

            # Конкретният предмет.
            "target": resolved_target,

            # Избраният вид целева позиция.
            "target_position": target_position,

            # Разрешава физическото изпълнение.
            "execute": True,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )

    # Връща общ резултат от:
    # - записването на корекцията;
    # - последващото движение.
    return json.dumps(
        {
            "ok": bool(move_result.get("ok")),
            "correction": correction_result,
            "movement": move_result,
        },
        ensure_ascii=False,
    )