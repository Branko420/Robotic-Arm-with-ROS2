#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване,
# измерване на време, уникални идентификатори
# и типови анотации.
import json
import time
import uuid
from typing import Any, Dict, Optional

# Декоратор от OpenAI Agents SDK, чрез който функциите
# се предоставят като инструменти на разговорния агент.
from agents import function_tool

# Импортира Redis ключовете и потоците, времената за изчакване
# и помощните функции за подготовка, сканиране и управление
# на физическия манипулатор.
from .runtime import (
    MOTION_POINTS_KEY,
    PLANNING_REQUEST_STREAM,
    PLANNING_RESULT_STREAM,
    PLANNING_TOOL_TIMEOUT_SEC,
    REDIS,
    SCAN_AFTER_HOME_SETTLE_SEC,
    SCAN_ALLOW_WITHOUT_PHYSICAL_ARM,
    SCAN_POSE_MIN_DWELL_SEC,
    WORKSPACE_SCAN_REQUEST_STREAM,
    WORKSPACE_SCAN_RESULT_STREAM,
    WORKSPACE_SCAN_TOOL_TIMEOUT_SEC,
    _move_arm_to_named_pose,
    _ping_physical_arm,
    _prepare_arm_for_scan,
    _request,
)


@function_tool
def scan_workspace_and_make_points() -> str:
    """
    Execute the physical scan sequence:

      HOME -> SCAN_POSE -> workspace scan -> HOME

    The motors remain enabled and no OE/release command is used. When the
    physical arm is unavailable and offline scanning is enabled, the scanner
    still runs without moving the arm.
    """

    # Проверява дали Raspberry Pi на физическия манипулатор
    # е достъпно в мрежата.
    arm_status = _ping_physical_arm(
        force=False
    )

    # Преобразува резултата от проверката
    # до булева стойност.
    arm_available = bool(
        arm_status.get("available")
    )

    # Резултат от подготовката на манипулатора
    # преди започване на сканирането.
    arm_prepare: Dict[str, Any]

    # Резултат от връщането към HOME след сканирането.
    # Първоначално няма изпълнена команда за връщане.
    return_home: Optional[Dict[str, Any]] = None

    # Когато физическият манипулатор е достъпен,
    # изпълнява последователността HOME -> SCAN_POSE.
    if arm_available:
        arm_prepare = _prepare_arm_for_scan()

        # Прекратява операцията, когато подготовката
        # на физическия манипулатор е неуспешна.
        if not arm_prepare.get("ok"):
            return json.dumps(
                {
                    "ok": False,
                    "state": "SCAN_ARM_PREPARATION_FAILED",
                    "arm_prepare": arm_prepare,
                },
                ensure_ascii=False,
            )

        # Изчаква кратко след достигане на позицията,
        # за да се установят манипулаторът и камерата.
        time.sleep(
            max(
                0.0,
                SCAN_AFTER_HOME_SETTLE_SEC,
            )
        )

    else:
        # Когато сканирането без физическа ръка
        # е забранено, операцията се прекратява.
        if not SCAN_ALLOW_WITHOUT_PHYSICAL_ARM:
            return json.dumps(
                {
                    **arm_status,
                    "ok": False,
                    "movement_sent": False,
                    "scan_started": False,
                },
                ensure_ascii=False,
            )

        # Отбелязва, че подготовката е пропусната,
        # защото физическият манипулатор не е достъпен.
        arm_prepare = {
            "ok": True,
            "state": "ARM_SCAN_POSE_SKIPPED_OFFLINE",
            "executed": False,
            "reason": "physical_arm_unavailable",
        }

    # Запазва момента, в който манипулаторът
    # е приет за влязъл в позицията за сканиране.
    scan_pose_entered_at = time.monotonic()

    # Изпраща заявка към workspace_scan_agent
    # за обработване на облака от точки
    # и създаване на точки за движение.
    scan_result = _request(
        WORKSPACE_SCAN_REQUEST_STREAM,
        WORKSPACE_SCAN_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"workspace-scan-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Действие за сканиране и създаване
            # на точки за движение.
            "action": "scan_workspace_and_make_points",

            # Описва предполагаемата позиция на ръката.
            "arm_pose": (
                "scan_pose"
                if arm_available
                else "unchanged"
            ),

            # Предава състоянието на физическия манипулатор.
            "physical_arm_available": arm_available,
        },
        WORKSPACE_SCAN_TOOL_TIMEOUT_SEC,
    )

    # Манипулаторът трябва да остане в scan_pose
    # поне за конфигурираното минимално време.
    #
    # Самото сканиране започва веднага, затова тук
    # се изчаква само оставащата част от периода.
    scan_pose_elapsed_sec = (
        time.monotonic()
        - scan_pose_entered_at
    )

    # Изчислява оставащото време
    # до минималния престой в scan_pose.
    scan_pose_remaining_sec = max(
        0.0,
        (
            SCAN_POSE_MIN_DWELL_SEC
            - scan_pose_elapsed_sec
        ),
    )

    # Изчаква оставащото време само когато
    # физическият манипулатор е достъпен.
    if (
        arm_available
        and scan_pose_remaining_sec > 0.0
    ):
        time.sleep(
            scan_pose_remaining_sec
        )

    # След получаване на резултата от сканирането
    # и изтичане на минималния престой връща ръката към HOME.
    if arm_available:
        return_home = _move_arm_to_named_pose(
            "home",
            "home",
        )

    # Добавя резултата от подготовката
    # към общия резултат от сканирането.
    scan_result["arm_prepare"] = arm_prepare

    # Добавя резултата от връщането към HOME.
    scan_result["arm_return_home"] = return_home

    # Добавя пълния резултат от ping проверката.
    scan_result["physical_arm"] = arm_status

    # Отбелязва дали сканирането е продължило
    # без физическия манипулатор.
    scan_result[
        "scan_continued_without_physical_arm"
    ] = not arm_available

    # Записва конфигурираното минимално време
    # за престой в scan_pose.
    scan_result["scan_pose_min_dwell_sec"] = (
        SCAN_POSE_MIN_DWELL_SEC
    )

    # Записва действителното време от влизането
    # в scan_pose до започване на връщането към HOME.
    scan_result[
        "scan_pose_elapsed_before_return_sec"
    ] = round(
        time.monotonic()
        - scan_pose_entered_at,
        3,
    )

    # При успешно сканиране създава кратко обобщение,
    # което може лесно да бъде използвано
    # от разговорния агент.
    if scan_result.get("ok"):
        # Получава списъка с откритите предмети.
        object_rows = scan_result.get("objects")

        # Списък с удобни за потребителя имена.
        object_names = []

        # Проверява дали върнатата стойност
        # действително е списък.
        if isinstance(object_rows, list):
            for row in object_rows:
                # Игнорира елементи, които не са речници.
                if not isinstance(row, dict):
                    continue

                # Избира първото налично подходящо име
                # на предмета.
                name = (
                    row.get("display_name_bg")
                    or row.get("point_label")
                    or row.get("label")
                    or row.get("point")
                )

                # Добавя валидното име към списъка.
                if name:
                    object_names.append(
                        str(name)
                    )

        # Създава структурирани факти
        # за отговора на разговорния агент.
        scan_result["response_facts"] = {
            "operation": "workspace_scan",

            # Използва върнатия брой или броя
            # на извлечените имена.
            "object_count": int(
                scan_result.get("count")
                or len(object_names)
            ),

            # Имена на откритите предмети.
            "objects": object_names,

            # При работа без физическа ръка се приема,
            # че няма необходимост от връщане към HOME.
            "returned_home": bool(
                not arm_available
                or (
                    return_home
                    and return_home.get("ok")
                )
            ),
        }

    # Проверява дали връщането към HOME е неуспешно,
    # когато физическият манипулатор е бил достъпен.
    if arm_available and not (
        return_home
        and return_home.get("ok")
    ):
        # Запазва първоначалното състояние
        # на сканирането преди грешката при връщане.
        original_state = scan_result.get("state")

        scan_result[
            "scan_result_before_home_error"
        ] = {
            "ok": scan_result.get("ok"),
            "state": original_state,
        }

        # Общата операция се приема за неуспешна,
        # защото ръката не е върната към HOME.
        scan_result["ok"] = False

        # Разграничава успешно сканиране с неуспешно връщане
        # от неуспешно сканиране и неуспешно връщане.
        scan_result["state"] = (
            "SCAN_COMPLETE_HOME_RETURN_FAILED"
            if scan_result.get(
                "scan_result_before_home_error",
                {},
            ).get("ok")
            else "SCAN_FAILED_AND_HOME_RETURN_FAILED"
        )

    # При напълно успешно сканиране синхронизира
    # началните позиции на предметите
    # с историята за последващо връщане.
    if scan_result.get("ok"):
        scan_result["traceback_sync"] = _request(
            PLANNING_REQUEST_STREAM,
            PLANNING_RESULT_STREAM,
            {
                # Уникален идентификатор на заявката.
                "request_id": (
                    f"sync-traceback-"
                    f"{uuid.uuid4().hex[:10]}"
                ),

                # Действие за синхронизиране
                # на историята на предметите.
                "action": "sync_traceback",

                # Заявката не предизвиква движение.
                "execute": False,
            },
            PLANNING_TOOL_TIMEOUT_SEC,
        )

        # При сканиране без физическа ръка
        # добавя предупреждение към резултата.
        if not arm_available:
            scan_result["warning"] = {
                "state": "PHYSICAL_ARM_UNAVAILABLE",
                "message": (
                    "Workspace scan completed without "
                    "moving the physical arm."
                ),
            }

    # Връща пълния резултат като JSON текст.
    return json.dumps(
        scan_result,
        ensure_ascii=False,
    )


@function_tool
def list_workspace_points() -> str:
    """
    Return the known physical objects and their unique instance identities.
    Internal virtual traceback points are omitted from the normal object list.
    """

    # Прочита документа с точките за движение от Redis.
    raw = REDIS.get(
        MOTION_POINTS_KEY
    )

    # Връща грешка, когато все още няма
    # записани точки от работното пространство.
    if not raw:
        return json.dumps(
            {
                "ok": False,
                "state": "NO_WORKSPACE_POINTS",
            },
            ensure_ascii=False,
        )

    try:
        # Преобразува JSON текста в Python речник.
        document = json.loads(raw)

    except Exception as exc:
        # Връща грешка при невалиден JSON документ.
        return json.dumps(
            {
                "ok": False,
                "state": "INVALID_WORKSPACE_POINTS",
                "error": str(exc),
            },
            ensure_ascii=False,
        )

    # Списък с физическите предмети,
    # които ще бъдат върнати на разговорния агент.
    points = []

    # Обхожда всички точки за движение
    # от последното сканиране.
    for point in document.get(
        "motion_points",
        [],
    ):
        # Пропуска вътрешните виртуални точки,
        # използвани само за историята на преместванията.
        if point.get("traceback_point"):
            continue

        # Добавя основната информация
        # за конкретния физически предмет.
        points.append(
            {
                # Уникален идентификатор
                # на конкретния екземпляр.
                "instance_id": point.get(
                    "instance_id"
                ),

                # Пореден номер при наличие
                # на няколко еднакви предмета.
                "instance_index": point.get(
                    "instance_index"
                ),

                # Общ брой предмети
                # със същото описание.
                "instance_count": point.get(
                    "instance_count"
                ),

                # Семантичен ключ за предмета.
                "semantic_key": point.get(
                    "semantic_key"
                ),

                # Потребителско име на български.
                "display_name_bg": point.get(
                    "display_name_bg"
                ),

                # Основно име без пространствено уточнение.
                "base_display_name_bg": point.get(
                    "base_display_name_bg"
                ),

                # Пространствени описания като
                # ляв, десен, близък или далечен.
                "spatial_aliases": point.get(
                    "spatial_aliases",
                    [],
                ),

                # Име на точката за движение.
                "point": point.get("point"),

                # Кратък етикет на предмета.
                "label": point.get("label"),

                # Роля на точката.
                "role": point.get("role"),

                # Открити цветове.
                "colors": point.get(
                    "colors",
                    [],
                ),

                # Допълнителни имена,
                # чрез които предметът може да бъде избран.
                "aliases": point.get(
                    "aliases",
                    [],
                ),

                # Централна позиция на предмета.
                "center": point.get("center"),

                # Измерени размери по X, Y и Z.
                "dimensions": point.get(
                    "dimensions"
                ),

                # Височина на предмета.
                #
                # Предпочита размерността по Z,
                # а при липса използва полето height.
                "height": (
                    (
                        point.get("dimensions")
                        or {}
                    ).get("z")
                    or point.get("height")
                ),

                # Позиция на горната повърхност.
                "top_position": point.get(
                    "top_position"
                ),

                # Позиция за хващане.
                "grasp_position": point.get(
                    "grasp_position"
                ),

                # Безопасна позиция за подход.
                "approach_position": point.get(
                    "approach_position"
                ),

                # Указва дали предметът
                # блокира безопасно движение.
                "safety_blocker": point.get(
                    "safety_blocker",
                    False,
                ),

                # Указва дали предметът
                # се следи от историята.
                "tracked_object": point.get(
                    "tracked_object",
                    False,
                ),

                # Дълбочина на запазената история
                # на преместванията.
                "traceback_depth": point.get(
                    "traceback_depth",
                ),
            }
        )

    # Връща броя и информацията
    # за всички известни физически предмети.
    return json.dumps(
        {
            "ok": True,
            "count": len(points),
            "points": points,
        },
        ensure_ascii=False,
    )