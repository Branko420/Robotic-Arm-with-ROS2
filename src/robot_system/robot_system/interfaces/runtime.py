#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за:
# - JSON обработване;
# - математически операции;
# - променливи на средата;
# - стартиране на външни процеси;
# - измерване на време;
# - създаване на уникални идентификатори.
import json
import math
import os
import subprocess
import time
import uuid
from typing import Any, Dict, Optional

# Клиент за комуникация с Redis.
import redis

# Зареждане на настройки от .env файлове.
from dotenv import load_dotenv


# Проверява последователно възможните местоположения
# на конфигурационния .env файл.
for env_path in (
    os.getenv("ROBOT_SYSTEM_ENV_FILE"),
    "/arm_ws/.env",
    "/workspace/.env",
):
    # Зарежда първия съществуващ файл.
    #
    # override=False означава, че вече зададените
    # променливи на средата няма да бъдат презаписани.
    if env_path and os.path.exists(env_path):
        load_dotenv(
            env_path,
            override=False,
        )


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    """
    Прочита булева стойност от променлива на средата.
    """

    # Получава текстовата стойност на променливата.
    value = os.getenv(name)

    # При липсваща променлива връща стойността по подразбиране.
    if value is None:
        return default

    # Приема няколко стандартни текстови представяния на True.
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Адрес на Redis сървъра.
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://127.0.0.1:6379/0",
)

# Redis поток за заявки към планиращия агент.
PLANNING_REQUEST_STREAM = os.getenv(
    "PLANNING_REQUEST_STREAM",
    "stream:planning_requests",
)

# Redis поток за резултатите от планиращия агент.
PLANNING_RESULT_STREAM = os.getenv(
    "PLANNING_RESULT_STREAM",
    "stream:planning_results",
)

# Redis поток за заявки към агента
# за сканиране на работното пространство.
WORKSPACE_SCAN_REQUEST_STREAM = os.getenv(
    "WORKSPACE_SCAN_REQUEST_STREAM",
    "stream:workspace_scan_requests",
)

# Redis поток за резултатите от сканирането.
WORKSPACE_SCAN_RESULT_STREAM = os.getenv(
    "WORKSPACE_SCAN_RESULT_STREAM",
    "stream:workspace_scan_results",
)

# Redis ключ, съдържащ точките за движение
# към откритите предмети.
MOTION_POINTS_KEY = os.getenv(
    "MOTION_POINTS_KEY",
    "workspace:motion_points",
)

# Redis ключ за историята на преместванията
# и първоначалните позиции на предметите.
OBJECT_TRACEBACK_KEY = os.getenv(
    "OBJECT_TRACEBACK_KEY",
    "workspace:object_traceback",
)


# Максимално време за изчакване на стандартна
# заявка към планиращия агент.
PLANNING_TOOL_TIMEOUT_SEC = float(
    os.getenv(
        "PLANNING_TOOL_TIMEOUT_SEC",
        "150",
    )
)

# Максимално време за изчакване
# на резултат от сканирането.
WORKSPACE_SCAN_TOOL_TIMEOUT_SEC = float(
    os.getenv(
        "WORKSPACE_SCAN_TOOL_TIMEOUT_SEC",
        "90",
    )
)

# Максимално време за изпълнение
# на връщане на предмет по запазената му история.
TRACEBACK_RETURN_TIMEOUT_SEC = float(
    os.getenv(
        "TRACEBACK_RETURN_TIMEOUT_SEC",
        "600",
    )
)

# Изчакване след достигане на началната позиция,
# преди продължаване към сканиране.
SCAN_AFTER_HOME_SETTLE_SEC = float(
    os.getenv(
        "SCAN_AFTER_HOME_SETTLE_SEC",
        "0.8",
    )
)

# Минимално време, през което манипулаторът трябва
# да остане в позицията за сканиране.
SCAN_POSE_MIN_DWELL_SEC = max(
    0.0,
    float(
        os.getenv(
            "SCAN_POSE_MIN_DWELL_SEC",
            "7.0",
        )
    ),
)


# IP адрес или име на Raspberry Pi,
# управляващо физическия манипулатор.
PHYSICAL_ARM_PING_HOST = os.getenv(
    "PHYSICAL_ARM_PING_HOST",
    os.getenv(
        "PLANNER_PI_IP",
        "192.168.188.144",
    ),
)

# Максимално време за изчакване на един ping отговор.
PHYSICAL_ARM_PING_TIMEOUT_SEC = max(
    0.2,
    float(
        os.getenv(
            "PHYSICAL_ARM_PING_TIMEOUT_SEC",
            "1.0",
        )
    ),
)

# Брой изпращани ping пакети.
PHYSICAL_ARM_PING_COUNT = max(
    1,
    int(
        os.getenv(
            "PHYSICAL_ARM_PING_COUNT",
            "1",
        )
    ),
)

# Време, за което резултатът от последната
# проверка за достъпност може да бъде използван повторно.
PHYSICAL_ARM_STATUS_CACHE_SEC = max(
    0.0,
    float(
        os.getenv(
            "PHYSICAL_ARM_STATUS_CACHE_SEC",
            "2.0",
        )
    ),
)

# Определя дали физическо движение е разрешено
# само при успешно откриване на Raspberry Pi.
PHYSICAL_ARM_REQUIRE_AVAILABLE = _env_bool(
    "PHYSICAL_ARM_REQUIRE_AVAILABLE",
    True,
)

# Определя дали сканирането е разрешено,
# когато физическият манипулатор не е достъпен.
SCAN_ALLOW_WITHOUT_PHYSICAL_ARM = _env_bool(
    "SCAN_ALLOW_WITHOUT_PHYSICAL_ARM",
    True,
)

# Определя дали трябва да бъде изпратена команда за HOME,
# когато физическият манипулатор не е достъпен.
SCAN_MOVE_HOME_WHEN_ARM_UNAVAILABLE = _env_bool(
    "SCAN_MOVE_HOME_WHEN_ARM_UNAVAILABLE",
    False,
)


# Създава общ Redis клиент.
REDIS = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)


# Кеш за последния резултат от ping проверката.
_PING_CACHE: Dict[str, Any] = {
    # Монотонното време на последната проверка.
    "checked_monotonic": 0.0,

    # Последният върнат резултат.
    "result": None,
}


def _stream_tail(stream: str) -> str:
    """
    Връща идентификатора на последния запис
    от зададен Redis поток.
    """

    try:
        # Прочита най-новия запис в потока.
        rows = REDIS.xrevrange(
            stream,
            count=1,
        )

        # При празен поток връща началния идентификатор.
        return rows[0][0] if rows else "0-0"

    except Exception:
        # При грешка приема, че четенето
        # трябва да започне от началото.
        return "0-0"


def _wait_for_result(
    stream: str,
    request_id: str,
    after_id: str,
    timeout_sec: float,
) -> Dict[str, Any]:
    """
    Изчаква резултат с конкретен request_id
    от зададен Redis поток.
    """

    # Изчислява крайния момент за изчакване.
    deadline = time.monotonic() + timeout_sec

    # Започва четенето след подадения Redis идентификатор.
    last_id = after_id

    while time.monotonic() < deadline:
        # Изчислява оставащото време в милисекунди.
        remaining_ms = max(
            1,
            int(
                (
                    deadline
                    - time.monotonic()
                )
                * 1000
            ),
        )

        try:
            # Изчаква нови записи в Redis потока.
            rows = REDIS.xread(
                {
                    stream: last_id,
                },
                count=20,
                block=min(
                    500,
                    remaining_ms,
                ),
            )

        except Exception as exc:
            # Връща структуриран резултат
            # при недостъпен Redis сървър.
            return {
                "ok": False,
                "request_id": request_id,
                "state": "REDIS_UNAVAILABLE",
                "error": str(exc),
            }

        # Обхожда получените Redis записи.
        for _stream_name, entries in rows:
            for entry_id, fields in entries:
                # Запазва последния обработен идентификатор.
                last_id = entry_id

                try:
                    # Декодира JSON съдържанието на резултата.
                    payload = json.loads(
                        fields.get(
                            "payload",
                            "{}",
                        )
                    )

                except Exception:
                    # Игнорира повредени или невалидни записи.
                    continue

                # Връща само резултата,
                # съответстващ на текущата заявка.
                if payload.get("request_id") == request_id:
                    return payload

    # Връща грешка при изтичане
    # на максималното време за изчакване.
    return {
        "ok": False,
        "request_id": request_id,
        "state": "TOOL_TIMEOUT",
        "error": (
            f"No result from {stream} "
            f"within {timeout_sec:.1f}s"
        ),
    }


def _request(
    request_stream: str,
    result_stream: str,
    payload: Dict[str, Any],
    timeout_sec: float,
) -> Dict[str, Any]:
    """
    Изпраща заявка чрез Redis поток
    и изчаква съответния резултат.
    """

    # Използва съществуващ request_id или създава нов.
    request_id = payload.setdefault(
        "request_id",
        f"request-{uuid.uuid4().hex[:10]}",
    )

    # Запазва последния запис в резултатния поток,
    # за да не бъде върнат стар резултат.
    result_tail = _stream_tail(result_stream)

    try:
        # Добавя заявката към Redis потока.
        REDIS.xadd(
            request_stream,
            {
                "payload": json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            },
        )

    except Exception as exc:
        # Връща грешка, когато заявката
        # не може да бъде публикувана.
        return {
            "ok": False,
            "request_id": request_id,
            "state": "REDIS_UNAVAILABLE",
            "error": str(exc),
        }

    # Изчаква отговор със същия request_id.
    return _wait_for_result(
        result_stream,
        request_id,
        result_tail,
        timeout_sec,
    )


def _ping_physical_arm(
    force: bool = False,
) -> Dict[str, Any]:
    """
    Проверява чрез ping дали Raspberry Pi
    на физическия манипулатор е достъпно.
    """

    # Текущо монотонно време.
    now = time.monotonic()

    # Последният кеширан резултат.
    cached = _PING_CACHE.get("result")

    # Времето на последната проверка.
    checked = float(
        _PING_CACHE.get(
            "checked_monotonic",
            0.0,
        )
    )

    # Използва кеширания резултат, когато:
    # - не е поискана принудителна проверка;
    # - има предишен резултат;
    # - кешът все още е актуален.
    if (
        not force
        and cached is not None
        and now - checked
        <= PHYSICAL_ARM_STATUS_CACHE_SEC
    ):
        return dict(cached)

    # Командата ping приема цяло число
    # за времето на изчакване.
    timeout_whole_seconds = max(
        1,
        int(
            math.ceil(
                PHYSICAL_ARM_PING_TIMEOUT_SEC
            )
        ),
    )

    # Създава системната ping команда.
    command = [
        "ping",
        "-c",
        str(PHYSICAL_ARM_PING_COUNT),
        "-W",
        str(timeout_whole_seconds),
        PHYSICAL_ARM_PING_HOST,
    ]

    try:
        # Изпълнява ping командата без извеждане
        # на стандартния изход и грешките.
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=(
                PHYSICAL_ARM_PING_TIMEOUT_SEC
                + PHYSICAL_ARM_PING_COUNT
                + 1.0
            ),
        )

        # Код 0 означава успешен ping отговор.
        available = completed.returncode == 0

        # Записва причината за резултата.
        reason = (
            "ping_ok"
            if available
            else "ping_failed"
        )

    except FileNotFoundError:
        # Системната команда ping не е налична.
        available = False
        reason = "ping_command_not_found"

    except subprocess.TimeoutExpired:
        # Системната команда не е приключила
        # в определеното време.
        available = False
        reason = "ping_timeout"

    except Exception as exc:
        # Обработва останалите неочаквани грешки.
        available = False
        reason = (
            f"ping_error:"
            f"{type(exc).__name__}"
        )

    # Създава структуриран резултат.
    result = {
        "ok": available,
        "available": available,
        "state": (
            "PHYSICAL_ARM_AVAILABLE"
            if available
            else "PHYSICAL_ARM_UNAVAILABLE"
        ),
        "host": PHYSICAL_ARM_PING_HOST,
        "reason": reason,
        "checked_at": time.time(),
    }

    # Актуализира кеша.
    _PING_CACHE["checked_monotonic"] = now
    _PING_CACHE["result"] = dict(result)

    return result


def _physical_arm_guard(
    execute: bool,
) -> Optional[Dict[str, Any]]:
    """
    Проверява дали физическото движение е разрешено.

    Връща None при разрешено изпълнение или речник
    с грешка, когато физическият манипулатор е недостъпен.
    """

    # При цифрова визуализация не се изисква
    # физическият манипулатор да бъде достъпен.
    if not execute:
        return None

    # Пропуска проверката, когато тя е изключена
    # чрез конфигурацията.
    if not PHYSICAL_ARM_REQUIRE_AVAILABLE:
        return None

    # Проверява достъпността на Raspberry Pi.
    status = _ping_physical_arm(
        force=False
    )

    # Позволява физическото движение
    # при успешно открит манипулатор.
    if status.get("available"):
        return None

    # Връща информация, че команда
    # за физическо движение не е изпратена.
    return {
        **status,
        "ok": False,
        "movement_sent": False,
    }


def _resolve_workspace_target(
    query: str,
) -> Dict[str, Any]:
    """
    Изпраща заявка към планиращия агент
    за определяне на конкретен предмет.
    """

    return _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"resolve-target-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Действие за определяне на цел.
            "action": "resolve_target",

            # Потребителското описание на предмета.
            "target": query,

            # Определянето на цел не предизвиква движение.
            "execute": False,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )


def _resolved_identifier(
    resolution: Dict[str, Any],
    fallback: str,
) -> str:
    """
    Извлича устойчив идентификатор
    от резултата за определяне на цел.
    """

    # Получава данните за определения предмет.
    target = resolution.get("target") or {}

    # Предпочита:
    # 1. уникален instance_id;
    # 2. име на точката;
    # 3. първоначално подаденото описание.
    return str(
        target.get("instance_id")
        or target.get("point")
        or fallback
    )


def _with_target_role(
    result: Dict[str, Any],
    role: str,
) -> Dict[str, Any]:
    """
    Добавя роля към резултат за определяне на цел.

    Ролята може например да бъде source или destination.
    """

    # Копира резултата, без да променя оригиналния речник.
    output = dict(result)

    # Добавя ролята на целта.
    output["target_role"] = role

    return output


def _move_arm_to_named_pose(
    action: str,
    target: str,
) -> Dict[str, Any]:
    """
    Изпраща команда за придвижване
    към именувана позиция.
    """

    return _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на движението.
            "request_id": (
                f"{action}-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Името на действието.
            "action": action,

            # Името на целевата позиция.
            "target": target,

            # Разрешава физическото изпълнение.
            "execute": True,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )


def _pose_status(
    pose_name: str,
) -> Dict[str, Any]:
    """
    Проверява дали манипулаторът се намира
    в конкретна именувана позиция.
    """

    return _request(
        PLANNING_REQUEST_STREAM,
        PLANNING_RESULT_STREAM,
        {
            # Уникален идентификатор на заявката.
            "request_id": (
                f"pose-status-"
                f"{uuid.uuid4().hex[:10]}"
            ),

            # Действие за проверка на позиция.
            "action": "pose_status",

            # Име на проверяваната позиция.
            "target": pose_name,

            # Проверката не предизвиква движение.
            "execute": False,
        },
        PLANNING_TOOL_TIMEOUT_SEC,
    )


def _prepare_arm_for_scan() -> Dict[str, Any]:
    """
    Подготвя манипулатора за сканиране.

    Последователността е:
    1. проверка дали ръката вече е в HOME;
    2. движение към HOME при необходимост;
    3. движение към scan_pose.
    """

    # Проверява дали манипулаторът вече
    # се намира в началната позиция.
    status_before = _pose_status("home")

    # Приема, че HOME е достигната само когато:
    # - заявката е успешна;
    # - at_pose е True.
    already_home = bool(
        status_before.get("ok")
        and status_before.get("at_pose")
    )

    if already_home:
        # Не изпраща излишно движение,
        # когато HOME вече е достигната.
        home_before = {
            "ok": True,
            "state": "HOME_ALREADY_REACHED",
            "skipped": True,
            "pose_status": status_before,
        }

    else:
        # Премества ръката към началната позиция.
        home_before = _move_arm_to_named_pose(
            "home",
            "home",
        )

        # Прекратява подготовката,
        # когато HOME не може да бъде достигната.
        if not home_before.get("ok"):
            return {
                "ok": False,
                "state": (
                    "SCAN_HOME_PREPARATION_FAILED"
                ),
                "home_before": home_before,
                "pose_status": status_before,
            }

    # След HOME премества манипулатора
    # към позицията за сканиране.
    scan_pose = _move_arm_to_named_pose(
        "scan_pose",
        "scan_pose",
    )

    # Връща грешка, когато позицията
    # за сканиране не може да бъде достигната.
    if not scan_pose.get("ok"):
        return {
            "ok": False,
            "state": (
                "SCAN_POSE_PREPARATION_FAILED"
            ),
            "home_before": home_before,
            "scan_pose": scan_pose,
        }

    # Връща успешен резултат след достигане
    # на позицията за сканиране.
    return {
        "ok": True,
        "state": "SCAN_POSE_REACHED",
        "home_was_skipped": already_home,
        "home_before": home_before,
        "scan_pose": scan_pose,
    }