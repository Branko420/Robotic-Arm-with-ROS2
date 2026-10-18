"""
Настройки на планиращия модул и Redis ключове,
зареждани чрез променливи на средата.
"""

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Клас за работа с файлови пътища.
from pathlib import Path

# Помощни функции за зареждане на текстови,
# булеви и числови стойности от средата.
from .common import env, env_bool, env_float

# Безопасни диапазони на физическите сервомотори.
from .constants import SAFE_SERVO_RANGE


# Директорията config в основната директория
# на Python пакета.
PACKAGE_CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "config"
)


# Адрес за свързване към Redis.
#
# По подразбиране се използва локален Redis сървър,
# база с индекс 0.
REDIS_URL = env(
    "REDIS_URL",
    "redis://127.0.0.1:6379/0",
)


# Redis ключ, съдържащ актуалните точки
# за движение в работното пространство.
MOTION_POINTS_KEY = env(
    "MOTION_POINTS_KEY",
    "workspace:motion_points",
)


# Redis Stream за входящите заявки
# към планиращия агент.
PLANNING_REQUEST_STREAM = env(
    "PLANNING_REQUEST_STREAM",
    "stream:planning_requests",
)


# Redis Stream за резултатите,
# публикувани от планиращия агент.
PLANNING_RESULT_STREAM = env(
    "PLANNING_RESULT_STREAM",
    "stream:planning_results",
)


# Redis ключ за натрупаните корекции
# по X, Y и Z на отделните цели.
TARGET_OFFSETS_KEY = env(
    "TARGET_OFFSETS_KEY",
    "workspace:target_offsets",
)


# Redis ключ за последно избраната
# цел на планиращия агент.
LAST_TARGET_KEY = env(
    "LAST_TARGET_KEY",
    "workspace:last_planning_target",
)


# Redis ключ за предмета,
# който в момента се държи от хващача.
HELD_OBJECT_KEY = env(
    "HELD_OBJECT_KEY",
    "workspace:held_object",
)


# Redis ключ за текущата декартова позиция
# на крайния механизъм.
CURRENT_POSE_KEY = env(
    "CURRENT_POSE_KEY",
    "workspace:current_end_effector_pose",
)


# Redis ключ за именувани или запазени
# декартови точки.
SAVED_POINTS_KEY = env(
    "SAVED_POINTS_KEY",
    "workspace:saved_cartesian_points",
)


# Максимално допустима разлика в градуси,
# използвана при проверка дали манипулаторът
# се намира в определена именувана позиция.
#
# Минималната разрешена стойност е 0.1 градуса.
POSE_MATCH_TOLERANCE_DEG = max(
    0.1,
    env_float(
        "POSE_MATCH_TOLERANCE_DEG",
        2.0,
    ),
)


# Redis ключ за историята на преместванията
# на отделните предмети.
OBJECT_TRACEBACK_KEY = env(
    "OBJECT_TRACEBACK_KEY",
    "workspace:object_traceback",
)


# ROS 2 тема за визуалните маркери,
# представящи историята на предметите в RViz.
TRACEBACK_MARKER_TOPIC = env(
    "TRACEBACK_MARKER_TOPIC",
    "/digital_twin/object_traceback_markers",
)


# Redis ключ за регистъра с устойчивите
# идентификатори на физическите екземпляри.
OBJECT_INSTANCE_REGISTRY_KEY = env(
    "OBJECT_INSTANCE_REGISTRY_KEY",
    "workspace:object_instances",
)


# Представка за Redis ключовете,
# използвани за кеширане на резултатите
# от отделните заявки.
REQUEST_RESULT_PREFIX = env(
    "PLANNING_REQUEST_RESULT_PREFIX",
    "planning:request_result",
)


# Продължителност на съхранение на кеширан
# резултат от заявка в секунди.
#
# Минималната допустима стойност е 60 секунди.
REQUEST_RESULT_TTL_SEC = max(
    60,
    int(
        env_float(
            "PLANNING_REQUEST_RESULT_TTL_SEC",
            3600,
        )
    ),
)


# Определя дали движенията, изпълнени без активна
# физическа ръка, трябва да бъдат записвани
# в историята на предметите.
RECORD_SIMULATED_MOVES = env_bool(
    "PLANNER_RECORD_SIMULATED_MOVES",
    False,
)


# Физически ъгъл на сервомотора при отворен хващач
# по време на последователност за хващане.
#
# Стойността се ограничава до безопасния диапазон
# на сервомотора на хващача.
PICK_GRIPPER_OPEN_SERVO_DEG = max(
    SAFE_SERVO_RANGE["gripper"][0],
    min(
        SAFE_SERVO_RANGE["gripper"][1],
        env_float(
            "PLANNER_PICK_GRIPPER_OPEN_SERVO_DEG",
            70.0,
        ),
    ),
)


# Физически ъгъл на сервомотора при затворен хващач
# по време на последователност за хващане.
#
# Стойността се ограничава до безопасния диапазон
# на сервомотора на хващача.
PICK_GRIPPER_CLOSE_SERVO_DEG = max(
    SAFE_SERVO_RANGE["gripper"][0],
    min(
        SAFE_SERVO_RANGE["gripper"][1],
        env_float(
            "PLANNER_GRIPPER_CLOSE_SERVO_DEG",
            130.0,
        ),
    ),
)


# Вертикален просвет в метри, използван
# при подход и оттегляне над предмета.
#
# Не се допуска отрицателна стойност.
PICK_VERTICAL_CLEARANCE_M = max(
    0.0,
    env_float(
        "PLANNER_PICK_VERTICAL_CLEARANCE_M",
        0.03,
    ),
)


# Продължителност на ръчно зададено
# ставно движение в секунди.
#
# Минималната допустима стойност е 0.2 секунди.
MANUAL_JOINT_MOVE_DURATION_SEC = max(
    0.2,
    env_float(
        "PLANNER_MANUAL_JOINT_MOVE_DURATION_SEC",
        3.0,
    ),
)