"""
Публични функционални инструменти, предоставени
на разговорния агент в реално време.
"""

# Инструменти за проверка и директно управление
# на физическия роботизиран манипулатор.
from .arm_tools import (
    check_physical_arm_availability,
    get_robot_joint_angles,
    set_robot_joint_angles,
)

# Инструменти за работа с историята
# на преместваните предмети.
from .history_tools import (
    list_object_movement_history,
    return_object_to_previous_position,
)

# Инструменти за избор на предмети,
# движение, хващане, преместване и поставяне.
from .manipulation_tools import (
    control_robot_in_rviz,
    correct_last_robot_target,
    move_held_object_to_target,
    pick_and_place_in_rviz,
    place_held_object_at_target,
    resolve_workspace_target,
)

# Инструменти за сканиране на работното пространство
# и получаване на откритите точки.
from .scan_tools import (
    list_workspace_points,
    scan_workspace_and_make_points,
)


# Определя публичните имена, които се експортират
# от пакета при използване на:
#
# from robot_system.interfaces.planning_tools import *
#
# Списъкът съдържа всички инструменти,
# достъпни за разговорния агент.
__all__ = [
    # Проверка на физическия манипулатор.
    "check_physical_arm_availability",

    # Основно управление на цифровия
    # и физическия манипулатор.
    "control_robot_in_rviz",

    # Относителна корекция на последната
    # или текущата целева позиция.
    "correct_last_robot_target",

    # Получаване на последните командвани
    # сервомоторни стойности.
    "get_robot_joint_angles",

    # Получаване на историята
    # на преместванията на предметите.
    "list_object_movement_history",

    # Получаване на точките
    # от работното пространство.
    "list_workspace_points",

    # Преместване на вече хванат предмет,
    # без освобождаване на хващача.
    "move_held_object_to_target",

    # Изпълнение на пълна последователност
    # за хващане и поставяне.
    "pick_and_place_in_rviz",

    # Поставяне на предмет,
    # който вече се държи от хващача.
    "place_held_object_at_target",

    # Определяне на конкретен предмет
    # от потребителско описание.
    "resolve_workspace_target",

    # Връщане на предмет към предишна
    # или начална позиция.
    "return_object_to_previous_position",

    # Сканиране на работното пространство
    # и създаване на точки за движение.
    "scan_workspace_and_make_points",

    # Директно задаване на стойности
    # на една или повече стави.
    "set_robot_joint_angles",

    # Изтриване на запазена именувана точка.
    "delete_saved_robot_point",

    # Получаване на текущата декартова позиция
    # на крайния изпълнителен механизъм.
    "get_current_robot_position",

    # Получаване на списък
    # със запазените точки.
    "list_saved_robot_points",

    # Преместване на манипулатора
    # към запазена точка.
    "move_to_saved_robot_point",

    # Запазване на текущата позиция
    # като именувана точка.
    "save_current_robot_point",
]


# Инструменти за работа с потребителски
# запазени декартови точки.
from .point_tools import (
    delete_saved_robot_point,
    get_current_robot_position,
    list_saved_robot_points,
    move_to_saved_robot_point,
    save_current_robot_point,
)