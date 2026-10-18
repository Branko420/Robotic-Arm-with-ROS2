# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Модул за синтактичен анализ на Python файлове
# чрез абстрактно синтактично дърво (Abstract Syntax Tree, AST).
import ast

# Помощен модул за работа с информация
# за налични Python модули.
import importlib.util

# Обектно-ориентиран интерфейс за работа
# с файлови пътища.
from pathlib import Path


# Определя кореновата директория на проекта.
#
# __file__ сочи към текущия файл, resolve() връща
# абсолютния му път, а parents[2] избира директорията
# две нива над него.
ROOT = Path(__file__).resolve().parents[2]

# Път до основния Python пакет robot_system.
PACKAGE = ROOT / "robot_system"


# Директории от предишната структура на проекта,
# които вече не трябва да съществуват.
#
# Проверката гарантира, че кодът е организиран
# според новото разделение по отговорности.
REMOVED_DIRECTORIES = {
    "agent_tools",
    "ai_agents",
    "coordinator",
    "execution",
    "motion",
    "perception",
    "safety",
    "world",
}


def _module_exists(module: str) -> bool:
    """
    Проверява дали вътрешен модул от пакета robot_system
    съществува като Python файл или пакетна директория.
    """

    # Външните модули не се проверяват спрямо
    # файловата структура на robot_system.
    if not module.startswith("robot_system"):
        return True

    # Премахва началната част robot_system
    # и разделя останалия модулен път.
    parts = module.split(".")[1:]

    # Създава файлов път до очаквания модул.
    candidate = PACKAGE.joinpath(*parts)

    # Модулът се счита за съществуващ, когато:
    # - има съответен .py файл;
    # - или има директория с __init__.py.
    return (
        candidate.with_suffix(".py").exists()
        or (candidate / "__init__.py").exists()
    )


def main() -> None:
    """
    Проверява дали директориалната структура
    и вътрешните import пътища на проекта са валидни.
    """

    # Събира имената на всички директории,
    # намиращи се директно в пакета robot_system.
    present = {
        path.name
        for path in PACKAGE.iterdir()
        if path.is_dir()
    }

    # Проверява дали някоя от премахнатите директории
    # все още присъства в проекта.
    #
    # При неуспех assert показва точното множество
    # на намерените забранени директории.
    assert not (
        REMOVED_DIRECTORIES & present
    ), REMOVED_DIRECTORIES & present

    # Обхожда рекурсивно всички Python файлове
    # в пакета robot_system.
    for path in PACKAGE.rglob("*.py"):
        # Прочита файла и го преобразува
        # до абстрактно синтактично дърво.
        tree = ast.parse(
            path.read_text(),
            filename=str(path),
        )

        # Обхожда всички синтактични елементи
        # в дървото на текущия файл.
        for node in ast.walk(tree):
            # Проверява import конструкции от вида:
            # from robot_system... import ...
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
            ):
                # Проверява дали посоченият вътрешен модул
                # действително съществува.
                assert _module_exists(
                    node.module
                ), f"{path}: missing {node.module}"

    # Задължителни директории според текущото
    # разделение на системата по отговорности.
    expected = {
        "agents",
        "interfaces",
        "planning",
        "scanning",
        "realtime",
        "storage",
        "visualization",
        "config",
    }

    # Проверява дали всички задължителни директории
    # присъстват в пакета.
    #
    # При грешка се показват липсващите директории.
    assert expected <= present, expected - present

    # Потвърждава успешната проверка
    # на основната директориална структура.
    print(
        "Clean responsibility-based directory layout: OK"
    )

    # Проверява дали старият RViz модул
    # trajectory_joint_state_bridge.py е премахнат.
    assert not (
        PACKAGE
        / "visualization/trajectory_joint_state_bridge.py"
    ).exists()

    # Потвърждава, че няма вътрешни import пътища
    # към липсващи модули.
    print(
        "No stale internal import paths: OK"
    )

    # Потвърждава, че остарелият RViz preview модул
    # вече не присъства.
    print(
        "No obsolete RViz preview node: OK"
    )


# Стартира main() само когато файлът
# се изпълнява директно.
if __name__ == "__main__":
    main()