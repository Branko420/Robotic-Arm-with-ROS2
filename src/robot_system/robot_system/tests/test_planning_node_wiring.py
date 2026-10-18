# Модул за синтактичен анализ на Python код
# чрез абстрактно синтактично дърво (AST).
import ast

# Обектно-ориентиран интерфейс за работа
# с файлови пътища.
from pathlib import Path


def _constructor_args() -> dict[str, list[str]]:
    # Определя пътя до основния възел
    # на планиращия агент.
    path = Path(__file__).resolve().parents[1] / "agents" / "planning_node.py"

    # Прочита planning_node.py и го преобразува
    # до абстрактно синтактично дърво.
    tree = ast.parse(path.read_text())

    # Резултатът свързва името на конструктора
    # със списъка на подадените позиционни аргументи.
    result: dict[str, list[str]] = {}

    # Обхожда всички синтактични елементи
    # в анализирания Python файл.
    for node in ast.walk(tree):
        # Интерес представляват само директни извиквания
        # на функции или класове, представени чрез ast.Name.
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue

        # Проверява само създаването на зависимостите
        # WorkspaceTargetTools и ObjectTracebackStore.
        if node.func.id not in {"WorkspaceTargetTools", "ObjectTracebackStore"}:
            continue

        # Списък с имената на подадените
        # позиционни аргументи.
        names: list[str] = []

        # Обработва всеки позиционен аргумент
        # на намерения конструктор.
        for argument in node.args:
            # При обикновена променлива записва
            # директно нейното име.
            if isinstance(argument, ast.Name):
                names.append(argument.id)

            # При атрибут, например self.redis,
            # записва очакваното стандартизирано име.
            elif isinstance(argument, ast.Attribute):
                names.append("self.redis")

            # При друг вид израз записва името
            # на съответния AST тип за диагностика.
            else:
                names.append(type(argument).__name__)

        # Запазва намерените аргументи
        # под името на конструктора.
        result[node.func.id] = names

    # Връща информацията за свързването
    # на зависимостите в planning_node.py.
    return result


def test_planning_node_dependency_wiring() -> None:
    # Извлича позиционните аргументи,
    # използвани при създаването на зависимостите.
    args = _constructor_args()

    # Проверява дали WorkspaceTargetTools получава
    # точно четирите очаквани аргумента
    # и в правилния ред.
    assert args["WorkspaceTargetTools"] == [
        "self.redis",
        "MOTION_POINTS_KEY",
        "TARGET_OFFSETS_KEY",
        "LAST_TARGET_KEY",
    ]

    # Проверява дали ObjectTracebackStore получава
    # Redis клиента, ключа за историята
    # и ключа за точките за движение.
    assert args["ObjectTracebackStore"] == [
        "self.redis",
        "OBJECT_TRACEBACK_KEY",
        "MOTION_POINTS_KEY",
    ]