from __future__ import annotations

# Динамично зареждане на Python модули
# директно от конкретни файлове.
import importlib.util

# Достъп до регистъра на заредените Python модули.
import sys

# Създаване на временни модулни обекти.
import types

# Обектно-ориентиран интерфейс
# за работа с файлови пътища.
from pathlib import Path


# Определя кореновата директория на проекта.
ROOT = Path(__file__).resolve().parents[2]

# Път до директорията с модулите
# от слоя за планиране.
PLANNING = ROOT / "robot_system/planning"


# Създава временен пакет robot_system.
#
# Това позволява тестът да зареди необходимите модули
# директно от файловата система, без пакетът да бъде
# предварително инсталиран.
pkg = types.ModuleType("robot_system")

# Определя директорията, в която Python трябва
# да търси подмодулите на robot_system.
pkg.__path__ = [str(ROOT / "robot_system")]

# Регистрира временния пакет само ако модул
# със същото име все още не е зареден.
sys.modules.setdefault("robot_system", pkg)


# Създава временен подпакет robot_system.planning.
sub = types.ModuleType("robot_system.planning")

# Определя директорията с модулите за планиране.
sub.__path__ = [str(PLANNING)]

# Регистрира подпакета в системния регистър.
sys.modules.setdefault("robot_system.planning", sub)


# Зарежда модулите, необходими за работа
# на WorkspaceTargetTools.
for name in ("common", "instances", "targets"):
    # Създава пълното пакетно име на модула.
    module_name = f"robot_system.planning.{name}"

    # Създава спецификация за зареждане
    # на модула от съответния Python файл.
    spec = importlib.util.spec_from_file_location(
        module_name,
        PLANNING / f"{name}.py",
    )

    # Създава нов модулен обект
    # според подготвената спецификация.
    module = importlib.util.module_from_spec(spec)

    # Регистрира модула преди неговото изпълнение,
    # за да работят относителните import конструкции.
    sys.modules[module_name] = module

    # Изпълнява съдържанието на модула.
    spec.loader.exec_module(module)


# Получава класа WorkspaceTargetTools
# от динамично заредения модул targets.
Targets = sys.modules[
    "robot_system.planning.targets"
].WorkspaceTargetTools

# Получава изключението, използвано при наличие
# на повече от една подходяща цел.
Ambiguous = sys.modules[
    "robot_system.planning.targets"
].AmbiguousTargetError


class FakeRedis:
    # Минимална тестова реализация на Redis клиент.
    #
    # Данните се пазят локално в речник,
    # без връзка към реален Redis сървър.
    def __init__(self) -> None:
        self.data = {}

    # Връща стойността за зададения ключ.
    def get(self, key):
        return self.data.get(key)

    # Записва стойност за зададения ключ.
    def set(self, key, value):
        self.data[key] = value


# Създава тестовия Redis клиент.
redis = FakeRedis()

# Създава инструмента за разрешаване на цели.
#
# Подадените ключове представляват:
# - документа с точките;
# - корекциите на целите;
# - последно запомнената цел.
targets = Targets(redis, "motion", "offsets", "last")


# Тестово работно пространство с:
# - два зелени предмета;
# - един син предмет.
points = [
    {
        "point": "point_green_1",
        "instance_id": "green_object_1",
        "colors": ["green"],
        "x": 0.22,
        "y": 0.08,
        "z": 0.03,
        "center": {"x": 0.22, "y": 0.08, "z": 0.03},
    },
    {
        "point": "point_green_2",
        "instance_id": "green_object_2",
        "colors": ["green"],
        "x": 0.27,
        "y": -0.07,
        "z": 0.03,
        "center": {"x": 0.27, "y": -0.07, "z": 0.03},
    },
    {
        "point": "point_blue_1",
        "instance_id": "blue_object_1",
        "colors": ["blue"],
        "x": 0.25,
        "y": 0.00,
        "z": 0.03,
        "center": {"x": 0.25, "y": 0.00, "z": 0.03},
    },
]


# Опитва да разреши заявката „синия“.
#
# Тъй като има само един син предмет,
# не трябва да се изисква допълнително уточнение.
blue = targets.resolve_target(points, "синия")

# Проверява дали е намерена цел.
assert blue is not None

# Проверява дали е избран единственият син предмет.
assert blue["instance_id"] == "blue_object_1"


# Запомня синия предмет като последно избрана цел.
targets.remember_target(blue)


# Проверява дали българските и английските местоимения
# се разрешават към последно запомнения предмет.
for reference in ("го", "него", "този", "it", "that"):
    resolved = targets.resolve_target(points, reference)

    # Местоимението трябва да бъде успешно разрешено.
    assert resolved is not None

    # Резултатът трябва да сочи към синия предмет.
    assert resolved["instance_id"] == "blue_object_1"


# Опитва да разреши заявката „зеления“.
#
# Тъй като има два зелени предмета,
# системата трябва да отчете нееднозначност.
try:
    targets.resolve_target(points, "зеления")

except Ambiguous as exc:
    # Проверява дали изключението съдържа
    # точно двата зелени кандидата.
    assert len(exc.candidates) == 2

else:
    # Ако не е повдигнато изключение,
    # поведението е неправилно, защото системата
    # не трябва произволно да избира един от предметите.
    raise AssertionError("Two green objects must remain ambiguous")


# Потвърждава, че единственият син предмет
# се разрешава без допълнителен въпрос.
print("Unique blue object resolves without a question: OK")

# Потвърждава, че местоименията използват
# последно запомнената цел.
print("Pronouns resolve to the remembered object: OK")

# Потвърждава, че двата зелени предмета
# остават нееднозначни.
print("Two real green objects remain ambiguous: OK")