# Работа с файлови пътища.
from pathlib import Path

# Динамично зареждане на Python модули от конкретни файлове.
import importlib.util

# Математически функции, включително изчисляване
# на евклидово разстояние между две точки.
import math

# Достъп до регистъра на вече заредените Python модули.
import sys

# Създаване на временни модулни обекти.
import types


# Определя кореновата директория на проекта.
#
# __file__ сочи към текущия тестов файл,
# resolve() връща абсолютния път,
# а parents[2] избира директорията две нива нагоре.
ROOT = Path(__file__).resolve().parents[2]

# Път до директорията с модулите за планиране.
PLANNING = ROOT / "robot_system/planning"


# Създава временен пакет robot_system.
#
# Това позволява модулите да бъдат заредени директно
# от файловата система, без целият ROS 2 пакет
# да бъде предварително инсталиран.
pkg = types.ModuleType("robot_system")

# Задава пътя, в който Python трябва да търси
# подмодулите на временния пакет.
pkg.__path__ = [str(ROOT / "robot_system")]

# Регистрира пакета само ако вече не съществува
# в sys.modules.
sys.modules.setdefault("robot_system", pkg)


# Създава временния подпакет robot_system.planning.
sub = types.ModuleType("robot_system.planning")

# Задава директорията с модулите за планиране.
sub.__path__ = [str(PLANNING)]

# Регистрира подпакета в системния модулен регистър.
sys.modules.setdefault("robot_system.planning", sub)


# Зарежда необходимите модули common.py и ik.py
# в правилния пакетен контекст.
for name in ("common", "ik"):
    # Създава пълното име на модула.
    module_name = f"robot_system.planning.{name}"

    # Създава спецификация за зареждане
    # на модула от конкретния .py файл.
    spec = importlib.util.spec_from_file_location(module_name, PLANNING / f"{name}.py")

    # Създава нов модулен обект според спецификацията.
    module = importlib.util.module_from_spec(spec)

    # Регистрира модула преди изпълнението му,
    # за да работят вътрешните относителни import конструкции.
    sys.modules[module_name] = module

    # Изпълнява съдържанието на модула.
    spec.loader.exec_module(module)


# Създава екземпляр на геометричния решател
# на обратната кинематика.
solver = sys.modules["robot_system.planning.ik"].GeometricIKSolver()

# Създава начално ставно състояние,
# при което всички ставни ъгли са 0 градуса.
current = {name: 0.0 for name in solver.limits_deg}

# Декартова цел, зададена чрез координати X, Y и Z.
target = (0.36777, -0.04087, 0.02308)

# Решава обратната кинематика за зададената цел,
# като използва текущото ставно състояние.
solution, _ = solver.solve(*target, current_joint_deg=current)

# Проверява дали е намерено валидно IK решение.
assert solution is not None

# Проверява дали достигнатата от решението позиция
# съвпада с целевата с грешка под 1 микрометър.
assert math.dist(solution["reached"], target) < 1e-6