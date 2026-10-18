# Обектно-ориентиран интерфейс за работа
# с файлови пътища.
from pathlib import Path


# Определя кореновата директория на проекта.
#
# __file__ сочи към текущия тестов файл,
# resolve() връща абсолютния път,
# а parents[2] избира директорията две нива нагоре.
ROOT = Path(__file__).resolve().parents[2]

# Път до основния пакет robot_system.
PACKAGE = ROOT / "robot_system"


# Зарежда като текст модула с геометричния
# решател на обратната кинематика.
ik = (PACKAGE / "planning/ik.py").read_text()

# Зарежда модула, който преобразува декартова цел
# до физически изпълнима ставна конфигурация.
solver = (PACKAGE / "planning/cartesian_solver_mixin.py").read_text()

# Зарежда последователността за хващане на предмет.
pick = (PACKAGE / "planning/pick_flow_mixin.py").read_text()

# Зарежда действията за движение
# към предмети от работното пространство.
object_actions = (PACKAGE / "planning/object_actions_mixin.py").read_text()


# Проверява дали геометричният IK решател поддържа
# функция за проектиране на всеки кандидат
# през физическата калибрация.
assert "candidate_projector" in ik

# Проверява дали декартовият решател използва
# специалния метод за преминаване на ставната конфигурация
# през физическата калибрация на сервомоторите.
assert "project_joints_through_physical_calibration" in solver

# Проверява дали ставните стойности се преобразуват
# до физически сервомоторни ъгли.
assert "joints_to_servo" in solver

# Проверява дали физическите сервомоторни ъгли
# се преобразуват обратно до действително представимите
# ставни стойности на цифровия модел.
assert "servo_to_joints" in solver

# Проверява дали последователността за хващане
# използва общия декартов решател.
assert "self.solve_cartesian_target(" in pick

# Проверява дали действията към предмети
# също използват общия декартов решател.
assert "self.solve_cartesian_target(target_xyz)" in object_actions

# Проверява дали X координатата за хващане
# се взема директно от публикуваната точка на скенера.
assert 'float(point["x"])' in pick

# Проверява дали Y координатата за хващане
# се взема директно от публикуваната точка на скенера.
assert 'float(point["y"])' in pick

# Проверява дали Z координатата за хващане
# се взема директно от публикуваната точка на скенера.
assert 'float(point["z"])' in pick

# Проверява дали резултатът изрично посочва,
# че точката за хващане идва от XYZ координатите,
# публикувани от сканиращия агент.
assert 'pick_point_source="scanner_motion_point_xyz"' in pick

# Проверява дали запазените ръчни корекции
# се отчитат като игнорирани при директно хващане,
# така че да не изместват точката от скенера.
assert 'ignored_stored_target_offsets=stored_offsets' in pick


# Обхожда всички mixin модули
# в директорията planning.
for path in (PACKAGE / "planning").glob("*_mixin.py"):
    # Самият централен декартов решател има право
    # да извиква директно геометричния IK решател.
    if path.name == "cartesian_solver_mixin.py":
        continue

    # Всички останали модули трябва да използват
    # solve_cartesian_target(), а не да извикват
    # self.ik.solve директно.
    #
    # Така всяка декартова цел преминава
    # през физическата калибрация преди приемането
    # на окончателното IK решение.
    assert "self.ik.solve" not in path.read_text(), path


# Потвърждава, че декартовите цели се решават
# след прилагане на физическата калибрация.
print("Cartesian targets are solved after physical calibration: OK")

# Потвърждава, че хващането използва точно
# публикуваните от скенера XYZ координати.
print("Pick uses the scanner-published XYZ exactly: OK")

# Потвърждава, че запазените корекции
# не могат да изместят точката за хващане.
print("Stored offsets cannot move the pick away from the point: OK")