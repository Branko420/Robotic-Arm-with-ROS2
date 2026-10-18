# Обектно-ориентиран интерфейс за работа с файлови пътища.
from pathlib import Path

# Определя кореновата директория на проекта.
ROOT = Path(__file__).resolve().parents[2]

# Зарежда като текст възела на агента
# за сканиране на работното пространство.
SCAN = (ROOT / "robot_system/agents/workspace_scan_node.py").read_text()

# Зарежда като текст конфигурацията и обработката
# на облака от точки.
CONFIG = (ROOT / "robot_system/scanning/cloud_processor.py").read_text()

# Зарежда като текст геометричния решател
# на обратната кинематика.
IK = (ROOT / "robot_system/planning/ik.py").read_text()

# Зарежда като текст последователността
# за хващане на предмет.
PICK = (ROOT / "robot_system/planning/pick_flow_mixin.py").read_text()

# Проверява дали възелът за сканиране използва
# конфигурирания Redis ключ за корекциите на целите.
assert "cfg.target_offsets_key" in SCAN

# Проверява дали конфигурацията поддържа изчистване
# на старите корекции преди ново сканиране.
assert "CLEAR_TARGET_OFFSETS_BEFORE_SCAN" in CONFIG

# Проверява дали IK решателят съдържа твърда
# максимална допустима грешка на целта.
assert "PLANNER_HARD_MAX_TARGET_ERROR_M" in IK

# Проверява дали режимът за приблизително решение
# е изключен по подразбиране.
assert 'env_bool("PLANNER_ALLOW_BEST_EFFORT", False)' in IK

# Проверява дали X координатата за хващане
# се взема директно от точката на скенера.
assert 'float(point["x"])' in PICK

# Проверява дали Y координатата за хващане
# се взема директно от точката на скенера.
assert 'float(point["y"])' in PICK

# Проверява дали Z координатата за хващане
# се взема директно от точката на скенера.
assert 'float(point["z"])' in PICK

# Проверява дали резултатът изрично отбелязва,
# че точката за хващане идва от XYZ координатите,
# публикувани от сканиращия агент.
assert 'pick_point_source="scanner_motion_point_xyz"' in PICK

# Проверява дали при значително разминаване между
# зададената и достигнатата цел се връща отделно
# състояние TARGET_MISMATCH за конкретната стъпка.
assert "PICK_{step_name.upper()}_TARGET_MISMATCH" in PICK