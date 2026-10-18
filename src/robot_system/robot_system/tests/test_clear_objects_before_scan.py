# Обектно-ориентиран интерфейс за работа с файлови пътища.
from pathlib import Path


# Определя кореновата директория на проекта.
#
# __file__ сочи към текущия тестов файл,
# resolve() връща абсолютния му път,
# а parents[2] избира директорията две нива нагоре.
ROOT = Path(__file__).resolve().parents[2]

# Зарежда като текст основния ROS 2 възел
# на агента за сканиране на работното пространство.
NODE = (ROOT / "robot_system/agents/workspace_scan_node.py").read_text()

# Зарежда като текст модула за обработване
# на облака от точки и конфигурацията на сканирането.
CONFIG = (ROOT / "robot_system/scanning/cloud_processor.py").read_text()


# Проверява дали конфигурацията поддържа настройка
# за изчистване на предишното състояние на предметите
# преди започване на ново сканиране.
assert "CLEAR_OBJECT_STATE_BEFORE_SCAN" in CONFIG

# Проверява дали в конфигурацията присъства
# Redis ключът за регистъра на отделните
# физически екземпляри на предметите.
assert "workspace:object_instances" in CONFIG

# Проверява дали в конфигурацията присъства
# Redis ключът за последно избраната цел.
assert "workspace:last_planning_target" in CONFIG

# Проверява дали възелът съдържа отделен метод
# за изчистване на състоянието от предишното сканиране.
assert "def clear_previous_scan_state" in NODE

# Проверява дали при започване на сканиране
# се публикува празно състояние SCANNING.
#
# Това предотвратява използването на остарели предмети,
# докато новият облак от точки се обработва.
assert 'self.publish_empty("SCANNING")' in NODE

# Проверява дали възелът използва конфигурирания
# Redis ключ за регистъра на екземплярите.
assert "cfg.object_instance_registry_key" in NODE

# Проверява дали възелът използва конфигурирания
# Redis ключ за последно избраната цел.
assert "cfg.last_target_key" in NODE

# Проверява дали при изчистване на предишния резултат
# се публикува и празен облак с избрани компоненти.
assert "self.publish_empty_clusters()" in NODE


# Намира началната позиция на метода scan_workspace
# в текстовото съдържание на възела.
scan_start = NODE.index("def scan_workspace")

# Намира позицията на изчакването за нов облак от точки,
# като търсенето започва от метода scan_workspace.
wait_start = NODE.index("cloud_msg, error = self.wait_for_cloud()", scan_start)

# Намира позицията на извикването, което изчиства
# предишното състояние на сканирането.
clear_start = NODE.index("self.clear_previous_scan_state(request_id)", scan_start)

# Проверява дали предишното състояние се изчиства
# преди възелът да започне да чака нов облак от точки.
#
# Така старите предмети и цели не остават активни
# по време на подготовката за новото сканиране.
assert clear_start < wait_start


# Redis ключове, чието съдържание трябва да бъде запазено
# при изчистване на резултатите от предишното сканиране.
for protected_key in (
    # Състояние на предмета, който в момента
    # се държи от хващача.
    "workspace:held_object",

    # История на преместванията на предметите.
    "workspace:object_traceback",

    # Натрупани ръчни корекции на целите.
    "workspace:target_offsets",

    # Потребителски запазени декартови точки.
    "workspace:saved_cartesian_points",
):
    # Проверява дали възелът не изтрива
    # нито един от защитените Redis ключове.
    assert protected_key not in NODE