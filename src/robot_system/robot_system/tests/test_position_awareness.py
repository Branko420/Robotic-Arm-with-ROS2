# Обектно-ориентиран интерфейс за работа с файлови пътища.
from pathlib import Path

# Определя основната директория robot_system.
#
# Текущият тестов файл се намира в поддиректория,
# затова parents[1] избира директорията на пакета.
ROOT = Path(__file__).resolve().parents[1]


def test_pick_sequence_and_gripper_values():
    # Зарежда последователността за хващане на предмет.
    text=(ROOT/'planning/pick_flow_mixin.py').read_text()

    # Зарежда общите настройки на планиращия агент.
    settings=(ROOT/'planning/settings.py').read_text()

    # Проверява дали последователността съдържа
    # позиция за подход на 3 cm над предмета.
    assert '("approach_3cm", above_xyz)' in text

    # Проверява дали хващането се изпълнява
    # в централната XYZ точка на предмета.
    assert '("grasp_center", pick_xyz)' in text

    # Проверява дали след хващането се изпълнява
    # вертикално оттегляне до позицията на 3 cm.
    assert '("retreat_3cm", above_xyz)' in text

    # Проверява дали височината за вертикален просвет
    # се управлява чрез конфигурационната константа.
    assert 'PICK_VERTICAL_CLEARANCE_M' in text

    # Проверява дали настройките съдържат
    # стойността 130° за затворено положение на хващача.
    assert '130.0' in settings

    # Проверява дали настройките съдържат
    # стойността 70° за отворено положение на хващача.
    assert '70.0' in settings


def test_current_pose_and_saved_points():
    # Зарежда модула за текущото състояние
    # и запазените декартови точки.
    state=(ROOT/'planning/state_mixin.py').read_text()

    # Зарежда маршрутизатора на действията
    # към съответните обработващи методи.
    router=(ROOT/'planning/action_router_mixin.py').read_text()

    # Проверява дали модулът за състояние поддържа:
    # - получаване на текущата декартова позиция;
    # - постоянно съхраняване на текущата позиция;
    # - записване на именувана точка;
    # - зареждане на запазените точки.
    for token in ('current_cartesian_pose', 'persist_current_cartesian_pose', 'save_current_point', 'load_saved_points'):
        assert token in state

    # Проверява дали маршрутизаторът поддържа:
    # - относително движение;
    # - записване на текуща точка;
    # - движение до запазена точка;
    # - изтриване на запазена точка.
    for token in ('move_relative', 'save_current_point', 'move_to_saved_point', 'delete_saved_point'):
        assert token in router


def test_barge_in_enabled():
    # Зарежда средата за изпълнение
    # на разговорния агент в реално време.
    text=(ROOT/'realtime/runtime.py').read_text()

    # Проверява дали Realtime моделът позволява
    # текущият отговор да бъде прекъснат от нова реч.
    assert '"interrupt_response": True' in text

    # Проверява дали съществува методът
    # за прекъсване на текущия аудиоотговор.
    assert 'interrupt_assistant_output' in text

    # Проверява дали прекъсването се обозначава
    # и маршрутизира като realtime_barge_in.
    assert 'realtime_barge_in' in text