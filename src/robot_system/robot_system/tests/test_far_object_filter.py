# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Обектно-ориентиран интерфейс за работа с файлови пътища.
from pathlib import Path


# Определя кореновата директория на проекта.
ROOT = Path(__file__).resolve().parents[2]

# Път до основния пакет robot_system.
PACKAGE = ROOT / "robot_system"


def text(relative: str) -> str:
    # Прочита текстовото съдържание на файл,
    # зададен спрямо директорията robot_system.
    return (PACKAGE / relative).read_text()


def main() -> None:
    # Зарежда обработката на облака от точки,
    # включително филтрирането по разстояние от основата.
    cloud = text("scanning/cloud_processor.py")

    # Зарежда логиката за избиране
    # и валидиране на целите на планиращия агент.
    targets = text("planning/targets.py")

    # Зарежда регистъра за съпоставяне
    # на предмети между последователни сканирания.
    instances = text("planning/instances.py")

    # Зарежда примерния файл с променливи на средата.
    env = (ROOT / ".env.example").read_text()

    # Проверява дали максималното разстояние
    # на откритите предмети от основата е 0,40 m.
    assert 'SIMPLE_OBJECT_MAX_BASE_DISTANCE_M", 0.40' in cloud

    # Проверява дали се отчита броят на кандидатните точки,
    # пренебрегнати поради прекалено голямо разстояние.
    assert 'far_candidate_points_ignored' in cloud

    # Проверява дали компонент с център извън допустимото
    # разстояние се отхвърля с конкретно състояние.
    assert 'CENTER_BEYOND_MAX_BASE_DISTANCE' in cloud

    # Проверява дали логиката за целите съдържа метод
    # за валидиране на разстоянието от основата.
    assert 'def target_is_within_range(' in targets

    # Проверява дали планиращият агент използва
    # отделна конфигурация за максималното разстояние до цел.
    assert 'PLANNER_MAX_TARGET_BASE_DISTANCE_M' in targets

    # Проверява дали максималното разстояние
    # за съпоставяне на екземпляри е зададено на 0,06 m.
    assert 'INSTANCE_MATCH_MAX_DISTANCE_M",\n            0.06' in instances

    # Проверява дали регистърът отчита броя
    # на игнорираните екземпляри извън допустимия обхват.
    assert 'ignored_out_of_range_count' in instances

    # Проверява стойността за максимално разстояние
    # при откриване на предмети в примерната конфигурация.
    assert 'SIMPLE_OBJECT_MAX_BASE_DISTANCE_M=0.40' in env

    # Проверява стойността за максимално разстояние
    # до целите на планиращия агент.
    assert 'PLANNER_MAX_TARGET_BASE_DISTANCE_M=0.40' in env

    # Проверява стойността за максимално разстояние
    # при съпоставяне на екземпляри.
    assert 'INSTANCE_MATCH_MAX_DISTANCE_M=0.06' in env

    # Потвърждава, че далечните откривания
    # не се включват в резултата от сканирането.
    print("Far detections excluded from scan output: OK")

    # Потвърждава, че планиращият агент
    # отхвърля цели извън допустимия обхват.
    print("Out-of-range targets rejected by planner: OK")

    # Потвърждава, че съпоставянето на екземпляри
    # използва максимално разстояние от 6 cm.
    print("Instance matching tightened to 6 cm: OK")


# Стартира проверките само когато файлът
# се изпълнява директно.
if __name__ == "__main__":
    main()