# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Обектно-ориентиран интерфейс за работа с файлови пътища.
from pathlib import Path


# Определя кореновата директория на проекта.
#
# __file__ съдържа пътя до текущия тестов файл,
# resolve() го преобразува до абсолютен път,
# а parents[2] избира директорията две нива нагоре.
ROOT = Path(__file__).resolve().parents[2]

# Път до основния Python пакет robot_system.
PACKAGE = ROOT / "robot_system"


def main() -> None:
    # Зарежда основния възел на планиращия агент.
    planning = (PACKAGE / "agents/planning_node.py").read_text()

    # Зарежда модула, който свързва планирането
    # с цифровия модел и физическия манипулатор.
    motion = (PACKAGE / "planning/motion_bridge.py").read_text()

    # Зарежда модула за публикуване и интерполация
    # на ставното състояние на роботизирания модел.
    model = (PACKAGE / "planning/robot_model_state.py").read_text()

    # Зарежда конфигурацията за инсталиране
    # и регистрираните входни точки на ROS 2 пакета.
    setup = (ROOT / "setup.py").read_text()

    # Зарежда декларацията с зависимостите
    # и метаданните на ROS 2 пакета.
    package_xml = (ROOT / "package.xml").read_text()

    # Проверява дали planning_agent създава
    # собствен издател за ставното състояние.
    assert 'self.joint_state_pub = self.create_publisher(' in planning

    # Проверява дали ставното състояние се публикува
    # директно в стандартната ROS 2 тема /joint_states.
    assert '"/joint_states"' in planning

    # Проверява дали planning_agent вече не използва
    # съобщения от тип JointTrajectory.
    assert 'JointTrajectory' not in planning

    # Проверява дали MotionBridge предава целевите
    # ставни стойности към модела чрез set_targets().
    assert 'self.model.set_targets(' in motion

    # Проверява дали модулът за състоянието
    # създава стандартно съобщение JointState.
    assert 'message = JointState()' in model

    # Проверява дали остарялата входна точка
    # rviz_trajectory_preview е премахната от setup.py.
    assert 'rviz_trajectory_preview' not in setup

    # Проверява дали зависимостта trajectory_msgs
    # е премахната от package.xml.
    assert 'trajectory_msgs' not in package_xml

    # Проверява дали старият междинен възел
    # за преобразуване на траектория към JointState
    # вече не съществува във файловата структура.
    assert not (PACKAGE / "visualization/trajectory_joint_state_bridge.py").exists()

    # Потвърждава, че планиращият агент публикува
    # директно актуалните ставни стойности.
    print("planning_agent publishes /joint_states directly: OK")

    # Потвърждава, че интерполацията на цифровия модел
    # остава вътре в слоя за планиране.
    print("RobotModel interpolation stays inside planning layer: OK")

    # Потвърждава премахването на остарелия preview възел
    # и темите, базирани на JointTrajectory.
    print("Preview node and trajectory topics removed: OK")


# Стартира проверката само когато файлът
# се изпълнява директно.
if __name__ == "__main__":
    main()