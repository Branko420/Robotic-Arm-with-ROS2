from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# Име на ROS 2 пакета, съдържащ URDF модела и RViz конфигурацията.
PACKAGE_NAME = "arm_description_pkg"


def _is_true(value: LaunchConfiguration) -> PythonExpression:
    """
    Преобразува стойността на launch аргумент в булево условие.

    Аргументите на launch файла се получават като текстови стойности.
    Функцията проверява дали подадената стойност е равна на „true“,
    без значение дали е изписана с главни или малки букви.
    """
    return PythonExpression(["'", value, "'.lower() == 'true'"])


def generate_launch_description() -> LaunchDescription:
    """
    Създава описанието за стартиране на цифровия модел на манипулатора.

    Стартират се:
    - robot_state_publisher за публикуване на трансформациите на робота;
    - joint_state_publisher_gui при ръчно управление на ставите;
    - RViz2 за визуализиране на цифровия модел.
    """

    # Определяне на директорията, в която е инсталиран ROS 2 пакетът.
    package_share = Path(get_package_share_directory(PACKAGE_NAME))

    # Път до URDF файла, описващ механичната структура на манипулатора.
    urdf_path = package_share / "urdf" / "arm.urdf"

    # Път до предварително създадената RViz конфигурация.
    rviz_path = package_share / "rviz" / "display.rviz"

    # Зареждане на цялото съдържание на URDF файла като текст.
    robot_description = urdf_path.read_text(encoding="utf-8")

    # Launch аргумент за включване на графичния интерфейс
    # за ръчно задаване на ставните положения.
    use_gui = LaunchConfiguration("use_gui")

    # Launch аргумент за включване или изключване на RViz2.
    use_rviz = LaunchConfiguration("use_rviz")

    # Връщане на пълното описание на възлите и аргументите,
    # които трябва да бъдат стартирани.
    return LaunchDescription(
        [
            # Аргумент за стартиране на joint_state_publisher_gui.
            DeclareLaunchArgument(
                "use_gui",
                default_value="false",
                description=(
                    "Използва joint_state_publisher_gui вместо planning_agent. "
                    "Не трябва да се включва, когато planning_agent публикува "
                    "ставните положения в /joint_states."
                ),
            ),

            # Аргумент за стартиране на RViz2.
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description=(
                    "Стартира RViz2 с конфигурацията за визуализиране "
                    "на роботизирания манипулатор."
                ),
            ),

            # Възелът robot_state_publisher зарежда URDF модела
            # и публикува трансформациите между отделните звена.
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": robot_description,
                    }
                ],
            ),

            # Графичен интерфейс за ръчно задаване на ставните положения.
            #
            # Възелът се стартира само при use_gui:=true.
            # Не трябва да работи едновременно с planning_agent,
            # защото двата компонента ще публикуват в /joint_states.
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                output="screen",
                condition=IfCondition(_is_true(use_gui)),
                parameters=[
                    {
                        "robot_description": robot_description,
                    }
                ],
            ),

            # Стартиране на RViz2 с предварително зададената
            # конфигурация display.rviz.
            #
            # Възелът се стартира само при use_rviz:=true.
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=[
                    "-d",
                    str(rviz_path),
                ],
                condition=IfCondition(_is_true(use_rviz)),
            ),
        ]
    )