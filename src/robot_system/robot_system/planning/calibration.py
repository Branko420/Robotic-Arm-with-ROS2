#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване,
# математически операции, файлови пътища и типови анотации.
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

# Помощна функция за ограничаване на стойност
# в зададен минимален и максимален диапазон.
from .common import clamp

# Константи за реда на ставите, съответствието между
# стави и сервомотори и безопасните сервомоторни диапазони.
from .constants import (
    JOINT_ORDER,
    JOINT_TO_SERVO,
    SAFE_SERVO_RANGE,
)


class PiecewiseCalibrator:
    """
    Извършва участъково линейно преобразуване между:

    - ставен ъгъл в цифровия модел и RViz;
    - физическа сервомоторна стойност.

    Калибрацията използва измерени двойки от вида:

        [ъгъл в RViz, физически ъгъл на сервомотора]
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
    ) -> None:
        # Режим на работа на калибратора.
        #
        # piecewise използва участъкова линейна интерполация,
        # а manual връща директно стойността default_offset.
        self.mode = str(
            cfg.get(
                "mode",
                "piecewise",
            )
        )

        # Стандартно отместване, използвано при липса
        # на достатъчно калибрационни точки.
        self.default_offset = float(
            cfg.get(
                "default_offset",
                90.0,
            )
        )

        # Стандартен коефициент на мащабиране.
        self.default_scale = float(
            cfg.get(
                "default_scale",
                1.0,
            )
        )

        # Преобразува всички калибрационни точки
        # до двойки от числа с плаваща запетая.
        self.points = [
            (
                float(rviz_deg),
                float(servo_deg),
            )
            for rviz_deg, servo_deg
            in cfg.get("points", [])
        ]

        # Подрежда точките по нарастващ RViz ъгъл.
        self.points.sort(
            key=lambda row: row[0]
        )

    @staticmethod
    def _interpolate(
        value: float,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
    ) -> float:
        """
        Извършва линейна интерполация между две точки.
        """

        # Координати на първата точка.
        x0, y0 = p0

        # Координати на втората точка.
        x1, y1 = p1

        # Предотвратява деление на нула,
        # когато двете X координати са еднакви.
        if abs(x1 - x0) < 1e-9:
            return y0

        # Изчислява относителното положение
        # на стойността между двете точки.
        ratio = (
            value - x0
        ) / (
            x1 - x0
        )

        # Връща интерполираната Y стойност.
        return y0 + ratio * (
            y1 - y0
        )

    def convert(
        self,
        rviz_deg: float,
    ) -> float:
        """
        Преобразува ъгъл от RViz в градуси
        до физическа сервомоторна стойност.
        """

        # Гарантира, че входната стойност е float.
        rviz_deg = float(rviz_deg)

        # При ръчен режим винаги връща
        # конфигурираното стандартно отместване.
        if self.mode == "manual":
            return self.default_offset

        # При по-малко от две точки не може
        # да се извърши участъкова интерполация.
        if len(self.points) < 2:
            return (
                self.default_offset
                + self.default_scale
                * rviz_deg
            )

        # При стойност под първата калибрационна точка
        # извършва линейна екстраполация чрез първите две точки.
        if rviz_deg <= self.points[0][0]:
            return self._interpolate(
                rviz_deg,
                self.points[0],
                self.points[1],
            )

        # При стойност над последната калибрационна точка
        # извършва линейна екстраполация чрез последните две точки.
        if rviz_deg >= self.points[-1][0]:
            return self._interpolate(
                rviz_deg,
                self.points[-2],
                self.points[-1],
            )

        # Търси участъка, в който попада подаденият ъгъл.
        for p0, p1 in zip(
            self.points,
            self.points[1:],
        ):
            if p0[0] <= rviz_deg <= p1[0]:
                return self._interpolate(
                    rviz_deg,
                    p0,
                    p1,
                )

        # Резервно линейно преобразуване,
        # ако не е намерен подходящ участък.
        return (
            self.default_offset
            + self.default_scale
            * rviz_deg
        )

    def inverse(
        self,
        servo_deg: float,
    ) -> float:
        """
        Извършва обратното преобразуване:

        физическа сервомоторна стойност
        към съответния RViz ъгъл в градуси.
        """

        # Гарантира, че входната стойност е float.
        servo_deg = float(servo_deg)

        # При недостатъчно точки използва
        # стандартното линейно преобразуване.
        if len(self.points) < 2:
            # Предотвратява деление на нула
            # при нулев коефициент на мащабиране.
            scale = (
                self.default_scale
                if abs(self.default_scale) > 1e-9
                else 1.0
            )

            return (
                servo_deg
                - self.default_offset
            ) / scale

        # Разменя местата на физическата и RViz стойността,
        # за да се извърши обратна интерполация.
        inverse_points = sorted(
            [
                (
                    servo,
                    rviz,
                )
                for rviz, servo
                in self.points
            ],
            key=lambda row: row[0],
        )

        # Екстраполация под първата физическа стойност.
        if servo_deg <= inverse_points[0][0]:
            return self._interpolate(
                servo_deg,
                inverse_points[0],
                inverse_points[1],
            )

        # Екстраполация над последната физическа стойност.
        if servo_deg >= inverse_points[-1][0]:
            return self._interpolate(
                servo_deg,
                inverse_points[-2],
                inverse_points[-1],
            )

        # Търси участъка, в който попада
        # физическата сервомоторна стойност.
        for p0, p1 in zip(
            inverse_points,
            inverse_points[1:],
        ):
            if p0[0] <= servo_deg <= p1[0]:
                return self._interpolate(
                    servo_deg,
                    p0,
                    p1,
                )

        # Резервно обратно линейно преобразуване.
        scale = (
            self.default_scale
            if abs(self.default_scale) > 1e-9
            else 1.0
        )

        return (
            servo_deg
            - self.default_offset
        ) / scale


def load_calibrators(
    path: Path,
) -> Dict[str, PiecewiseCalibrator]:
    """
    Зарежда калибраторите за всички сервомотори
    от JSON конфигурационен файл.
    """

    # При липсващ файл връща празен речник.
    if not path.exists():
        return {}

    try:
        # Прочита и декодира JSON документа.
        document = json.loads(
            path.read_text()
        )

    except Exception:
        # При невалиден или недостъпен файл
        # не създава калибратори.
        return {}

    # Речник с калибраторите по име на сервомотор.
    result: Dict[
        str,
        PiecewiseCalibrator,
    ] = {}

    # Зарежда конфигурацията за всеки използван сервомотор.
    for servo in (
        "base",
        "shoulder",
        "elbow",
        "wrist",
        "wrist_rotation",
        "gripper",
    ):
        # Получава конфигурационния раздел
        # за съответния сервомотор.
        row = document.get(
            servo,
            {},
        )

        # Създава калибратор само при валиден речник.
        if isinstance(row, dict):
            result[servo] = PiecewiseCalibrator(
                row
            )

    return result


class PoseLibrary:
    """
    Зарежда именувани синхронизирани позиции
    от permanent_points.json.

    Позициите могат да бъдат зададени като:
    - физически сервомоторни стойности;
    - RViz ставни ъгли в градуси.
    """

    # Вградени резервни позиции,
    # използвани при липсващ или невалиден файл.
    DEFAULT_POSES = {
        # Нормална начална позиция на манипулатора.
        "home": {
            "type": "servo_pose",
            "purpose": "normal_home",
            "angles": {
                "base": 92.0,
                "shoulder": 141.6,
                "elbow": 0.0,
                "wrist": 180.0,
                "wrist_rotation": 90.0,
                "gripper": 70.0,
            },
        },

        # Позиция, при която манипулаторът
        # освобождава изгледа на RGB-D камерата.
        "scan_pose": {
            "type": "servo_pose",
            "purpose": "camera_workspace_scan",
            "angles": {
                "base": 0.0,
                "shoulder": 31.5,
                "elbow": 90.4,
                "wrist": 180.0,
                "wrist_rotation": 90.0,
                "gripper": 70.0,
            },
        },
    }

    def __init__(
        self,
        path: Path,
        calibrators: Mapping[
            str,
            PiecewiseCalibrator,
        ],
    ) -> None:
        # Път до файла с постоянните позиции.
        self.path = path

        # Локално копие на заредените калибратори.
        self.calibrators = dict(
            calibrators
        )

        # Зарежда документа с именуваните позиции.
        self.document = self._load_document()

    def _load_document(
        self,
    ) -> Dict[str, Any]:
        """
        Зарежда документа с именуваните позиции.

        При липсващ или невалиден файл използва
        вградените позиции DEFAULT_POSES.
        """

        # Опитва да прочете зададения JSON файл.
        if self.path.exists():
            try:
                loaded = json.loads(
                    self.path.read_text()
                )

                # Приема само JSON обект.
                if isinstance(
                    loaded,
                    dict,
                ):
                    return loaded

            except Exception:
                # При грешка преминава
                # към резервните позиции.
                pass

        # Създава дълбоко копие на DEFAULT_POSES
        # чрез JSON сериализация и десериализация.
        return json.loads(
            json.dumps(
                self.DEFAULT_POSES
            )
        )

    def _row(
        self,
        name: str = "home",
    ) -> Dict[str, Any]:
        """
        Връща конфигурационния запис
        за конкретна именувана позиция.
        """

        # Поддържа както формат с основен ключ poses,
        # така и директно записани позиции.
        poses = self.document.get(
            "poses",
            self.document,
        )

        # Търси позицията в заредения документ.
        row = poses.get(name)

        # При липса използва вградената позиция
        # със същото име.
        if not isinstance(row, dict):
            row = self.DEFAULT_POSES.get(name)

        # Ако позицията не съществува и във вградените
        # стойности, прекратява с KeyError.
        if not isinstance(row, dict):
            raise KeyError(
                f"Unknown permanent pose: {name}"
            )

        return row

    def joint_radians(
        self,
        name: str = "home",
    ) -> Dict[str, float]:
        """
        Връща ставните стойности на именувана позиция
        в радиани за използване в RViz.
        """

        # Зарежда конфигурацията на позицията.
        row = self._row(name)

        # Определя в какво пространство
        # са зададени входните стойности.
        space = str(
            row.get(
                "space",
                "servo_deg",
            )
        ).strip().lower()

        # Получава речника с ъглите.
        angles = row.get(
            "angles",
            {},
        )

        # Резултатни ставни стойности в радиани.
        output: Dict[
            str,
            float,
        ] = {}

        # Когато стойностите са зададени като RViz градуси,
        # директно ги преобразува в радиани.
        if space == "rviz_deg":
            for joint in JOINT_ORDER:
                output[joint] = math.radians(
                    float(
                        angles.get(
                            joint,
                            0.0,
                        )
                    )
                )

            return output

        # При сервомоторни стойности извършва
        # обратно калибрационно преобразуване.
        for joint in JOINT_ORDER:
            # Получава физическия сервомотор,
            # съответстващ на ставата.
            servo_name = JOINT_TO_SERVO[
                joint
            ]

            # Прочита физическата стойност.
            servo_deg = float(
                angles.get(
                    servo_name,
                    90.0,
                )
            )

            # Получава калибратора за сервомотора.
            calibrator = self.calibrators.get(
                servo_name
            )

            # Преобразува физическия ъгъл
            # до RViz ъгъл в градуси.
            rviz_deg = (
                calibrator.inverse(servo_deg)
                if calibrator is not None
                else servo_deg - 90.0
            )

            # RViz и ROS 2 използват радиани.
            output[joint] = math.radians(
                rviz_deg
            )

        return output

    def gripper_radians(
        self,
        name: str = "home",
    ) -> float:
        """
        Връща положението на хващача
        в RViz радиани.
        """

        # Зарежда именуваната позиция.
        row = self._row(name)

        # Определя пространството на стойностите.
        space = str(
            row.get(
                "space",
                "servo_deg",
            )
        ).strip().lower()

        # Получава речника с ъглите.
        angles = row.get(
            "angles",
            {},
        )

        # При RViz градуси използва директно
        # стойността на ставата на хващача.
        if space == "rviz_deg":
            return math.radians(
                float(
                    angles.get(
                        "joint_1_gripper",
                        angles.get(
                            "gripper",
                            -10.0,
                        ),
                    )
                )
            )

        # При физически стойности прочита ъгъла
        # на сервомотора на хващача.
        servo_deg = float(
            angles.get(
                "gripper",
                80.0,
            )
        )

        # Получава калибратора на хващача.
        calibrator = self.calibrators.get(
            "gripper"
        )

        # Извършва обратно преобразуване
        # до RViz градуси.
        rviz_deg = (
            calibrator.inverse(servo_deg)
            if calibrator is not None
            else servo_deg - 80.0
        )

        return math.radians(rviz_deg)

    def servo_angles(
        self,
        name: str = "home",
    ) -> Dict[str, float]:
        """
        Връща физическите сервомоторни стойности
        за зададена именувана позиция.
        """

        # Зарежда конфигурацията на позицията.
        row = self._row(name)

        # Определя пространството
        # на съхранените стойности.
        space = str(
            row.get(
                "space",
                "servo_deg",
            )
        ).strip().lower()

        # Получава речника с ъглите.
        angles = row.get(
            "angles",
            {},
        )

        # Когато стойностите вече са физически градуси,
        # само ги ограничава до безопасните диапазони.
        if space == "servo_deg":
            return {
                key: clamp(
                    float(value),
                    *SAFE_SERVO_RANGE[key],
                )
                for key, value
                in angles.items()
                if key in SAFE_SERVO_RANGE
            }

        # Преобразува RViz градусите
        # до ставни стойности в радиани.
        joints = {
            joint: math.radians(
                float(
                    angles.get(
                        joint,
                        0.0,
                    )
                )
            )
            for joint in JOINT_ORDER
        }

        # Резултатни физически сервомоторни стойности.
        result: Dict[
            str,
            float,
        ] = {}

        # Преобразува всяка RViz става
        # до съответния физически сервомотор.
        for (
            joint,
            servo_name,
        ) in JOINT_TO_SERVO.items():
            # Преобразува радианите обратно в градуси.
            rviz_deg = math.degrees(
                joints[joint]
            )

            # Получава калибратора.
            calibrator = self.calibrators.get(
                servo_name
            )

            # Изчислява физическата стойност.
            servo_deg = (
                calibrator.convert(rviz_deg)
                if calibrator is not None
                else 90.0 + rviz_deg
            )

            # Ограничава стойността
            # до безопасния диапазон.
            result[servo_name] = clamp(
                servo_deg,
                *SAFE_SERVO_RANGE[
                    servo_name
                ],
            )

        # Получава RViz положението на хващача в градуси.
        gripper_rviz_deg = math.degrees(
            self.gripper_radians(name)
        )

        # Получава калибратора на хващача.
        gripper_calibrator = (
            self.calibrators.get(
                "gripper"
            )
        )

        # Преобразува RViz стойността
        # до физическа сервомоторна стойност.
        gripper_servo_deg = (
            gripper_calibrator.convert(
                gripper_rviz_deg
            )
            if gripper_calibrator is not None
            else 80.0 + gripper_rviz_deg
        )

        # Добавя ограничената стойност на хващача.
        result["gripper"] = clamp(
            gripper_servo_deg,
            *SAFE_SERVO_RANGE["gripper"],
        )

        return result

    def describe(
        self,
        name: str = "home",
    ) -> Dict[str, Any]:
        """
        Връща подробно описание на именувана позиция
        във всички използвани представяния.
        """

        # Получава ставните стойности в радиани.
        joints = self.joint_radians(name)

        # Получава оригиналния конфигурационен запис.
        row = self._row(name)

        return {
            # Име на позицията.
            "name": name,

            # Предназначение на позицията.
            "purpose": row.get(
                "purpose"
            ),

            # Ставни стойности в RViz градуси.
            "joint_degrees": {
                joint: round(
                    math.degrees(value),
                    3,
                )
                for joint, value
                in joints.items()
            },

            # Положение на хващача в RViz градуси.
            "gripper_rviz_degrees": round(
                math.degrees(
                    self.gripper_radians(name)
                ),
                3,
            ),

            # Физически сервомоторни стойности.
            "servo_degrees": {
                key: round(
                    value,
                    3,
                )
                for key, value
                in self.servo_angles(
                    name
                ).items()
            },
        }