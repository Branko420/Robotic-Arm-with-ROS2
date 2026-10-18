#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Типови анотации за речници, асоциативни структури,
# незадължителни стойности и кортежи.
from typing import Any, Dict, Mapping, Optional, Tuple


class PlanningCartesianSolverMixin:
    """
    Смесен клас за решаване на декартови цели.

    Класът свързва геометричния решател на обратната кинематика
    с физическата калибрация на сервомоторите, така че избраното
    решение да бъде съобразено с реалното движение на манипулатора.
    """

    def project_joints_through_physical_calibration(
        self,
        joints: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Проектира ставните стойности през физическата калибрация.

        Последователността е:
        1. преобразуване от ставни стойности в радиани
           към физически сервомоторни стойности;
        2. обратно преобразуване от сервомоторните стойности
           към синхронизираните ставни стойности за RViz.

        Така се отчита реалното участъково калибриране
        на отделните сервомотори.
        """

        # Преобразува ставните стойности от цифровия модел
        # към физически сервомоторни ъгли в градуси.
        servo_degrees = (
            self.motion.real.joints_to_servo(
                joints
            )
        )

        # Преобразува получените физически сервомоторни стойности
        # обратно към ставни стойности в радиани за RViz.
        projected = (
            self.motion.real.servo_to_joints(
                servo_degrees
            )
        )

        # Запазва всички стави от оригиналното решение,
        # които не са били включени в калибрационното преобразуване.
        for name, value in joints.items():
            projected.setdefault(
                name,
                float(value),
            )

        # Връща проектираното през физическата
        # калибрация ставно решение.
        return projected

    def solve_cartesian_target(
        self,
        xyz: Tuple[float, float, float],
        *,
        current_joint_deg: Optional[
            Mapping[str, float]
        ] = None,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:
        """
        Решава обратната кинематика за зададена декартова цел.

        Аргументи:
        - xyz — целева позиция по X, Y и Z;
        - current_joint_deg — текущи ставни стойности в градуси.
          При липса се използват актуалните стойности
          на планиращия агент.

        Връща:
        - успешно ставно решение и None;
        - None и структуриран резултат за грешка.
        """

        # Извиква геометричния решател на обратната кинематика.
        return self.ik.solve(
            # Разгъва координатите X, Y и Z
            # като отделни аргументи.
            *xyz,

            # Използва подадените текущи ставни стойности
            # или ги получава от актуалното състояние.
            current_joint_deg=(
                self.current_joint_deg()
                if current_joint_deg is None
                else current_joint_deg
            ),

            # Всеки кандидат за IK решение се проектира
            # през физическата калибрация преди оценяване.
            candidate_projector=(
                self.project_joints_through_physical_calibration
            ),
        )