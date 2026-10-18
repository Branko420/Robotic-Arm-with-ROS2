#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за математически операции
# и типови анотации.
import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

# Помощни функции за прочитане на булеви
# и числови настройки от променливите на средата.
from .common import env_bool, env_float


class GeometricIKSolver:
    """
    Геометричен решател на обратната кинематика
    за роботизирания манипулатор.

    Решателят търси допустима конфигурация на ставите,
    чрез която крайният механизъм да достигне зададена
    декартова позиция по X, Y и Z.

    При търсенето се отчитат:
    - геометричните размери на звената;
    - ограниченията на ставите;
    - текущата ставна конфигурация;
    - физическата калибрация;
    - грешката между желаната и достигнатата позиция.
    """

    def __init__(self) -> None:
        # Допустими диапазони на ставите в градуси.
        #
        # Стойностите могат да бъдат променени
        # чрез съответните променливи на средата.
        self.limits_deg = {
            # Ограничения на ставата за въртене на основата.
            "joint_6_base": (
                env_float(
                    "PLANNER_BASE_MIN_DEG",
                    -84.5,
                ),
                env_float(
                    "PLANNER_BASE_MAX_DEG",
                    86.5,
                ),
            ),

            # Ограничения на ставата на рамото.
            "joint_5_shoulder": (
                env_float(
                    "PLANNER_SHOULDER_MIN_DEG",
                    -90.0,
                ),
                env_float(
                    "PLANNER_SHOULDER_MAX_DEG",
                    90.0,
                ),
            ),

            # Ограничения на ставата на лакътя.
            "joint_4_elbow": (
                env_float(
                    "PLANNER_ELBOW_MIN_DEG",
                    -83.9,
                ),
                env_float(
                    "PLANNER_ELBOW_MAX_DEG",
                    71.4,
                ),
            ),

            # Ограничения на ставата за наклон на китката.
            "joint_3_wrist_pitch": (
                env_float(
                    "PLANNER_WRIST_MIN_DEG",
                    -85.0,
                ),
                env_float(
                    "PLANNER_WRIST_MAX_DEG",
                    80.0,
                ),
            ),
        }

        # Височина на оста на рамото
        # спрямо координатната система на основата.
        self.shoulder_height = env_float(
            "PLANNER_SHOULDER_HEIGHT",
            0.1162,
        )

        # Дължина на горната част на рамото.
        self.l1 = env_float(
            "PLANNER_UPPER_ARM_LENGTH",
            0.1041,
        )

        # Дължина на предмишницата.
        self.l2 = env_float(
            "PLANNER_FOREARM_LENGTH",
            0.1463,
        )

        # Дължина от китката до крайния механизъм.
        self.l3 = env_float(
            "PLANNER_WRIST_LENGTH",
            0.1779,
        )

        # Знаков коефициент за въртенето на основата.
        #
        # Позволява посоката на математическия модел
        # да бъде съгласувана с посоката на ROS 2 ставата.
        self.base_sign = env_float(
            "PLANNER_BASE_SIGN",
            1.0,
        )

        # Знаков коефициент за рамото.
        self.shoulder_sign = env_float(
            "PLANNER_SHOULDER_SIGN",
            -1.0,
        )

        # Знаков коефициент за лакътя.
        self.elbow_sign = env_float(
            "PLANNER_ELBOW_SIGN",
            -1.0,
        )

        # Знаков коефициент за китката.
        self.wrist_sign = env_float(
            "PLANNER_WRIST_SIGN",
            1.0,
        )

        # Минимална ориентация на последното звено,
        # която ще бъде проверена от решателя.
        self.phi_min_deg = env_float(
            "PLANNER_PHI_MIN_DEG",
            -120.0,
        )

        # Максимална ориентация на последното звено.
        self.phi_max_deg = env_float(
            "PLANNER_PHI_MAX_DEG",
            160.0,
        )

        # Стъпка на обхождане на възможните ориентации.
        self.phi_step_deg = env_float(
            "PLANNER_PHI_STEP_DEG",
            2.0,
        )

        # Максимална допустима грешка на позицията,
        # зададена чрез общата конфигурация.
        self.max_position_error = max(
            0.0,
            env_float(
                "PLANNER_MAX_POSITION_ERROR",
                0.01,
            ),
        )

        # Твърда горна граница за допустимата
        # декартова грешка в метри.
        self.hard_max_target_error = max(
            0.0005,
            env_float(
                "PLANNER_HARD_MAX_TARGET_ERROR_M",
                0.008,
            ),
        )

        # Разрешава използване на най-близкото решение,
        # когато точната цел не може да бъде достигната.
        self.allow_best_effort = env_bool(
            "PLANNER_ALLOW_BEST_EFFORT",
            False,
        )

    def within_limit(
        self,
        joint: str,
        degrees: float,
    ) -> bool:
        """
        Проверява дали даден ставен ъгъл
        е в допустимия диапазон.
        """

        # Получава минималната и максималната
        # стойност за съответната става.
        low, high = self.limits_deg[joint]

        # Връща True само когато ъгълът
        # е между двете граници.
        return low <= degrees <= high

    def fk_tool(
        self,
        joint_rad: Mapping[str, float],
    ) -> Tuple[float, float, float]:
        """
        Изчислява правата кинематика на крайния механизъм.

        Приема ставни стойности в радиани и връща
        достигнатата декартова позиция:
        (x, y, z).
        """

        # Ъгъл на въртене на основата.
        q_base = joint_rad["joint_6_base"]

        # Абсолютен ъгъл на първото подвижно звено.
        a1 = (
            joint_rad["joint_5_shoulder"]
            / self.shoulder_sign
        )

        # Абсолютен ъгъл на второто звено.
        #
        # Към ъгъла на рамото се добавя
        # относителният ъгъл на лакътя.
        a2 = (
            a1
            + joint_rad["joint_4_elbow"]
            / self.elbow_sign
        )

        # Абсолютен ъгъл на последното звено.
        #
        # Към предишната ориентация се добавя
        # относителният наклон на китката.
        a3 = (
            a2
            + joint_rad["joint_3_wrist_pitch"]
            / self.wrist_sign
        )

        # Изчислява хоризонталното радиално разстояние
        # от оста на основата до крайния механизъм.
        radial = (
            self.l1 * math.sin(a1)
            + self.l2 * math.sin(a2)
            + self.l3 * math.sin(a3)
        )

        # Изчислява височината на крайния механизъм.
        z = (
            self.shoulder_height
            + self.l1 * math.cos(a1)
            + self.l2 * math.cos(a2)
            + self.l3 * math.cos(a3)
        )

        # Преобразува радиалната позиция
        # до декартови координати X и Y.
        return (
            radial * math.cos(q_base),
            radial * math.sin(q_base),
            z,
        )

    def solve(
        self,
        x: float,
        y: float,
        z: float,
        current_joint_deg: Mapping[str, float],
        candidate_projector: Optional[
            Callable[
                [Mapping[str, float]],
                Mapping[str, float],
            ]
        ] = None,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:
        """
        Търси ставно решение за зададената декартова цел.

        Аргументи:
        - x, y, z — желаната позиция на крайния механизъм;
        - current_joint_deg — текущите ставни ъгли в градуси;
        - candidate_projector — незадължителна функция,
          която проектира кандидата през физическата калибрация.

        Връща:
        - най-доброто допустимо решение и диагностичен кандидат;
        - None и най-близкия кандидат при неуспех.
        """

        # Изчислява необходимото въртене на основата
        # според координатите X и Y.
        base_rad = (
            self.base_sign
            * math.atan2(y, x)
        )

        # Изчислява хоризонталното разстояние
        # от основата до целта.
        radial = math.hypot(x, y)

        # Най-доброто решение, което спазва
        # всички ограничения на ставите.
        best_valid: Optional[
            Dict[str, Any]
        ] = None

        # Оценка на най-доброто валидно решение.
        best_score = float("inf")

        # Най-близкият кандидат независимо
        # от ограниченията на ставите.
        best_any: Optional[
            Dict[str, Any]
        ] = None

        # Най-малката намерена позиционна грешка.
        best_any_error = float("inf")

        # Започва обхождането от минималната
        # допустима ориентация на последното звено.
        phi = self.phi_min_deg

        # Проверява последователно всички ориентации
        # до зададената максимална стойност.
        while phi <= self.phi_max_deg + 1e-9:
            # Преобразува ориентацията в радиани.
            phi_rad = math.radians(phi)

            # Изчислява радиалната позиция на китката,
            # като изважда приноса на последното звено.
            wr = (
                radial
                - self.l3 * math.sin(phi_rad)
            )

            # Изчислява вертикалната позиция на китката
            # спрямо оста на рамото.
            wz = (
                z
                - self.shoulder_height
                - self.l3 * math.cos(phi_rad)
            )

            # Знаменател от закона за косинусите.
            denominator = (
                2
                * self.l1
                * self.l2
            )

            # Изчислява косинуса на относителния
            # ъгъл между първите две звена.
            d_value = (
                (
                    wr * wr
                    + wz * wz
                    - self.l1 * self.l1
                    - self.l2 * self.l2
                )
                / denominator
            )

            # Реално геометрично решение съществува
            # само когато стойността е в [-1, 1].
            if -1.0 <= d_value <= 1.0:
                # Проверява двата възможни клона:
                # лакът нагоре и лакът надолу.
                for branch in (
                    1.0,
                    -1.0,
                ):
                    # Изчислява относителния ъгъл
                    # между горната част на рамото и предмишницата.
                    q2 = (
                        branch
                        * math.acos(d_value)
                    )

                    # Изчислява ъгъла на първото звено
                    # в равнинния кинематичен модел.
                    theta1 = (
                        math.atan2(wz, wr)
                        - math.atan2(
                            self.l2 * math.sin(q2),
                            (
                                self.l1
                                + self.l2
                                * math.cos(q2)
                            ),
                        )
                    )

                    # Абсолютна ориентация
                    # на второто звено.
                    theta2 = theta1 + q2

                    # Преобразува равнинните ъгли
                    # към използваната от робота система.
                    a1 = (
                        math.pi / 2.0
                        - theta1
                    )

                    a2 = (
                        math.pi / 2.0
                        - theta2
                    )

                    # Ориентацията на последното звено
                    # е текущата стойност phi.
                    a3 = phi_rad

                    # Създава суров кандидат
                    # със ставни стойности в радиани.
                    raw_candidate = {
                        "joint_6_base": base_rad,

                        "joint_5_shoulder": (
                            self.shoulder_sign
                            * a1
                        ),

                        "joint_4_elbow": (
                            self.elbow_sign
                            * (a2 - a1)
                        ),

                        "joint_3_wrist_pitch": (
                            self.wrist_sign
                            * (a3 - a2)
                        ),

                        # Завъртането на китката не участва
                        # в позиционното IK решение.
                        "joint_2_wrist_roll": 0.0,
                    }

                    try:
                        # По желание проектира кандидата
                        # през физическата калибрация.
                        candidate = (
                            dict(
                                candidate_projector(
                                    raw_candidate
                                )
                            )
                            if candidate_projector
                            is not None
                            else dict(raw_candidate)
                        )

                    except Exception:
                        # Игнорира кандидат, който не може
                        # да бъде преобразуван успешно.
                        continue

                    # Проверява дали проектираният кандидат
                    # съдържа всички необходими стави.
                    if not all(
                        name in candidate
                        for name in (
                            "joint_6_base",
                            "joint_5_shoulder",
                            "joint_4_elbow",
                            "joint_3_wrist_pitch",
                        )
                    ):
                        continue

                    # Преобразува всички ставни стойности
                    # от радиани в градуси.
                    degrees = {
                        name: math.degrees(
                            float(value)
                        )
                        for name, value
                        in candidate.items()
                    }

                    # Изчислява действително достигнатата позиция
                    # чрез правата кинематика.
                    reached = self.fk_tool(
                        candidate
                    )

                    # Разлика между достигнатата
                    # и желаната позиция по всяка ос.
                    dx = reached[0] - x
                    dy = reached[1] - y
                    dz = reached[2] - z

                    # Евклидова позиционна грешка.
                    error = math.sqrt(
                        dx * dx
                        + dy * dy
                        + dz * dz
                    )

                    # Събира цялата информация
                    # за текущия кандидат.
                    row = {
                        "candidate": candidate,
                        "raw_candidate": raw_candidate,

                        # Показва дали е използвана
                        # физическата калибрация.
                        "candidate_projected": (
                            candidate_projector
                            is not None
                        ),

                        # Ставни стойности в градуси.
                        "degs": degrees,

                        # Реално достигната позиция.
                        "reached": reached,

                        # Позиционна грешка в метри.
                        "error": error,

                        # Проверяваната ориентация.
                        "phi": phi,
                    }

                    # Запазва най-близкия кандидат,
                    # независимо от ставните ограничения.
                    if error < best_any_error:
                        best_any_error = error
                        best_any = row

                    # Отхвърля кандидата, когато някоя
                    # от ставите е извън безопасния диапазон.
                    if not all(
                        self.within_limit(
                            name,
                            degrees[name],
                        )
                        for name
                        in self.limits_deg
                    ):
                        continue

                    # Основната част на оценката
                    # е позиционната грешка.
                    score = error * 1000.0

                    # Добавя наказание за голяма промяна
                    # спрямо текущото положение на ставите.
                    #
                    # Така се предпочитат по-плавни
                    # и по-къси движения.
                    for name in self.limits_deg:
                        if name in current_joint_deg:
                            score += (
                                0.02
                                * abs(
                                    degrees[name]
                                    - current_joint_deg[
                                        name
                                    ]
                                )
                            )

                    # Добавя малко наказание
                    # за голям наклон на китката.
                    score += (
                        0.01
                        * abs(
                            degrees[
                                "joint_3_wrist_pitch"
                            ]
                        )
                    )

                    # Запазва кандидата,
                    # когато оценката му е по-добра.
                    if score < best_score:
                        best_score = score
                        best_valid = row

            # Преминава към следващата
            # възможна ориентация.
            phi += self.phi_step_deg

        # Не е открит кандидат, който едновременно
        # да удовлетворява геометрията и ставните ограничения.
        if best_valid is None:
            return None, best_any

        # Избира по-строгата от двете
        # конфигурирани граници за грешка.
        accepted_error = min(
            self.max_position_error,
            self.hard_max_target_error,
        )

        # При достатъчно малка грешка
        # приема намереното решение.
        if best_valid["error"] <= accepted_error:
            return best_valid, best_any

        # При разрешен режим best effort приема
        # решение до твърдата максимална грешка.
        if (
            self.allow_best_effort
            and best_valid["error"]
            <= self.hard_max_target_error
        ):
            return best_valid, best_any

        # Връща неуспех, когато най-доброто валидно
        # решение все още има прекалено голяма грешка.
        return None, best_valid