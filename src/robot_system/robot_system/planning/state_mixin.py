#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за:
# - JSON сериализация и десериализация;
# - математически операции;
# - времеви стойности;
# - типови анотации.
import json
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Ред на ставите в цифровия модел
# и безопасните физически диапазони на сервомоторите.
from .constants import JOINT_ORDER, SAFE_SERVO_RANGE

# Redis ключове и настройки за:
# - текущата декартова позиция;
# - държания предмет;
# - толерантността за именувани позиции;
# - запазените потребителски точки.
from .settings import (
    CURRENT_POSE_KEY,
    HELD_OBJECT_KEY,
    POSE_MATCH_TOLERANCE_DEG,
    SAVED_POINTS_KEY,
)

# Изключение, използвано при повече от една
# възможна цел за дадено описание.
from .targets import AmbiguousTargetError


class PlanningStateMixin:
    """
    Съдържа логиката за състоянието на планиращия агент.

    Модулът управлява:
    - текущата декартова позиция на крайния механизъм;
    - запазени именувани точки;
    - текущите ставни и сервомоторни стойности;
    - състоянието на държания предмет;
    - проверка за достигане на именувана позиция;
    - изчисляване на безопасна транзитна височина.
    """

    def persist_current_cartesian_pose(
        self,
        xyz: Tuple[float, float, float],
        *,
        source: str,
    ) -> Dict[str, Any]:
        """
        Записва текущата декартова позиция
        на крайния механизъм в Redis.
        """

        # Създава структуриран запис на позицията.
        payload = {
            # Координата по оста X в метри.
            "x": round(
                float(xyz[0]),
                6,
            ),

            # Координата по оста Y в метри.
            "y": round(
                float(xyz[1]),
                6,
            ),

            # Координата по оста Z в метри.
            "z": round(
                float(xyz[2]),
                6,
            ),

            # Координатна система на позицията.
            "frame": "base_link",

            # Източникът, който е обновил позицията.
            "source": source,

            # Време на последното обновяване.
            "updated_at": time.time(),
        }

        # Записва позицията като JSON документ в Redis.
        self.redis.set(
            CURRENT_POSE_KEY,
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
        )

        return payload

    def persist_current_pose_from_joint_state(
        self,
        msg: Any,
        *,
        source: str = "joint_states",
    ) -> Optional[Dict[str, Any]]:
        """
        Изчислява и записва декартовата позиция
        от ROS 2 съобщение със ставни състояния.
        """

        # Свързва имената на ставите
        # с техните позиции в радиани.
        positions = {
            name: float(value)
            for name, value in zip(
                msg.name,
                msg.position,
            )
        }

        # Не може да изчисли правата кинематика,
        # ако липсва някоя от основните стави.
        if not all(
            name in positions
            for name in JOINT_ORDER
        ):
            return None

        # Изчислява декартовата позиция
        # чрез правата кинематика.
        xyz = self.ik.fk_tool(
            {
                name: positions[name]
                for name in JOINT_ORDER
            }
        )

        # Записва получената позиция в Redis.
        return self.persist_current_cartesian_pose(
            xyz,
            source=source,
        )

    def current_cartesian_pose(
        self,
    ) -> Dict[str, Any]:
        """
        Връща най-надеждната налична текуща
        декартова позиция на крайния механизъм.

        Приоритет:
        1. последното ROS 2 ставно състояние;
        2. последната.

        Приоритет:
        1. последното ROS 2 ставно състоя позиция, записана в Redis;
        3. изчислената начална позиция home.
        """

        # Предпочита актуалното ROS 2 ставно състояние.
        if self.latest_joint_state is not None:
            current = (
                self.persist_current_pose_from_joint_state(
                    self.latest_joint_state,
                    source="latest_joint_state",
                )
            )

            if current is not None:
                return current

        # При липса на пълно ставно състояние
        # проверява последната позиция в Redis.
        raw = self.redis.get(
            CURRENT_POSE_KEY
        )

        if raw:
            try:
                value = json.loads(raw)

                # Приема само речник с трите координати.
                if (
                    isinstance(value, dict)
                    and all(
                        key in value
                        for key in (
                            "x",
                            "y",
                            "z",
                        )
                    )
                ):
                    return value

            except Exception:
                # При невалиден JSON преминава
                # към резервната начална позиция.
                pass

        # Изчислява декартовата позиция
        # на предварително дефинираната home позиция.
        home_xyz = self.ik.fk_tool(
            self.poses.joint_radians(
                "home"
            )
        )

        # Записва и връща резервната позиция.
        return self.persist_current_cartesian_pose(
            home_xyz,
            source="home_fallback",
        )

    @staticmethod
    def normalize_saved_point_name(
        name: Any,
    ) -> str:
        """
        Проверява и нормализира името
        на запазена декартова точка.
        """

        # Преобразува стойността до текст
        # и премахва излишните интервали.
        value = str(
            name or ""
        ).strip()

        # Не допуска записване на точка без име.
        if not value:
            raise ValueError(
                "Saved point name is required"
            )

        return value

    def load_saved_points(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Зарежда всички запазени декартови точки от Redis.
        """

        # Прочита JSON документа от Redis.
        raw = self.redis.get(
            SAVED_POINTS_KEY
        )

        # При липсваща стойност връща празен речник.
        if not raw:
            return {}

        try:
            # Декодира документа.
            value = json.loads(raw)

        except Exception:
            # При невалиден JSON приема,
            # че няма запазени точки.
            return {}

        # Приема само речникова структура.
        return (
            value
            if isinstance(value, dict)
            else {}
        )

    def save_current_point(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """
        Запазва текущата декартова позиция
        под зададено потребителско име.
        """

        # Проверява името на точката.
        point_name = (
            self.normalize_saved_point_name(
                name
            )
        )

        # Зарежда вече запазените точки.
        points = self.load_saved_points()

        # Получава текущата декартова позиция.
        pose = self.current_cartesian_pose()

        # Създава или заменя точката със същото име.
        points[point_name] = {
            "name": point_name,
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "frame": "base_link",
            "saved_at": time.time(),
        }

        # Записва обновения документ в Redis.
        self.redis.set(
            SAVED_POINTS_KEY,
            json.dumps(
                points,
                ensure_ascii=False,
            ),
        )

        # Връща новозаписаната точка.
        return points[point_name]

    def get_saved_point(
        self,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Търси запазена декартова точка по име.

        Търсенето първо е точно, а след това
        без значение от главни и малки букви.
        """

        # Проверява подаденото име.
        wanted = self.normalize_saved_point_name(
            name
        )

        # Зарежда всички запазени точки.
        points = self.load_saved_points()

        # Проверява за точно съвпадение.
        if wanted in points:
            return points[wanted]

        # Нормализира името до малки букви.
        lowered = wanted.lower()

        # Проверява за съвпадение
        # без значение от регистъра.
        for key, value in points.items():
            if key.lower() == lowered:
                return value

        return None

    def delete_saved_point(
        self,
        name: str,
    ) -> bool:
        """
        Изтрива запазена декартова точка по име.
        """

        # Проверява подаденото име.
        wanted = self.normalize_saved_point_name(
            name
        )

        # Зарежда запазените точки.
        points = self.load_saved_points()

        # Намира действителния ключ,
        # без значение от главни и малки букви.
        actual = next(
            (
                key
                for key in points
                if (
                    key.lower()
                    == wanted.lower()
                )
            ),
            None,
        )

        # Точката не съществува.
        if actual is None:
            return False

        # Премахва намерената точка.
        del points[actual]

        # Записва обновения документ.
        self.redis.set(
            SAVED_POINTS_KEY,
            json.dumps(
                points,
                ensure_ascii=False,
            ),
        )

        return True

    def current_joint_deg(
        self,
    ) -> Dict[str, float]:
        """
        Връща текущите ставни позиции в градуси.
        """

        # При липсващо ROS 2 ставно състояние
        # връща празен речник.
        if self.latest_joint_state is None:
            return {}

        # Преобразува позициите
        # от радиани в градуси.
        return {
            name: math.degrees(
                float(position)
            )
            for name, position in zip(
                self.latest_joint_state.name,
                self.latest_joint_state.position,
            )
        }

    def current_servo_degrees(
        self,
    ) -> Tuple[
        Dict[str, float],
        Dict[str, Any],
    ]:
        """
        Връща най-доброто налично отворено състояние
        на командваните физически сервомотори.

        Приоритет:
        1. текущо интерполираното състояние от Raspberry Pi;
        2. последните изпратени сервомоторни стойности;
        3. стойности, изведени от текущото RViz състояние;
        4. сервомоторните стойности на home позицията.

        Състоянието е отворено, защото системата не използва
        физическа обратна връзка от енкодери на ставите.
        """

        # Статус, върнат от физическия изпълнител.
        status: Dict[str, Any] = {}

        # Събрани физически сервомоторни стойности.
        servo: Dict[str, float] = {}

        # При активно физическо управление
        # първо пита Raspberry Pi за текущия статус.
        if self.motion.real.enabled:
            try:
                status = (
                    self.motion.real.request_status()
                )

                # Текущи интерполирани стойности,
                # отчетени от изпълнителя.
                current = status.get(
                    "current_angles"
                )

                if isinstance(current, dict):
                    for key, value in current.items():
                        # Запазва само познати сервомотори.
                        if key in SAFE_SERVO_RANGE:
                            servo[key] = round(
                                float(value),
                                3,
                            )

            except Exception as exc:
                # При проблем със статуса създава
                # диагностичен резултат.
                status = {
                    "ok": False,
                    "state": (
                        "PHYSICAL_STATUS_UNAVAILABLE"
                    ),
                    "error": str(exc),
                }

        # Допълва липсващите стойности
        # с последните изпратени команди.
        for key, value in (
            self.motion.real
            .last_commanded_angles
            .items()
        ):
            servo.setdefault(
                key,
                round(
                    float(value),
                    3,
                ),
            )

        # Получава текущите RViz ставни позиции.
        current_joints = self.current_joint_deg()

        # Когато са налични всички основни стави,
        # извежда физически сервомоторни стойности от тях.
        if all(
            joint in current_joints
            for joint in JOINT_ORDER
        ):
            # Преобразува текущите стави
            # обратно от градуси в радиани.
            rviz_radians = {
                joint: math.radians(
                    current_joints[joint]
                )
                for joint in JOINT_ORDER
            }

            # Прилага физическата калибрация.
            derived = (
                self.motion.real.joints_to_servo(
                    rviz_radians
                )
            )

            # Допълва само липсващите стойности.
            for key, value in derived.items():
                servo.setdefault(
                    key,
                    round(
                        float(value),
                        3,
                    ),
                )

        # Хващачът се обработва отделно,
        # защото не е част от JOINT_ORDER.
        if (
            self.latest_joint_state is not None
            and "joint_1_gripper"
            in self.latest_joint_state.name
        ):
            # Намира индекса на ставата на хващача.
            index = (
                self.latest_joint_state.name.index(
                    "joint_1_gripper"
                )
            )

            # Преобразува RViz позицията
            # от радиани в градуси.
            rviz_deg = math.degrees(
                float(
                    self.latest_joint_state.position[
                        index
                    ]
                )
            )

            # Получава калибратора на хващача.
            calibrator = (
                self.motion.real.calibrators.get(
                    "gripper"
                )
            )

            # Преобразува RViz стойността
            # до физическа сервомоторна стойност.
            gripper_servo = (
                calibrator.convert(rviz_deg)
                if calibrator is not None
                else 80.0 + rviz_deg
            )

            # Ограничава стойността
            # до безопасния диапазон.
            servo.setdefault(
                "gripper",
                round(
                    max(
                        SAFE_SERVO_RANGE[
                            "gripper"
                        ][0],
                        min(
                            SAFE_SERVO_RANGE[
                                "gripper"
                            ][1],
                            gripper_servo,
                        ),
                    ),
                    3,
                ),
            )

        # Зарежда сервомоторните стойности
        # на home позицията като последен резервен вариант.
        home = self.poses.servo_angles(
            "home"
        )

        for key, value in home.items():
            servo.setdefault(
                key,
                round(
                    float(value),
                    3,
                ),
            )

        # Подрежда сервомоторите в постоянен ред.
        ordered = {
            key: servo[key]
            for key in (
                "base",
                "shoulder",
                "elbow",
                "wrist",
                "wrist_rotation",
                "gripper",
            )
            if key in servo
        }

        return ordered, status

    def is_at_named_pose(
        self,
        pose_name: str,
    ) -> Tuple[
        bool,
        Dict[str, float],
    ]:
        """
        Проверява дали текущото ставно състояние
        съответства на именувана позиция.
        """

        # Получава текущите стави в градуси.
        current = self.current_joint_deg()

        # Получава целевите стави в радиани.
        target_rad = self.poses.joint_radians(
            pose_name
        )

        # Грешка за всяка отделна става.
        errors: Dict[str, float] = {}

        # Сравнява текущата и желаната стойност.
        for joint_name, target_value in target_rad.items():
            # При липсваща става проверката е неуспешна.
            if joint_name not in current:
                return False, {
                    "missing_joint": joint_name
                }

            # Изчислява абсолютната грешка в градуси.
            errors[joint_name] = abs(
                current[joint_name]
                - math.degrees(target_value)
            )

        # Позицията се счита за достигната,
        # когато всички стави са в допустимата толерантност.
        return (
            bool(errors)
            and max(errors.values())
            <= POSE_MATCH_TOLERANCE_DEG,
            errors,
        )

    def load_held_object(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Зарежда състоянието на предмета,
        който в момента се държи от хващача.
        """

        # Прочита състоянието от Redis.
        raw = self.redis.get(
            HELD_OBJECT_KEY
        )

        # Няма регистриран държан предмет.
        if not raw:
            return None

        try:
            # Декодира JSON документа.
            value = json.loads(raw)

        except Exception:
            return None

        # Приема само речниково състояние.
        return (
            value
            if isinstance(value, dict)
            else None
        )

    @staticmethod
    def point_display_name(
        point: Mapping[str, Any],
    ) -> str:
        """
        Връща най-подходящото име
        за показване на предмета.
        """

        return str(
            point.get("display_name_bg")
            or point.get("label")
            or point.get("point")
            or point.get("instance_id")
            or "object"
        )

    def save_held_object(
        self,
        point: Dict[str, Any],
    ) -> None:
        """
        Запазва предмет като текущо държан
        от хващача.
        """

        # Изчислява нормализираната геометрия
        # на предмета.
        geometry = self.targets.object_geometry(
            point
        )

        # Полета, които трябва да бъдат запазени
        # в моментната снимка на предмета.
        snapshot_fields = (
            "instance_id",
            "instance_index",
            "instance_count",
            "semantic_key",
            "point",
            "label",
            "safe_label",
            "display_name_bg",
            "base_display_name_bg",
            "role",
            "colors",
            "aliases",
            "dimensions",
            "raw_dimensions",
            "height",
            "center",
            "position",
            "top_position",
            "bottom_position",
            "grasp_position",
            "approach_position",
        )

        # Копира само наличните полета.
        snapshot = {
            key: point.get(key)
            for key in snapshot_fields
            if point.get(key) is not None
        }

        # Добавя изчислената геометрия.
        snapshot["geometry"] = geometry

        # Текущият център първоначално съвпада
        # с геометричния център на предмета.
        snapshot["current_center"] = dict(
            geometry["center"]
        )

        # Записва момента на хващане.
        snapshot["picked_at"] = time.time()

        # Съхранява състоянието в Redis.
        self.redis.set(
            HELD_OBJECT_KEY,
            json.dumps(
                snapshot,
                ensure_ascii=False,
            ),
        )

    def save_held_state(
        self,
        held: Mapping[str, Any],
    ) -> None:
        """
        Записва обновено състояние
        на държания предмет.
        """

        self.redis.set(
            HELD_OBJECT_KEY,
            json.dumps(
                dict(held),
                ensure_ascii=False,
            ),
        )

    def hydrate_held_object(
        self,
        held: Mapping[str, Any],
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Допълва запазеното състояние на държания предмет
        с актуални данни от работното пространство.

        Запазените данни имат приоритет пред данните
        от последното сканиране.
        """

        # Създава заявка за предмета чрез
        # устойчивия му идентификатор или име.
        query = str(
            held.get("instance_id")
            or held.get("point")
            or ""
        )

        # Актуалният предмет от работното пространство.
        live = None

        if query:
            try:
                # Опитва да намери съответстващия
                # актуален предмет.
                live = self.targets.resolve_target(
                    points,
                    query,
                )

            except AmbiguousTargetError:
                # При двусмислена цел използва
                # само запазеното състояние.
                live = None

        # Започва с актуалните данни,
        # когато такива са намерени.
        merged = dict(
            live or {}
        )

        # Запазеното състояние има приоритет.
        merged.update(
            dict(held)
        )

        # Получава актуализирания център
        # на държания предмет.
        current_center = held.get(
            "current_center"
        )

        if isinstance(
            current_center,
            Mapping,
        ):
            # Изчислява текущата геометрия
            # преди преместването.
            old_geometry = (
                self.targets.object_geometry(
                    merged
                )
            )

            old_center = old_geometry[
                "center"
            ]

            # Изчислява отместването между
            # стария и актуалния център.
            delta = (
                float(
                    current_center.get(
                        "x",
                        old_center["x"],
                    )
                )
                - old_center["x"],
                float(
                    current_center.get(
                        "y",
                        old_center["y"],
                    )
                )
                - old_center["y"],
                float(
                    current_center.get(
                        "z",
                        old_center["z"],
                    )
                )
                - old_center["z"],
            )

            # Премества всички свързани
            # геометрични позиции със същото отместване.
            for key in (
                "center",
                "position",
                "top_position",
                "bottom_position",
                "grasp_position",
                "approach_position",
            ):
                row = merged.get(key)

                if isinstance(row, Mapping):
                    merged[key] = {
                        "x": (
                            float(
                                row.get(
                                    "x",
                                    0.0,
                                )
                            )
                            + delta[0]
                        ),
                        "y": (
                            float(
                                row.get(
                                    "y",
                                    0.0,
                                )
                            )
                            + delta[1]
                        ),
                        "z": (
                            float(
                                row.get(
                                    "z",
                                    0.0,
                                )
                            )
                            + delta[2]
                        ),
                    }

            # Актуализира директните координати.
            merged["x"] = float(
                current_center.get(
                    "x",
                    0.0,
                )
            )
            merged["y"] = float(
                current_center.get(
                    "y",
                    0.0,
                )
            )
            merged["z"] = float(
                current_center.get(
                    "z",
                    0.0,
                )
            )

            # Актуализира централната позиция.
            merged["center"] = {
                "x": merged["x"],
                "y": merged["y"],
                "z": merged["z"],
            }

        return merged

    def update_held_object_from_gripper(
        self,
        held: Mapping[str, Any],
        gripper_xyz: Tuple[
            float,
            float,
            float,
        ],
    ) -> Dict[str, Any]:
        """
        Актуализира позицията на държания предмет
        спрямо новата позиция на хващача.
        """

        # Създава изменяемо копие
        # на запазеното състояние.
        hydrated = dict(held)

        # Получава геометрията на предмета.
        geometry = self.targets.object_geometry(
            hydrated
        )

        # Отместване между центъра на предмета
        # и позицията на хващача.
        offset = geometry["grasp_offset"]

        # Изчислява новия център на предмета.
        current_center = {
            "x": (
                gripper_xyz[0]
                - offset["x"]
            ),
            "y": (
                gripper_xyz[1]
                - offset["y"]
            ),
            "z": (
                gripper_xyz[2]
                - offset["z"]
            ),
        }

        # Записва новия център.
        hydrated[
            "current_center"
        ] = current_center

        # Записва последната позиция
        # на хващача.
        hydrated[
            "last_gripper_position"
        ] = self.xyz_dict(
            gripper_xyz
        )

        # Добавя време на обновяване.
        hydrated["updated_at"] = time.time()

        # Записва обновеното състояние в Redis.
        self.save_held_state(
            hydrated
        )

        return hydrated

    def clear_held_object(
        self,
    ) -> None:
        """
        Премахва състоянието за държан предмет.
        """

        self.redis.delete(
            HELD_OBJECT_KEY
        )

    @staticmethod
    def point_to_segment_distance(
        point_xy: Tuple[float, float],
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
    ) -> float:
        """
        Изчислява минималното двумерно разстояние
        между точка и отсечка.
        """

        # Координати на проверяваната точка.
        px, py = point_xy

        # Начална точка на отсечката.
        ax, ay = start_xy

        # Крайна точка на отсечката.
        bx, by = end_xy

        # Вектор на отсечката.
        dx = bx - ax
        dy = by - ay

        # Квадрат на дължината на отсечката.
        length_sq = (
            dx * dx
            + dy * dy
        )

        # При почти нулева дължина
        # отсечката се разглежда като точка.
        if length_sq <= 1e-12:
            return math.hypot(
                px - ax,
                py - ay,
            )

        # Изчислява проекцията на точката
        # върху отсечката.
        t = max(
            0.0,
            min(
                1.0,
                (
                    (
                        (px - ax) * dx
                        + (py - ay) * dy
                    )
                    / length_sq
                ),
            ),
        )

        # Координати на най-близката точка
        # върху отсечката.
        nearest_x = ax + t * dx
        nearest_y = ay + t * dy

        # Връща евклидовото разстояние.
        return math.hypot(
            px - nearest_x,
            py - nearest_y,
        )

    def height_aware_transit_z(
        self,
        workspace_points: List[
            Dict[str, Any]
        ],
        held_object: Mapping[str, Any],
        start_gripper_xyz: Tuple[
            float,
            float,
            float,
        ],
        end_gripper_xyz: Tuple[
            float,
            float,
            float,
        ],
        minimum_z: float,
        ignored_ids: Optional[
            set[str]
        ] = None,
    ) -> Tuple[
        Optional[float],
        List[Dict[str, Any]],
    ]:
        """
        Изчислява безопасна височина за хоризонтално
        преместване над измерените препятствия.

        Само предметите, чиито хоризонтални размери
        пресичат коридора между началната и крайната
        позиция, влияят върху необходимата височина.
        """

        # Геометрия на държания предмет.
        held_geometry = (
            self.targets.object_geometry(
                held_object
            )
        )

        # Приблизителен хоризонтален радиус
        # на държания предмет.
        held_radius = held_geometry[
            "radius_xy"
        ]

        # Вертикално отместване между центъра
        # на предмета и хващача.
        grasp_offset_z = held_geometry[
            "grasp_offset"
        ]["z"]

        # Начална минимална височина.
        required_z = float(
            minimum_z
        )

        # Списък с препятствията,
        # които пресичат транзитния коридор.
        blockers: List[
            Dict[str, Any]
        ] = []

        # Идентификатори, които не трябва
        # да участват в проверката.
        ignored = (
            ignored_ids
            or set()
        )

        # Начална хоризонтална позиция
        # на хващача.
        start_xy = (
            float(
                start_gripper_xyz[0]
            ),
            float(
                start_gripper_xyz[1]
            ),
        )

        # Крайна хоризонтална позиция
        # на хващача.
        end_xy = (
            float(
                end_gripper_xyz[0]
            ),
            float(
                end_gripper_xyz[1]
            ),
        )

        # Проверява всички точки
        # в работното пространство.
        for point in workspace_points:
            # Устойчив идентификатор
            # на текущата точка.
            point_id = str(
                point.get("instance_id")
                or point.get("point")
                or ""
            )

            # Пропуска изрично игнорираните предмети.
            if point_id in ignored:
                continue

            # Виртуалните точки и точките от историята
            # не представляват физически препятствия.
            if (
                point.get("virtual_point")
                or point.get(
                    "traceback_point"
                )
            ):
                continue

            # Получава геометрията
            # на възможното препятствие.
            geometry = (
                self.targets.object_geometry(
                    point
                )
            )

            # Радиус на безопасния коридор,
            # включващ двата предмета и допълнителен запас.
            corridor_radius = (
                geometry["radius_xy"]
                + held_radius
                + self.transit_corridor_margin
            )

            # Изчислява минималното разстояние
            # от центъра на препятствието
            # до траекторията по XY.
            distance = (
                self.point_to_segment_distance(
                    (
                        geometry["center"]["x"],
                        geometry["center"]["y"],
                    ),
                    start_xy,
                    end_xy,
                )
            )

            # Предметът не пресича коридора.
            if distance > corridor_radius:
                continue

            # Изчислява необходимата височина
            # на хващача над препятствието.
            obstacle_required_z = (
                geometry["top_z"]
                + held_geometry["height"]
                / 2.0
                + grasp_offset_z
                + self.transit_clearance
            )

            # Увеличава общата транзитна височина,
            # ако текущото препятствие е по-високо.
            if obstacle_required_z > required_z:
                required_z = obstacle_required_z

            # Записва диагностичните данни
            # за препятствието.
            blockers.append(
                {
                    "point": point.get(
                        "point"
                    ),
                    "instance_id": point.get(
                        "instance_id"
                    ),
                    "top_z": geometry[
                        "top_z"
                    ],
                    "required_gripper_z": (
                        obstacle_required_z
                    ),
                    "corridor_distance": distance,
                }
            )

        # Не допуска транзит над максимално
        # достижимата безопасна височина.
        if required_z > self.maximum_transit_z:
            return None, blockers

        return required_z, blockers