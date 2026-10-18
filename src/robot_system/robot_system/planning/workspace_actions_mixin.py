#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартна библиотека за математически операции
# и типови анотации.
import math
from typing import Any, Dict, List, Mapping, Tuple

# Помощна функция за зареждане на числова стойност
# от променливите на средата.
from .common import env_float

# Безопасни физически диапазони на сервомоторите.
from .constants import SAFE_SERVO_RANGE

# Настройки, свързани с:
# - последно избраната цел;
# - продължителността на ръчно движение;
# - толерантността при именувани позиции.
from .settings import (
    LAST_TARGET_KEY,
    MANUAL_JOINT_MOVE_DURATION_SEC,
    POSE_MATCH_TOLERANCE_DEG,
)


class WorkspaceActionsMixin:
    """
    Съдържа действията на планиращия агент,
    свързани с предметите и точките
    в работното пространство.

    Поддържаните операции включват:
    - синхронизиране на точките от историята;
    - разрешаване на естественоезикова цел;
    - връщане на предмет до предишна позиция;
    - хващане и поставяне на предмет;
    - поставяне на предмет, който вече се държи.
    """

    def handle_sync_traceback(
        self,
        request_id: str,
        request: Dict[str, Any],
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Връща текущото обобщение на историята
        на предметите и броя на наличните точки.

        Самото добавяне на виртуалните точки от историята
        към документа с работното пространство се извършва
        преди извикването на този обработващ метод.
        """

        # Получава обобщението на историята
        # за всички проследявани предмети.
        summary = self.traceback.history_summary()

        # Връща успешен резултат с:
        # - историята на предметите;
        # - броя на точките в текущия документ.
        return self.result(
            request_id,
            True,
            "TRACEBACK_POINTS_SYNCED",
            traceback=summary,
            point_count=len(points),
        )

    def handle_resolve_target(
        self,
        request_id: str,
        request: Dict[str, Any],
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Разрешава потребителско описание
        до конкретен предмет или точка
        от работното пространство.
        """

        # Получава името на целта.
        #
        # Предпочита полето target, а при липса
        # използва source.
        target_name = str(
            request.get("target")
            or request.get("source")
            or ""
        ).strip()

        # Не допуска заявка без зададена цел.
        if not target_name:
            return self.result(
                request_id,
                False,
                "TARGET_REQUIRED",
            )

        # Опитва да намери конкретна точка
        # според естественоезиковото описание.
        point = self.targets.resolve_target(
            points,
            target_name,
        )

        # При липса на съвпадение връща
        # списък с достъпните физически точки.
        if point is None:
            return self.result(
                request_id,
                False,
                "TARGET_NOT_FOUND",
                target=target_name,
                available=[
                    {
                        # Устойчив идентификатор
                        # на физическия екземпляр.
                        "instance_id": row.get(
                            "instance_id"
                        ),

                        # Потребителско име на български.
                        "display_name_bg": row.get(
                            "display_name_bg"
                        ),

                        # Име на точката.
                        "point": row.get("point"),
                    }
                    for row in points

                    # Виртуалните точки от историята
                    # не се показват като основни налични цели.
                    if not row.get("virtual_point")
                ],
            )

        # Запомня намерената точка като последно
        # използвана цел за последващи команди
        # като „вземи го“ или „премести същия“.
        self.targets.remember_target(point)

        # Връща основните данни
        # за разрешената конкретна цел.
        return self.result(
            request_id,
            True,
            "TARGET_RESOLVED",
            query=target_name,
            target={
                # Устойчив идентификатор.
                "instance_id": point.get(
                    "instance_id"
                ),

                # Пореден номер сред еднакви предмети.
                "instance_index": point.get(
                    "instance_index"
                ),

                # Общ брой еднакви екземпляри.
                "instance_count": point.get(
                    "instance_count"
                ),

                # Семантичен ключ на предмета.
                "semantic_key": point.get(
                    "semantic_key"
                ),

                # Име на точката.
                "point": point.get("point"),

                # Име за показване на български.
                "display_name_bg": point.get(
                    "display_name_bg"
                ),

                # Основно име без пространствено уточнение.
                "base_display_name_bg": point.get(
                    "base_display_name_bg"
                ),

                # Пространствени псевдоними,
                # например „левия“ или „най-близкия“.
                "spatial_aliases": point.get(
                    "spatial_aliases",
                    [],
                ),

                # Роля на точката.
                "role": point.get("role"),

                # Показва дали точката блокира
                # планираното движение.
                "safety_blocker": point.get(
                    "safety_blocker",
                    False,
                ),

                # Геометричен център на целта.
                "center": point.get("center"),
            },
        )

    def handle_return_object(
        self,
        request_id: str,
        request: Dict[str, Any],
        action: str,
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Връща предмет назад по неговата
        запазена история на преместванията.

        В зависимост от заявката предметът може да бъде:
        - върнат само една позиция назад;
        - върнат до първоначалната си позиция.
        """

        # Получава името на предмета.
        #
        # По подразбиране се използва
        # най-близкият предмет.
        target_name = str(
            request.get("target")
            or request.get("source")
            or "nearest_object"
        ).strip()

        # Определя дали движението трябва
        # да бъде действително изпълнено.
        #
        # При липсващо поле execute се използва
        # обратната стойност на preview_only_default.
        execute = bool(
            request.get(
                "execute",
                not self.preview_only_default,
            )
        )

        # Определя дали предметът трябва да се върне
        # през цялата история до произхода.
        #
        # Това се активира чрез:
        # - all_the_way_to_origin=True;
        # - действието return_object_to_origin.
        all_the_way = bool(
            request.get(
                "all_the_way_to_origin",
                False,
            )
            or action == "return_object_to_origin"
        )

        # Разрешава описанието до конкретен
        # физически екземпляр.
        resolved_source = self.targets.resolve_target(
            points,
            target_name,
        )

        # При липса на предмет операцията
        # не може да бъде изпълнена.
        if resolved_source is None:
            return self.result(
                request_id,
                False,
                "TARGET_NOT_FOUND",
                target=target_name,
            )

        # Създава устойчиво име за търсене
        # в историята на предмета.
        stable_target = str(
            resolved_source.get("instance_id")
            or resolved_source.get("point")
            or target_name
        )

        # Изпълнява общата логика
        # за връщане на предмет.
        return self.execute_return_object(
            request_id=request_id,
            target_name=stable_target,
            execute=execute,
            all_the_way_to_origin=all_the_way,
        )

    def handle_pick_and_place(
        self,
        request_id: str,
        request: Dict[str, Any],
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Обработва заявка за цялостно хващане
        и поставяне на предмет.
        """

        # Получава предмета източник.
        #
        # По подразбиране се използва
        # най-близкият предмет.
        source_name = str(
            request.get("source")
            or "nearest_object"
        ).strip()

        # Получава крайната цел.
        #
        # По подразбиране се използва
        # най-близката повърхност за поставяне.
        destination_name = str(
            request.get("destination")
            or "nearest_surface"
        ).strip()

        # Разрешава предмета източник.
        source = self.targets.resolve_target(
            points,
            source_name,
        )

        # Разрешава крайната цел.
        destination = self.targets.resolve_target(
            points,
            destination_name,
        )

        # Връща грешка, когато предметът
        # източник не е намерен.
        if source is None:
            return self.result(
                request_id,
                False,
                "SOURCE_NOT_FOUND",
                source=source_name,

                # Връща имената на всички налични точки
                # за диагностични цели.
                available=[
                    point.get("point")
                    for point in points
                ],
            )

        # Връща грешка, когато крайната цел
        # не е намерена.
        if destination is None:
            return self.result(
                request_id,
                False,
                "DESTINATION_NOT_FOUND",
                destination=destination_name,
                available=[
                    point.get("point")
                    for point in points
                ],
            )

        # Не допуска действие върху предмет
        # или цел, обозначени като опасни препятствия.
        if (
            source.get("safety_blocker")
            or destination.get("safety_blocker")
        ):
            return self.result(
                request_id,
                False,
                "SAFETY_BLOCKED",
            )

        # Определя дали последователността
        # трябва да бъде изпълнена или само визуализирана.
        execute = bool(
            request.get(
                "execute",
                not self.preview_only_default,
            )
        )

        # Изпълнява общата последователност
        # за хващане и поставяне.
        return self.execute_pick_and_place_sequence(
            request_id,
            source,
            destination,
            execute,

            # Предава актуалните точки,
            # за да бъдат отчетени препятствията
            # при изчисляване на транзитната височина.
            workspace_points=points,
        )

    def handle_place_held(
        self,
        request_id: str,
        request: Dict[str, Any],
        points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Поставя предмет, който вече се държи
        от хващача, върху избрана крайна цел.
        """

        # Зарежда запазеното състояние
        # на текущо държания предмет.
        held_state = self.load_held_object()

        # Операцията не може да продължи,
        # когато няма регистриран държан предмет.
        if held_state is None:
            return self.result(
                request_id,
                False,
                "NO_HELD_OBJECT",
            )

        # Получава името на крайната цел.
        #
        # Приоритет:
        # 1. destination;
        # 2. target;
        # 3. най-близката повърхност.
        destination_name = str(
            request.get("destination")
            or request.get("target")
            or "nearest_surface"
        ).strip()

        # Разрешава крайната цел
        # от актуалните точки.
        destination = self.targets.resolve_target(
            points,
            destination_name,
        )

        # Връща грешка, когато целта липсва.
        if destination is None:
            return self.result(
                request_id,
                False,
                "DESTINATION_NOT_FOUND",
                destination=destination_name,
            )

        # Не допуска поставяне върху точка,
        # обозначена като опасно препятствие.
        if destination.get("safety_blocker"):
            return self.result(
                request_id,
                False,
                "SAFETY_BLOCKED",
            )

        # Обединява запазеното състояние
        # на държания предмет с актуалните
        # данни от работното пространство.
        held_object = self.hydrate_held_object(
            held_state,
            points,
        )

        # Определя дали движението трябва
        # да бъде изпълнено или само визуализирано.
        execute = bool(
            request.get(
                "execute",
                not self.preview_only_default,
            )
        )

        # Изпълнява специализираната последователност
        # за поставяне на вече държан предмет.
        return self.execute_place_held_sequence(
            request_id=request_id,
            held_object=held_object,
            destination=destination,
            execute=execute,
            workspace_points=points,
        )