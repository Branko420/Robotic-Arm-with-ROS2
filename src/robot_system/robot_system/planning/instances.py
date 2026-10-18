#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване,
# математически операции, регулярни изрази,
# измерване на време и типови анотации.
import json
import math
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Помощна функция за прочитане на числова
# конфигурационна стойност от променливите на средата.
from .common import env_float


class ObjectInstanceRegistry:
    """
    Назначава устойчиви идентификатори на отделните физически
    екземпляри, без да приема конкретен тип, форма или цвят.

    Примери:
      bottle_1, bottle_2
      red_object_1, red_object_2
      screwdriver_1
      unknown_object_3

    Семантичната основа се извлича от метаданните на скенера:
    semantic_key, class_name, object_type, category, label,
    safe_label или point.
    """

    # Съответствие между текстови редни и бройни числителни
    # на български език и техните числови стойности.
    NUMBER_WORDS = {
        "първия": 1,
        "първият": 1,
        "първо": 1,
        "едно": 1,
        "втория": 2,
        "вторият": 2,
        "второ": 2,
        "две": 2,
        "третия": 3,
        "третият": 3,
        "трето": 3,
        "три": 3,
        "четвъртия": 4,
        "четвъртият": 4,
        "четири": 4,
        "петия": 5,
        "петият": 5,
        "пет": 5,
    }

    # Общи думи, които не носят достатъчно семантична
    # информация и се премахват при създаване на ключ.
    GENERIC_WORDS = {
        "point",
        "object",
        "item",
        "thing",
        "detected",
        "tracked",
        "предмет",
        "обект",
        "точка",
    }

    def __init__(
        self,
        redis_client: Any,
        registry_key: str,
    ) -> None:
        """
        Инициализира регистъра на екземплярите.

        redis_client се използва за постоянно съхранение,
        а registry_key определя Redis ключа на регистъра.
        """

        # Redis клиентът, използван за четене и запис.
        self.redis = redis_client

        # Redis ключът, под който се съхранява регистърът.
        self.registry_key = registry_key

        # Максимално допустимо разстояние между ново открит предмет
        # и очакваната позиция на съществуващ екземпляр.
        self.match_max_distance = env_float(
            "INSTANCE_MATCH_MAX_DISTANCE_M",
            0.06,
        )

        # Максимално допустимо хоризонтално разстояние
        # на предмет от основата на манипулатора.
        self.max_base_distance = max(
            0.0,
            env_float(
                "PLANNER_MAX_TARGET_BASE_DISTANCE_M",
                env_float(
                    "SIMPLE_OBJECT_MAX_BASE_DISTANCE_M",
                    0.40,
                ),
            ),
        )

    @staticmethod
    def _copy(value: Any) -> Any:
        """
        Създава дълбоко копие чрез JSON сериализация.
        """

        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
            )
        )

    @staticmethod
    def normalize(value: Any) -> str:
        """
        Нормализира текст за използване като устойчив ключ.

        Текстът се преобразува до малки букви, премахват се
        специалните символи и интервалите се заменят с долни черти.
        """

        # Преобразува входната стойност до нормализиран текст.
        text = str(value or "").strip().lower()

        # Премахва символи, които не са букви, цифри,
        # интервали, тирета или долни черти.
        text = re.sub(
            r"[^0-9a-zа-яё_\- ]+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Заменя поредици от интервали и тирета с долна черта.
        text = re.sub(
            r"[\s\-]+",
            "_",
            text,
        )

        # Обединява последователните долни черти
        # и премахва началните и крайните.
        return re.sub(
            r"_+",
            "_",
            text,
        ).strip("_")

    @staticmethod
    def center(
        point: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        """
        Извлича централната позиция на предмета.

        Поддържат се полетата center, position
        или директни координати x, y и z.
        """

        # Предпочита вложения център, след това позицията,
        # а при липса използва самия речник.
        row = (
            point.get("center")
            or point.get("position")
            or point
        )

        return (
            float(
                row.get(
                    "x",
                    point.get("x", 0.0),
                )
            ),
            float(
                row.get(
                    "y",
                    point.get("y", 0.0),
                )
            ),
            float(
                row.get(
                    "z",
                    point.get("z", 0.0),
                )
            ),
        )

    @staticmethod
    def distance(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        """
        Изчислява евклидовото разстояние между две 3D позиции.
        """

        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2
        )

    def load(self) -> Dict[str, Any]:
        """
        Зарежда регистъра на екземплярите от Redis.
        """

        try:
            # Прочита JSON документа от Redis.
            raw = self.redis.get(
                self.registry_key
            )

            # Декодира документа или използва празен речник.
            document = (
                json.loads(raw)
                if raw
                else {}
            )

        except Exception:
            # При грешка използва празен регистър.
            document = {}

        # Гарантира, че документът е речник.
        if not isinstance(document, dict):
            document = {}

        # Добавя задължителните полета,
        # когато липсват в по-стар документ.
        document.setdefault("version", 1)
        document.setdefault("next_index", {})
        document.setdefault("instances", {})

        return document

    def save(
        self,
        document: Dict[str, Any],
    ) -> None:
        """
        Записва регистъра на екземплярите в Redis.
        """

        # Добавя времето на последното обновяване.
        document["updated_at"] = time.time()

        # Записва документа като JSON текст.
        self.redis.set(
            self.registry_key,
            json.dumps(
                document,
                ensure_ascii=False,
            ),
        )

    def _strip_instance_suffix(
        self,
        value: str,
    ) -> str:
        """
        Премахва пореден номер от края на име на екземпляр.
        """

        # Нормализира входното име.
        value = self.normalize(value)

        # Премахва окончания като:
        # object_1, object_номер_1 или object_#1.
        value = re.sub(
            r"(?:_номер)?_\d+$|_#\d+$|_\d+$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip("_")

        return value

    def _clean_semantic_text(
        self,
        value: Any,
    ) -> str:
        """
        Почиства текст за създаване на семантичен ключ.
        """

        # Премахва номера на екземпляра.
        text = self._strip_instance_suffix(
            str(value or "")
        )

        # Оставя само токените, които не са общи думи.
        tokens = [
            token
            for token in text.split("_")
            if (
                token
                and token not in self.GENERIC_WORDS
            )
        ]

        return "_".join(tokens)

    def semantic_key(
        self,
        point: Mapping[str, Any],
    ) -> str:
        """
        Създава общ семантичен ключ за семейство предмети.

        Приоритет:
        1. изрични метаданни от скенера;
        2. клас, категория или тип;
        3. устойчиво потребителско име;
        4. име на точката от скенера;
        5. обща резервна стойност.

        В този метод няма твърдо зададени форми или цветове.
        """

        # Полетата се проверяват последователно
        # според тяхната надеждност и конкретност.
        explicit_fields = (
            "semantic_key",
            "class_name",
            "object_class",
            "object_type",
            "category",
            "type",
            "semantic_label",
            "safe_label",
            "label",
            "display_name_bg",
            "point",
        )

        # Връща първата валидна почистена стойност.
        for field in explicit_fields:
            cleaned = self._clean_semantic_text(
                point.get(field)
            )

            if cleaned:
                return cleaned

        # При липса на основно поле проверява
        # допълнителните имена на предмета.
        aliases = point.get("aliases") or []

        for alias in aliases:
            cleaned = self._clean_semantic_text(
                alias
            )

            if cleaned:
                return cleaned

        # Резервен ключ за непознат предмет.
        return "unknown_object"

    def base_display_name(
        self,
        point: Mapping[str, Any],
        semantic_key: str,
    ) -> str:
        """
        Определя основното потребителско име на предмета,
        без пореден номер на екземпляр.
        """

        # Избира първото налично подходящо име.
        value = str(
            point.get("base_display_name_bg")
            or point.get("display_name_bg")
            or point.get("label")
            or point.get("class_name")
            or point.get("object_type")
            or semantic_key.replace("_", " ")
        ).strip()

        # Премахва номер от края на името.
        value = re.sub(
            r"(?:\s*#?\s*\d+|\s+номер\s+\d+)$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

        # При празен резултат използва семантичния ключ.
        return (
            value
            or semantic_key.replace("_", " ")
        )

    @staticmethod
    def external_tracking_id(
        point: Mapping[str, Any],
    ) -> Optional[str]:
        """
        Връща постоянен идентификатор, предоставен
        от външната система за проследяване.
        """

        # Поддържа няколко възможни имена
        # на полето за устойчив идентификатор.
        for key in (
            "persistent_id",
            "tracking_id",
            "track_id",
            "object_uuid",
            "uuid",
        ):
            value = point.get(key)

            if (
                value is not None
                and str(value).strip()
            ):
                return str(value).strip()

        return None

    def _new_instance(
        self,
        document: Dict[str, Any],
        semantic_key: str,
        point: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Създава нов устойчив екземпляр за открит предмет.
        """

        # Получава следващия свободен пореден номер
        # за конкретното семантично семейство.
        next_index = int(
            document["next_index"].get(
                semantic_key,
                1,
            )
        )

        # Увеличава брояча за следващ нов екземпляр.
        document["next_index"][
            semantic_key
        ] = next_index + 1

        # Създава идентификатор като cube_1 или bottle_2.
        instance_id = (
            f"{semantic_key}_{next_index}"
        )

        # Извлича текущия център на предмета.
        center = self.center(point)

        # Създава записа за новия екземпляр.
        row = {
            "instance_id": instance_id,
            "semantic_key": semantic_key,
            "instance_index": next_index,
            "expected_center": {
                "x": center[0],
                "y": center[1],
                "z": center[2],
            },
            "scanner_point": point.get("point"),
            "external_tracking_id": (
                self.external_tracking_id(point)
            ),
            "base_display_name": (
                self.base_display_name(
                    point,
                    semantic_key,
                )
            ),
            "created_at": time.time(),
            "last_seen_at": time.time(),
            "active": True,
        }

        # Добавя екземпляра към регистъра.
        document["instances"][
            instance_id
        ] = row

        return row

    def update_center(
        self,
        instance_id: str,
        center: Tuple[float, float, float],
    ) -> None:
        """
        Актуализира очаквания център на съществуващ екземпляр.
        """

        # Зарежда регистъра.
        document = self.load()

        # Търси съществуващия екземпляр.
        row = document["instances"].get(
            instance_id
        )

        # Прекратява, ако идентификаторът не съществува.
        if not isinstance(row, dict):
            return

        # Записва новия очакван център.
        row["expected_center"] = {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        }

        # Отбелязва, че позицията е обновена
        # от планиращия агент.
        row[
            "updated_by_planning_agent_at"
        ] = time.time()

        # Маркира екземпляра като активен.
        row["active"] = True

        # Записва обновения регистър.
        self.save(document)

    def point_is_within_range(
        self,
        point: Mapping[str, Any],
    ) -> bool:
        """
        Проверява дали предметът е валиден
        и се намира в допустимия работен обсег.
        """

        # Уважава изричното поле within_reach от скенера.
        if point.get("within_reach") is False:
            return False

        try:
            # Извлича централната позиция.
            center = self.center(point)

        except (
            TypeError,
            ValueError,
            KeyError,
        ):
            return False

        # Отхвърля невалидни числови координати.
        if not all(
            math.isfinite(value)
            for value in center
        ):
            return False

        # Проверява хоризонталното разстояние
        # от оста на основата.
        return (
            self.max_base_distance <= 0.0
            or math.hypot(
                center[0],
                center[1],
            )
            <= self.max_base_distance
        )

    def assign_document(
        self,
        points_document: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Назначава устойчиви идентификатори на всички
        физически предмети в документ с точки за движение.
        """

        # Създава дълбоко копие, за да не променя
        # директно подадения документ.
        output = self._copy(
            dict(points_document)
        )

        # Извлича първоначалните точки.
        original_points = list(
            output.get(
                "motion_points",
                [],
            )
        )

        # Запазва:
        # - виртуалните точки;
        # - точките от историята;
        # - физическите предмети в допустимия обсег.
        points = [
            point
            for point in original_points
            if (
                point.get("virtual_point")
                or point.get("traceback_point")
                or self.point_is_within_range(point)
            )
        ]

        # Изчислява броя на игнорираните точки
        # извън допустимия обсег.
        ignored_out_of_range = (
            len(original_points)
            - len(points)
        )

        # Индекси на реалните физически предмети.
        physical_indices = [
            index
            for index, point in enumerate(points)
            if (
                not point.get("virtual_point")
                and not point.get(
                    "traceback_point"
                )
            )
        ]

        # Зарежда текущия регистър.
        document = self.load()

        # Първоначално маркира всички известни
        # екземпляри като неактивни.
        for row in document[
            "instances"
        ].values():
            row["active"] = False

        # Групира физическите предмети
        # според техния семантичен ключ.
        grouped: Dict[str, List[int]] = {}

        for index in physical_indices:
            key = self.semantic_key(
                points[index]
            )

            grouped.setdefault(
                key,
                [],
            ).append(index)

        # Обработва всяко семантично семейство поотделно.
        for semantic_key, indices in grouped.items():
            # Съществуващи регистрирани екземпляри
            # от същото семейство.
            existing = [
                row
                for row in document[
                    "instances"
                ].values()
                if (
                    row.get("semantic_key")
                    == semantic_key
                )
            ]

            # Съответствие между индекс на открит предмет
            # и регистриран екземпляр.
            assigned: Dict[
                int,
                Dict[str, Any],
            ] = {}

            # Идентификатори, които вече са използвани
            # в текущото сканиране.
            used_instances: set[str] = set()

            # Запазва идентичност само при реален постоянен
            # идентификатор от външна система за проследяване.
            #
            # Името на точката от скенера не се приема за устойчиво,
            # защото редът на предметите може да се промени.
            for point_index in indices:
                tracking_id = (
                    self.external_tracking_id(
                        points[point_index]
                    )
                )

                # Без устойчив външен идентификатор
                # преминава към пространствено съпоставяне.
                if tracking_id is None:
                    continue

                for row in existing:
                    instance_id = str(
                        row.get("instance_id")
                    )

                    # Един екземпляр не може да бъде
                    # назначен на два предмета.
                    if instance_id in used_instances:
                        continue

                    # Съпоставя еднаквите външни
                    # идентификатори.
                    if (
                        row.get(
                            "external_tracking_id"
                        )
                        == tracking_id
                    ):
                        assigned[
                            point_index
                        ] = row

                        used_instances.add(
                            instance_id
                        )

                        break

            # Създава всички възможни двойки между
            # останалите открити предмети и известните екземпляри.
            pairs: List[
                Tuple[
                    float,
                    int,
                    Dict[str, Any],
                ]
            ] = []

            for point_index in indices:
                # Пропуска вече назначения предмет.
                if point_index in assigned:
                    continue

                # Център на текущото откритие.
                point_center = self.center(
                    points[point_index]
                )

                for row in existing:
                    instance_id = str(
                        row.get("instance_id")
                    )

                    # Пропуска вече използван екземпляр.
                    if instance_id in used_instances:
                        continue

                    # Очакваната позиция на екземпляра.
                    expected = (
                        row.get("expected_center")
                        or {}
                    )

                    try:
                        expected_center = (
                            float(expected["x"]),
                            float(expected["y"]),
                            float(expected["z"]),
                        )

                    except Exception:
                        continue

                    # Добавя двойката и пространственото разстояние.
                    pairs.append(
                        (
                            self.distance(
                                point_center,
                                expected_center,
                            ),
                            point_index,
                            row,
                        )
                    )

            # Най-близките двойки се разглеждат първи.
            pairs.sort(
                key=lambda item: item[0]
            )

            # Индекси на предмети, които вече са назначени.
            used_points = set(assigned)

            # Назначава най-близките валидни съответствия.
            for distance, point_index, row in pairs:
                instance_id = str(
                    row.get("instance_id")
                )

                # Отхвърля двойката, когато:
                # - разстоянието е прекалено голямо;
                # - предметът вече е назначен;
                # - екземплярът вече е използван.
                if (
                    distance > self.match_max_distance
                    or point_index in used_points
                    or instance_id in used_instances
                ):
                    continue

                assigned[point_index] = row
                used_points.add(point_index)
                used_instances.add(instance_id)

            # Създава нови екземпляри за предметите,
            # които не са съпоставени със съществуващ запис.
            for point_index in indices:
                row = assigned.get(
                    point_index
                )

                if row is None:
                    row = self._new_instance(
                        document,
                        semantic_key,
                        points[point_index],
                    )

                    assigned[point_index] = row

                # Актуализира очакваната позиция
                # според текущото сканиране.
                center = self.center(
                    points[point_index]
                )

                row["expected_center"] = {
                    "x": center[0],
                    "y": center[1],
                    "z": center[2],
                }

                # Запазва текущото име на сканираната точка.
                row["scanner_point"] = (
                    points[point_index].get(
                        "point"
                    )
                )

                # Актуализира външния идентификатор,
                # когато такъв е предоставен.
                tracking_id = (
                    self.external_tracking_id(
                        points[point_index]
                    )
                )

                if tracking_id is not None:
                    row[
                        "external_tracking_id"
                    ] = tracking_id

                # Запазва съществуващото потребителско име
                # или създава ново.
                row["base_display_name"] = (
                    row.get(
                        "base_display_name"
                    )
                    or self.base_display_name(
                        points[point_index],
                        semantic_key,
                    )
                )

                # Актуализира времето на последното откриване.
                row["last_seen_at"] = time.time()

                # Маркира екземпляра като активен.
                row["active"] = True

            # Списък с всички предмети
            # от текущото семантично семейство.
            group_rows = [
                (
                    point_index,
                    assigned[point_index],
                )
                for point_index in indices
            ]

            # Брой еднакви предмети в текущото сканиране.
            group_count = len(group_rows)

            # Подрежда предметите по оста Y,
            # за да определи левия и десния.
            by_left = sorted(
                group_rows,
                key=lambda item: self.center(
                    points[item[0]]
                )[1],
                reverse=True,
            )

            # Подрежда предметите според хоризонталното
            # им разстояние от основата.
            by_distance = sorted(
                group_rows,
                key=lambda item: math.hypot(
                    self.center(
                        points[item[0]]
                    )[0],
                    self.center(
                        points[item[0]]
                    )[1],
                ),
            )

            # Добавя уникалните имена и пространствени псевдоними.
            for point_index, row in group_rows:
                point = points[point_index]

                # Устойчив идентификатор на екземпляра.
                instance_id = str(
                    row["instance_id"]
                )

                # Пореден номер на екземпляра.
                instance_index = int(
                    row["instance_index"]
                )

                # Основно потребителско име.
                base_name = str(
                    row["base_display_name"]
                )

                # Пространствени описания на предмета.
                spatial_aliases: List[str] = []

                # Пространствени псевдоними са необходими
                # само при повече от един еднакъв предмет.
                if group_count > 1:
                    # Предметът с най-голяма Y координата
                    # се приема за най-ляв.
                    if by_left[0][0] == point_index:
                        spatial_aliases.extend(
                            [
                                f"левия {base_name}",
                                f"най-левия {base_name}",
                                f"left {base_name}",
                            ]
                        )

                    # Предметът с най-малка Y координата
                    # се приема за най-десен.
                    if by_left[-1][0] == point_index:
                        spatial_aliases.extend(
                            [
                                f"десния {base_name}",
                                f"най-десния {base_name}",
                                f"right {base_name}",
                            ]
                        )

                    # Първият в подреждането по разстояние
                    # е най-близкият до основата.
                    if (
                        by_distance[0][0]
                        == point_index
                    ):
                        spatial_aliases.extend(
                            [
                                f"най-близкия {base_name}",
                                f"nearest {base_name}",
                            ]
                        )

                    # Последният е най-далечният.
                    if (
                        by_distance[-1][0]
                        == point_index
                    ):
                        spatial_aliases.extend(
                            [
                                f"най-далечния {base_name}",
                                f"farthest {base_name}",
                            ]
                        )

                # Копира съществуващите псевдоними.
                aliases = list(
                    point.get("aliases")
                    or []
                )

                # Добавя идентификатора и няколко
                # варианта с пореден номер.
                aliases.extend(
                    [
                        instance_id,
                        (
                            f"{base_name} номер "
                            f"{instance_index}"
                        ),
                        (
                            f"{base_name} "
                            f"{instance_index}"
                        ),
                        (
                            f"инстанция {instance_index} "
                            f"на {base_name}"
                        ),
                    ]
                )

                # Добавя пространствените описания.
                aliases.extend(
                    spatial_aliases
                )

                # Записва идентификационните полета
                # в точката от работното пространство.
                point["instance_id"] = instance_id
                point["instance_index"] = (
                    instance_index
                )
                point["instance_count"] = (
                    group_count
                )
                point["semantic_key"] = (
                    semantic_key
                )
                point["scanner_point"] = (
                    point.get("point")
                )
                point[
                    "base_display_name_bg"
                ] = base_name
                point["spatial_aliases"] = (
                    spatial_aliases
                )

                # Премахва повтарящите се псевдоними,
                # като запазва първоначалния им ред.
                point["aliases"] = list(
                    dict.fromkeys(aliases)
                )

                # При повече от един еднакъв предмет
                # показваното име включва поредния номер.
                if group_count > 1:
                    point["display_name_bg"] = (
                        f"{base_name} номер "
                        f"{instance_index}"
                    )

        # Записва актуализирания регистър в Redis.
        self.save(document)

        # Заменя точките с филтрирания
        # и обогатен списък.
        output["motion_points"] = points

        # Добавя диагностична информация
        # за филтрираните точки.
        output[
            "ignored_out_of_range_count"
        ] = ignored_out_of_range

        # Добавя използваното максимално разстояние.
        output[
            "max_base_distance_m"
        ] = self.max_base_distance

        # Добавя Redis ключа на регистъра.
        output[
            "instance_registry_key"
        ] = self.registry_key

        # Време на последното обновяване.
        output[
            "instance_registry_updated_at"
        ] = time.time()

        return output