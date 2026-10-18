#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за:
# - JSON сериализация и десериализация;
# - математически операции;
# - регулярни изрази;
# - типови анотации.
import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Помощна функция за прочитане на числова
# конфигурационна стойност от променливите на средата.
from .common import env_float

# Регистърът назначава устойчиви идентификатори
# на отделните физически екземпляри.
from .instances import ObjectInstanceRegistry


class AmbiguousTargetError(Exception):
    """
    Изключение при съвпадение на естественоезикова
    заявка с повече от един физически екземпляр.
    """

    def __init__(
        self,
        query: str,
        candidates: List[Mapping[str, Any]],
    ) -> None:
        # Запазва оригиналната потребителска заявка.
        self.query = str(query)

        # Създава съкратено описание на всички
        # възможни кандидати за цел.
        self.candidates = [
            {
                # Устойчив идентификатор на екземпляра.
                "instance_id": point.get("instance_id"),

                # Пореден номер на екземпляра.
                "instance_index": point.get("instance_index"),

                # Брой еднакви екземпляри.
                "instance_count": point.get("instance_count"),

                # Име за показване на български език.
                "display_name_bg": point.get("display_name_bg"),

                # Име на точката от сканиращия модул.
                "point": point.get("point"),

                # Центърът се взема от center или
                # се създава от директните координати.
                "center": (
                    point.get("center")
                    or {
                        "x": point.get("x"),
                        "y": point.get("y"),
                        "z": point.get("z"),
                    }
                ),

                # Пространствени псевдоними като
                # „левия“, „десния“ и „най-близкия“.
                "spatial_aliases": point.get(
                    "spatial_aliases",
                    [],
                ),
            }
            for point in candidates
        ]

        # Инициализира базовото изключение
        # с диагностично съобщение.
        super().__init__(
            f"Ambiguous target {self.query!r}: "
            f"{len(self.candidates)} candidates"
        )


class WorkspaceTargetTools:
    """
    Съдържа помощните средства за намиране,
    валидиране и геометрично обработване на цели
    в работното пространство.

    Класът поддържа:
    - естественоезиково търсене на предмети;
    - препратки към последно използваната цел;
    - избор по цвят, номер и пространствено положение;
    - откриване на двусмислени цели;
    - нормализиране на измерената геометрия;
    - изчисляване на позиции за подход и поставяне;
    - съхраняване на корекции по X, Y и Z.
    """

    # Думи и фрази, които се интерпретират
    # като препратка към последно избрания предмет.
    REFERENCE_QUERIES = {
        "го",
        "го_вземи",
        "него",
        "нея",
        "това",
        "този",
        "тази",
        "същия",
        "същият",
        "същото",
        "същия_обект",
        "последния_обект",
        "it",
        "that",
        "this",
        "the_object",
        "last_object",
    }

    def __init__(
        self,
        redis_client: Any,
        motion_points_key: str,
        offsets_key: str,
        last_target_key: str = "workspace:last_planning_target",
    ) -> None:
        """
        Инициализира инструментите за избор на цели.
        """

        # Redis клиентът, използван за четене
        # и запис на състоянието.
        self.redis = redis_client

        # Redis ключът с актуалните точки
        # от работното пространство.
        self.motion_points_key = motion_points_key

        # Redis ключът с натрупаните корекции
        # на отделните цели.
        self.offsets_key = offsets_key

        # Redis ключът с последно избраната цел.
        self.last_target_key = last_target_key

        # Максимално допустимо хоризонтално разстояние
        # на целта от основата на манипулатора.
        self.max_target_base_distance = max(
            0.0,
            env_float(
                "PLANNER_MAX_TARGET_BASE_DISTANCE_M",
                env_float(
                    "SIMPLE_OBJECT_MAX_BASE_DISTANCE_M",
                    0.40,
                ),
            ),
        )

    def load_motion_points(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Зарежда актуалния документ с точки
        от работното пространство.
        """

        # Прочита документа от Redis.
        raw = self.redis.get(
            self.motion_points_key
        )

        # При липсваща стойност връща None.
        if not raw:
            return None

        try:
            # Декодира JSON документа.
            document = json.loads(raw)

        except Exception:
            # При невалиден JSON връща None.
            return None

        # Приема само речников документ.
        return (
            document
            if isinstance(document, dict)
            else None
        )

    def last_target(
        self,
    ) -> Optional[str]:
        """
        Връща устойчивия идентификатор
        на последно избраната цел.
        """

        try:
            # Прочита стойността от Redis.
            value = self.redis.get(
                self.last_target_key
            )

        except Exception:
            # При Redis грешка приема,
            # че няма запомнена цел.
            return None

        # Нормализира стойността до текст.
        text = str(
            value or ""
        ).strip()

        return text or None

    @staticmethod
    def _point_identity(
        point: Mapping[str, Any],
    ) -> str:
        """
        Връща най-устойчивия наличен
        идентификатор на дадена точка.
        """

        return str(
            point.get("instance_id")
            or point.get("object_id")
            or point.get("point")
            or id(point)
        )

    def _deduplicate_points(
        self,
        points: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Премахва повтарящите се точки
        според устойчивия им идентификатор.
        """

        # Речникът гарантира по един запис
        # за всеки уникален идентификатор.
        unique: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for point in points:
            unique[
                self._point_identity(point)
            ] = point

        return list(
            unique.values()
        )

    def remember_target(
        self,
        point: Mapping[str, Any],
    ) -> None:
        """
        Запазва конкретната точка
        като последно избрана цел.
        """

        # Получава устойчивата идентичност.
        identity = self._point_identity(
            point
        )

        try:
            # Записва идентификатора в Redis.
            self.redis.set(
                self.last_target_key,
                identity,
            )

        except Exception:
            # Не прекратява основната операция
            # при проблем със запомнянето.
            pass

    @staticmethod
    def normalize_query(
        value: Any,
    ) -> str:
        """
        Нормализира естественоезикова заявка
        за сравнение с имена и псевдоними.
        """

        # Преобразува стойността до малки букви.
        text = str(
            value or ""
        ).strip().lower()

        # Премахва неподдържаните символи.
        text = re.sub(
            r"[^0-9a-zа-яё_\- ]+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Заменя интервали и тирета с долна черта.
        text = re.sub(
            r"[\s\-]+",
            "_",
            text,
        )

        # Обединява поредните долни черти.
        return re.sub(
            r"_+",
            "_",
            text,
        ).strip("_")

    def target_is_within_range(
        self,
        point: Mapping[str, Any],
    ) -> bool:
        """
        Проверява дали целта се намира
        в допустимия хоризонтален обсег.
        """

        # Уважава изрично зададеното поле
        # за недостижим предмет.
        if point.get("within_reach") is False:
            return False

        # Предпочита центъра, след това позицията,
        # а накрая директните координати.
        row = (
            point.get("center")
            or point.get("position")
            or point
        )

        # При липсващи X или Y координати
        # не отхвърля целта автоматично.
        if (
            "x" not in row
            or "y" not in row
        ):
            return True

        try:
            # Преобразува координатите до float.
            x = float(row["x"])
            y = float(row["y"])

        except (
            TypeError,
            ValueError,
            KeyError,
        ):
            return False

        # Отхвърля NaN и безкрайни стойности.
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
        ):
            return False

        # Проверява хоризонталното разстояние
        # от основата на манипулатора.
        return (
            self.max_target_base_distance <= 0.0
            or math.hypot(x, y)
            <= self.max_target_base_distance
        )

    def resolve_target(
        self,
        points: List[Dict[str, Any]],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Намира конкретна цел според естественоезикова заявка.

        Поддържат се:
        - точни и частични съвпадения;
        - цветове;
        - поредни номера;
        - instance_id;
        - ляв, десен, най-близък и най-далечен;
        - препратка към последно използваната цел.
        """

        # Запазва само точките в допустимия обсег
        # и премахва дублираните записи.
        points = self._deduplicate_points(
            [
                point
                for point in points
                if self.target_is_within_range(
                    point
                )
            ]
        )

        # Няма възможни цели.
        if not points:
            return None

        # Оригинална заявка за проверки,
        # зависещи от естествения текст.
        original_query = str(
            query or ""
        ).strip().lower()

        # Нормализирана заявка за сравнение.
        q = self.normalize_query(
            original_query
        )

        # При референтна дума използва
        # последно запомнената цел.
        if q in self.REFERENCE_QUERIES:
            remembered = self.last_target()

            if remembered:
                original_query = remembered.lower()
                q = self.normalize_query(
                    remembered
                )

        # Речник с канонични цветове
        # и техните български и английски варианти.
        color_words = {
            "green": {
                "green",
                "зелен",
                "зеления",
                "зеленият",
                "зеленото",
                "зелена",
                "зелената",
                "зелено",
            },
            "blue": {
                "blue",
                "син",
                "синия",
                "синият",
                "синьото",
                "синята",
                "синя",
                "синьо",
            },
            "yellow": {
                "yellow",
                "жълт",
                "жълтия",
                "жълто",
            },
            "red": {
                "red",
                "червен",
                "червения",
                "червено",
            },
            "white": {
                "white",
                "бял",
                "белия",
                "бяла",
                "бяло",
            },
            "black": {
                "black",
                "черен",
                "черния",
                "черна",
                "черно",
            },
        }

        # Определя цветовете, споменати
        # в потребителската заявка.
        requested_colors = {
            canonical
            for canonical, variants
            in color_words.items()
            if any(
                word in original_query
                for word in variants
            )
        }

        # При споменат цвят първо филтрира
        # предметите според цветовите им метаданни.
        if requested_colors:
            color_candidates = [
                point
                for point in points
                if (
                    {
                        self.normalize_query(color)
                        for color
                        in point.get("colors", [])
                    }
                    & requested_colors
                )
            ]

            # Премахва повторенията.
            color_candidates = (
                self._deduplicate_points(
                    color_candidates
                )
            )

            # При точно един предмет
            # го връща незабавно.
            if len(color_candidates) == 1:
                return color_candidates[0]

            # При няколко предмета ограничава
            # следващото търсене до тях.
            if color_candidates:
                points = color_candidates

        # Кандидати с точно съвпадение.
        exact: List[
            Dict[str, Any]
        ] = []

        # Кандидати с частично съвпадение
        # и изчислена оценка.
        fuzzy: List[
            Tuple[int, Dict[str, Any]]
        ] = []

        # Сравнява заявката с всички имена
        # и псевдоними на всяка точка.
        for point in points:
            names = {
                self.normalize_query(
                    point.get("instance_id")
                ),
                self.normalize_query(
                    point.get("point")
                ),
                self.normalize_query(
                    point.get("label")
                ),
                self.normalize_query(
                    point.get("safe_label")
                ),
                self.normalize_query(
                    point.get("display_name_bg")
                ),
                self.normalize_query(
                    point.get("base_display_name_bg")
                ),
                self.normalize_query(
                    point.get("role")
                ),
                *[
                    self.normalize_query(alias)
                    for alias
                    in point.get("aliases", [])
                ],
            }

            # Премахва празния низ.
            names.discard("")

            # Нормализира цветовете на точката.
            colors = {
                self.normalize_query(color)
                for color
                in point.get("colors", [])
            }

            # При точно съвпадение добавя
            # предмета към точните кандидати.
            if q in names:
                exact.append(point)
                continue

            # Оценка за частично съвпадение.
            score = 0

            for name in names:
                # Дава по-висока оценка,
                # когато единият текст съдържа другия.
                if (
                    name
                    and (
                        name in q
                        or q in name
                    )
                ):
                    score = max(
                        score,
                        min(
                            len(name),
                            len(q),
                        ),
                    )

                else:
                    # Оценява броя на общите думи.
                    score = max(
                        score,
                        len(
                            set(q.split("_"))
                            & set(name.split("_"))
                        )
                        * 3,
                    )

            # Споменатият цвят увеличава или намалява
            # оценката на кандидата.
            if requested_colors:
                score += (
                    20
                    if colors & requested_colors
                    else -20
                )

            # Запазва само положителните съвпадения.
            if score > 0:
                fuzzy.append(
                    (
                        score,
                        point,
                    )
                )

        # Точните съвпадения имат приоритет.
        candidates = exact

        # При липса на точни съвпадения
        # избира кандидатите с най-висока оценка.
        if (
            not candidates
            and fuzzy
        ):
            best_score = max(
                score
                for score, _ in fuzzy
            )

            candidates = [
                point
                for score, point in fuzzy
                if score == best_score
            ]

        # Специална заявка за най-близкия предмет.
        if (
            not candidates
            and q in {
                "nearest_object",
                "nearest",
                "object",
                "предмет",
                "обект",
            }
        ):
            candidates = [
                point
                for point in points
                if (
                    not point.get("safety_blocker")
                    and not point.get(
                        "virtual_point"
                    )
                )
            ]

            if candidates:
                return min(
                    candidates,
                    key=lambda point: math.hypot(
                        float(
                            point.get(
                                "x",
                                0.0,
                            )
                        ),
                        float(
                            point.get(
                                "y",
                                0.0,
                            )
                        ),
                    ),
                )

        # Специална заявка за най-близката
        # повърхност за поставяне.
        if (
            not candidates
            and q in {
                "nearest_surface",
                "nearest_destination",
                "placement_surface",
                "повърхност",
                "място_за_поставяне",
            }
        ):
            candidates = [
                point
                for point in points
                if point.get("role")
                in {
                    "placement_surface",
                    "reference_surface",
                    "support_surface",
                }
            ]

            if candidates:
                return min(
                    candidates,
                    key=lambda point: math.hypot(
                        float(
                            point.get(
                                "x",
                                0.0,
                            )
                        ),
                        float(
                            point.get(
                                "y",
                                0.0,
                            )
                        ),
                    ),
                )

        # При споменат цвят извършва
        # допълнително филтриране на кандидатите.
        if (
            requested_colors
            and candidates
        ):
            filtered = [
                point
                for point in candidates
                if (
                    {
                        self.normalize_query(color)
                        for color
                        in point.get("colors", [])
                    }
                    & requested_colors
                )
            ]

            if filtered:
                candidates = filtered

        # Няма намерена цел.
        if not candidates:
            return None

        # Точното съвпадение по устойчив instance_id
        # винаги има най-висок приоритет.
        instance_matches = [
            point
            for point in candidates
            if (
                q
                == self.normalize_query(
                    point.get("instance_id")
                )
            )
        ]

        if len(instance_matches) == 1:
            return instance_matches[0]

        # Проверява за числов селектор:
        # „номер 2“, „#2“, „cube 2“ и други.
        requested_index: Optional[int] = None

        number_match = re.search(
            r"(?:номер|number|no|#)?[_\s-]*(\d+)(?:\D|$)",
            original_query,
            flags=re.IGNORECASE,
        )

        if number_match:
            requested_index = int(
                number_match.group(1)
            )

        else:
            # Проверява за текстови числителни.
            for word, index in (
                ObjectInstanceRegistry
                .NUMBER_WORDS
                .items()
            ):
                if word in original_query:
                    requested_index = index
                    break

        # Избира предмета със съответния
        # пореден номер.
        if requested_index is not None:
            indexed = [
                point
                for point in candidates
                if int(
                    point.get(
                        "instance_index",
                        -1,
                    )
                )
                == requested_index
            ]

            if len(indexed) == 1:
                return indexed[0]

        # Проверява пространствените псевдоними,
        # създадени от регистъра на екземплярите.
        selector_aliases = [
            point
            for point in candidates
            if any(
                (
                    self.normalize_query(alias)
                    in q
                )
                or (
                    q
                    in self.normalize_query(alias)
                )
                for alias
                in point.get(
                    "spatial_aliases",
                    [],
                )
            )
        ]

        if len(selector_aliases) == 1:
            return selector_aliases[0]

        # Избира най-левия предмет
        # според най-голямата Y координата.
        if any(
            word in original_query
            for word in (
                "левия",
                "ляв",
                "отляво",
                "left",
            )
        ):
            return max(
                candidates,
                key=lambda point: float(
                    point.get(
                        "y",
                        0.0,
                    )
                ),
            )

        # Избира най-десния предмет
        # според най-малката Y координата.
        if any(
            word in original_query
            for word in (
                "десния",
                "десен",
                "дясно",
                "отдясно",
                "right",
            )
        ):
            return min(
                candidates,
                key=lambda point: float(
                    point.get(
                        "y",
                        0.0,
                    )
                ),
            )

        # Избира най-близкия предмет
        # до основата на манипулатора.
        if any(
            word in original_query
            for word in (
                "най-близ",
                "близкия",
                "nearest",
            )
        ):
            return min(
                candidates,
                key=lambda point: math.hypot(
                    float(
                        point.get(
                            "x",
                            0.0,
                        )
                    ),
                    float(
                        point.get(
                            "y",
                            0.0,
                        )
                    ),
                ),
            )

        # Избира най-далечния предмет
        # от основата на манипулатора.
        if any(
            word in original_query
            for word in (
                "най-далеч",
                "далечния",
                "farthest",
            )
        ):
            return max(
                candidates,
                key=lambda point: math.hypot(
                    float(
                        point.get(
                            "x",
                            0.0,
                        )
                    ),
                    float(
                        point.get(
                            "y",
                            0.0,
                        )
                    ),
                ),
            )

        # Не избира автоматично между няколко
        # еднакви физически екземпляра.
        unique_instances = {
            str(
                point.get("instance_id")
                or point.get("point")
                or id(point)
            )
            for point in candidates
        }

        # При повече от един уникален кандидат
        # изисква уточнение от потребителя.
        if len(unique_instances) > 1:
            raise AmbiguousTargetError(
                original_query,
                candidates,
            )

        # Връща единствения останал кандидат.
        return candidates[0]

    @staticmethod
    def _finite_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Преобразува стойност до крайно число
        с плаваща запетая.
        """

        try:
            result = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return float(default)

        # При NaN или безкрайна стойност
        # използва резервната стойност.
        return (
            result
            if math.isfinite(result)
            else float(default)
        )

    @classmethod
    def object_geometry(
        cls,
        point: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Нормализира пълната измерена геометрия на предмета.

        Позициите за горна повърхност и хващане,
        предоставени от сканиращия модул, имат приоритет.
        Измерените размери и центърът се използват
        като проверка и резервен вариант.

        Всички действия на планиращия агент използват
        тази обща геометрична структура.
        """

        # Предпочита вложения център, след това
        # позицията, а накрая директните координати.
        center_row = (
            point.get("center")
            or point.get("position")
            or point
        )

        # Нормализира координатата X на центъра.
        center_x = cls._finite_float(
            center_row.get(
                "x",
                point.get("x", 0.0),
            )
        )

        # Нормализира координатата Y на центъра.
        center_y = cls._finite_float(
            center_row.get(
                "y",
                point.get("y", 0.0),
            )
        )

        # Нормализира координатата Z на центъра.
        center_z = cls._finite_float(
            center_row.get(
                "z",
                point.get("z", 0.0),
            )
        )

        # Обработени размери на предмета.
        dimensions = (
            point.get("dimensions")
            or {}
        )

        # Необработени измерени размери.
        raw_dimensions = (
            point.get("raw_dimensions")
            or {}
        )

        # Ширина по оста X.
        width = max(
            0.0,
            cls._finite_float(
                dimensions.get(
                    "x",
                    raw_dimensions.get(
                        "x",
                        point.get(
                            "width",
                            0.0,
                        ),
                    ),
                )
            ),
        )

        # Дълбочина по оста Y.
        depth = max(
            0.0,
            cls._finite_float(
                dimensions.get(
                    "y",
                    raw_dimensions.get(
                        "y",
                        point.get(
                            "depth",
                            0.0,
                        ),
                    ),
                )
            ),
        )

        # Височина по оста Z.
        height = max(
            0.0,
            cls._finite_float(
                dimensions.get(
                    "z",
                    raw_dimensions.get(
                        "z",
                        point.get(
                            "height",
                            0.0,
                        ),
                    ),
                )
            ),
        )

        # Теоретична горна повърхност
        # според центъра и височината.
        geometric_top_z = (
            center_z
            + height / 2.0
        )

        # Теоретична долна повърхност.
        geometric_bottom_z = (
            center_z
            - height / 2.0
        )

        # Възможни измервания на горната повърхност.
        top_candidates = [
            geometric_top_z
        ]

        top_row = point.get(
            "top_position"
        )

        # Добавя горната позиция от скенера.
        if isinstance(
            top_row,
            Mapping,
        ):
            top_candidates.append(
                cls._finite_float(
                    top_row.get("z"),
                    geometric_top_z,
                )
            )

        # Добавя директното поле top_z.
        if point.get("top_z") is not None:
            top_candidates.append(
                cls._finite_float(
                    point.get("top_z"),
                    geometric_top_z,
                )
            )

        # Избира най-високата измерена
        # или изчислена горна граница.
        top_z = max(
            top_candidates
        )

        # Възможни измервания на долната повърхност.
        bottom_candidates = [
            geometric_bottom_z
        ]

        bottom_row = point.get(
            "bottom_position"
        )

        # Добавя долната позиция от скенера.
        if isinstance(
            bottom_row,
            Mapping,
        ):
            bottom_candidates.append(
                cls._finite_float(
                    bottom_row.get("z"),
                    geometric_bottom_z,
                )
            )

        # Добавя директното поле bottom_z.
        if point.get("bottom_z") is not None:
            bottom_candidates.append(
                cls._finite_float(
                    point.get("bottom_z"),
                    geometric_bottom_z,
                )
            )

        # Избира най-ниската измерена
        # или изчислена долна граница.
        bottom_z = min(
            bottom_candidates
        )

        # Измерена височина между
        # горната и долната граница.
        measured_span = max(
            0.0,
            top_z - bottom_z,
        )

        # При липсваща височина използва
        # измерения вертикален диапазон.
        if (
            height <= 0.0
            and measured_span > 0.0
        ):
            height = measured_span

        # Ако измереният диапазон е по-голям,
        # увеличава височината до него.
        elif measured_span > height:
            height = measured_span

        # Позиция за хващане от скенера.
        grasp_row = point.get(
            "grasp_position"
        )

        if isinstance(
            grasp_row,
            Mapping,
        ):
            raw_grasp = {
                "x": cls._finite_float(
                    grasp_row.get("x"),
                    center_x,
                ),
                "y": cls._finite_float(
                    grasp_row.get("y"),
                    center_y,
                ),
                "z": cls._finite_float(
                    grasp_row.get("z"),
                    center_z,
                ),
            }

        else:
            # При липсваща позиция за хващане
            # използва геометричния център.
            raw_grasp = {
                "x": center_x,
                "y": center_y,
                "z": center_z,
            }

        # Скенерът може да постави grasp_position.z
        # малко над или под действителния център.
        #
        # Запазва измерените X и Y, но за реалното
        # хващане използва Z координатата на центъра.
        grasp = {
            "x": raw_grasp["x"],
            "y": raw_grasp["y"],
            "z": center_z,
        }

        # Незадължителна позиция за подход.
        approach_row = point.get(
            "approach_position"
        )

        approach = None

        if isinstance(
            approach_row,
            Mapping,
        ):
            approach = {
                "x": cls._finite_float(
                    approach_row.get("x"),
                    center_x,
                ),
                "y": cls._finite_float(
                    approach_row.get("y"),
                    center_y,
                ),
                "z": cls._finite_float(
                    approach_row.get("z"),
                    top_z,
                ),
            }

        # Приблизителен хоризонтален радиус,
        # използван при проверка на коридора.
        radius_xy = max(
            width,
            depth,
        ) / 2.0

        # Връща общата нормализирана геометрия.
        return {
            # Център на предмета.
            "center": {
                "x": center_x,
                "y": center_y,
                "z": center_z,
            },

            # Размери по трите оси.
            "dimensions": {
                "x": width,
                "y": depth,
                "z": height,
            },

            # Нормализирана височина.
            "height": height,

            # Най-висока точка.
            "top_z": top_z,

            # Най-ниска точка.
            "bottom_z": bottom_z,

            # Позиция на горната повърхност.
            "top": {
                "x": center_x,
                "y": center_y,
                "z": top_z,
            },

            # Позиция на долната повърхност.
            "bottom": {
                "x": center_x,
                "y": center_y,
                "z": bottom_z,
            },

            # Реално използвана позиция за хващане.
            "grasp": grasp,

            # Оригинална позиция за хващане от скенера.
            "raw_grasp": raw_grasp,

            # Източник на Z координатата за хващане.
            "grasp_z_source": "center.z",

            # Отместване между центъра на предмета
            # и позицията на хващане.
            "grasp_offset": {
                "x": grasp["x"] - center_x,
                "y": grasp["y"] - center_y,
                "z": 0.0,
            },

            # Незадължителна позиция за подход.
            "approach": approach,

            # Хоризонтален радиус на предмета.
            "radius_xy": radius_xy,

            # Описание на източника на височината.
            "height_source": (
                "measured_dimensions_and_top"
                if height > 0.0
                else "center_only_fallback"
            ),
        }

    @classmethod
    def position_for_mode(
        cls,
        point: Dict[str, Any],
        mode: str,
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """
        Връща декартова позиция според
        избрания режим на движение.
        """

        # Нормализира режима.
        normalized_mode = str(
            mode
        ).strip().lower()

        # Получава общата геометрия.
        geometry = cls.object_geometry(
            point
        )

        # Център на предмета.
        if normalized_mode in {
            "center",
            "object_center",
        }:
            row = geometry["center"]

        # Горна повърхност.
        elif normalized_mode == "top":
            row = geometry["top"]

        # Позиция за хващане.
        elif normalized_mode == "grasp":
            row = geometry["grasp"]

        # Позиция за подход.
        elif normalized_mode == "approach":
            row = (
                geometry["approach"]
                or geometry["top"]
            )

        else:
            # Съответствие между режим
            # и конфигурационно поле.
            key_map = {
                "place": "place_position",
                "destination": (
                    "destination_position"
                ),
            }

            key = key_map.get(
                normalized_mode
            )

            configured = (
                point.get(key)
                if key
                else None
            )

            # Използва конфигурираната позиция,
            # когато е налична.
            if isinstance(
                configured,
                Mapping,
            ):
                row = configured

            else:
                # Резервната позиция е
                # горната повърхност.
                row = geometry["top"]

        # Връща нормализираните координати.
        return (
            cls._finite_float(
                row.get("x")
            ),
            cls._finite_float(
                row.get("y")
            ),
            cls._finite_float(
                row.get("z")
            ),
        )

    @classmethod
    def above_object_position(
        cls,
        point: Dict[str, Any],
        clearance: float,
        minimum_clearance: float,
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """
        Връща позиция над измерената горна
        повърхност, а не над центъра на предмета.
        """

        # Получава геометрията на предмета.
        geometry = cls.object_geometry(
            point
        )

        # Избира по-голямата стойност
        # между желания и минималния просвет.
        required_clearance = max(
            float(clearance),
            float(minimum_clearance),
        )

        # Позицията по Z е над измерената
        # горна повърхност.
        target_z = (
            geometry["top_z"]
            + required_clearance
        )

        # Проверява конфигурираната
        # позиция за подход.
        configured = geometry.get(
            "approach"
        )

        if isinstance(
            configured,
            Mapping,
        ):
            # Не допуска позицията да бъде
            # по-ниска от конфигурираната.
            target_z = max(
                target_z,
                cls._finite_float(
                    configured.get("z"),
                    target_z,
                ),
            )

        return (
            geometry["center"]["x"],
            geometry["center"]["y"],
            target_z,
        )

    @classmethod
    def held_object_approach_position(
        cls,
        held_object: Mapping[str, Any],
        destination: Mapping[str, Any],
        clearance: float,
        minimum_clearance: float,
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """
        Изчислява позиция на хващача, при която
        долната част на държания предмет остава
        над измерената горна повърхност на целта.
        """

        # Геометрия на държания предмет.
        held_geometry = cls.object_geometry(
            held_object
        )

        # Геометрия на крайната цел.
        destination_geometry = (
            cls.object_geometry(
                destination
            )
        )

        # Избира необходимия безопасен просвет.
        required_clearance = max(
            float(clearance),
            float(minimum_clearance),
        )

        # Вертикално отместване между центъра
        # на предмета и хващача.
        grasp_offset_z = held_geometry[
            "grasp_offset"
        ]["z"]

        # Изчислява необходимата височина
        # на хващача.
        gripper_z = (
            destination_geometry["top_z"]
            + required_clearance
            + held_geometry["height"] / 2.0
            + grasp_offset_z
        )

        # Проверява конфигурираната позиция
        # за подход към крайната цел.
        configured = destination_geometry.get(
            "approach"
        )

        if isinstance(
            configured,
            Mapping,
        ):
            # Не допуска позицията да бъде
            # по-ниска от конфигурираната.
            gripper_z = max(
                gripper_z,
                cls._finite_float(
                    configured.get("z"),
                    gripper_z,
                ),
            )

        return (
            destination_geometry["center"]["x"],
            destination_geometry["center"]["y"],
            gripper_z,
        )

    @classmethod
    def placement_targets(
        cls,
        held_object: Mapping[str, Any],
        destination: Mapping[str, Any],
        place_clearance: float,
        approach_height: float,
        minimum_clearance: float,
    ) -> Dict[
        str,
        Tuple[float, float, float],
    ]:
        """
        Изчислява крайния център на държания предмет,
        позицията на хващача за поставяне и безопасната
        позиция за подход.

        Използват се измерените височини
        и на двата предмета.
        """

        # Геометрия на държания предмет.
        held_geometry = cls.object_geometry(
            held_object
        )

        # Геометрия на крайната цел.
        destination_geometry = (
            cls.object_geometry(
                destination
            )
        )

        # Отместване между центъра на предмета
        # и хващача.
        grasp_offset = held_geometry[
            "grasp_offset"
        ]

        # Краен център на предмета върху целта.
        object_center = (
            destination_geometry["center"]["x"],
            destination_geometry["center"]["y"],
            (
                destination_geometry["top_z"]
                + held_geometry["height"] / 2.0
                + max(
                    0.0,
                    float(place_clearance),
                )
            ),
        )

        # Позиция на хващача при поставяне.
        gripper_place = (
            object_center[0]
            + grasp_offset["x"],
            object_center[1]
            + grasp_offset["y"],
            object_center[2]
            + grasp_offset["z"],
        )

        # Изчислява безопасната височина
        # за подход над позицията за поставяне.
        safe_approach_z = max(
            (
                gripper_place[2]
                + max(
                    0.0,
                    float(approach_height),
                )
            ),
            (
                destination_geometry["top_z"]
                + max(
                    0.0,
                    float(minimum_clearance),
                )
                + held_geometry["height"] / 2.0
                + grasp_offset["z"]
            ),
        )

        # Позиция на хващача над крайната цел.
        gripper_approach = (
            gripper_place[0],
            gripper_place[1],
            safe_approach_z,
        )

        return {
            "object_center": object_center,
            "gripper_place": gripper_place,
            "gripper_approach": gripper_approach,
        }

    def load_offsets(
        self,
        target: str,
    ) -> Dict[str, float]:
        """
        Зарежда натрупаните корекции по X, Y и Z
        за конкретна цел.
        """

        try:
            # Зарежда всички корекции от Redis.
            all_offsets = json.loads(
                self.redis.get(
                    self.offsets_key
                )
                or "{}"
            )

        except Exception:
            # При Redis или JSON грешка
            # използва празен речник.
            all_offsets = {}

        # Получава корекциите за конкретната цел.
        row = all_offsets.get(
            target,
            {},
        )

        # Връща нормализирани числови стойности.
        return {
            "x": float(
                row.get(
                    "x",
                    0.0,
                )
            ),
            "y": float(
                row.get(
                    "y",
                    0.0,
                )
            ),
            "z": float(
                row.get(
                    "z",
                    0.0,
                )
            ),
        }

    def update_offsets(
        self,
        target: str,
        dx: float,
        dy: float,
        dz: float,
    ) -> Dict[str, float]:
        """
        Добавя нова корекция към натрупаните
        отмествания на конкретна цел.
        """

        try:
            # Зарежда текущите корекции.
            all_offsets = json.loads(
                self.redis.get(
                    self.offsets_key
                )
                or "{}"
            )

        except Exception:
            # При грешка започва
            # с празен документ.
            all_offsets = {}

        # Получава текущите корекции на целта.
        current = all_offsets.get(
            target,
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
        )

        # Добавя новите стойности
        # към съществуващите корекции.
        current = {
            "x": (
                float(
                    current.get(
                        "x",
                        0.0,
                    )
                )
                + dx
            ),
            "y": (
                float(
                    current.get(
                        "y",
                        0.0,
                    )
                )
                + dy
            ),
            "z": (
                float(
                    current.get(
                        "z",
                        0.0,
                    )
                )
                + dz
            ),
        }

        # Обновява документа.
        all_offsets[target] = current

        # Записва всички корекции в Redis.
        self.redis.set(
            self.offsets_key,
            json.dumps(
                all_offsets
            ),
        )

        # Връща новите натрупани стойности.
        return current