#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за:
# - JSON сериализация и десериализация;
# - обработване на текст чрез регулярни изрази;
# - времеви стойности;
# - типови анотации.
import json
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Помощна функция за прочитане на числова
# конфигурационна стойност от променливите на средата.
from .common import env_float


class ObjectTracebackStore:
    """
    Persistent movement history for objects.

    The[object Object],[object Object],[object Object],[object Object],[object Object],[object Object],[object Object],[object Object],[object Object],[object Object],[object Object] first source point is copied directly from workspace_scan_agent and is
    kept as the immutable origin. Every successful pick-and-place appends a new
    location. A return operation pops one location only after the physical and
    RViz sequence finishes successfully.
    """

    # Класът съхранява постоянна история
    # на преместванията на отделните предмети.
    #
    # Първата позиция от workspace_scan_agent се запазва
    # като неизменен произход. Всяко успешно преместване
    # добавя ново местоположение към стека.
    #
    # При връщане последната позиция се премахва
    # само след успешно приключване на движението.

    # Полета, съдържащи пространствени позиции,
    # които трябва да бъдат преместени заедно
    # с центъра на предмета.
    POSITION_FIELDS = (
        "center",
        "position",
        "top_position",
        "grasp_position",
        "approach_position",
        "place_position",
        "destination_position",
    )

    def __init__(
        self,
        redis_client: Any,
        history_key: str,
        motion_points_key: str,
    ) -> None:
        """
        Инициализира хранилището за историята
        на преместванията.
        """

        # Redis клиент за постоянно съхранение.
        self.redis = redis_client

        # Redis ключ за документа с историята.
        self.history_key = history_key

        # Redis ключ за актуалните точки
        # от работното пространство.
        self.motion_points_key = motion_points_key

        # Максимален брой местоположения,
        # съхранявани за един предмет.
        #
        # Минималната стойност е две позиции:
        # произход и поне едно следващо местоположение.
        self.max_locations = max(
            2,
            int(
                env_float(
                    "TRACEBACK_MAX_LOCATIONS_PER_OBJECT",
                    30,
                )
            ),
        )

    @staticmethod
    def _copy(value: Any) -> Any:
        """
        Създава дълбоко копие чрез JSON
        сериализация и десериализация.
        """

        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
            )
        )

    @staticmethod
    def _normalize(value: Any) -> str:
        """
        Нормализира текстова стойност за сравнение
        и използване като устойчив ключ.
        """

        # Преобразува стойността до малки букви
        # и премахва началните и крайните интервали.
        text = str(
            value or ""
        ).strip().lower()

        # Премахва неподдържаните специални символи.
        text = re.sub(
            r"[^0-9a-zа-яё_\- ]+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Заменя интервалите и тиретата
        # с долни черти.
        text = re.sub(
            r"[\s\-]+",
            "_",
            text,
        )

        # Обединява последователните долни черти.
        return re.sub(
            r"_+",
            "_",
            text,
        ).strip("_")

    @staticmethod
    def _safe_id(value: Any) -> str:
        """
        Създава безопасен текстов идентификатор.
        """

        # Нормализира подадената стойност.
        text = ObjectTracebackStore._normalize(
            value
        )

        # При празен резултат използва общото име object.
        return text or "object"

    @staticmethod
    def _point_id(
        point: Mapping[str, Any],
    ) -> str:
        """
        Връща най-подходящия устойчив
        идентификатор на предмета.
        """

        return str(
            point.get("instance_id")
            or point.get("point")
            or point.get("safe_label")
            or point.get("label")
            or "object"
        )

    @staticmethod
    def _display_name(
        point: Mapping[str, Any],
    ) -> str:
        """
        Връща потребителското име
        на предмета за визуализация.
        """

        return str(
            point.get("display_name_bg")
            or point.get("label")
            or point.get("point")
            or "обект"
        )

    @staticmethod
    def _center(
        point: Mapping[str, Any],
    ) -> Tuple[float, float, float]:
        """
        Извлича централната позиция на предмета.
        """

        # Предпочита полето center, след това position,
        # а при липса използва директните координати.
        position = (
            point.get("center")
            or point.get("position")
            or point
        )

        return (
            float(
                position.get(
                    "x",
                    point.get("x", 0.0),
                )
            ),
            float(
                position.get(
                    "y",
                    point.get("y", 0.0),
                )
            ),
            float(
                position.get(
                    "z",
                    point.get("z", 0.0),
                )
            ),
        )

    def load(self) -> Dict[str, Any]:
        """
        Зарежда документа с историята от Redis.
        """

        try:
            # Прочита JSON документа от Redis.
            raw = self.redis.get(
                self.history_key
            )

            # При липсваща стойност използва празен речник.
            document = (
                json.loads(raw)
                if raw
                else {}
            )

        except Exception:
            # При Redis или JSON грешка
            # използва празен документ.
            document = {}

        # Гарантира речникова структура.
        if not isinstance(document, dict):
            document = {}

        # Добавя задължителните полета,
        # когато липсват в по-стар документ.
        document.setdefault("version", 1)
        document.setdefault("objects", {})

        return document

    def save(
        self,
        document: Dict[str, Any],
    ) -> None:
        """
        Записва документа с историята в Redis.
        """

        # Добавя време на последното обновяване.
        document["updated_at"] = time.time()

        # Сериализира и записва документа.
        self.redis.set(
            self.history_key,
            json.dumps(
                document,
                ensure_ascii=False,
            ),
        )

    def _identity_names(
        self,
        entry: Mapping[str, Any],
    ) -> List[str]:
        """
        Извлича всички имена и псевдоними,
        чрез които може да бъде намерен предметът.
        """

        # Получава запазената идентичност
        # на предмета.
        identity = (
            entry.get("identity")
            or {}
        )

        # Събира всички възможни имена.
        names = [
            entry.get("object_id"),
            identity.get("instance_id"),
            identity.get("point"),
            identity.get("label"),
            identity.get("safe_label"),
            identity.get("display_name_bg"),
            *(
                identity.get("aliases")
                or []
            ),
        ]

        # Нормализира и премахва празните стойности.
        return [
            self._normalize(name)
            for name in names
            if self._normalize(name)
        ]

    def resolve_object_id(
        self,
        query: str,
    ) -> Optional[str]:
        """
        Намира устойчив идентификатор на предмет
        според име, псевдоним или частично съвпадение.
        """

        # Зарежда документа с историята.
        document = self.load()

        # Нормализира потребителската заявка.
        q = self._normalize(query)

        # Празната заявка не може да бъде разрешена.
        if not q:
            return None

        # Точни съвпадения.
        exact: List[str] = []

        # Частични съвпадения и техните оценки.
        fuzzy: List[
            Tuple[int, str]
        ] = []

        # Проверява всички регистрирани предмети.
        for object_id, entry in (
            document["objects"].items()
        ):
            # Получава всички имена на предмета.
            names = self._identity_names(
                entry
            )

            # Точното съвпадение има приоритет.
            if q in names:
                exact.append(object_id)
                continue

            # Оценка на частичното съвпадение.
            score = 0

            # Думи от потребителската заявка.
            q_tokens = set(
                q.split("_")
            )

            for name in names:
                # По-висока оценка, когато единият
                # текст съдържа другия.
                if (
                    q in name
                    or name in q
                ):
                    score = max(
                        score,
                        min(
                            len(q),
                            len(name),
                        ),
                    )

                else:
                    # Оценка според броя общи думи.
                    score = max(
                        score,
                        len(
                            q_tokens
                            & set(
                                name.split("_")
                            )
                        )
                        * 3,
                    )

            # Запазва само положителните съвпадения.
            if score > 0:
                fuzzy.append(
                    (
                        score,
                        object_id,
                    )
                )

        # Връща първото точно съвпадение.
        if exact:
            return exact[0]

        # При частични съвпадения връща
        # предмета с най-висока оценка.
        if fuzzy:
            fuzzy.sort(
                key=lambda row: row[0],
                reverse=True,
            )
            return fuzzy[0][1]

        return None

    def relocate_point(
        self,
        point: Mapping[str, Any],
        new_center: Tuple[float, float, float],
        *,
        object_id: Optional[str] = None,
        traceback_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Премества копие на предмета до нов център,
        като запазва относителните позиции
        на всички негови геометрични полета.
        """

        # Създава дълбоко копие на предмета.
        moved = self._copy(
            dict(point)
        )

        # Получава стария център.
        old_center = self._center(
            point
        )

        # Изчислява преместването по трите оси.
        delta = (
            float(new_center[0])
            - old_center[0],
            float(new_center[1])
            - old_center[1],
            float(new_center[2])
            - old_center[2],
        )

        # Премества всяко вложено позиционно поле.
        for field in self.POSITION_FIELDS:
            row = moved.get(field)

            if (
                isinstance(row, dict)
                and all(
                    axis in row
                    for axis in (
                        "x",
                        "y",
                        "z",
                    )
                )
            ):
                row["x"] = (
                    float(row["x"])
                    + delta[0]
                )
                row["y"] = (
                    float(row["y"])
                    + delta[1]
                )
                row["z"] = (
                    float(row["z"])
                    + delta[2]
                )

        # Актуализира директните координати.
        moved["x"] = float(
            new_center[0]
        )
        moved["y"] = float(
            new_center[1]
        )
        moved["z"] = float(
            new_center[2]
        )

        # Актуализира центъра.
        moved["center"] = {
            "x": float(new_center[0]),
            "y": float(new_center[1]),
            "z": float(new_center[2]),
        }

        # Отбелязва, че позицията е създадена
        # от историята на планиращия агент.
        moved[
            "location_source"
        ] = "planning_agent_traceback"

        # Отбелязва предмета като проследяван.
        moved["tracked_object"] = True

        # Добавя устойчивия идентификатор,
        # когато е предоставен.
        if object_id is not None:
            moved[
                "traceback_object_id"
            ] = object_id

        # Добавя дълбочината в историята,
        # когато е предоставена.
        if traceback_depth is not None:
            moved[
                "traceback_depth"
            ] = traceback_depth

        return moved

    def _origin_location(
        self,
        source: Mapping[str, Any],
        object_id: str,
    ) -> Dict[str, Any]:
        """
        Създава неизменната начална позиция
        на предмета от данните на сканиращия агент.
        """

        # Получава центъра на първоначалната точка.
        center = self._center(
            source
        )

        return {
            # Уникален идентификатор на началната позиция.
            "location_id": (
                f"{object_id}:origin"
            ),

            # Показва, че позицията идва
            # от първоначалното сканиране.
            "kind": "workspace_scan_origin",

            # Център на произхода.
            "center": {
                "x": center[0],
                "y": center[1],
                "z": center[2],
            },

            # Пълно копие на точката
            # с добавени данни за историята.
            "point": self.relocate_point(
                source,
                center,
                object_id=object_id,
                traceback_depth=0,
            ),

            # Началната позиция няма предходна цел.
            "destination_ref": None,

            # Време на създаване.
            "timestamp": time.time(),
        }

    def ensure_object(
        self,
        source: Mapping[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        str,
        Dict[str, Any],
    ]:
        """
        Гарантира, че предметът има запис
        в историята и начална позиция.
        """

        # Зарежда текущия документ.
        document = self.load()

        # Получава устойчив идентификатор
        # на предмета.
        object_id = self._point_id(
            source
        )

        # Търси съществуващ запис.
        entry = document[
            "objects"
        ].get(object_id)

        # Създава нов запис, когато предметът
        # все още не присъства в историята.
        if not isinstance(entry, dict):
            entry = {
                # Устойчив идентификатор.
                "object_id": object_id,

                # Запазена семантична идентичност.
                "identity": {
                    "instance_id": source.get(
                        "instance_id"
                    ),
                    "instance_index": source.get(
                        "instance_index"
                    ),
                    "semantic_key": source.get(
                        "semantic_key"
                    ),
                    "point": source.get("point"),
                    "label": source.get("label"),
                    "safe_label": source.get(
                        "safe_label"
                    ),
                    "display_name_bg": source.get(
                        "display_name_bg"
                    ),
                    "aliases": list(
                        source.get("aliases")
                        or []
                    ),
                    "colors": list(
                        source.get("colors")
                        or []
                    ),
                    "role": source.get("role"),

                    # Създава независимо копие
                    # на размерите.
                    "dimensions": self._copy(
                        source.get("dimensions")
                        or {}
                    ),
                },

                # Първата позиция е неизменният произход.
                "locations": [
                    self._origin_location(
                        source,
                        object_id,
                    )
                ],

                # Списък с всички операции
                # по преместване и връщане.
                "moves": [],

                # Време на създаване на записа.
                "created_at": time.time(),
            }

            # Добавя предмета към документа.
            document["objects"][
                object_id
            ] = entry

            # Записва новия документ.
            self.save(document)

        # Гарантира наличието на основните списъци.
        entry.setdefault(
            "locations",
            [],
        )
        entry.setdefault(
            "moves",
            [],
        )

        # Ако записът няма позиции,
        # възстановява началната позиция.
        if not entry["locations"]:
            entry["locations"].append(
                self._origin_location(
                    source,
                    object_id,
                )
            )
            self.save(document)

        return (
            document,
            object_id,
            entry,
        )

    def get_return_plan(
        self,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Създава план за връщане на предмета
        към предишното му местоположение.
        """

        # Определя устойчивия идентификатор.
        object_id = self.resolve_object_id(
            query
        )

        # Няма предмет с такава история.
        if object_id is None:
            return None

        # Зарежда актуалния документ.
        document = self.load()

        # Получава записа за предмета.
        entry = document[
            "objects"
        ].get(object_id)

        if not isinstance(entry, dict):
            return None

        # Получава стека с местоположения.
        locations = (
            entry.get("locations")
            or []
        )

        # При една или нула позиции
        # предметът се счита за намиращ се в произхода.
        if len(locations) < 2:
            return {
                "object_id": object_id,
                "at_origin": True,
                "depth": 0,
                "entry": self._copy(entry),
            }

        # Последната позиция е текущата.
        current = locations[-1]

        # Предпоследната позиция е целта за връщане.
        previous = locations[-2]

        return {
            "object_id": object_id,
            "at_origin": False,

            # Брой премествания от произхода.
            "depth": len(locations) - 1,

            # Текущо местоположение.
            "current": self._copy(
                current
            ),

            # Предишно местоположение.
            "previous": self._copy(
                previous
            ),

            # Пълно копие на записа.
            "entry": self._copy(entry),
        }

    def record_move(
        self,
        source: Mapping[str, Any],
        destination: Mapping[str, Any],
        placed_center: Tuple[float, float, float],
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Записва успешно преместване на предмет
        към ново местоположение.
        """

        # Гарантира наличието на запис
        # и начална позиция за предмета.
        document, object_id, entry = (
            self.ensure_object(source)
        )

        # Зарежда документа повторно, защото
        # ensure_object може да е създал и записал нов документ.
        document = self.load()

        # Получава актуалния запис.
        entry = document[
            "objects"
        ][object_id]

        # Получава стека с позиции.
        locations = entry["locations"]

        # Последната позиция преди преместването.
        before = locations[-1]

        # Дълбочина на новата позиция.
        depth = len(locations)

        # Пореден номер на операцията.
        sequence_number = (
            len(
                entry.get("moves")
                or []
            )
            + 1
        )

        # Създава копие на предмета
        # на новата позиция.
        moved_point = self.relocate_point(
            source,
            placed_center,
            object_id=object_id,
            traceback_depth=depth,
        )

        # Идентификатор на целевия предмет
        # или повърхност.
        destination_ref = self._point_id(
            destination
        )

        # Създава новото местоположение.
        location = {
            "location_id": (
                f"{object_id}:location:"
                f"{sequence_number}"
            ),
            "kind": "placed",
            "center": {
                "x": float(
                    placed_center[0]
                ),
                "y": float(
                    placed_center[1]
                ),
                "z": float(
                    placed_center[2]
                ),
            },
            "point": moved_point,
            "destination_ref": destination_ref,
            "timestamp": time.time(),
        }

        # Добавя новата позиция към стека.
        locations.append(location)

        # Ограничава броя съхранявани позиции.
        if len(locations) > self.max_locations:
            # Началната позиция никога не се премахва.
            del locations[
                1:
                len(locations)
                - self.max_locations
                + 1
            ]

        # Добавя подробен запис за операцията.
        entry["moves"].append(
            {
                "kind": "pick_and_place",
                "request_id": request_id,
                "from_location_id": (
                    before.get("location_id")
                ),
                "to_location_id": (
                    location["location_id"]
                ),
                "from_center": before.get(
                    "center"
                ),
                "to_center": location[
                    "center"
                ],
                "destination_ref": (
                    destination_ref
                ),
                "timestamp": time.time(),
            }
        )

        # Обновява времето на предмета.
        entry["updated_at"] = time.time()

        # Записва историята.
        self.save(document)

        # Обновява документа с точки
        # и добавя виртуалните позиции от историята.
        merged = self.overlay_motion_points(
            self._load_motion_points_or_empty(),
            persist=True,
        )

        return {
            "object_id": object_id,

            # Текуща дълбочина в историята.
            "traceback_depth": (
                len(locations) - 1
            ),

            # Първоначален център.
            "origin": locations[0][
                "center"
            ],

            # Текущ център.
            "current": locations[-1][
                "center"
            ],

            # Брой виртуални точки
            # от историята за предмета.
            "trace_points": sum(
                1
                for point in merged.get(
                    "motion_points",
                    [],
                )
                if (
                    point.get(
                        "traceback_object_id"
                    )
                    == object_id
                    and point.get(
                        "traceback_point"
                    )
                )
            ),
        }

    def commit_return(
        self,
        object_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Потвърждава успешно връщане на предмета
        към предишната позиция.
        """

        # Зарежда историята.
        document = self.load()

        # Получава записа на предмета.
        entry = document[
            "objects"
        ].get(object_id)

        # Не допуска връщане на непознат предмет.
        if not isinstance(entry, dict):
            raise KeyError(
                f"No traceback entry for {object_id}"
            )

        # Получава стека с позиции.
        locations = (
            entry.get("locations")
            or []
        )

        # При липса на предишна позиция
        # предметът вече е в произхода.
        if len(locations) < 2:
            return {
                "object_id": object_id,
                "traceback_depth": 0,
                "at_origin": True,
                "current": (
                    locations[0]["center"]
                    if locations
                    else None
                ),
            }

        # Премахва последното местоположение.
        removed = locations.pop()

        # Новата текуща позиция е предишната
        # позиция от стека.
        current = locations[-1]

        # Добавя запис за операцията по връщане.
        entry.setdefault(
            "moves",
            [],
        ).append(
            {
                "kind": "return_previous",
                "request_id": request_id,
                "from_location_id": (
                    removed.get("location_id")
                ),
                "to_location_id": (
                    current.get("location_id")
                ),
                "from_center": removed.get(
                    "center"
                ),
                "to_center": current.get(
                    "center"
                ),
                "timestamp": time.time(),
            }
        )

        # Обновява времето на записа.
        entry["updated_at"] = time.time()

        # Записва историята.
        self.save(document)

        # Обновява виртуалните точки
        # в работното пространство.
        self.overlay_motion_points(
            self._load_motion_points_or_empty(),
            persist=True,
        )

        return {
            "object_id": object_id,
            "traceback_depth": (
                len(locations) - 1
            ),
            "at_origin": (
                len(locations) == 1
            ),
            "current": current.get(
                "center"
            ),
            "returned_from": removed.get(
                "center"
            ),
        }

    def _load_motion_points_or_empty(
        self,
    ) -> Dict[str, Any]:
        """
        Зарежда документа с точки
        или създава празен документ.
        """

        try:
            # Прочита актуалните точки от Redis.
            raw = self.redis.get(
                self.motion_points_key
            )

            document = (
                json.loads(raw)
                if raw
                else {}
            )

        except Exception:
            # При Redis или JSON грешка
            # използва празен документ.
            document = {}

        # Гарантира речникова структура.
        if not isinstance(document, dict):
            document = {}

        # Гарантира наличието на списък с точки.
        document.setdefault(
            "motion_points",
            [],
        )

        return document

    def _trace_point(
        self,
        entry: Mapping[str, Any],
        location: Mapping[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        """
        Създава виртуална точка за стара позиция
        на предмета.
        """

        # Устойчив идентификатор на предмета.
        object_id = str(
            entry.get("object_id")
            or "object"
        )

        # Запазена идентичност на предмета.
        identity = (
            entry.get("identity")
            or {}
        )

        # Име за показване.
        display_name = str(
            identity.get("display_name_bg")
            or identity.get("label")
            or object_id
        )

        # Създава копие на предмета
        # от конкретното местоположение.
        point = self._copy(
            location.get("point")
            or {}
        )

        # Уникално име на виртуалната точка.
        trace_name = (
            f"trace_"
            f"{self._safe_id(object_id)}_"
            f"{index}"
        )

        # Заменя името на точката.
        point["point"] = trace_name

        # Добавя английски диагностичен етикет.
        point["label"] = (
            f"previous location {index} "
            f"of {object_id}"
        )

        # Безопасен етикет.
        point["safe_label"] = trace_name

        # Име на български език.
        point["display_name_bg"] = (
            f"стара позиция {index} "
            f"на {display_name}"
        )

        # Определя ролята на виртуалната точка.
        point["role"] = "traceback_location"

        # Точката не представлява физически предмет.
        point["virtual_point"] = True

        # Отбелязва я като точка от историята.
        point["traceback_point"] = True

        # Свързва точката с предмета.
        point[
            "traceback_object_id"
        ] = object_id

        # Пореден индекс в историята.
        point["traceback_index"] = index

        # Идентификатор на местоположението.
        point[
            "traceback_location_id"
        ] = location.get("location_id")

        # Вид на местоположението.
        point["traceback_kind"] = (
            location.get("kind")
        )

        # Директно копие на центъра.
        point[
            "traceback_direct_center"
        ] = self._copy(
            location.get("center")
        )

        # Псевдоними за естественоезиково избиране.
        point["aliases"] = [
            (
                f"стара позиция {index} "
                f"на {display_name}"
            ),
            (
                f"предишна позиция {index} "
                f"на {display_name}"
            ),
            (
                f"точка {index} "
                f"на {display_name}"
            ),
            (
                f"trace {index} "
                f"{object_id}"
            ),
        ]

        # Виртуалната точка не блокира движенията.
        point["safety_blocker"] = False

        return point

    def overlay_motion_points(
        self,
        points_document: Mapping[str, Any],
        *,
        persist: bool,
    ) -> Dict[str, Any]:
        """
        Добавя текущите и предишните позиции
        от историята към документа с точки.
        """

        # Създава независимо копие
        # на подадения документ.
        document = self._copy(
            dict(points_document)
        )

        # Премахва вече добавените точки от историята,
        # за да не се натрупват дублирани записи.
        points = [
            point
            for point in document.get(
                "motion_points",
                [],
            )
            if not point.get(
                "traceback_point"
            )
        ]

        # Зарежда историята.
        history = self.load()

        # Обобщена информация за всички предмети.
        summary: Dict[str, Any] = {}

        # Обработва всеки проследяван предмет.
        for object_id, entry in (
            history["objects"].items()
        ):
            # Получава всички местоположения.
            locations = (
                entry.get("locations")
                or []
            )

            if not locations:
                continue

            # Създава копие на текущия предмет
            # на последната му позиция.
            current_point = self._copy(
                locations[-1].get("point")
                or {}
            )

            # Добавя дълбочината в историята.
            current_point[
                "traceback_depth"
            ] = len(locations) - 1

            # Отбелязва предмета като проследяван.
            current_point[
                "tracked_object"
            ] = True

            # Добавя устойчивия идентификатор.
            current_point[
                "traceback_object_id"
            ] = object_id

            # Показва дали съществуваща точка
            # е заменена с актуалната позиция.
            replaced = False

            # Търси същия предмет
            # сред актуалните точки.
            for index, point in enumerate(
                points
            ):
                if (
                    str(
                        point.get("instance_id")
                        or ""
                    )
                    == object_id
                    or str(
                        point.get("point")
                        or ""
                    )
                    == object_id
                ):
                    points[index] = current_point
                    replaced = True
                    break

            # Ако предметът не е открит
            # в актуалното сканиране, го добавя.
            if not replaced:
                points.append(
                    current_point
                )

            # Добавя всички предишни позиции
            # като виртуални точки.
            for index, location in enumerate(
                locations[:-1]
            ):
                points.append(
                    self._trace_point(
                        entry,
                        location,
                        index,
                    )
                )

            # Създава обобщение за предмета.
            summary[object_id] = {
                "traceback_depth": (
                    len(locations) - 1
                ),
                "origin": locations[0].get(
                    "center"
                ),
                "current": locations[-1].get(
                    "center"
                ),
                "saved_previous_locations": max(
                    0,
                    len(locations) - 1,
                ),
            }

        # Записва обновения списък с точки.
        document["motion_points"] = points

        # Добавя обобщението на историята.
        document[
            "object_traceback"
        ] = summary

        # Време на последното обновяване.
        document[
            "traceback_updated_at"
        ] = time.time()

        # По желание записва обновения документ в Redis.
        if persist:
            self.redis.set(
                self.motion_points_key,
                json.dumps(
                    document,
                    ensure_ascii=False,
                ),
            )

        return document

    def history_summary(
        self,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Връща обобщена история за всички предмети
        или само за конкретно избран предмет.
        """

        # Зарежда документа.
        document = self.load()

        # При подадена заявка определя
        # устойчивия идентификатор на предмета.
        object_id = (
            self.resolve_object_id(query)
            if query
            else None
        )

        # При заявка за непознат предмет
        # връща структуриран неуспешен резултат.
        if (
            query
            and object_id is None
        ):
            return {
                "ok": False,
                "state": (
                    "TRACEBACK_OBJECT_NOT_FOUND"
                ),
                "query": query,
                "objects": [],
            }

        # Избира само заявения предмет
        # или всички налични предмети.
        selected = (
            {
                object_id:
                document["objects"][object_id]
            }
            if object_id
            else document["objects"]
        )

        # Списък с обобщенията.
        objects = []

        for key, entry in selected.items():
            # Местоположения на предмета.
            locations = (
                entry.get("locations")
                or []
            )

            # Запазена идентичност.
            identity = (
                entry.get("identity")
                or {}
            )

            # Добавя обобщението на предмета.
            objects.append(
                {
                    "object_id": key,
                    "display_name_bg": (
                        identity.get(
                            "display_name_bg"
                        )
                    ),
                    "aliases": (
                        identity.get("aliases")
                        or []
                    ),
                    "traceback_depth": max(
                        0,
                        len(locations) - 1,
                    ),
                    "at_origin": (
                        len(locations) <= 1
                    ),

                    # Подробен списък
                    # на всички местоположения.
                    "locations": [
                        {
                            "index": index,
                            "location_id": (
                                location.get(
                                    "location_id"
                                )
                            ),
                            "kind": location.get(
                                "kind"
                            ),
                            "center": location.get(
                                "center"
                            ),
                            "destination_ref": (
                                location.get(
                                    "destination_ref"
                                )
                            ),
                            "timestamp": (
                                location.get(
                                    "timestamp"
                                )
                            ),

                            # Последната позиция
                            # е текущото местоположение.
                            "current": (
                                index
                                == len(locations) - 1
                            ),
                        }
                        for index, location
                        in enumerate(locations)
                    ],

                    # Общ брой записани операции.
                    "move_count": len(
                        entry.get("moves")
                        or []
                    ),
                }
            )

        return {
            "ok": True,
            "state": "TRACEBACK_HISTORY",
            "objects": objects,
        }

    def marker_locations(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Връща позициите, необходими
        за визуализация на историята в RViz.
        """

        # Зарежда историята.
        document = self.load()

        # Резултатен списък с маркерни позиции.
        output: List[
            Dict[str, Any]
        ] = []

        # Обработва всеки предмет.
        for object_id, entry in (
            document["objects"].items()
        ):
            # Получава стека с местоположения.
            locations = (
                entry.get("locations")
                or []
            )

            # Получава името за показване.
            display = self._display_name(
                entry.get("identity")
                or {}
            )

            # Създава запис за всяка позиция.
            for index, location in enumerate(
                locations
            ):
                output.append(
                    {
                        "object_id": object_id,
                        "display": display,
                        "index": index,

                        # Последната позиция е текущата.
                        "current": (
                            index
                            == len(locations) - 1
                        ),

                        # Позицията с индекс 0
                        # е първоначалният произход.
                        "origin": index == 0,

                        # Център на маркера.
                        "center": location.get(
                            "center"
                        ),
                    }
                )

        return output