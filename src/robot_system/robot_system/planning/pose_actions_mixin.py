#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартна библиотека за математически операции
# и типови анотации.
import math
from typing import Any, Dict, List, Mapping, Tuple

# Помощна функция за прочитане на числова
# конфигурационна стойност от променливите на средата.
from .common import env_float

# Безопасни диапазони на физическите сервомотори.
from .constants import SAFE_SERVO_RANGE

# Конфигурационни стойности за:
# - последно избраната цел;
# - продължителността на ръчно ставно движение;
# - толерантността при сравнение с именувана позиция.
from .settings import (
    LAST_TARGET_KEY,
    MANUAL_JOINT_MOVE_DURATION_SEC,
    POSE_MATCH_TOLERANCE_DEG,
)


class PoseActionsMixin:
    """
    Съдържа действията, свързани с именувани позиции,
    корекции на целеви точки и историята на движенията.

    Поддържаните операции включват:
    - проверка дали манипулаторът е в зададена позиция;
    - движение до начална позиция;
    - движение до позиция за сканиране;
    - добавяне на корекции към предмет;
    - извеждане на историята за преместване на предмети.
    """

    def handle_pose_status(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Проверява дали манипулаторът се намира
        в зададена именувана позиция.
        """

        # Получава името на позицията от заявката.
        #
        # При липсваща стойност се проверява
        # началната позиция home.
        pose_name = str(
            request.get("target")
            or "home"
        ).strip()

        # Разрешава проверка само за предварително
        # поддържаните именувани позиции.
        if pose_name not in {
            "home",
            "scan_pose",
        }:
            return self.result(
                request_id,
                False,
                "UNKNOWN_NAMED_POSE",
                target=pose_name,
            )

        # Сравнява текущите ставни стойности
        # със стойностите на именуваната позиция.
        at_pose, errors = self.is_at_named_pose(
            pose_name
        )

        # Връща резултата от проверката,
        # допустимата толерантност и грешката
        # за всяка отделна става.
        return self.result(
            request_id,
            True,
            "POSE_STATUS",
            target=pose_name,
            at_pose=at_pose,
            tolerance_deg=(
                POSE_MATCH_TOLERANCE_DEG
            ),
            joint_errors_deg=errors,
        )

    def handle_home(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Изпълнява движение до началната позиция home.
        """

        # Използва общия метод за изпълнение
        # на именувана позиция.
        return self.execute_named_pose(
            request_id=request_id,
            pose_name="home",

            # По подразбиране движението се изпълнява,
            # освен ако execute не е зададено като False.
            execute=bool(
                request.get(
                    "execute",
                    True,
                )
            ),
        )

    def handle_scan_pose(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Изпълнява движение до позицията за сканиране.
        """

        # Използва общия метод за именувани позиции,
        # като отбелязва, че движението е заявено
        # във връзка със сканиране на работното пространство.
        return self.execute_named_pose(
            request_id=request_id,
            pose_name="scan_pose",
            execute=bool(
                request.get(
                    "execute",
                    True,
                )
            ),
            requested_for_scan=True,
        )

    def handle_correction(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Обработва пространствена корекция по X, Y и Z.

        При посочен предмет корекцията се запазва
        към неговия устойчив идентификатор.

        При липсващ предмет заявката се преобразува
        в относително движение на манипулатора.
        """

        # Получава изрично зададената цел.
        explicit_target = str(
            request.get("target")
            or ""
        ).strip()

        # Получава корекциите по трите координатни оси.
        dx = float(
            request.get(
                "dx",
                0.0,
            )
        )
        dy = float(
            request.get(
                "dy",
                0.0,
            )
        )
        dz = float(
            request.get(
                "dz",
                0.0,
            )
        )

        # Когато няма изрично посочен предмет,
        # корекцията се интерпретира като относително
        # движение спрямо текущата позиция.
        if not explicit_target:
            # Създава копие на заявката,
            # за да не променя оригиналния речник.
            relative_request = dict(request)

            # Променя действието на move_relative.
            relative_request[
                "action"
            ] = "move_relative"

            # Запазва желаното поведение за изпълнение.
            relative_request[
                "execute"
            ] = bool(
                request.get(
                    "execute",
                    True,
                )
            )

            # Предава заявката обратно към основния
            # маршрутизатор за обработване.
            return self.handle_request(
                request_id,
                relative_request,
            )

        # Зарежда актуалните точки
        # от работното пространство.
        correction_doc = (
            self.targets.load_motion_points()
        )

        # Не може да приложи корекция към предмет,
        # когато няма налични точки.
        if correction_doc is None:
            return self.result(
                request_id,
                False,
                "NO_WORKSPACE_POINTS",
            )

        # Назначава устойчиви идентификатори
        # на физическите екземпляри.
        correction_doc = (
            self.instances.assign_document(
                correction_doc
            )
        )

        # Добавя виртуалните точки от историята
        # към актуалния документ.
        correction_doc = (
            self.traceback.overlay_motion_points(
                correction_doc,
                persist=True,
            )
        )

        # Определя конкретния предмет,
        # към който трябва да се приложи корекцията.
        correction_point = (
            self.targets.resolve_target(
                correction_doc.get(
                    "motion_points",
                    [],
                ),
                explicit_target,
            )
        )

        # Връща грешка, когато предметът
        # не може да бъде намерен.
        if correction_point is None:
            return self.result(
                request_id,
                False,
                "TARGET_NOT_FOUND",
                target=explicit_target,
            )

        # Създава устойчив ключ за предмета.
        #
        # Предпочита instance_id, след това името
        # на точката, а при липса използва
        # първоначално подадената цел.
        target = str(
            correction_point.get(
                "instance_id"
            )
            or correction_point.get("point")
            or explicit_target
        )

        # Добавя корекциите към съхранените
        # отмествания на конкретния предмет.
        offsets = self.targets.update_offsets(
            target,
            dx,
            dy,
            dz,
        )

        # Връща новите натрупани корекции.
        return self.result(
            request_id,
            True,
            "OFFSET_UPDATED",
            target=target,
            offsets=offsets,
        )

    def handle_traceback_history(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Връща обобщение на историята за преместване
        на всички предмети или на конкретно избран предмет.
        """

        # Получава целта от заявката.
        #
        # При празна стойност подава None,
        # което означава история за всички предмети.
        summary = self.traceback.history_summary(
            str(
                request.get("target")
                or ""
            ).strip()
            or None
        )

        # Връща състоянието и подробното
        # обобщение на историята.
        return self.result(
            request_id,
            bool(
                summary.get("ok")
            ),
            str(
                summary.get(
                    "state",
                    "TRACEBACK_HISTORY",
                )
            ),
            traceback=summary,
        )