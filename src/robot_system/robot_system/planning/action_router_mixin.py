#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Типови анотации за общи речници, списъци
# и стойности, които могат да липсват.
from typing import Any, Dict, List, Optional

# Смесени класове (mixins), съдържащи обработващите методи
# за отделните групи действия на планиращия агент.
from .held_actions_mixin import HeldActionsMixin
from .object_actions_mixin import ObjectActionsMixin
from .pose_actions_mixin import PoseActionsMixin
from .status_actions_mixin import StatusActionsMixin
from .workspace_actions_mixin import WorkspaceActionsMixin


class PlanningActionRouterMixin(
    StatusActionsMixin,
    PoseActionsMixin,
    WorkspaceActionsMixin,
    HeldActionsMixin,
    ObjectActionsMixin,
):
    """
    Маршрутизира входящите заявки към съответния
    обработващ метод на планиращия агент.

    Класът обединява действията от няколко смесени класа
    и избира правилния обработващ метод според полето action.
    """

    # Допустими имена за заявка на текущите
    # командвани ставни стойности.
    _JOINT_STATUS_ACTIONS = {
        "joint_status",
        "get_joint_angles",
        "servo_status",
    }

    # Допустими имена за заявка на текущата
    # декартова позиция на крайния механизъм.
    _CURRENT_POSE_ACTIONS = {
        "current_pose",
        "cartesian_status",
    }

    # Допустими имена за директно задаване
    # на сервомоторни или ставни стойности.
    _MANUAL_JOINT_ACTIONS = {
        "set_servo_angles",
        "set_joint_angles",
        "manual_joint_move",
    }

    # Допустими имена за движение
    # към началната позиция HOME.
    _HOME_ACTIONS = {
        "home",
        "point_0",
        "initial",
        "initial_position",
    }

    # Допустими имена за движение
    # към позицията за сканиране.
    _SCAN_POSE_ACTIONS = {
        "scan_pose",
        "scan_position",
        "prepare_scan",
        "scan_home",
    }

    # Допустими имена за получаване
    # на историята на преместванията.
    _TRACEBACK_HISTORY_ACTIONS = {
        "traceback_history",
        "list_traceback",
        "object_history",
    }

    # Допустими имена за синхронизиране
    # на началните позиции и историята на предметите.
    _SYNC_TRACEBACK_ACTIONS = {
        "sync_traceback",
        "refresh_traceback_points",
    }

    # Допустими имена за връщане на предмет
    # към предишна или начална позиция.
    _RETURN_ACTIONS = {
        "return_object",
        "return_to_previous",
        "undo_object_move",
        "return_object_to_origin",
    }

    # Допустими имена за поставяне
    # на предмет, който вече се държи от хващача.
    _PLACE_HELD_ACTIONS = {
        "place_held_at",
        "place_held_object",
        "release_held_at",
    }

    def _load_workspace_points(
        self,
        request_id: str,
    ) -> tuple[
        Optional[List[Dict[str, Any]]],
        Optional[Dict[str, Any]],
    ]:
        """
        Зарежда актуалните точки от работното пространство.

        При успех връща списъка с точки и None.
        При грешка връща None и структуриран резултат за грешка.
        """

        # Зарежда последния документ с точки за движение.
        points_doc = self.targets.load_motion_points()

        # При липса на документ връща грешка,
        # че работното пространство все още не е сканирано.
        if points_doc is None:
            return None, self.result(
                request_id,
                False,
                "NO_WORKSPACE_POINTS",
            )

        # Добавя или актуализира уникалните идентификатори
        # на отделните екземпляри на предметите.
        points_doc = self.instances.assign_document(
            points_doc
        )

        # Наслагва виртуалните точки от историята
        # върху актуалния документ и записва резултата.
        points_doc = self.traceback.overlay_motion_points(
            points_doc,
            persist=True,
        )

        # Извлича списъка с точките за движение.
        points = points_doc.get(
            "motion_points",
            [],
        )

        # Обновява RViz маркерите
        # за точките от историята.
        self.publish_traceback_markers()

        return points, None

    def handle_request(
        self,
        request_id: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Обработва една входяща заявка към планиращия агент.

        Полето action определя кой обработващ метод
        трябва да бъде извикан.
        """

        # Получава действието от заявката.
        #
        # При липсващо поле се използва move.
        # Стойността се нормализира до малки букви.
        action = str(
            request.get("action", "move")
        ).strip().lower()

        # Връща текущите командвани ставни стойности.
        if action in self._JOINT_STATUS_ACTIONS:
            return self.handle_joint_status(
                request_id,
                request,
            )

        # Връща текущата декартова позиция
        # на крайния механизъм.
        if action in self._CURRENT_POSE_ACTIONS:
            return self.handle_current_pose(
                request_id,
                request,
            )

        # Запазва текущата декартова позиция
        # под зададено име.
        if action == "save_current_point":
            return self.handle_save_current_point(
                request_id,
                request,
            )

        # Връща всички запазени потребителски точки.
        if action == "list_saved_points":
            return self.handle_list_saved_points(
                request_id,
                request,
            )

        # Изтрива запазена точка по име.
        if action == "delete_saved_point":
            return self.handle_delete_saved_point(
                request_id,
                request,
            )

        # Премества манипулатора
        # към предварително запазена точка.
        if action == "move_to_saved_point":
            return self.handle_move_to_saved_point(
                request_id,
                request,
            )

        # Изпълнява относително декартово движение
        # спрямо текущата позиция.
        if action == "move_relative":
            return self.handle_move_relative(
                request_id,
                request,
            )

        # Изпълнява директно движение
        # чрез зададени ставни стойности.
        if action in self._MANUAL_JOINT_ACTIONS:
            return self.handle_manual_joint_move(
                request_id,
                request,
            )

        # Проверява дали манипулаторът се намира
        # в зададена именувана позиция.
        if action == "pose_status":
            return self.handle_pose_status(
                request_id,
                request,
            )

        # Изпълнява движение към HOME.
        if action in self._HOME_ACTIONS:
            return self.handle_home(
                request_id,
                request,
            )

        # Изпълнява движение
        # към позицията за сканиране.
        if action in self._SCAN_POSE_ACTIONS:
            return self.handle_scan_pose(
                request_id,
                request,
            )

        # Записва корекция на целева позиция.
        if action == "correct":
            return self.handle_correction(
                request_id,
                request,
            )

        # Връща историята на преместванията.
        if action in self._TRACEBACK_HISTORY_ACTIONS:
            return self.handle_traceback_history(
                request_id,
                request,
            )

        # Следващите действия изискват
        # актуални точки от работното пространство.
        points, error = self._load_workspace_points(
            request_id
        )

        # При грешка прекратява обработването.
        if error is not None:
            return error

        # Указва на проверяващите инструменти за типове,
        # че points не може да бъде None след успешното зареждане.
        assert points is not None

        # Синхронизира историята
        # с актуалните точки от сканирането.
        if action in self._SYNC_TRACEBACK_ACTIONS:
            return self.handle_sync_traceback(
                request_id,
                request,
                points,
            )

        # Определя конкретен предмет
        # от потребителско описание.
        if action == "resolve_target":
            return self.handle_resolve_target(
                request_id,
                request,
                points,
            )

        # Връща предмет към предишна
        # или начална позиция.
        if action in self._RETURN_ACTIONS:
            return self.handle_return_object(
                request_id,
                request,
                action,
                points,
            )

        # Изпълнява пълна последователност
        # за хващане и поставяне.
        if action == "pick_and_place":
            return self.handle_pick_and_place(
                request_id,
                request,
                points,
            )

        # Поставя предмета, който вече
        # се намира в хващача.
        if action in self._PLACE_HELD_ACTIONS:
            return self.handle_place_held(
                request_id,
                request,
                points,
            )

        # Премества вече държан предмет,
        # без да отваря хващача.
        if action == "move_held_to":
            return self.handle_move_held_to(
                request_id,
                request,
                points,
            )

        # Всички останали действия към предмет
        # се обработват от общия обектен обработващ метод.
        return self.handle_object_action(
            request_id,
            request,
            action,
            points,
        )