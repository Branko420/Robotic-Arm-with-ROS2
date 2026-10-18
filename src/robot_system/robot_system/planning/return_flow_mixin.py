#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Типови анотации за общи стойности,
# речници и списъци.
from typing import Any, Dict, List

# Определя дали движенията, изпълнени само в симулация,
# трябва да бъдат записвани в историята на предметите.
from .settings import RECORD_SIMULATED_MOVES


class PlanningReturnFlowMixin:
    """
    Съдържа логиката за връщане на преместен предмет
    към негова предишна позиция или до първоначалното
    му местоположение.

    Пътят за връщане се изгражда чрез историята
    на движенията, съхранявана от traceback модула.
    """

    def execute_return_object(
        self,
        request_id: str,
        target_name: str,
        execute: bool,
        all_the_way_to_origin: bool,
    ) -> Dict[str, Any]:
        """
        Връща предмет назад по неговата история на движенията.

        Аргументи:
        - request_id — идентификатор на текущата заявка;
        - target_name — име или идентификатор на предмета;
        - execute — определя дали движението да бъде изпълнено;
        - all_the_way_to_origin — определя дали предметът да бъде
          върнат само една стъпка назад или до първоначалното място.
        """

        # Определя устойчивия идентификатор на предмета
        # чрез неговото име, псевдоним или instance_id.
        object_id = self.traceback.resolve_object_id(
            target_name
        )

        # При липса на история за предмета
        # операцията не може да бъде изпълнена.
        if object_id is None:
            return self.result(
                request_id,
                False,
                "NO_OBJECT_TRACEBACK",
                target=target_name,
            )

        # Списък с успешно изпълнените
        # стъпки за връщане на предмета.
        completed_steps: List[
            Dict[str, Any]
        ] = []

        # Продължава да връща предмета назад,
        # докато не достигне първоначалната позиция
        # или докато не бъде изпълнена една стъпка.
        while True:
            # Получава следващата стъпка
            # от плана за връщане.
            plan = self.traceback.get_return_plan(
                object_id
            )

            # При липса на валиден план
            # връща грешка за липсваща история.
            if plan is None:
                return self.result(
                    request_id,
                    False,
                    "NO_OBJECT_TRACEBACK",
                    target=target_name,
                )

            # Проверява дали предметът вече се намира
            # в първоначалната си позиция.
            if plan.get("at_origin"):
                # Когато преди това са изпълнени стъпки,
                # предметът току-що е върнат до началото.
                if completed_steps:
                    return self.result(
                        request_id,
                        True,
                        "OBJECT_RETURNED_TO_ORIGIN",
                        target=object_id,
                        returned_steps=completed_steps,
                        traceback=(
                            self.traceback.history_summary(
                                object_id
                            )
                        ),
                        executed=execute,
                    )

                # Когато няма изпълнени стъпки,
                # предметът вече е бил в началната позиция.
                return self.result(
                    request_id,
                    True,
                    "OBJECT_ALREADY_AT_ORIGIN",
                    target=object_id,
                    traceback=(
                        self.traceback.history_summary(
                            object_id
                        )
                    ),
                    executed=execute,
                )

            # Текущо местоположение на предмета.
            current_location = plan["current"]

            # Предишно местоположение,
            # към което предметът трябва да бъде върнат.
            previous_location = plan["previous"]

            # Описание на предмета
            # в текущото му местоположение.
            source = (
                current_location.get("point")
                or {}
            )

            # Описание на целта,
            # представляваща предишното местоположение.
            destination = (
                previous_location.get("point")
                or {}
            )

            # Извлича точния център
            # на предишната позиция.
            previous_center = self._optional_xyz(
                previous_location.get("center")
            )

            # При невалидни координати
            # прекратява операцията.
            if previous_center is None:
                return self.result(
                    request_id,
                    False,
                    "TRACEBACK_LOCATION_INVALID",
                    target=object_id,
                )

            # Изпълнява стандартна последователност
            # за хващане и поставяне на предмета
            # върху предишната му позиция.
            step_result = (
                self.execute_pick_and_place_sequence(
                    request_id=request_id,
                    source=source,
                    destination=destination,
                    execute=execute,

                    # Използва точния запазен център,
                    # вместо текущия център на целта.
                    destination_center_override=(
                        previous_center
                    ),

                    # Указва, че движението е част
                    # от операция за връщане.
                    history_mode="return",

                    # Предава устойчивия идентификатор
                    # на предмета в историята.
                    traceback_object_id=object_id,

                    # Зарежда актуалните точки
                    # от работното пространство.
                    workspace_points=(
                        (
                            self.targets.load_motion_points()
                            or {}
                        ).get("motion_points")
                        or []
                    ),
                )
            )

            # При неуспешна стъпка
            # прекратява цялата операция.
            if not step_result.get("ok"):
                return self.result(
                    request_id,
                    False,
                    "OBJECT_RETURN_FAILED",
                    target=object_id,

                    # Връща подробния резултат
                    # от неуспешната стъпка.
                    failed_step=step_result,

                    # Добавя вече успешно изпълнените стъпки.
                    returned_steps=completed_steps,
                )

            # Записва успешно изпълнената стъпка
            # в локалния резултат.
            completed_steps.append(
                {
                    # Позицията, от която е взет предметът.
                    "from": current_location.get(
                        "center"
                    ),

                    # Позицията, към която е върнат.
                    "to": previous_location.get(
                        "center"
                    ),

                    # Състоянието на изпълнената
                    # последователност.
                    "result_state": (
                        step_result.get("state")
                    ),
                }
            )

            # Историята се променя само когато:
            # - движението действително се изпълнява;
            # - физическият манипулатор е активен
            #   или е разрешен запис на симулирани движения.
            history_will_mutate = bool(
                execute
                and (
                    self.motion.real.enabled
                    or RECORD_SIMULATED_MOVES
                )
            )

            # При предварителна визуализация или работа само в RViz
            # историята не се променя. Затова се изпълнява само една
            # планирана стъпка, за да не се получи безкраен цикъл.
            #
            # Също така приключва след една стъпка, когато потребителят
            # не е поискал връщане до първоначалната позиция.
            if (
                not history_will_mutate
                or not all_the_way_to_origin
            ):
                # Определя състоянието според начина,
                # по който е изпълнена операцията.
                state = (
                    # Само предварителна визуализация.
                    "OBJECT_RETURN_PREVIEWED"
                    if not execute

                    # Реално или симулирано изпълнение.
                    else (
                        # Историята е обновена и предметът
                        # е върнат една позиция назад.
                        "OBJECT_RETURNED_TO_PREVIOUS"
                        if history_will_mutate

                        # Движението е показано в RViz,
                        # но историята не е променена.
                        else "OBJECT_RETURN_RVIZ_ONLY"
                    )
                )

                # Връща резултат след една стъпка назад.
                return self.result(
                    request_id,
                    True,
                    state,
                    target=object_id,
                    returned_steps=completed_steps,
                    traceback=(
                        self.traceback.history_summary(
                            object_id
                        )
                    ),
                    executed=execute,
                )