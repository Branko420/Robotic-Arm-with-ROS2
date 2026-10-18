#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Типови анотации за общи стойности, речници,
# списъци, незадължителни стойности и координатни кортежи.
from typing import Any, Dict, List, Optional, Tuple

# Конфигурационни стойности за:
# - последно избраната цел;
# - отворено и затворено положение на хващача;
# - вертикален просвет при хващане;
# - записване на симулираните движения в историята.
from .settings import (
    LAST_TARGET_KEY,
    PICK_GRIPPER_CLOSE_SERVO_DEG,
    PICK_GRIPPER_OPEN_SERVO_DEG,
    PICK_VERTICAL_CLEARANCE_M,
    RECORD_SIMULATED_MOVES,
)


class PlanningTransferFlowMixin:
    """
    Съдържа пълната последователност за хващане
    и преместване на предмет от една позиция до друга.

    Последователността включва:
    - изчисляване на геометрията на предметите;
    - определяне на позиции за подход, хващане и поставяне;
    - проверка за препятствия по хоризонталния път;
    - решаване на обратната кинематика за всяка стъпка;
    - изпълнение в RViz и върху физическия манипулатор;
    - управление на хващача;
    - обновяване на историята на предмета.
    """

    def execute_pick_and_place_sequence(
        self,
        request_id: str,
        source: Dict[str, Any],
        destination: Dict[str, Any],
        execute: bool,
        destination_center_override: Optional[
            Tuple[float, float, float]
        ] = None,
        history_mode: str = "record",
        traceback_object_id: Optional[str] = None,
        workspace_points: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Изпълнява или визуализира цялостна последователност
        за хващане и поставяне на предмет.

        Аргументи:
        - request_id — идентификатор на заявката;
        - source — предметът, който трябва да бъде преместен;
        - destination — предметът или повърхността за поставяне;
        - execute — определя дали движението да бъде изпълнено;
        - destination_center_override — директно зададен краен център;
        - history_mode — режим за запис или връщане в историята;
        - traceback_object_id — идентификатор при връщане на предмет;
        - workspace_points — незадължителен списък с актуални точки.
        """

        # Получава име на предмета източник.
        source_name = str(
            source.get("point")
            or source.get("label")
            or "source"
        )

        # Получава име на крайната цел.
        destination_name = str(
            destination.get("point")
            or destination.get("label")
            or "destination"
        )

        # Създава устойчив ключ за предмета източник.
        source_key = str(
            source.get("instance_id")
            or source.get("point")
            or source_name
        )

        # Създава устойчив ключ за крайната цел.
        destination_key = str(
            destination.get("instance_id")
            or destination.get("point")
            or destination_name
        )

        # Зарежда натрупаните ръчни корекции
        # за предмета източник.
        source_offsets = self.targets.load_offsets(
            source_key
        )

        # Зарежда натрупаните ръчни корекции
        # за крайната цел.
        destination_offsets = self.targets.load_offsets(
            destination_key
        )

        # Нормализира измерената геометрия
        # на предмета, който ще бъде хванат.
        source_geometry = self.targets.object_geometry(
            source
        )

        # Нормализира измерената геометрия
        # на крайната цел.
        destination_geometry = self.targets.object_geometry(
            destination
        )

        # Позиция на хващача при реалното хващане.
        #
        # По X и Y използва измерената позиция за хващане,
        # а по Z използва центъра на предмета.
        source_grasp = (
            source_geometry["grasp"]["x"],
            source_geometry["grasp"]["y"],
            source_geometry["center"]["z"],
        )

        # Позиция за вертикален подход над предмета.
        source_approach = (
            source_grasp[0],
            source_grasp[1],
            source_grasp[2]
            + PICK_VERTICAL_CLEARANCE_M,
        )

        # Отместване между геометричния център
        # на предмета и позицията на хващане.
        grasp_offset = (
            source_geometry["grasp_offset"]["x"],
            source_geometry["grasp_offset"]["y"],
            source_geometry["grasp_offset"]["z"],
        )

        # Проверява дали е подаден директен краен център.
        #
        # Той се използва основно при връщане на предмет
        # към позиция, записана в историята.
        direct_center = (
            destination_center_override
            or self._optional_xyz(
                destination.get(
                    "traceback_direct_center"
                )
            )
        )

        if direct_center is not None:
            # При директно зададен център не изчислява
            # поставянето спрямо геометрията на целта.
            destination_object_center = direct_center

            # Изключва натрупаните корекции на крайната цел,
            # защото координатите идват директно от историята.
            destination_offsets = {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }

            # Изчислява позицията на хващача при поставяне
            # чрез отместването спрямо центъра на предмета.
            destination_place = (
                direct_center[0] + grasp_offset[0],
                direct_center[1] + grasp_offset[1],
                direct_center[2] + grasp_offset[2],
            )

            # Изчислява позиция за подход
            # над директно зададеното място.
            destination_approach = (
                destination_place[0],
                destination_place[1],
                destination_place[2]
                + self.place_approach_height,
            )

        else:
            # Изчислява позициите за поставяне
            # според геометрията и височината на двата предмета.
            placement = self.targets.placement_targets(
                source,
                destination,
                place_clearance=self.place_clearance,
                approach_height=self.place_approach_height,
                minimum_clearance=(
                    self.minimum_above_clearance
                ),
            )

            # Краен център на поставения предмет.
            destination_object_center = placement[
                "object_center"
            ]

            # Позиция на хващача при освобождаването.
            destination_place = placement[
                "gripper_place"
            ]

            # Безопасна позиция за подход
            # над крайната цел.
            destination_approach = placement[
                "gripper_approach"
            ]

        # Използва изрично подадените точки,
        # когато такива са налични.
        #
        # В противен случай зарежда последния документ
        # с точки от работното пространство.
        all_points = (
            workspace_points
            if workspace_points is not None
            else (
                (
                    self.targets.load_motion_points()
                    or {}
                ).get("motion_points")
                or []
            )
        )

        # Идентификатор на предмета източник.
        source_id = str(
            source.get("instance_id")
            or source.get("point")
            or ""
        )

        # Идентификатор на крайната цел.
        destination_id = str(
            destination.get("instance_id")
            or destination.get("point")
            or ""
        )

        # Минималната транзитна височина трябва да бъде
        # поне толкова висока, колкото двете позиции за подход.
        minimum_transit_z = max(
            source_approach[2],
            destination_approach[2],
        )

        # Изчислява безопасна хоризонтална височина,
        # като отчита препятствията между двете позиции.
        transit_z, transit_blockers = (
            self.height_aware_transit_z(
                all_points,
                source,
                source_approach,
                destination_approach,
                minimum_transit_z,

                # Самият предмет и крайната цел
                # не се разглеждат като препятствия.
                ignored_ids={
                    source_id,
                    destination_id,
                },
            )
        )

        # Прекратява планирането, когато необходимата
        # височина е над максимално допустимата.
        if transit_z is None:
            return self.result(
                request_id,
                False,
                "TRANSIT_CLEARANCE_UNREACHABLE",
                blockers=transit_blockers,
                maximum_transit_z=self.maximum_transit_z,
                source_geometry=source_geometry,
                destination_geometry=destination_geometry,
            )

        def apply_offsets(
            xyz: Tuple[float, float, float],
            offsets: Dict[str, float],
        ) -> Tuple[float, float, float]:
            """
            Добавя натрупаните корекции по X, Y и Z
            към дадена декартова позиция.
            """

            return (
                xyz[0] + offsets["x"],
                xyz[1] + offsets["y"],
                xyz[2] + offsets["z"],
            )

        # Прилага ръчните корекции
        # към позицията за подход към предмета.
        source_approach_offset = apply_offsets(
            source_approach,
            source_offsets,
        )

        # Прилага ръчните корекции
        # към позицията за хващане.
        source_grasp_offset = apply_offsets(
            source_grasp,
            source_offsets,
        )

        # Прилага корекциите на крайната цел
        # към позицията за подход.
        destination_approach_offset = apply_offsets(
            destination_approach,
            destination_offsets,
        )

        # Прилага корекциите на крайната цел
        # към позицията за поставяне.
        destination_place_offset = apply_offsets(
            destination_place,
            destination_offsets,
        )

        # Основната последователност при вземане на предмета.
        #
        # Всеки запис съдържа:
        # - име на стъпката;
        # - декартова позиция;
        # - незадължително действие на хващача.
        sequence_spec = [
            (
                "source_approach",
                source_approach_offset,
                None,
            ),
            (
                "source_grasp",
                source_grasp_offset,
                "close",
            ),
            (
                "source_retreat",
                source_approach_offset,
                None,
            ),
        ]

        # Добавя вертикално повдигане над предмета,
        # когато безопасната транзитна височина
        # е осезаемо по-висока от позицията за подход.
        if (
            transit_z
            > source_approach_offset[2] + 0.005
        ):
            sequence_spec.append(
                (
                    "source_transit",
                    (
                        source_approach_offset[0],
                        source_approach_offset[1],
                        transit_z,
                    ),
                    None,
                )
            )

        # Добавя транзитна позиция над крайната цел,
        # когато е необходимо спускане от по-голяма височина.
        if (
            transit_z
            > destination_approach_offset[2] + 0.005
        ):
            sequence_spec.append(
                (
                    "destination_transit",
                    (
                        destination_approach_offset[0],
                        destination_approach_offset[1],
                        transit_z,
                    ),
                    None,
                )
            )

        # Добавя стъпките за приближаване,
        # поставяне и оттегляне от крайната цел.
        sequence_spec.extend(
            [
                (
                    "destination_approach",
                    destination_approach_offset,
                    None,
                ),
                (
                    "destination_place",
                    destination_place_offset,
                    "open",
                ),
                (
                    "destination_retreat",
                    destination_approach_offset,
                    None,
                ),
            ]
        )

        # Списък с успешно решените
        # ставни конфигурации за всяка стъпка.
        solved = []

        # Началното състояние за първото IK решение
        # се взема от текущото ставно състояние.
        current_for_step = self.current_joint_deg()

        # Решава цялата последователност предварително,
        # преди да бъде изпратено каквото и да е движение.
        for (
            step_name,
            xyz,
            gripper_action,
        ) in sequence_spec:
            # Валидира декартовата позиция.
            xyz = self.validate_xyz(xyz)

            # Решава обратната кинематика,
            # като използва предходното решение
            # като начално състояние за следващата стъпка.
            solution, debug = (
                self.solve_cartesian_target(
                    xyz,
                    current_joint_deg=current_for_step,
                )
            )

            # При липса на решение прекратява
            # цялата последователност преди изпълнение.
            if solution is None:
                # Публикува визуален маркер
                # на неуспешната цел в RViz.
                self.publish_plan_markers(
                    xyz,
                    None,
                    f"IK FAILED: {step_name}",
                )

                return self.result(
                    request_id,
                    False,
                    (
                        "PICK_PLACE_"
                        f"{step_name.upper()}"
                        "_IK_FAILED"
                    ),
                    target_xyz=self.xyz_dict(xyz),
                    closest_error=(
                        None
                        if debug is None
                        else debug.get("error")
                    ),
                )

            # Запазва решението за последващо изпълнение.
            solved.append(
                (
                    step_name,
                    xyz,
                    gripper_action,
                    solution,
                )
            )

            # Следващата стъпка започва
            # от текущото намерено решение.
            current_for_step = dict(
                solution["degs"]
            )

        # Запомня крайната цел като последно
        # използван предмет или повърхност.
        self.redis.set(
            LAST_TARGET_KEY,
            destination_key,
        )

        # Публикува маркери за началната
        # и крайната позиция в RViz.
        self.publish_pick_place_markers(
            source_approach,
            destination_approach,
            source_name,
            destination_name,
        )

        # При реално изпълнение първо отваря хващача.
        if execute:
            if not self.motion.execute_gripper(
                close=False,
                servo_angle_override=(
                    PICK_GRIPPER_OPEN_SERVO_DEG
                ),
            ):
                return self.result(
                    request_id,
                    False,
                    "PICK_PLACE_CANCELLED",
                )

            # Изпълнява последователно
            # всички предварително решени стъпки.
            for (
                step_name,
                _xyz,
                gripper_action,
                solution,
            ) in solved:
                # Проверява за външна команда stop.
                if self.cancel_event.is_set():
                    return self.result(
                        request_id,
                        False,
                        "PICK_PLACE_CANCELLED",
                    )

                # Изпълнява ставното движение
                # за текущата стъпка.
                ok, _ = self.motion.execute_joints(
                    solution["candidate"],
                    label=(
                        f"pick_place:{step_name}"
                    ),
                )

                # Прекратява при неуспешно движение.
                if not ok:
                    return self.result(
                        request_id,
                        False,
                        "PICK_PLACE_CANCELLED",
                    )

                # След достигане на позицията за хващане
                # затваря хващача.
                if gripper_action == "close":
                    if not self.motion.execute_gripper(
                        close=True,
                        servo_angle_override=(
                            PICK_GRIPPER_CLOSE_SERVO_DEG
                        ),
                    ):
                        return self.result(
                            request_id,
                            False,
                            "PICK_PLACE_CANCELLED",
                        )

                    # Записва предмета като текущо държан.
                    self.save_held_object(source)

                # След достигане на позицията за поставяне
                # отваря хващача.
                elif gripper_action == "open":
                    if not self.motion.execute_gripper(
                        close=False,
                        servo_angle_override=(
                            PICK_GRIPPER_OPEN_SERVO_DEG
                        ),
                    ):
                        return self.result(
                            request_id,
                            False,
                            "PICK_PLACE_CANCELLED",
                        )

                    # Изчиства състоянието
                    # за държан предмет.
                    self.clear_held_object()

        # При предварителна визуализация публикува
        # първата намерена конфигурация в RViz.
        if not execute and solved:
            self.motion.publish_rviz_joints(
                solved[0][3]["candidate"],
                self.motion.timings.rviz_move_sec,
            )

        # Резултат от обновяването
        # на историята на предмета.
        trace_update = None

        # Крайна позиция на хващача
        # след прилагане на корекциите.
        final_destination_target = apply_offsets(
            destination_place,
            destination_offsets,
        )

        # Краен център на поставения предмет
        # след прилагане на корекциите.
        final_object_center = apply_offsets(
            destination_object_center,
            destination_offsets,
        )

        # Историята се обновява само когато:
        # - операцията е изпълнена;
        # - физическият манипулатор е активен
        #   или е разрешен запис на симулирани движения.
        history_committed = bool(
            execute
            and (
                self.motion.real.enabled
                or RECORD_SIMULATED_MOVES
            )
        )

        if history_committed:
            # При нормално преместване добавя
            # ново местоположение към историята.
            if history_mode == "record":
                trace_update = (
                    self.traceback.record_move(
                        source=source,
                        destination=destination,
                        placed_center=(
                            final_object_center
                        ),
                        request_id=request_id,
                    )
                )

                # Обновява последния известен център
                # в регистъра на екземплярите.
                if trace_update.get("object_id"):
                    self.instances.update_center(
                        str(
                            trace_update["object_id"]
                        ),
                        final_object_center,
                    )

            # При връщане премахва последното
            # местоположение от стека.
            elif (
                history_mode == "return"
                and traceback_object_id is not None
            ):
                trace_update = (
                    self.traceback.commit_return(
                        traceback_object_id,
                        request_id,
                    )
                )

                # Получава новия текущ център
                # след връщането.
                current_center = self._optional_xyz(
                    trace_update.get("current")
                )

                # Обновява центъра в регистъра.
                if current_center is not None:
                    self.instances.update_center(
                        traceback_object_id,
                        current_center,
                    )

            # Обновява маркерите за историята в RViz.
            self.publish_traceback_markers()

        # След завършено изпълнение гарантира,
        # че няма останало състояние за държан предмет.
        if execute:
            self.clear_held_object()

        # Връща подробен резултат
        # за цялата последователност.
        return self.result(
            request_id,
            True,
            (
                "PICK_PLACE_EXECUTED"
                if execute
                else "PICK_PLACE_PREVIEWED"
            ),

            # Име на преместения предмет.
            source=source_name,

            # Име на крайната цел.
            destination=destination_name,

            # Изпълнените или планираните стъпки.
            sequence=[
                row[0]
                for row in solved
            ],

            # Крайна позиция на хващача.
            destination_place_xyz=self.xyz_dict(
                final_destination_target
            ),

            # Краен център на предмета.
            destination_object_center_xyz=(
                self.xyz_dict(
                    final_object_center
                )
            ),

            # Отместване между хващача
            # и центъра на предмета.
            grasp_offset_xyz=self.xyz_dict(
                grasp_offset
            ),

            # Използвана позиция за хващане.
            source_pick_xyz=self.xyz_dict(
                source_grasp_offset
            ),

            # Източник на вертикалната координата
            # при хващането.
            source_pick_z_source="center.z",

            # Резултат от обновяването на историята.
            trace_update=trace_update,

            # Показва дали историята е променена.
            history_committed=history_committed,

            # Показва дали движението е изпълнено.
            executed=execute,

            # Показва дали системата работи
            # само с цифровия модел.
            simulation_only=(
                not self.motion.real.enabled
            ),

            # Времеви настройки на движенията.
            timings=self.timing_dict(),

            # Измерена геометрия на предмета.
            source_geometry=source_geometry,

            # Измерена геометрия на крайната цел.
            destination_geometry=destination_geometry,

            # Показва, че е използвана проверка
            # за височината на препятствията.
            height_aware=True,

            # Изчислена безопасна транзитна височина.
            transit_z=transit_z,

            # Предметите, пресичащи транзитния коридор.
            transit_blockers=transit_blockers,

            # При execute=True предметът е освободен
            # в крайната позиция.
            object_released=bool(execute),

            # Структурирани факти за разговорния агент.
            response_facts={
                "operation": "pick_and_place",
                "object": self.point_display_name(
                    source
                ),
                "destination": (
                    self.point_display_name(
                        destination
                    )
                ),
                "object_released": bool(execute),
                "object_still_held": False,
            },
        )