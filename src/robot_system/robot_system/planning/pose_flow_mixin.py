#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Типови анотации за общи стойности и речници.
from typing import Any, Dict


class PlanningPoseFlowMixin:
    """
    Съдържа логиката за изпълнение на предварително
    дефинирани именувани позиции на манипулатора.

    Позициите се зареждат от библиотеката с постоянни точки
    и се синхронизират между цифровия модел в RViz
    и физическия манипулатор.
    """

    def execute_named_pose(
        self,
        request_id: str,
        pose_name: str,
        execute: bool,
        requested_for_scan: bool = False,
    ) -> Dict[str, Any]:
        """
        Изпълнява или визуализира зададена именувана позиция.

        Аргументи:
        - request_id — идентификатор на заявката;
        - pose_name — име на постоянната позиция;
        - execute — определя дали движението да бъде изпълнено;
        - requested_for_scan — показва дали позицията е заявена
          като част от сканиране на работното пространство.
        """

        # Получава ставните стойности на именуваната позиция
        # в радиани за използване от RViz и кинематичния модел.
        joints = self.poses.joint_radians(
            pose_name
        )

        # Получава физическите сервомоторни стойности,
        # свързани със същата именувана позиция.
        servo_angles = self.poses.servo_angles(
            pose_name
        )

        # Създава подробно описание на позицията,
        # включващо ставни и сервомоторни стойности.
        description = self.poses.describe(
            pose_name
        )

        # Изчислява декартовата позиция на крайния механизъм
        # чрез правата кинематика.
        pose_xyz = self.ik.fk_tool(
            joints
        )

        # Публикува маркери в RViz.
        #
        # Желаната и достигнатата позиция са еднакви,
        # защото именуваната позиция е зададена директно
        # чрез ставни стойности.
        self.publish_plan_markers(
            pose_xyz,
            pose_xyz,
            pose_name,
        )

        # Публикува отделно положението на хващача
        # към цифровия модел в RViz.
        self.motion.publish_rviz_gripper_position(
            self.poses.gripper_radians(
                pose_name
            )
        )

        # При реално изпълнение изпраща ставните
        # и физическите сервомоторни стойности.
        if execute:
            ok, sent_servo_angles = (
                self.motion.execute_joints(
                    joints,
                    label=pose_name,

                    # Използва точните сервомоторни стойности
                    # от конфигурацията на именуваната позиция.
                    servo_override=servo_angles,

                    # Изчаква потвърждение от Raspberry Pi,
                    # че физическото движение е приключило.
                    wait_for_physical_completion=True,
                )
            )

            # Връща грешка, ако движението е прекратено
            # или физическият манипулатор не го завърши успешно.
            if not ok:
                return self.result(
                    request_id,
                    False,
                    f"{pose_name.upper()}_CANCELLED",
                )

            # Запазва изчислената декартова позиция
            # като текущо състояние на манипулатора.
            self.persist_current_cartesian_pose(
                pose_xyz,
                source=f"named_pose:{pose_name}",
            )

        else:
            # При предварителна визуализация физическите
            # стойности се връщат, без да се изпращат към ръката.
            sent_servo_angles = servo_angles

            # Публикува ставната позиция само в RViz.
            self.motion.publish_rviz_joints(
                joints,
                self.motion.timings.rviz_move_sec,
            )

        # Създава състояние на резултата според това
        # дали позицията е изпълнена или само визуализирана.
        state = (
            f"{pose_name.upper()}_REACHED"
            if execute
            else f"{pose_name.upper()}_PREVIEWED"
        )

        # Връща подробен структуриран резултат.
        return self.result(
            request_id,
            True,
            state,

            # Име на изпълнената позиция.
            target=pose_name,

            # Подробно описание на позицията.
            pose=description,

            # Показва дали позицията е заявена
            # във връзка със сканиране.
            requested_for_scan=requested_for_scan,

            # Изпратените или изчислените
            # физически сервомоторни стойности.
            physical_servo_degrees=(
                sent_servo_angles
            ),

            # Последният отговор за завършено физическо движение.
            #
            # Полето се попълва само когато има реално изпълнение
            # и физическият манипулатор е активиран.
            physical_completion_reply=(
                self.motion.real.last_completion_reply
                if (
                    execute
                    and self.motion.real.enabled
                )
                else None
            ),

            # Показва дали движението е било изпълнено.
            executed=execute,

            # Показва дали системата работи
            # само чрез цифровия модел в RViz.
            simulation_only=(
                not self.motion.real.enabled
            ),

            # Времеви настройки на движението.
            timings=self.timing_dict(),
        )