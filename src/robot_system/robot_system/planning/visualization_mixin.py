#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за:
# - математически операции;
# - времеви стойности;
# - типови анотации.
import math
import time
from typing import Any, Dict, Optional, Tuple

# ROS 2 съобщение за представяне на тримерна точка.
from geometry_msgs.msg import Point

# ROS 2 съобщения за публикуване на единични
# и групирани визуални маркери в RViz.
from visualization_msgs.msg import Marker, MarkerArray

# Изключения, свързани с недостъпен физически
# манипулатор и неуспешно изпълнение на команда.
from .arm_client import (
    PhysicalArmCommandError,
    PhysicalArmUnavailableError,
)

# Подредба на основните стави
# на кинематичния модел.
from .constants import JOINT_ORDER


class PlanningVisualizationMixin:
    """
    Съдържа помощните методи за визуализация,
    валидиране на координати и създаване
    на структурирани резултати.

    Модулът управлява:
    - маркерите за историята на предметите;
    - маркерите за хващане и поставяне;
    - маркерите за целта на планирането;
    - командата за задържане на текущата позиция;
    - преобразуването на координати;
    - извеждането на времевите настройки.
    """

    def publish_traceback_markers(self) -> None:
        """
        Публикува в RViz маркери за всички запазени
        местоположения от историята на предметите.

        Текущите, началните и междинните позиции
        се представят с различни цветове.
        """

        # Създава масив, който ще съдържа
        # всички маркери от историята.
        array = MarkerArray()

        # Създава специален маркер за изтриване
        # на всички по-рано публикувани маркери.
        delete = Marker()
        delete.header.frame_id = "base_link"
        delete.header.stamp = self.get_clock().now().to_msg()
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        # Начален идентификатор за новите маркери.
        marker_id = 1

        # Обхожда всички местоположения,
        # предоставени от хранилището за историята.
        for row in self.traceback.marker_locations():
            # Получава центъра на местоположението.
            center = row.get("center") or {}

            try:
                # Преобразува координатите до числови стойности.
                x = float(center["x"])
                y = float(center["y"])
                z = float(center["z"])
            except Exception:
                # Пропуска записи с липсващи
                # или невалидни координати.
                continue

            # Създава сферичен маркер
            # за конкретното местоположение.
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "object_traceback_points"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Задава позицията на маркера.
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0

            # Задава размерите на сферата.
            marker.scale.x = 0.025
            marker.scale.y = 0.025
            marker.scale.z = 0.025

            # Маркерът е напълно непрозрачен.
            marker.color.a = 1.0

            # Текущата позиция се показва в зелено.
            if row.get("current"):
                marker.color.r = 0.1
                marker.color.g = 1.0
                marker.color.b = 0.2

            # Първоначалната позиция се показва в синьо.
            elif row.get("origin"):
                marker.color.r = 0.2
                marker.color.g = 0.5
                marker.color.b = 1.0

            # Междинните стари позиции
            # се показват в жълто-оранжево.
            else:
                marker.color.r = 1.0
                marker.color.g = 0.75
                marker.color.b = 0.1

            # Добавя сферичния маркер към масива.
            array.markers.append(marker)

            # Създава текстов маркер
            # над съответната позиция.
            text = Marker()

            # Използва същия времеви печат
            # и координатна система като сферата.
            text.header = marker.header
            text.ns = "object_traceback_text"

            # Използва различен диапазон на идентификаторите,
            # за да няма конфликт със сферичните маркери.
            text.id = marker_id + 10000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            # Поставя текста малко над сферата.
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 0.035
            text.pose.orientation.w = 1.0

            # Задава размера на текста.
            text.scale.z = 0.022

            # Текстът се показва в бяло.
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            # Създава надпис с името и индекса
            # на местоположението.
            text.text = (
                f"{row.get('display')}: "
                f"{row.get('index')}"
                + (" current" if row.get("current") else "")
            )

            # Добавя текстовия маркер.
            array.markers.append(text)

            # Увеличава идентификатора
            # за следващото местоположение.
            marker_id += 1

        # Публикува целия масив в темата
        # за маркерите на историята.
        self.traceback_marker_pub.publish(array)

    @staticmethod
    def validate_xyz(
        xyz: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Проверява и нормализира тримерни координати.

        Всяка координата трябва да бъде крайно число,
        което не е NaN или безкрайност.
        """

        # Преобразува всички стойности до float.
        clean = tuple(float(value) for value in xyz)

        # Проверява броя на координатите
        # и валидността на числовите стойности.
        if len(clean) != 3 or not all(
            math.isfinite(value)
            for value in clean
        ):
            raise ValueError(
                f"Invalid target coordinates: {xyz!r}"
            )

        # Връща нормализирания тримерен кортеж.
        return clean[0], clean[1], clean[2]

    @staticmethod
    def _optional_xyz(
        value: Any,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Опитва да извлече тримерни координати
        от речник, списък или кортеж.

        При невалидна структура връща None.
        """

        # Обработва структура от вида:
        # {"x": ..., "y": ..., "z": ...}.
        if isinstance(value, dict):
            try:
                return (
                    float(value["x"]),
                    float(value["y"]),
                    float(value["z"]),
                )
            except Exception:
                return None

        # Обработва списък или кортеж
        # с поне три елемента.
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return (
                    float(value[0]),
                    float(value[1]),
                    float(value[2]),
                )
            except Exception:
                return None

        # Неподдържана структура.
        return None

    def publish_hold_position(self) -> bool:
        """
        Изпраща текущото ставно състояние като команда
        за задържане на текущата позиция.

        Командата се публикува към RViz и при активен
        физически изпълнител се изпраща към Raspberry Pi.
        """

        # Получава текущите ставни стойности в градуси.
        current = self.current_joint_deg()

        # Не може да създаде команда при липсващо състояние.
        if not current:
            return False

        # Преобразува ставните стойности
        # от градуси в радиани.
        joints = {
            name: math.radians(
                current.get(name, 0.0)
            )
            for name in JOINT_ORDER
        }

        # Публикува кратък преход към същата позиция
        # в цифровия модел.
        self.motion.publish_rviz_joints(
            joints,
            0.2,
        )

        # При изключен физически манипулатор
        # няма физическа команда за изпращане.
        if not self.motion.real.enabled:
            return False

        try:
            # Изпраща текущите стави към физическия
            # изпълнител с кратка продължителност.
            self.motion.real.send_joints(
                joints,
                duration_sec=0.2,
            )
            return True

        except (
            PhysicalArmUnavailableError,
            PhysicalArmCommandError,
        ) as exc:
            # Записва предупреждение при недостъпна
            # ръка или неуспешна физическа команда.
            self.get_logger().warning(
                f"Could not send physical hold command: {exc}"
            )
            return False

    def publish_pick_place_markers(
        self,
        source: Tuple[float, float, float],
        destination: Tuple[float, float, float],
        source_name: str,
        destination_name: str,
    ) -> None:
        """
        Визуализира началната и крайната позиция
        на операция за хващане и поставяне.

        Между двете точки се публикува линия,
        а над всяка точка се показва текстов етикет.
        """

        # Създава масив с маркери.
        array = MarkerArray()

        # Изтрива предишните маркери
        # от същата тема.
        delete = Marker()
        delete.header.frame_id = "base_link"
        delete.header.stamp = self.get_clock().now().to_msg()
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        # Създава линия между началната
        # и крайната позиция.
        line = Marker()
        line.header.frame_id = "base_link"
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = "pick_place_path"
        line.id = 10
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD

        # Дебелина на линията.
        line.scale.x = 0.012

        # Зелен цвят на траекторията.
        line.color.r = 0.2
        line.color.g = 1.0
        line.color.b = 0.3
        line.color.a = 1.0

        # Двете крайни точки на линията.
        line.points = [
            Point(
                x=source[0],
                y=source[1],
                z=source[2],
            ),
            Point(
                x=destination[0],
                y=destination[1],
                z=destination[2],
            ),
        ]

        # Добавя линията към масива.
        array.markers.append(line)

        # Създава сферичен и текстов маркер
        # за източника и крайната цел.
        for marker_id, xyz, name, red, green, blue in (
            # Началната точка се показва
            # в червено-оранжев цвят.
            (
                11,
                source,
                source_name,
                1.0,
                0.3,
                0.1,
            ),

            # Крайната точка се показва
            # в зелен цвят.
            (
                12,
                destination,
                destination_name,
                0.1,
                1.0,
                0.3,
            ),
        ):
            # Създава сферичен маркер.
            marker = Marker()
            marker.header = line.header
            marker.ns = "pick_place_targets"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Позиция на маркера.
            marker.pose.position.x = xyz[0]
            marker.pose.position.y = xyz[1]
            marker.pose.position.z = xyz[2]
            marker.pose.orientation.w = 1.0

            # Размер на сферичния маркер.
            marker.scale.x = 0.04
            marker.scale.y = 0.04
            marker.scale.z = 0.04

            # Задава подадения цвят.
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = 1.0

            # Добавя сферата към масива.
            array.markers.append(marker)

            # Създава текстов етикет
            # за съответната позиция.
            text = Marker()
            text.header = line.header
            text.ns = "pick_place_text"
            text.id = marker_id + 10
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            # Поставя текста над сферата.
            text.pose.position.x = xyz[0]
            text.pose.position.y = xyz[1]
            text.pose.position.z = xyz[2] + 0.06
            text.pose.orientation.w = 1.0

            # Размер на текста.
            text.scale.z = 0.03

            # Бял цвят на текста.
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            # Име на предмета или целта.
            text.text = name

            # Добавя текста към масива.
            array.markers.append(text)

        # Публикува маркерите в основната тема
        # за визуализация на плана.
        self.marker_pub.publish(array)

    def publish_plan_markers(
        self,
        target: Optional[Tuple[float, float, float]],
        reached: Optional[Tuple[float, float, float]],
        label: str,
    ) -> None:
        """
        Визуализира желаната цел и разликата
        между целевата и достигнатата позиция.

        При налична цел се публикуват сфера и текст.
        При налична достигната позиция се публикува
        и линия между двете точки.
        """

        # Създава масив с маркери.
        array = MarkerArray()

        # Изтрива предишните маркери.
        delete = Marker()
        delete.header.frame_id = "base_link"
        delete.header.stamp = self.get_clock().now().to_msg()
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        # Публикува целевия маркер
        # само когато е подадена цел.
        if target is not None:
            # Създава червена сфера
            # на желаната позиция.
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "planner_target"
            marker.id = 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Позиция на целта.
            marker.pose.position.x = target[0]
            marker.pose.position.y = target[1]
            marker.pose.position.z = target[2]
            marker.pose.orientation.w = 1.0

            # Размер на целевия маркер.
            marker.scale.x = 0.035
            marker.scale.y = 0.035
            marker.scale.z = 0.035

            # Червен цвят.
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker.color.a = 1.0

            # Добавя целевата сфера.
            array.markers.append(marker)

            # Създава текстов етикет
            # над целевата позиция.
            text = Marker()
            text.header = marker.header
            text.ns = "planner_text"
            text.id = 2
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            # Поставя текста над целта.
            text.pose.position.x = target[0]
            text.pose.position.y = target[1]
            text.pose.position.z = target[2] + 0.06
            text.pose.orientation.w = 1.0

            # Размер на текста.
            text.scale.z = 0.035

            # Бял цвят на текста.
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            # Показва етикета и координатите
            # с точност до три знака.
            text.text = (
                f"PLAN: {label}\n"
                f"({target[0]:.3f}, "
                f"{target[1]:.3f}, "
                f"{target[2]:.3f})"
            )

            # Добавя текста към масива.
            array.markers.append(text)

        # При налични цел и достигната позиция
        # публикува линия между тях.
        if target is not None and reached is not None:
            line = Marker()
            line.header.frame_id = "base_link"
            line.header.stamp = self.get_clock().now().to_msg()
            line.ns = "planner_error"
            line.id = 3
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD

            # Дебелина на линията.
            line.scale.x = 0.008

            # Жълт цвят на линията.
            line.color.r = 1.0
            line.color.g = 1.0
            line.color.b = 0.0
            line.color.a = 1.0

            # Линия от желаната към достигнатата точка.
            line.points = [
                Point(
                    x=target[0],
                    y=target[1],
                    z=target[2],
                ),
                Point(
                    x=reached[0],
                    y=reached[1],
                    z=reached[2],
                ),
            ]

            # Добавя линията към масива.
            array.markers.append(line)

        # Публикува всички създадени маркери.
        self.marker_pub.publish(array)

    def timing_dict(self) -> Dict[str, float]:
        """
        Връща използваните времеви настройки
        за движенията и управлението на хващача.
        """

        # Получава конфигурационния обект
        # с времевите стойности.
        timings = self.motion.timings

        # Преобразува настройките до речник,
        # подходящ за включване в резултат.
        return {
            # Продължителност на движение в RViz.
            "rviz_move_sec": timings.rviz_move_sec,

            # Продължителност на физическото движение.
            "real_move_sec": timings.real_move_sec,

            # Допълнително изчакване след движение.
            "post_move_delay_sec": timings.post_move_delay_sec,

            # Продължителност на движение
            # на хващача в RViz.
            "rviz_gripper_sec": timings.rviz_gripper_sec,

            # Продължителност на физическо
            # движение на хващача.
            "real_gripper_sec": timings.real_gripper_sec,

            # Допълнително изчакване след
            # движение на хващача.
            "post_gripper_delay_sec": (
                timings.post_gripper_delay_sec
            ),
        }

    @staticmethod
    def xyz_dict(
        xyz: Tuple[float, float, float],
    ) -> Dict[str, float]:
        """
        Преобразува координатен кортеж
        до речник с полета x, y и z.
        """

        return {
            "x": xyz[0],
            "y": xyz[1],
            "z": xyz[2],
        }

    @staticmethod
    def result(
        request_id: str,
        ok: bool,
        state: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Създава стандартен структуриран резултат
        за заявка към планиращия агент.
        """

        # Създава основните задължителни полета.
        output = {
            # Показва дали операцията е успешна.
            "ok": ok,

            # Идентификатор на заявката.
            "request_id": request_id,

            # Текстово състояние на операцията.
            "state": state,

            # Време на създаване на резултата.
            "timestamp": time.time(),
        }

        # Добавя всички допълнително подадени полета.
        output.update(extra)

        return output