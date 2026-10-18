#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване, математически
# операции, UDP комуникация, системни процеси, нишки и време.
import json
import math
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# Помощни модули за калибрацията, конфигурацията
# и ограниченията на сервомоторите.
from .calibration import load_calibrators
from .common import clamp, env, env_bool, env_float
from .constants import (
    JOINT_TO_SERVO,
    SAFE_SERVO_RANGE,
    SERVO_TO_JOINT,
)


# Папка с конфигурационните файлове на пакета.
PACKAGE_CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "config"
)


class PhysicalArmUnavailableError(RuntimeError):
    """
    Грешка при опит за изпълнение на движение,
    когато физическият манипулатор не е достъпен.
    """


class PhysicalArmCommandError(RuntimeError):
    """
    Грешка при неуспешно изпращане или изпълнение
    на UDP команда към физическия манипулатор.
    """


class RealArmUdpClient:
    """
    UDP клиент за управление на физическия манипулатор.

    Класът:
    - проверява достъпността на Raspberry Pi;
    - преобразува ставни ъгли от RViz към сервомоторни стойности;
    - преобразува сервомоторни стойности обратно към RViz;
    - изпраща движения чрез UDP;
    - изчаква потвърждение за завършване на движението;
    - поддържа последните командвани сервомоторни стойности.
    """

    def __init__(
        self,
        logger: Any,
    ) -> None:
        # ROS 2 логерът, използван за диагностични съобщения.
        self.logger = logger

        # Определя дали изпращането към физическия
        # манипулатор е разрешено.
        self.enabled = env_bool(
            "PLANNER_SEND_TO_REAL",
            False,
        )

        # IP адрес на Raspberry Pi.
        self.ip = env(
            "PLANNER_PI_IP",
            "192.168.188.144",
        )

        # UDP порт на изпълняващия възел върху Raspberry Pi.
        self.port = int(
            env_float(
                "PLANNER_PI_PORT",
                9090,
            )
        )

        # Формат на изпращаните UDP пакети.
        self.packet_format = env(
            "PLANNER_UDP_PACKET_FORMAT",
            "duration_angles",
        ).strip().lower()

        # Път до файла с участъковата калибрация.
        self.calibration_path = Path(
            env(
                "PLANNER_CALIBRATION_FILE",
                str(
                    PACKAGE_CONFIG_DIR
                    / "arm_calibration_piecewise.json"
                ),
            )
        )

        # Зарежда калибрационните преобразуватели
        # за отделните сервомотори.
        self.calibrators = load_calibrators(
            self.calibration_path
        )

        # Създава UDP сокет.
        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        # Заключване, което предотвратява едновременно
        # изпращане и четене през един и същ UDP сокет.
        self.send_lock = threading.RLock()

        # Определя дали трябва да бъде извършена
        # проверка за достъпност преди движение.
        self.require_available = env_bool(
            "PHYSICAL_ARM_REQUIRE_AVAILABLE",
            True,
        )

        # Адрес, който ще бъде проверяван чрез ping.
        self.ping_host = env(
            "PHYSICAL_ARM_PING_HOST",
            self.ip,
        )

        # Максимално време за изчакване на ping отговор.
        self.ping_timeout_sec = max(
            0.2,
            env_float(
                "PHYSICAL_ARM_PING_TIMEOUT_SEC",
                1.0,
            ),
        )

        # Брой ping пакети.
        self.ping_count = max(
            1,
            int(
                env_float(
                    "PHYSICAL_ARM_PING_COUNT",
                    1,
                )
            ),
        )

        # Време, през което последният резултат
        # от ping проверката може да се използва повторно.
        self.status_cache_sec = max(
            0.0,
            env_float(
                "PHYSICAL_ARM_STATUS_CACHE_SEC",
                2.0,
            ),
        )

        # Момент на последната проверка.
        self._last_status_check = 0.0

        # Последният кеширан резултат от проверката.
        self._last_status: Optional[
            Dict[str, Any]
        ] = None

        # Състояние на последното UDP изпращане.
        self.last_send_ok = False

        # Текст на последната грешка.
        self.last_error: Optional[str] = None

        # Последният отговор за завършено движение.
        self.last_completion_reply: Optional[
            Dict[str, Any]
        ] = None

        # Последните командвани физически
        # сервомоторни стойности.
        self.last_commanded_angles: Dict[
            str,
            float,
        ] = {}

        # Максимално време за изчакване
        # на потвърждение за завършено движение.
        self.completion_timeout_sec = max(
            1.0,
            env_float(
                "PLANNER_PI_COMPLETION_TIMEOUT_SEC",
                30.0,
            ),
        )

    def close(self) -> None:
        """
        Затваря UDP сокета.
        """

        try:
            self.sock.close()
        except Exception:
            # Игнорира грешки при повторно
            # или неуспешно затваряне.
            pass

    def check_available(
        self,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Проверява дали Raspberry Pi на физическия
        манипулатор е достъпно чрез ping.
        """

        # Когато физическото изпращане е изключено,
        # ръката се счита за недостъпна.
        if not self.enabled:
            return {
                "available": False,
                "state": "PHYSICAL_ARM_DISABLED",
                "host": self.ping_host,
            }

        # Когато проверката е изключена, приема,
        # че манипулаторът е достъпен.
        if not self.require_available:
            return {
                "available": True,
                "state": "PHYSICAL_ARM_CHECK_DISABLED",
                "host": self.ping_host,
            }

        # Текущо монотонно време.
        now = time.monotonic()

        # Използва кеширания резултат, когато:
        # - не е поискана принудителна проверка;
        # - има предишен резултат;
        # - кешът все още е актуален.
        if (
            not force
            and self._last_status is not None
            and (
                now - self._last_status_check
                <= self.status_cache_sec
            )
        ):
            return dict(self._last_status)

        # Командата ping приема цяло число
        # за времето на изчакване.
        timeout_seconds = max(
            1,
            int(
                math.ceil(
                    self.ping_timeout_sec
                )
            ),
        )

        # Създава системната ping команда.
        command = [
            "ping",
            "-c",
            str(self.ping_count),
            "-W",
            str(timeout_seconds),
            self.ping_host,
        ]

        try:
            # Изпълнява ping командата без извеждане
            # на стандартния изход и грешките.
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=(
                    self.ping_timeout_sec
                    + self.ping_count
                    + 1.0
                ),
            )

            # Код за завършване 0 означава,
            # че Raspberry Pi е отговорило.
            available = (
                completed.returncode == 0
            )

            reason = (
                "ping_ok"
                if available
                else "ping_failed"
            )

        except FileNotFoundError:
            # Системната команда ping не е налична.
            available = False
            reason = "ping_command_not_found"

        except subprocess.TimeoutExpired:
            # Ping командата е надвишила
            # максималното време.
            available = False
            reason = "ping_timeout"

        except Exception as exc:
            # Обработва останалите неочаквани грешки.
            available = False
            reason = (
                f"ping_error:"
                f"{type(exc).__name__}"
            )

        # Създава структуриран резултат
        # от проверката.
        status = {
            "available": available,
            "state": (
                "PHYSICAL_ARM_AVAILABLE"
                if available
                else "PHYSICAL_ARM_UNAVAILABLE"
            ),
            "host": self.ping_host,
            "reason": reason,
            "checked_at": time.time(),
        }

        # Актуализира кеша.
        self._last_status_check = now
        self._last_status = dict(status)

        return status

    def ensure_available(self) -> None:
        """
        Проверява достъпността на физическия манипулатор
        и хвърля изключение при неуспех.
        """

        # Когато изпращането към реалната ръка
        # е изключено, не се изисква проверка.
        if not self.enabled:
            return

        status = self.check_available(
            force=False
        )

        # Прекратява изпълнението,
        # когато физическият манипулатор не отговаря.
        if not status.get("available"):
            raise PhysicalArmUnavailableError(
                f"Physical arm is unavailable "
                f"at {self.ping_host} "
                f"({status.get('reason', 'unknown')})"
            )

    def joints_to_servo(
        self,
        joints: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Преобразува ставните ъгли от RViz в радиани
        до физически сервомоторни стойности в градуси.
        """

        result: Dict[str, float] = {}

        # Обхожда съответствията между стави
        # и физически сервомотори.
        for joint, servo_name in JOINT_TO_SERVO.items():
            # Преобразува ставния ъгъл
            # от радиани в градуси.
            rviz_deg = math.degrees(
                float(joints[joint])
            )

            # Получава калибратора
            # за съответния сервомотор.
            calibrator = self.calibrators.get(
                servo_name
            )

            # Използва калибрацията, когато е налична.
            # При липса използва стандартно отместване 90°.
            servo_deg = (
                calibrator.convert(rviz_deg)
                if calibrator is not None
                else 90.0 + rviz_deg
            )

            # Ограничава стойността до безопасния диапазон.
            result[servo_name] = round(
                clamp(
                    servo_deg,
                    *SAFE_SERVO_RANGE[
                        servo_name
                    ],
                ),
                3,
            )

        return result

    def servo_to_joints(
        self,
        servo_angles: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Convert physical servo degrees to synchronized RViz radians.
        """

        # Речник с изчислените ставни стойности.
        joints: Dict[str, float] = {}

        # Обхожда съответствията между физически
        # сервомотори и ставите в URDF модела.
        for (
            servo_name,
            joint_name,
        ) in SERVO_TO_JOINT.items():
            # Пропуска сервомотора, когато
            # няма подадена негова стойност.
            if servo_name not in servo_angles:
                continue

            # Ограничава физическия ъгъл
            # до безопасния диапазон.
            servo_deg = clamp(
                float(
                    servo_angles[servo_name]
                ),
                *SAFE_SERVO_RANGE[
                    servo_name
                ],
            )

            # Получава калибратора
            # за съответния сервомотор.
            calibrator = self.calibrators.get(
                servo_name
            )

            # Изпълнява обратно калибрационно преобразуване.
            rviz_deg = (
                calibrator.inverse(servo_deg)
                if calibrator is not None
                else servo_deg - 90.0
            )

            # RViz използва радиани.
            joints[joint_name] = math.radians(
                rviz_deg
            )

        return joints

    def gripper_servo_to_rviz_rad(
        self,
        servo_deg: float,
    ) -> float:
        """
        Преобразува физическата стойност на хващача
        до съответната стойност в RViz в радиани.
        """

        # Ограничава стойността
        # до безопасния диапазон на хващача.
        servo_deg = clamp(
            float(servo_deg),
            *SAFE_SERVO_RANGE["gripper"],
        )

        # Получава калибратора на хващача.
        calibrator = self.calibrators.get(
            "gripper"
        )

        # При липса на калибратор използва
        # стандартно отместване от 80°.
        rviz_deg = (
            calibrator.inverse(servo_deg)
            if calibrator is not None
            else servo_deg - 80.0
        )

        return math.radians(rviz_deg)

    def request_status(
        self,
        timeout_sec: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Read the Pi executor's open-loop commanded state.

        These are not encoder measurements. They are the executor's current
        interpolated/commanded servo angles.
        """

        # При изключено физическо управление
        # не изпраща заявка.
        if not self.enabled:
            return {
                "ok": False,
                "state": "PHYSICAL_ARM_DISABLED",
            }

        # Проверява достъпността на Raspberry Pi.
        self.ensure_available()

        # Създава уникален идентификатор
        # на заявката за състояние.
        request_id = (
            f"status-{uuid.uuid4().hex}"
        )

        # UDP пакет за заявка на текущото състояние.
        packet = {
            "request_id": request_id,
            "command": "status",
        }

        # Краен момент за изчакване на отговор.
        deadline = (
            time.monotonic()
            + max(
                0.2,
                float(timeout_sec),
            )
        )

        # Запазва предишната timeout настройка
        # на сокета.
        previous_timeout = (
            self.sock.gettimeout()
        )

        try:
            # Заключва сокета за цялата последователност
            # от изпращане и получаване.
            with self.send_lock:
                # Кодира пакета като UTF-8 JSON.
                payload = json.dumps(
                    packet
                ).encode("utf-8")

                # Изпраща заявката към Raspberry Pi.
                self.sock.sendto(
                    payload,
                    (
                        self.ip,
                        self.port,
                    ),
                )

                # Изчаква съответстващ отговор.
                while (
                    time.monotonic()
                    < deadline
                ):
                    # Изчислява оставащото време.
                    remaining = max(
                        0.05,
                        (
                            deadline
                            - time.monotonic()
                        ),
                    )

                    # Използва кратки периоди
                    # за изчакване на UDP пакет.
                    self.sock.settimeout(
                        min(
                            0.25,
                            remaining,
                        )
                    )

                    try:
                        data, address = (
                            self.sock.recvfrom(8192)
                        )
                    except socket.timeout:
                        continue

                    # Игнорира пакети от друг IP адрес.
                    if address[0] != self.ip:
                        continue

                    try:
                        # Декодира отговора като JSON.
                        reply = json.loads(
                            data.decode("utf-8")
                        )
                    except Exception:
                        continue

                    # Очаква JSON обект.
                    if not isinstance(
                        reply,
                        dict,
                    ):
                        continue

                    # Игнорира отговори
                    # за други заявки.
                    if (
                        str(
                            reply.get(
                                "request_id"
                            )
                            or ""
                        )
                        != request_id
                    ):
                        continue

                    # Получава последните командвани
                    # сервомоторни стойности.
                    current = reply.get(
                        "current_angles"
                    )

                    if isinstance(
                        current,
                        dict,
                    ):
                        for (
                            key,
                            value,
                        ) in current.items():
                            # Записва само познати сервомотори.
                            if key in SAFE_SERVO_RANGE:
                                self.last_commanded_angles[
                                    key
                                ] = round(
                                    clamp(
                                        float(value),
                                        *SAFE_SERVO_RANGE[
                                            key
                                        ],
                                    ),
                                    3,
                                )

                    return reply

            # Не е получен валиден отговор
            # в определеното време.
            raise PhysicalArmCommandError(
                "Timed out waiting for "
                "Raspberry Pi status"
            )

        finally:
            # Възстановява предишния timeout
            # на UDP сокета.
            self.sock.settimeout(
                previous_timeout
            )

    def send_servo_angles(
        self,
        servo_angles: Mapping[str, float],
        duration_sec: float,
        *,
        label: str = "manual_servo_angles",
        wait_for_completion: bool = True,
    ) -> Dict[str, float]:
        """
        Изпраща директно зададени физически
        сервомоторни стойности.
        """

        # Филтрира непознатите имена
        # и ограничава всички стойности.
        clean = {
            key: round(
                clamp(
                    float(value),
                    *SAFE_SERVO_RANGE[key],
                ),
                3,
            )
            for (
                key,
                value,
            ) in servo_angles.items()
            if key in SAFE_SERVO_RANGE
        }

        # Изпраща почистените стойности.
        self._send(
            clean,
            duration_sec,
            label=label,
            wait_for_completion=(
                wait_for_completion
            ),
        )

        return clean

    def _wait_for_completion_reply(
        self,
        request_id: str,
        timeout_sec: float,
    ) -> Dict[str, Any]:
        """
        Изчаква потвърждение от Raspberry Pi,
        че движението е напълно завършено.
        """

        # Краен момент за изчакване.
        deadline = (
            time.monotonic()
            + max(
                1.0,
                timeout_sec,
            )
        )

        # Запазва предишната timeout настройка.
        previous_timeout = (
            self.sock.gettimeout()
        )

        try:
            while (
                time.monotonic()
                < deadline
            ):
                # Изчислява оставащото време.
                remaining = max(
                    0.05,
                    (
                        deadline
                        - time.monotonic()
                    ),
                )

                self.sock.settimeout(
                    min(
                        0.5,
                        remaining,
                    )
                )

                try:
                    # Получава следващия UDP отговор.
                    payload, address = (
                        self.sock.recvfrom(8192)
                    )
                except socket.timeout:
                    continue

                # Игнорира отговори от друг адрес.
                if address[0] != self.ip:
                    continue

                try:
                    # Декодира получения JSON пакет.
                    reply = json.loads(
                        payload.decode("utf-8")
                    )
                except Exception:
                    continue

                # Отговорът трябва да бъде JSON обект.
                if not isinstance(
                    reply,
                    dict,
                ):
                    continue

                # Игнорира отговори за други заявки.
                if (
                    str(
                        reply.get(
                            "request_id"
                        )
                        or ""
                    )
                    != request_id
                ):
                    continue

                # Нормализира състоянието
                # до главни букви.
                state = str(
                    reply.get("state")
                    or ""
                ).upper()

                # Потвърждението за приемане на командата
                # не означава завършено движение.
                if state in {
                    "MOVE_ACCEPTED",
                    "HOME_ACCEPTED",
                }:
                    continue

                # Движението е успешно завършено.
                if state == "MOTION_COMPLETED":
                    # Проверява допълнителното поле ok.
                    if not reply.get(
                        "ok",
                        False,
                    ):
                        raise PhysicalArmCommandError(
                            "Physical arm reported "
                            f"failure: {reply}"
                        )

                    # Запазва последния отговор.
                    self.last_completion_reply = (
                        dict(reply)
                    )

                    return dict(reply)

                # Обработва прекъснати или отхвърлени движения.
                if state in {
                    "MOTION_INTERRUPTED",
                    "MOVE_REJECTED",
                    "HOME_REJECTED",
                }:
                    raise PhysicalArmCommandError(
                        "Physical arm rejected/"
                        "interrupted command: "
                        f"{reply}"
                    )

            # Не е получено потвърждение
            # в рамките на определеното време.
            raise PhysicalArmCommandError(
                "Timed out waiting for "
                "Raspberry Pi motion completion"
            )

        finally:
            # Възстановява предишната timeout настройка.
            self.sock.settimeout(
                previous_timeout
            )

    def _send(
        self,
        angles: Mapping[str, float],
        duration_sec: float,
        *,
        label: str = "move",
        wait_for_completion: bool = False,
    ) -> bool:
        """
        Изпраща UDP команда за движение
        към физическия манипулатор.
        """

        # При изключено реално управление
        # не се изпраща пакет.
        if not self.enabled:
            self.last_send_ok = False
            self.last_error = (
                "physical_arm_disabled"
            )
            return False

        # Проверява физическата достъпност.
        self.ensure_available()

        # Филтрира и ограничава
        # сервомоторните стойности.
        clean_angles = {
            key: round(
                clamp(
                    value,
                    *SAFE_SERVO_RANGE[key],
                ),
                3,
            )
            for (
                key,
                value,
            ) in angles.items()
            if key in SAFE_SERVO_RANGE
        }

        # Не допуска изпращане
        # на празна команда.
        if not clean_angles:
            raise PhysicalArmCommandError(
                "No valid servo angles to send"
            )

        # Създава уникален идентификатор
        # на движението.
        request_id = (
            f"planner-{uuid.uuid4().hex}"
        )

        # При стария формат и без изчакване
        # изпраща само речника с ъглите.
        if (
            self.packet_format
            == "angles_only"
            and not wait_for_completion
        ):
            packet: Any = clean_angles

        else:
            # Създава пълен пакет с продължителност,
            # етикет и идентификатор.
            packet = {
                "request_id": request_id,
                "command": "move",
                "duration": float(
                    duration_sec
                ),
                "angles": clean_angles,
                "label": str(label),
            }

        try:
            # Кодира пакета като UTF-8 JSON.
            payload = json.dumps(
                packet
            ).encode("utf-8")

            # Изпращането и изчакването на отговор
            # използват един и същ заключен UDP сокет.
            with self.send_lock:
                sent = self.sock.sendto(
                    payload,
                    (
                        self.ip,
                        self.port,
                    ),
                )

                # Проверява дали целият пакет е изпратен.
                if sent != len(payload):
                    raise OSError(
                        "partial UDP send: "
                        f"{sent}/{len(payload)} bytes"
                    )

                # По желание изчаква окончателно
                # потвърждение за завършено движение.
                if wait_for_completion:
                    timeout_sec = max(
                        self.completion_timeout_sec,
                        float(duration_sec)
                        + 10.0,
                    )

                    self._wait_for_completion_reply(
                        request_id,
                        timeout_sec,
                    )

        except PhysicalArmCommandError:
            # Запазва неуспешното състояние
            # и препредава грешката.
            self.last_send_ok = False
            raise

        except Exception as exc:
            # Преобразува останалите грешки
            # до PhysicalArmCommandError.
            self.last_send_ok = False
            self.last_error = str(exc)

            raise PhysicalArmCommandError(
                "Could not send command to "
                f"{self.ip}:{self.port}: {exc}"
            ) from exc

        # Командата е изпратена успешно.
        self.last_send_ok = True
        self.last_error = None

        # Актуализира последните командвани стойности.
        self.last_commanded_angles.update(
            clean_angles
        )

        # Записва диагностично съобщение.
        self.logger.info(
            "physical arm command sent to "
            f"{self.ip}:{self.port}; "
            f"duration={duration_sec:.2f}s "
            f"wait_for_completion="
            f"{wait_for_completion} "
            f"angles={clean_angles}"
        )

        return True

    def send_joints(
        self,
        joints: Mapping[str, float],
        duration_sec: float,
        servo_override: Optional[
            Mapping[str, float]
        ] = None,
        *,
        label: str = "move",
        wait_for_completion: bool = False,
    ) -> Dict[str, float]:
        """
        Изпраща ставни стойности към физическия манипулатор.

        По подразбиране ставните стойности се преобразуват
        чрез калибрационния файл. При servo_override
        се използват директно подадените сервомоторни стойности.
        """

        # Използва директните стойности,
        # когато е подадено servo_override.
        angles = (
            dict(servo_override)
            if servo_override is not None
            else self.joints_to_servo(joints)
        )

        # Изпраща изчислените сервомоторни стойности.
        self._send(
            angles,
            duration_sec,
            label=label,
            wait_for_completion=(
                wait_for_completion
            ),
        )

        return angles

    def send_gripper(
        self,
        close: bool,
        duration_sec: float,
        *,
        servo_angle_override: Optional[
            float
        ] = None,
    ) -> Dict[str, float]:
        """
        Отваря или затваря хващача.
        """

        # При липса на директна стойност
        # използва конфигурацията от средата.
        if servo_angle_override is None:
            angle = env_float(
                (
                    "PLANNER_GRIPPER_CLOSE_SERVO_DEG"
                    if close
                    else "PLANNER_GRIPPER_OPEN_SERVO_DEG"
                ),
                (
                    120.0
                    if close
                    else 70.0
                ),
            )

        else:
            # Използва директно подадения физически ъгъл.
            angle = float(
                servo_angle_override
            )

        # Ограничава стойността
        # до безопасния диапазон.
        angle = clamp(
            angle,
            *SAFE_SERVO_RANGE["gripper"],
        )

        # Създава речник с командата
        # за сервомотора на хващача.
        angles = {
            "gripper": round(
                angle,
                3,
            )
        }

        # Изпраща командата.
        self._send(
            angles,
            duration_sec,
            label=(
                "gripper_close"
                if close
                else "gripper_open"
            ),
        )

        return angles