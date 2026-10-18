#!/usr/bin/env python3

# Позволява използването на отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за JSON обработване, математически операции,
# опашки, нишки, време и типови анотации.
import json
import math
import queue
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

# Библиотеки за числова обработка, ROS 2, Redis
# и работа с PointCloud2 съобщения.
import numpy as np
import rclpy
import redis
import sensor_msgs_py.point_cloud2 as pc2

# Многонишков ROS 2 изпълнител и основен клас за възел.
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# Настройки за качеството на обслужване при ROS 2 комуникацията.
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)

# ROS 2 типове съобщения.
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray


# Опитва да зареди променливите на средата от /workspace/.env.
# При липса на библиотеката или файла възелът продължава работа.
try:
    from dotenv import load_dotenv

    load_dotenv("/workspace/.env", override=True)
except Exception:
    pass


# Импортира конфигурацията, структурите от данни и алгоритмите
# за обработване на облака от точки.
from robot_system.scanning.cloud_processor import (
    MARKER_RGB,
    SOURCE_NAME,
    CloudSnapshot,
    ScanArtifacts,
    WorkspaceCloudProcessor,
    WorkspaceScanConfig,
    decode_redis,
)


class WorkspaceScanAgent(Node):
    """
    ROS 2 агент за сканиране на работното пространство.

    Агентът:
    - получава облак от точки от RGB-D камерата;
    - изчаква заявки за сканиране чрез Redis;
    - обработва облака от точки;
    - открива предметите;
    - създава точки за движение;
    - публикува резултатите в ROS 2 и Redis;
    - визуализира откритите предмети в RViz.
    """

    def __init__(self) -> None:
        # Инициализира ROS 2 възела с име workspace_scan_agent.
        super().__init__("workspace_scan_agent")

        # Зарежда конфигурацията на агента от променливите на средата.
        self.config = WorkspaceScanConfig.from_env()

        # Създава обработващия модул за облака от точки.
        self.processor = WorkspaceCloudProcessor(self.config)

        # Кратка локална препратка към конфигурацията.
        cfg = self.config

        # Създава връзка към Redis.
        #
        # decode_responses=True връща текстови стойности,
        # вместо последователности от байтове.
        self.redis = redis.Redis.from_url(
            cfg.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=None,
            health_check_interval=30,
        )

        # Заключване за безопасен достъп до последния облак
        # от няколко различни нишки.
        self.state_lock = threading.RLock()

        # Последният валиден облак от точки и времето на получаването му.
        self.latest_cloud: Optional[CloudSnapshot] = None

        # Събитие за спиране на работните нишки.
        self.stop_event = threading.Event()

        # Опашка със заявки за сканиране.
        #
        # Всеки елемент съдържа:
        # - идентификатор на заявката;
        # - съдържание на заявката.
        self.request_queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = (
            queue.Queue()
        )

        # Запазва идентификатора на последния запис в Redis потока,
        # за да не бъдат обработвани повторно стари заявки.
        self.last_request_stream_id = self.get_stream_tail(
            cfg.request_stream
        )

        # Време на последното автоматично сканиране.
        self.last_auto_scan = 0.0

        # Абонира се за облака от точки от RGB-D камерата.
        #
        # Използва qos_profile_sensor_data, който е предназначен
        # за сензорни потоци с висока честота.
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            cfg.point_cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        # QoS профил за RViz маркерите.
        #
        # TRANSIENT_LOCAL позволява последните маркери да бъдат
        # получени и от абонат, който се е свързал по-късно.
        marker_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Публикува JSON описание на откритите предмети.
        self.objects_pub = self.create_publisher(
            String,
            cfg.objects_json_topic,
            10,
        )

        # Публикува JSON документ с точките за движение.
        self.motion_points_pub = self.create_publisher(
            String,
            cfg.motion_points_json_topic,
            10,
        )

        # Публикува RViz маркери за откритите предмети
        # и техните центрове.
        self.marker_pub = self.create_publisher(
            MarkerArray,
            cfg.motion_point_markers_topic,
            marker_qos,
        )

        # Публикува обединен облак, съдържащ само точките
        # от избраните обектни клъстери.
        self.cluster_pub = self.create_publisher(
            PointCloud2,
            cfg.selected_clusters_topic,
            5,
        )

        # Нишка за четене на заявки от Redis.
        self.redis_thread = threading.Thread(
            target=self.redis_request_loop,
            name="workspace-scan-redis-reader",
            daemon=True,
        )

        # Нишка за последователно обработване на заявките за сканиране.
        self.worker_thread = threading.Thread(
            target=self.scan_worker_loop,
            name="workspace-scan-worker",
            daemon=True,
        )

        # Стартира двете работни нишки.
        self.redis_thread.start()
        self.worker_thread.start()

        # Създава таймер за автоматично сканиране,
        # когато периодът е по-голям от нула.
        self.auto_timer = (
            self.create_timer(
                max(0.2, cfg.auto_scan_period_sec),
                self.auto_scan_timer_callback,
            )
            if cfg.auto_scan_period_sec > 0.0
            else None
        )

        # Диагностични съобщения при успешно стартиране.
        self.get_logger().info(
            "✅ workspace_scan_agent ready; geometry moved to agent_tools"
        )
        self.get_logger().info(
            f"Point cloud: {cfg.point_cloud_topic}"
        )
        self.get_logger().info(
            f"Output frame: {cfg.target_frame}"
        )
        self.get_logger().info(
            f"Maximum object distance from base: "
            f"{cfg.max_base_distance:.3f} m"
        )

    def cloud_callback(self, msg: PointCloud2) -> None:
        """
        Обработва входящ облак от точки.

        Проверява наличието на координатните полета x, y и z,
        след което запазва последното валидно съобщение.
        """

        # Създава множество с имената на полетата в PointCloud2.
        names = {field.name for field in msg.fields}

        # Облак без x, y и z координати не може да бъде обработен.
        if not {"x", "y", "z"}.issubset(names):
            self.get_logger().warning(
                "Ignoring PointCloud2 without x/y/z fields: "
                f"{sorted(names)}"
            )
            return

        # Игнорира празни съобщения.
        if msg.width * msg.height <= 0 or not msg.data:
            return

        # Заключва състоянието при актуализиране на последния облак.
        with self.state_lock:
            # Проверява дали това е първият получен облак.
            first = self.latest_cloud is None

            # Запазва съобщението и монотонното време на получаването.
            self.latest_cloud = CloudSnapshot(
                msg=msg,
                received_monotonic=time.monotonic(),
            )

        # Извежда информация само при първия успешно получен облак.
        if first:
            self.get_logger().info(
                f"✅ cloud received; "
                f"frame={msg.header.frame_id}; "
                f"fields={sorted(names)}"
            )

    def destroy_node(self) -> bool:
        """
        Спира работните нишки и унищожава ROS 2 възела.
        """

        # Сигнализира на фоновите нишки да приключат.
        self.stop_event.set()

        # Извиква стандартното унищожаване на ROS 2 възела.
        return super().destroy_node()

    def auto_scan_timer_callback(self) -> None:
        """
        Добавя автоматична заявка за сканиране.

        Автоматична заявка се добавя само когато:
        - е изминал определеният период;
        - няма друга чакаща заявка.
        """

        cfg = self.config
        now = time.monotonic()

        # Предотвратява прекалено честото автоматично сканиране.
        if (
            now - self.last_auto_scan
            < cfg.auto_scan_period_sec * 0.8
        ):
            return

        # Добавя автоматична заявка само когато опашката е празна.
        if self.request_queue.empty():
            self.last_auto_scan = now

            self.request_queue.put(
                (
                    f"auto-scan-{time.time_ns()}",
                    {
                        "action": "scan_workspace_and_make_points",
                        "auto": True,
                    },
                )
            )

    def get_stream_tail(self, stream: str) -> str:
        """
        Връща идентификатора на последния запис в Redis поток.

        Ако потокът е празен или възникне грешка,
        се връща началният идентификатор 0-0.
        """

        try:
            # Прочита най-новия запис от потока.
            rows = self.redis.xrevrange(stream, count=1)

            # При наличен запис връща неговия идентификатор.
            if rows:
                return decode_redis(rows[0][0])

        except Exception as exc:
            self.get_logger().warning(
                f"Could not read Redis stream tail: {exc}"
            )

        return "0-0"

    def redis_request_loop(self) -> None:
        """
        Чете заявки за сканиране от Redis потока.

        Всяка нова заявка се декодира и се добавя
        към локалната опашка за обработване.
        """

        cfg = self.config

        # Продължава работа, докато възелът не бъде спрян.
        while not self.stop_event.is_set():
            try:
                # Изчаква нови записи в Redis потока.
                rows = self.redis.xread(
                    {
                        cfg.request_stream:
                            self.last_request_stream_id
                    },
                    count=20,
                    block=1000,
                )

            except Exception as exc:
                # При Redis грешка изчаква кратко,
                # преди да направи нов опит.
                if not self.stop_event.is_set():
                    self.get_logger().warning(
                        f"Redis XREAD failed: {exc}"
                    )
                    time.sleep(0.5)

                continue

            # Обхожда върнатите потоци и записи.
            for _stream, entries in rows:
                for entry_id, fields in entries:
                    # Декодира идентификатора на текущия запис.
                    entry_text = decode_redis(entry_id)

                    # Запазва идентификатора, за да продължи
                    # след този запис при следващото четене.
                    self.last_request_stream_id = entry_text

                    # Декодира ключовете и стойностите на Redis записа.
                    decoded = {
                        decode_redis(key): decode_redis(value)
                        for key, value in fields.items()
                    }

                    # Опитва да прочете JSON съдържанието на заявката.
                    try:
                        payload = json.loads(
                            decoded.get("payload", "{}")
                        )
                    except Exception:
                        payload = {}

                    # Използва подадения request_id или създава нов.
                    request_id = str(
                        payload.get("request_id")
                        or f"workspace-scan-{time.time_ns()}"
                    )

                    # Създава временен Redis ключ за заключване
                    # на конкретния запис.
                    lock_key = (
                        f"lock:workspace_scan_entry:{entry_text}"
                    )

                    try:
                        # NX позволява записване само ако ключът
                        # все още не съществува.
                        #
                        # Така една заявка не се обработва
                        # едновременно от няколко процеса.
                        if not self.redis.set(
                            lock_key,
                            "1",
                            nx=True,
                            ex=120,
                        ):
                            continue

                    except Exception:
                        # При проблем със заключването заявката
                        # все пак може да бъде обработена локално.
                        pass

                    # Добавя заявката към работната опашка.
                    self.request_queue.put(
                        (request_id, payload)
                    )

    def scan_worker_loop(self) -> None:
        """
        Обработва заявките за сканиране от локалната опашка.
        """

        cfg = self.config

        while not self.stop_event.is_set():
            try:
                # Изчаква заявка до 0.25 секунди.
                request_id, payload = (
                    self.request_queue.get(timeout=0.25)
                )

            except queue.Empty:
                continue

            try:
                # Изпълнява сканирането.
                result = self.scan_workspace(request_id)

            except Exception as exc:
                # Неочаквана грешка при сканирането
                # се записва и се връща структуриран резултат.
                self.get_logger().exception(
                    f"Workspace scan crashed: {exc}"
                )

                result = self.failure_result(
                    request_id,
                    "SCAN_CRASHED",
                    error=f"{type(exc).__name__}: {exc}",
                )

            # При автоматично сканиране не се публикува
            # резултат в потока за отговори.
            if not payload.get("auto", False):
                try:
                    self.redis.xadd(
                        cfg.result_stream,
                        {
                            "payload": json.dumps(
                                result,
                                ensure_ascii=False,
                            )
                        },
                    )

                except Exception as exc:
                    self.get_logger().error(
                        f"Could not publish Redis result: {exc}"
                    )

            # Отбелязва заявката като приключена.
            self.request_queue.task_done()

    def wait_for_cloud(
        self,
    ) -> Tuple[Optional[PointCloud2], Optional[str]]:
        """
        Изчаква актуален облак от точки.

        Връща:
        - PointCloud2 и None при успех;
        - None и NO_POINT_CLOUD, когато няма получен облак;
        - None и STALE_POINT_CLOUD, когато облакът е остарял.
        """

        cfg = self.config

        # Определя крайния момент за изчакване.
        deadline = (
            time.monotonic()
            + cfg.input_wait_timeout_sec
        )

        while (
            time.monotonic() < deadline
            and not self.stop_event.is_set()
        ):
            # Чете последния облак под заключване.
            with self.state_lock:
                snapshot = self.latest_cloud

            if snapshot is not None:
                # Проверява дали облакът е достатъчно актуален.
                if (
                    time.monotonic()
                    - snapshot.received_monotonic
                    <= cfg.cloud_stale_timeout_sec
                ):
                    return snapshot.msg, None

            # Кратко изчакване преди следващата проверка.
            time.sleep(0.05)

        # След изтичане на времето проверява
        # дали някога е бил получен облак.
        with self.state_lock:
            snapshot = self.latest_cloud

        return (
            (None, "NO_POINT_CLOUD")
            if snapshot is None
            else (None, "STALE_POINT_CLOUD")
        )

    def scan_workspace(
        self,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Изпълнява пълно сканиране на работното пространство.

        Последователността включва:
        - изчистване на старото състояние;
        - изчакване на актуален облак;
        - обработване на облака;
        - откриване на предмети;
        - публикуване на резултатите.
        """

        # При активирана настройка изчиства
        # резултатите от предишното сканиране.
        if self.config.clear_state_before_scan:
            self.clear_previous_scan_state(request_id)

        # Изчаква актуален облак от точки.
        cloud_msg, error = self.wait_for_cloud()

        # Връща неуспешен резултат при липса на валиден облак.
        if cloud_msg is None:
            self.publish_empty(
                error or "NO_POINT_CLOUD"
            )

            return self.failure_result(
                request_id,
                error or "NO_POINT_CLOUD",
            )

        # Предава облака към геометричния обработващ модул.
        artifacts = self.processor.process_cloud(cloud_msg)

        # Обработва неуспешен резултат от анализа на облака.
        if not artifacts.ok:
            self.publish_empty(artifacts.state)

            # Копира допълнителната диагностична информация.
            extra = dict(artifacts.details)

            # Добавя параметрите на равнината на масата,
            # когато такава е била открита.
            if artifacts.table_plane is not None:
                extra["table_plane"] = (
                    self.processor.table_plane_dict(
                        artifacts.table_plane
                    )
                )

            # Добавя до 30 отхвърлени компонента
            # за диагностични цели.
            if artifacts.rejected:
                extra["rejected"] = (
                    artifacts.rejected[:30]
                )

            return self.failure_result(
                request_id,
                artifacts.state,
                **extra,
            )

        # Публикува успешно откритите предмети.
        self.publish_success(artifacts)

        # Преобразува параметрите на равнината на масата
        # в речник, подходящ за JSON.
        plane_dict = (
            self.processor.table_plane_dict(
                artifacts.table_plane
            )
            if artifacts.table_plane is not None
            else None
        )

        # Връща пълния успешен резултат от сканирането.
        return {
            "ok": True,
            "request_id": request_id,
            "scan_state": "OK",
            "count": len(artifacts.objects),
            "objects": artifacts.objects,
            "motion_points": (
                artifacts.motion_document["motion_points"]
            ),
            "table_plane": plane_dict,
            "max_base_distance_m": (
                self.config.max_base_distance
            ),
            "far_candidate_points_ignored": (
                artifacts.details.get(
                    "far_candidate_points_ignored",
                    0,
                )
            ),
            "timestamp": time.time(),
        }

    def clear_previous_scan_state(
        self,
        request_id: str,
    ) -> None:
        """
        Изчиства състоянието от предишното сканиране.
        """

        cfg = self.config

        # Публикува временно състояние SCANNING
        # и премахва предишните предмети от RViz.
        self.publish_empty("SCANNING")

        # Redis ключове, които трябва да бъдат изтрити.
        keys_to_delete = [
            cfg.object_instance_registry_key,
            cfg.last_target_key,
        ]

        # При активирана настройка изчиства и натрупаните
        # корекции на целевите точки.
        if cfg.clear_target_offsets_before_scan:
            keys_to_delete.append(
                cfg.target_offsets_key
            )

        # Изтрива избраните Redis ключове.
        deleted = self.redis.delete(*keys_to_delete)

        # Публикува празен облак за избраните клъстери.
        self.publish_empty_clusters()

        self.get_logger().info(
            "Cleared previous workspace objects and stale "
            "target corrections before scan "
            f"request={request_id}; deleted_keys={deleted}"
        )

    def publish_success(
        self,
        artifacts: ScanArtifacts,
    ) -> None:
        """
        Публикува резултатите от успешно сканиране.
        """

        cfg = self.config

        # Преобразува документите за предметите
        # и точките за движение в JSON.
        objects_json = json.dumps(
            artifacts.objects_document,
            ensure_ascii=False,
        )
        motion_json = json.dumps(
            artifacts.motion_document,
            ensure_ascii=False,
        )

        # Публикува JSON документите в ROS 2.
        self.objects_pub.publish(
            String(data=objects_json)
        )
        self.motion_points_pub.publish(
            String(data=motion_json)
        )

        # Записва актуалното състояние в Redis.
        self.redis.set(
            cfg.workspace_state_key,
            objects_json,
        )
        self.redis.set(
            cfg.motion_points_key,
            motion_json,
        )

        # Публикува RViz маркерите и избраните клъстери.
        self.publish_markers(artifacts.objects)
        self.publish_selected_clusters(
            artifacts.selected_clusters
        )

        # Създава кратко текстово описание
        # на координатите на откритите предмети.
        summary = ", ".join(
            (
                f"{obj['point_label']}="
                f"({obj['x']:.3f},"
                f"{obj['y']:.3f},"
                f"{obj['z']:.3f})"
            )
            for obj in artifacts.objects
        )

        # Брой точки, игнорирани поради прекалена
        # отдалеченост от основата.
        far_ignored = artifacts.details.get(
            "far_candidate_points_ignored",
            0,
        )

        self.get_logger().info(
            f"✅ {len(artifacts.objects)} object centres; "
            f"ignored far candidate points={far_ignored}; "
            f"{summary}"
        )

        # По желание публикува диагностична информация
        # за отхвърлените компоненти.
        if (
            cfg.publish_rejected_debug
            and artifacts.rejected
        ):
            self.get_logger().info(
                f"Rejected {len(artifacts.rejected)} "
                "components: "
                + ", ".join(
                    item["reason"]
                    for item in artifacts.rejected[:8]
                )
            )

    def publish_empty(self, state: str) -> None:
        """
        Публикува празно състояние без открити предмети.

        Използва се при:
        - започване на ново сканиране;
        - липса на облак;
        - грешка при обработването;
        - липса на валидни предмети.
        """

        cfg = self.config

        # Празен документ за предметите.
        empty_objects = {
            "source": SOURCE_NAME,
            "scan_state": state,
            "frame": cfg.target_frame,
            "objects": [],
            "timestamp": time.time(),
        }

        # Празен документ за точките за движение.
        empty_points = {
            "source": SOURCE_NAME,
            "scan_state": state,
            "frame": cfg.target_frame,
            "motion_points": [],
            "timestamp": time.time(),
        }

        # Преобразува документите в JSON.
        objects_json = json.dumps(
            empty_objects,
            ensure_ascii=False,
        )
        points_json = json.dumps(
            empty_points,
            ensure_ascii=False,
        )

        # Публикува празните документи в ROS 2.
        self.objects_pub.publish(
            String(data=objects_json)
        )
        self.motion_points_pub.publish(
            String(data=points_json)
        )

        # Актуализира Redis състоянието.
        self.redis.set(
            cfg.workspace_state_key,
            objects_json,
        )
        self.redis.set(
            cfg.motion_points_key,
            points_json,
        )

        # Създава MarkerArray с команда за изтриване
        # на всички предишни RViz маркери.
        markers = MarkerArray()

        delete = Marker()
        delete.header.frame_id = cfg.target_frame
        delete.header.stamp = (
            self.get_clock().now().to_msg()
        )
        delete.action = Marker.DELETEALL

        markers.markers.append(delete)

        # Публикува командата за изтриване.
        self.marker_pub.publish(markers)

    def publish_markers(
        self,
        objects: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Създава и публикува RViz маркери за предметите.

        За всеки предмет се създават:
        - маркер за приблизителното тяло;
        - жълт маркер за центъра;
        - текстов етикет с координатите.
        """

        cfg = self.config
        array = MarkerArray()

        # Първият маркер премахва всички стари маркери.
        delete = Marker()
        delete.header.frame_id = cfg.target_frame
        delete.header.stamp = (
            self.get_clock().now().to_msg()
        )
        delete.action = Marker.DELETEALL

        array.markers.append(delete)

        # Уникален идентификатор за следващия маркер.
        marker_id = 1

        for obj in objects:
            # Маркер за приблизителното тяло на предмета.
            body = Marker()
            body.header.frame_id = cfg.target_frame
            body.header.stamp = (
                self.get_clock().now().to_msg()
            )
            body.ns = "simple_object_bodies"
            body.id = marker_id
            marker_id += 1

            body.type = Marker.SPHERE
            body.action = Marker.ADD

            # Поставя маркера в центъра на предмета.
            body.pose.position.x = float(obj["x"])
            body.pose.position.y = float(obj["y"])
            body.pose.position.z = float(obj["z"])
            body.pose.orientation.w = 1.0

            # Използва измерените размери на предмета
            # за мащаба на маркера.
            body.scale.x = float(
                obj["dimensions"]["x"]
            )
            body.scale.y = float(
                obj["dimensions"]["y"]
            )
            body.scale.z = float(
                obj["dimensions"]["z"]
            )

            # Избира цвета според класифицирания
            # цветови клас на предмета.
            r, g, b = MARKER_RGB.get(
                obj["colors"][0],
                MARKER_RGB["unknown"],
            )

            body.color.r = r
            body.color.g = g
            body.color.b = b
            body.color.a = 0.62

            array.markers.append(body)

            # Жълт сферичен маркер за точния център.
            center = Marker()
            center.header.frame_id = cfg.target_frame
            center.header.stamp = (
                self.get_clock().now().to_msg()
            )
            center.ns = "simple_object_centres"
            center.id = marker_id
            marker_id += 1

            center.type = Marker.SPHERE
            center.action = Marker.ADD

            center.pose.position.x = float(obj["x"])
            center.pose.position.y = float(obj["y"])
            center.pose.position.z = float(obj["z"])
            center.pose.orientation.w = 1.0

            center.scale.x = 0.012
            center.scale.y = 0.012
            center.scale.z = 0.012

            center.color.r = 1.0
            center.color.g = 1.0
            center.color.b = 0.0
            center.color.a = 1.0

            array.markers.append(center)

            # Текстов етикет с името и координатите.
            text = Marker()
            text.header.frame_id = cfg.target_frame
            text.header.stamp = (
                self.get_clock().now().to_msg()
            )
            text.ns = "simple_object_labels"
            text.id = marker_id
            marker_id += 1

            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = float(obj["x"])
            text.pose.position.y = float(obj["y"])

            # Поставя текста над горната част на предмета.
            text.pose.position.z = float(
                obj["top_position"]["z"] + 0.045
            )

            text.pose.orientation.w = 1.0
            text.scale.z = 0.026

            # Бял цвят на текста.
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            # Показва етикета и координатите.
            text.text = (
                f"{obj['point_label']}\n"
                f"x={obj['x']:.3f} "
                f"y={obj['y']:.3f} "
                f"z={obj['z']:.3f}"
            )

            array.markers.append(text)

        # Публикува всички маркери с едно съобщение.
        self.marker_pub.publish(array)

    def publish_empty_clusters(self) -> None:
        """
        Публикува празен PointCloud2 за избраните клъстери.
        """

        # Създава заглавната част на облака.
        header = Header()
        header.frame_id = self.config.target_frame
        header.stamp = (
            self.get_clock().now().to_msg()
        )

        # Създава и публикува празен XYZ облак.
        self.cluster_pub.publish(
            pc2.create_cloud_xyz32(header, [])
        )

    def publish_selected_clusters(
        self,
        clusters: Sequence[np.ndarray],
    ) -> None:
        """
        Обединява и публикува точките от валидните обектни клъстери.
        """

        cfg = self.config

        # Оставя само непразните клъстери.
        valid = [
            cluster
            for cluster in clusters
            if cluster.shape[0] > 0
        ]

        # Не публикува съобщение, ако няма валидни точки.
        if not valid:
            return

        # Обединява всички клъстери в един масив.
        combined = np.vstack(valid).astype(
            np.float32,
            copy=False,
        )

        # Създава заглавната част на PointCloud2 съобщението.
        header = Header()
        header.frame_id = cfg.target_frame
        header.stamp = (
            self.get_clock().now().to_msg()
        )

        # Преобразува NumPy масива в ROS 2 PointCloud2
        # и го публикува.
        self.cluster_pub.publish(
            pc2.create_cloud_xyz32(
                header,
                combined.tolist(),
            )
        )

    @staticmethod
    def failure_result(
        request_id: str,
        state: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Създава стандартен резултат за неуспешно сканиране.
        """

        # Основна структура на неуспешния резултат.
        result: Dict[str, Any] = {
            "ok": False,
            "request_id": request_id,
            "scan_state": state,
            "count": 0,
            "objects": [],
            "motion_points": [],
            "timestamp": time.time(),
        }

        # Добавя допълнителна диагностична информация.
        result.update(extra)

        return result


def main(
    args: Optional[Sequence[str]] = None,
) -> None:
    """
    Главна функция за стартиране на workspace_scan_agent.
    """

    # Инициализира ROS 2.
    rclpy.init(args=args)

    # Създава възела за сканиране.
    node = WorkspaceScanAgent()

    # Използва многонишков изпълнител с три работни нишки,
    # за да могат абонаментите, таймерите и обратните
    # извиквания да се обработват паралелно.
    executor = MultiThreadedExecutor(num_threads=3)

    # Добавя възела към изпълнителя.
    executor.add_node(node)

    try:
        # Стартира ROS 2 цикъла.
        executor.spin()

    except KeyboardInterrupt:
        # Позволява прекратяване чрез Ctrl+C.
        pass

    finally:
        # Сигнализира на фоновите нишки да приключат.
        node.stop_event.set()

        # Спира многонишковия изпълнител.
        executor.shutdown()

        # Освобождава ресурсите на възела.
        node.destroy_node()

        # Изключва ROS 2.
        rclpy.shutdown()


# Стартира main() само когато файлът е изпълнен директно.
if __name__ == "__main__":
    main()