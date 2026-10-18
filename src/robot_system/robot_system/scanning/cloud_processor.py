#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за:
# - математически операции;
# - работа с променливи на средата;
# - регулярни изрази;
# - времеви стойности;
# - специализирани структури от данни;
# - декларативни класове;
# - типови анотации.
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# NumPy се използва за ефективна обработка
# на облака от точки и цветовите стойности.
import numpy as np

# ROS 2 помощен модул за прочитане
# на съобщения от тип PointCloud2.
import sensor_msgs_py.point_cloud2 as pc2

# ROS 2 съобщение, съдържащо облак от точки.
from sensor_msgs.msg import PointCloud2


# Име на източника, което се записва
# в генерираните документи и точки.
SOURCE_NAME = "cloud_center_only_agent"


# Съответствие между английските имена
# на цветовете и техните български прилагателни.
COLOR_BG = {
    "red": "червен",
    "green": "зелен",
    "blue": "син",
    "yellow": "жълт",
    "orange": "оранжев",
    "purple": "лилав",
    "pink": "розов",
    "cyan": "циан",
    "white": "бял",
    "black": "черен",
    "gray": "сив",
    "unknown": "неизвестен",
}


# Нормализирани RGB стойности, използвани
# за визуални маркери на отделните цветове.
MARKER_RGB = {
    "red": (0.95, 0.08, 0.08),
    "green": (0.08, 0.90, 0.18),
    "blue": (0.08, 0.28, 0.95),
    "yellow": (0.95, 0.90, 0.08),
    "orange": (1.00, 0.42, 0.05),
    "purple": (0.60, 0.18, 0.88),
    "pink": (1.00, 0.30, 0.65),
    "cyan": (0.05, 0.90, 0.90),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
    "gray": (0.48, 0.48, 0.48),
    "unknown": (0.90, 0.45, 0.05),
}


def env(
    name: str,
    default: str,
) -> str:
    """
    Зарежда текстова стойност от променлива на средата.
    """

    return os.getenv(
        name,
        default,
    )


def env_float(
    name: str,
    default: float,
) -> float:
    """
    Зарежда число с плаваща запетая
    от променлива на средата.
    """

    try:
        return float(
            os.getenv(
                name,
                str(default),
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        # При липсваща или невалидна стойност
        # връща стойността по подразбиране.
        return default


def env_int(
    name: str,
    default: int,
) -> int:
    """
    Зарежда цяло число от променлива на средата.
    """

    try:
        return int(
            os.getenv(
                name,
                str(default),
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def env_bool(
    name: str,
    default: bool,
) -> bool:
    """
    Зарежда булева стойност от променлива на средата.
    """

    raw = os.getenv(name)

    # При липсваща стойност използва
    # предварително зададената конфигурация.
    if raw is None:
        return default

    # Разпознава най-често използваните
    # текстови представяния на истина.
    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def safe_label(
    value: str,
) -> str:
    """
    Преобразува текст до безопасен етикет,
    подходящ за идентификатор.
    """

    # Преобразува стойността до малки букви
    # и премахва излишните интервали.
    value = str(
        value or "object"
    ).strip().lower()

    # Заменя всички неподдържани символи
    # с долна черта.
    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    # Обединява последователните долни черти.
    value = re.sub(
        r"_+",
        "_",
        value,
    ).strip("_")

    return value or "object"


def decode_redis(
    value: Any,
) -> str:
    """
    Преобразува Redis bytes стойност
    до UTF-8 текст.
    """

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


@dataclass(frozen=True)
class TablePlane:
    """
    Представя равнината на работната маса
    чрез уравнението z = a*x + b*y + c.
    """

    # Коефициент пред X.
    a: float

    # Коефициент пред Y.
    b: float

    # Свободен    a: float

    # Коефициент пред Y.
    b: float

    # Свободен член.
    c: float

    # Източник или метод на изчисляване.
    source: str

    # Брой точки, използвани като вътрешни точки
    # при оценяването на равнината.
    inliers: int

    # Медиана на абсолютната остатъчна грешка.
    residual_median: float

    def z_at(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
    ) -> np.ndarray | float:
        """
        Изчислява височината на равнината
        за дадени X и Y координати.
        """

        return (
            self.a * x
            + self.b * y
            + self.c
        )


@dataclass
class CloudSnapshot:
    """
    Съхранява последно получен облак от точки
    и момента на неговото приемане.
    """

    # ROS 2 съобщението с облака от точки.
    msg: PointCloud2

    # Монотонен времеви печат на приемане.
    received_monotonic: float


@dataclass(frozen=True)
class WorkspaceScanConfig:
    """
    Съдържа пълната конфигурация
    за обработване на работното пространство.
    """

    # ROS 2 тема с входния облак от точки.
    point_cloud_topic: str

    # Координатна система на изходните точки.
    target_frame: str

    # ROS 2 тема за JSON документа с предметите.
    objects_json_topic: str

    # ROS 2 тема за JSON документа
    # с точките за движение.
    motion_points_json_topic: str

    # ROS 2 тема за визуалните маркери
    # на точките за движение.
    motion_point_markers_topic: str

    # ROS 2 тема с избраните облаци
    # на откритите предмети.
    selected_clusters_topic: str

    # Адрес на Redis.
    redis_url: str

    # Redis Stream за заявки за сканиране.
    request_stream: str

    # Redis Stream за резултати от сканиране.
    result_stream: str

    # Redis ключ за точките за движение.
    motion_points_key: str

    # Redis ключ за състоянието
    # на работното пространство.
    workspace_state_key: str

    # Redis ключ за регистъра на екземплярите.
    object_instance_registry_key: str

    # Redis ключ за последно избраната цел.
    last_target_key: str

    # Redis ключ за корекциите на целите.
    target_offsets_key: str

    # Определя дали състоянието на предметите
    # да се изчиства преди ново сканиране.
    clear_state_before_scan: bool

    # Определя дали ръчните корекции на целите
    # да се изчистват преди сканиране.
    clear_target_offsets_before_scan: bool

    # Минимална и максимална X координата.
    workspace_x_min: float
    workspace_x_max: float

    # Минимална и максимална Y координата.
    workspace_y_min: float
    workspace_y_max: float

    # Минимална и максимална Z координата.
    workspace_z_min: float
    workspace_z_max: float

    # Максимално хоризонтално разстояние
    # от основата на манипулатора.
    max_base_distance: float

    # Незадължително директно зададена
    # височина на масата.
    table_z_override_text: str

    # Размер на интервалите при хистограмата
    # за откриване на масата.
    table_histogram_bin: float

    # Начален допустим диапазон около масата.
    table_initial_band: float

    # Краен праг за вътрешни точки.
    table_final_inlier: float

    # Максимален допустим наклон
    # на равнината на масата.
    table_max_tilt: float

    # Минимална и максимална височина на предмет.
    object_min_height: float
    object_max_height: float

    # Минимален и максимален хоризонтален размер.
    object_min_xy_size: float
    object_max_xy_size: float

    # Минимален и максимален брой точки
    # в един предмет.
    object_min_points: int
    object_max_points: int

    # Максимален брой предмети в един резултат.
    max_objects: int

    # Размер на XY воксела за групиране.
    xy_voxel_size: float

    # Радиус на съседство между вокселите.
    xy_neighbour_range: int

    # Минимален брой заети воксели
    # за валиден компонент.
    min_occupied_voxels: int

    # Долна и горна перцентилна граница
    # при изчисляване на размерите.
    low_percentile: float
    high_percentile: float

    # Перцентил за височината
    # на горната повърхност.
    top_percentile: float

    # Височини за подход, хващане и поставяне.
    point_approach_height: float
    point_grasp_height_offset: float
    point_place_height_offset: float

    # Максимално допустима възраст
    # на входния облак.
    cloud_stale_timeout_sec: float

    # Максимално време за изчакване
    # на входен облак.
    input_wait_timeout_sec: float

    # Период за автоматично сканиране.
    auto_scan_period_sec: float

    # Определя дали отхвърлените компоненти
    # да се публикуват за диагностика.
    publish_rejected_debug: bool

    @classmethod
    def from_env(
        cls,
    ) -> "WorkspaceScanConfig":
        """
        Създава конфигурацията чрез стойности
        от променливите на средата.
        """

        return cls(
            # Входна ROS 2 тема с цветен облак от точки.
            point_cloud_topic=env(
                "SIMPLE_OBJECT_POINT_CLOUD_TOPIC",
                "/digital_twin/live_cloud_rgb",
            ),

            # Всички изходни координати се представят
            # спрямо основата на манипулатора.
            target_frame=env(
                "SIMPLE_OBJECT_TARGET_FRAME",
                "base_link",
            ),

            # Изходни ROS 2 теми.
            objects_json_topic=env(
                "OBJECTS_JSON_TOPIC",
                "/digital_twin/objects_json",
            ),
            motion_points_json_topic=env(
                "MOTION_POINTS_JSON_TOPIC",
                "/digital_twin/motion_points_json",
            ),
            motion_point_markers_topic=env(
                "MOTION_POINT_MARKERS_TOPIC",
                "/digital_twin/motion_point_markers",
            ),
            selected_clusters_topic=env(
                "SELECTED_CLUSTERS_TOPIC",
                "/digital_twin/selected_object_clusters",
            ),

            # Redis конфигурация.
            redis_url=env(
                "REDIS_URL",
                "redis://127.0.0.1:6379/0",
            ),
            request_stream=env(
                "WORKSPACE_SCAN_REQUEST_STREAM",
                "stream:workspace_scan_requests",
            ),
            result_stream=env(
                "WORKSPACE_SCAN_RESULT_STREAM",
                "stream:workspace_scan_results",
            ),
            motion_points_key=env(
                "MOTION_POINTS_KEY",
                "workspace:motion_points",
            ),
            workspace_state_key=env(
                "WORKSPACE_STATE_KEY",
                "workspace:state",
            ),
            object_instance_registry_key=env(
                "OBJECT_INSTANCE_REGISTRY_KEY",
                "workspace:object_instances",
            ),
            last_target_key=env(
                "LAST_TARGET_KEY",
                "workspace:last_planning_target",
            ),
            target_offsets_key=env(
                "TARGET_OFFSETS_KEY",
                "workspace:target_offsets",
            ),

            # Поведение при ново сканиране.
            clear_state_before_scan=env_bool(
                "CLEAR_OBJECT_STATE_BEFORE_SCAN",
                True,
            ),
            clear_target_offsets_before_scan=env_bool(
                "CLEAR_TARGET_OFFSETS_BEFORE_SCAN",
                True,
            ),

            # Граници на работното пространство.
            workspace_x_min=env_float(
                "SIMPLE_OBJECT_X_MIN",
                0.05,
            ),
            workspace_x_max=env_float(
                "SIMPLE_OBJECT_X_MAX",
                0.80,
            ),
            workspace_y_min=env_float(
                "SIMPLE_OBJECT_Y_MIN",
                -0.50,
            ),
            workspace_y_max=env_float(
                "SIMPLE_OBJECT_Y_MAX",
                0.50,
            ),
            workspace_z_min=env_float(
                "SIMPLE_OBJECT_Z_MIN",
                -0.08,
            ),
            workspace_z_max=env_float(
                "SIMPLE_OBJECT_Z_MAX",
                0.30,
            ),

            # Ограничение на разстоянието от основата.
            max_base_distance=max(
                0.0,
                env_float(
                    "SIMPLE_OBJECT_MAX_BASE_DISTANCE_M",
                    0.40,
                ),
            ),

            # Настройки за откриване на масата.
            table_z_override_text=env(
                "SIMPLE_OBJECT_TABLE_Z",
                "",
            ).strip(),
            table_histogram_bin=env_float(
                "SIMPLE_OBJECT_TABLE_BIN_M",
                0.0025,
            ),
            table_initial_band=env_float(
                "SIMPLE_OBJECT_TABLE_INITIAL_BAND_M",
                0.022,
            ),
            table_final_inlier=env_float(
                "SIMPLE_OBJECT_TABLE_INLIER_M",
                0.008,
            ),
            table_max_tilt=env_float(
                "SIMPLE_OBJECT_TABLE_MAX_SLOPE",
                0.15,
            ),

            # Геометрични ограничения за предметите.
            object_min_height=env_float(
                "SIMPLE_OBJECT_MIN_HEIGHT_M",
                0.008,
            ),
            object_max_height=env_float(
                "SIMPLE_OBJECT_MAX_HEIGHT_M",
                0.14,
            ),
            object_min_xy_size=env_float(
                "SIMPLE_OBJECT_MIN_XY_SIZE_M",
                0.014,
            ),
            object_max_xy_size=env_float(
                "SIMPLE_OBJECT_MAX_XY_SIZE_M",
                0.13,
            ),

            # Ограничения по брой точки.
            object_min_points=env_int(
                "SIMPLE_OBJECT_MIN_POINTS",
                24,
            ),
            object_max_points=env_int(
                "SIMPLE_OBJECT_MAX_POINTS",
                100000,
            ),
            max_objects=env_int(
                "SIMPLE_OBJECT_MAX_COUNT",
                20,
            ),

            # Настройки за XY групиране.
            xy_voxel_size=env_float(
                "SIMPLE_OBJECT_XY_VOXEL_M",
                0.010,
            ),
            xy_neighbour_range=env_int(
                "SIMPLE_OBJECT_XY_NEIGHBOUR_RANGE",
                1,
            ),
            min_occupied_voxels=env_int(
                "SIMPLE_OBJECT_MIN_OCCUPIED_VOXELS",
                3,
            ),

            # Перцентили за устойчиво изчисляване
            # на размерите и височината.
            low_percentile=env_float(
                "SIMPLE_OBJECT_LOW_PERCENTILE",
                2.0,
            ),
            high_percentile=env_float(
                "SIMPLE_OBJECT_HIGH_PERCENTILE",
                98.0,
            ),
            top_percentile=env_float(
                "SIMPLE_OBJECT_TOP_PERCENTILE",
                98.5,
            ),

            # Височини на генерираните точки.
            point_approach_height=env_float(
                "POINT_APPROACH_HEIGHT",
                0.10,
            ),
            point_grasp_height_offset=env_float(
                "POINT_GRASP_HEIGHT_OFFSET",
                0.018,
            ),
            point_place_height_offset=env_float(
                "POINT_PLACE_HEIGHT_OFFSET",
                0.025,
            ),

            # Времеви ограничения.
            cloud_stale_timeout_sec=env_float(
                "SIMPLE_OBJECT_CLOUD_STALE_SEC",
                5.0,
            ),
            input_wait_timeout_sec=env_float(
                "WORKSPACE_INPUT_WAIT_TIMEOUT_SEC",
                12.0,
            ),
            auto_scan_period_sec=env_float(
                "SIMPLE_OBJECT_AUTO_SCAN_PERIOD_SEC",
                0.0,
            ),

            # Диагностично поведение.
            publish_rejected_debug=env_bool(
                "SIMPLE_OBJECT_LOG_REJECTED",
                True,
            ),
        )


@dataclass
class ScanArtifacts:
    """
    Събира всички резултати
    от едно обработване на облака.
    """

    # Показва дали сканирането е успешно.
    ok: bool

    # Текстово състояние на сканирането.
    state: str

    # Открити предмети.
    objects: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # Документ с точките за движение.
    motion_document: Dict[str, Any] = field(
        default_factory=dict
    )

    # Документ с пълните данни за предметите.
    objects_document: Dict[str, Any] = field(
        default_factory=dict
    )

    # Облаци от точки на приетите предмети.
    selected_clusters: List[np.ndarray] = field(
        default_factory=list
    )

    # Оценена равнина на масата.
    table_plane: Optional[TablePlane] = None

    # Отхвърлени компоненти.
    rejected: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # Допълнителни диагностични данни.
    details: Dict[str, Any] = field(
        default_factory=dict
    )


class WorkspaceCloudProcessor:
    """
    Чиста услуга за обработване на облака от точки,
    използвана от workspace_scan_agent.
    """

    def __init__(
        self,
        config: WorkspaceScanConfig,
    ) -> None:
        """
        Инициализира обработващия модул
        с подадената конфигурация.
        """

        self.config = config

    def process_cloud(
        self,
        cloud_msg: PointCloud2,
    ) -> ScanArtifacts:
        """
        Изпълнява пълната обработка на един облак от точки.

        Основните етапи са:
        - прочитане на XYZ и RGB;
        - ограничаване до работното пространство;
        - оценяване на равнината на масата;
        - премахване на точките от масата;
        - групиране на предметите по XY;
        - изчисляване на центрове, размери и цветове;
        - генериране на точки за движение.
        """

        cfg = self.config

        # Прочита геометрията и цветовете
        # от ROS 2 съобщението.
        xyz, rgb = self.read_cloud(
            cloud_msg
        )

        # Облакът не съдържа достатъчно точки.
        if xyz.shape[0] < cfg.object_min_points:
            return ScanArtifacts(
                ok=False,
                state="EMPTY_POINT_CLOUD",
                details={
                    "cloud_points": int(
                        xyz.shape[0]
                    )
                },
            )

        # Създава маска за точките,
        # намиращи се в разрешения пространствен обем.
        workspace_mask = (
            (xyz[:, 0] >= cfg.workspace_x_min)
            & (xyz[:, 0] <= cfg.workspace_x_max)
            & (xyz[:, 1] >= cfg.workspace_y_min)
            & (xyz[:, 1] <= cfg.workspace_y_max)
            & (xyz[:, 2] >= cfg.workspace_z_min)
            & (xyz[:, 2] <= cfg.workspace_z_max)
        )

        # Запазва само точките
        # в работното пространство.
        xyz = xyz[workspace_mask]
        rgb = rgb[workspace_mask]

        # Няма достатъчно точки
        # след пространственото филтриране.
        if xyz.shape[0] < cfg.object_min_points:
            return ScanArtifacts(
                ok=False,
                state="NO_POINTS_IN_WORKSPACE",
                details={
                    "workspace_points": int(
                        xyz.shape[0]
                    )
                },
            )

        # Оценява равнината на работната маса.
        plane = self.estimate_table_plane(
            xyz
        )

        # Изчислява височината на равнината
        # под всяка точка.
        table_z = plane.z_at(
            xyz[:, 0],
            xyz[:, 1],
        )

        # Височина на точките над масата.
        heights = xyz[:, 2] - table_z

        # Оставя само точки с височина,
        # подходяща за предмет.
        height_mask = (
            (heights >= cfg.object_min_height)
            & (heights <= cfg.object_max_height)
            & np.isfinite(heights)
        )

        # Хоризонтално разстояние на всяка точка
        # от основата на манипулатора.
        base_distances = np.hypot(
            xyz[:, 0],
            xyz[:, 1],
        )

        # Ограничение на достижимия радиус.
        distance_mask = (
            base_distances
            <= cfg.max_base_distance
            if cfg.max_base_distance > 0.0
            else np.ones(
                xyz.shape[0],
                dtype=bool,
            )
        )

        # Брой точки, които са с валидна височина,
        # но извън допустимото разстояние.
        far_candidate_points_ignored = int(
            np.count_nonzero(
                height_mask
                & ~distance_mask
            )
        )

        # Общата маска за точки от предмети.
        object_mask = (
            height_mask
            & distance_mask
        )

        # Отделя XYZ, RGB и височината
        # на възможните предмети.
        object_xyz = xyz[object_mask]
        object_rgb = rgb[object_mask]
        object_heights = heights[
            object_mask
        ]

        # Няма достатъчно точки над масата.
        if (
            object_xyz.shape[0]
            < cfg.object_min_points
        ):
            return ScanArtifacts(
                ok=False,
                state="NO_OBJECT_POINTS_ABOVE_TABLE",
                table_plane=plane,
                details={
                    "candidate_points": int(
                        object_xyz.shape[0]
                    ),
                    "far_candidate_points_ignored": (
                        far_candidate_points_ignored
                    ),
                    "max_base_distance_m": (
                        cfg.max_base_distance
                    ),
                },
            )

        # Групира точките по близост в XY равнината.
        components = self.cluster_xy(
            object_xyz
        )

        # Приети обекти.
        candidates: List[
            Dict[str, Any]
        ] = []

        # Отхвърлени компоненти.
        rejected: List[
            Dict[str, Any]
        ] = []

        # Обработва всеки свързан компонент.
        for component_index, indices in enumerate(
            components
        ):
            # Точки и цветове на текущия компонент.
            cluster_xyz = object_xyz[indices]
            cluster_rgb = object_rgb[indices]
            cluster_heights = object_heights[
                indices
            ]

            # Опитва да създаде валиден предмет.
            obj, reason = self.object_from_cluster(
                cluster_xyz,
                cluster_rgb,
                cluster_heights,
                plane,
                component_index,
            )

            # Записва причината при отхвърляне.
            if obj is None:
                rejected.append(
                    {
                        "component": component_index,
                        "points": int(
                            cluster_xyz.shape[0]
                        ),
                        "reason": reason,
                    }
                )
                continue

            # Временно запазва оригиналния облак
            # за публикуване на избраните компоненти.
            obj["_cluster_xyz"] = cluster_xyz

            candidates.append(obj)

        # Сортира предметите според разстоянието
        # от основата на манипулатора.
        candidates.sort(
            key=lambda item: math.hypot(
                item["x"],
                item["y"],
            )
        )

        # Ограничава максималния брой предмети.
        candidates = candidates[
            : cfg.max_objects
        ]

        # Назначава имена, номера и псевдоними.
        objects = self.assign_names_and_aliases(
            candidates
        )

        # Събира облаците на приетите предмети.
        selected_clusters = [
            np.asarray(
                obj["_cluster_xyz"],
                dtype=np.float32,
            )
            for obj in objects
            if (
                isinstance(
                    obj.get("_cluster_xyz"),
                    np.ndarray,
                )
                and obj["_cluster_xyz"].shape[0] > 0
            )
        ]

        # Премахва временното вътрешно поле
        # от крайния JSON резултат.
        for obj in objects:
            obj.pop(
                "_cluster_xyz",
                None,
            )

        # Няма нито един валиден предмет.
        if not objects:
            return ScanArtifacts(
                ok=False,
                state="NO_VALID_OBJECT_CENTERS",
                table_plane=plane,
                rejected=rejected,
                details={
                    "far_candidate_points_ignored": (
                        far_candidate_points_ignored
                    ),
                    "max_base_distance_m": (
                        cfg.max_base_distance
                    ),
                },
            )

        # Създава документа с точките
        # за движение на планиращия агент.
        motion_document = self.build_motion_points(
            objects
        )

        # Създава пълния документ с предметите.
        objects_document = {
            "source": SOURCE_NAME,
            "max_base_distance_m": (
                cfg.max_base_distance
            ),
            "far_candidate_points_ignored": (
                far_candidate_points_ignored
            ),
            "scan_state": "OK",
            "frame": cfg.target_frame,

            # Обработващият модул не класифицира
            # конкретна тримерна форма.
            "supported_shapes": [
                "generic_object"
            ],

            # Параметри на измерената маса.
            "table_plane": (
                self.table_plane_dict(
                    plane
                )
            ),

            # Валидни предмети.
            "objects": objects,

            # Ограничен диагностичен списък
            # с отхвърлените компоненти.
            "rejected_components": rejected[
                :30
            ],

            # Време на създаване.
            "timestamp": time.time(),
        }

        # Връща всички резултати
        # от успешното сканиране.
        return ScanArtifacts(
            ok=True,
            state="OK",
            objects=objects,
            motion_document=motion_document,
            objects_document=objects_document,
            selected_clusters=selected_clusters,
            table_plane=plane,
            rejected=rejected,
            details={
                "far_candidate_points_ignored": (
                    far_candidate_points_ignored
                ),
                "max_base_distance_m": (
                    cfg.max_base_distance
                ),
            },
        )

    def read_cloud(
        self,
        msg: PointCloud2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прочита XYZ координатите и цветовете
        от ROS 2 съобщение PointCloud2.
        """

        # Налични полета в съобщението.
        available = {
            field.name
            for field in msg.fields
        }

        # Предпочита rgb, след това rgba.
        color_field = (
            "rgb"
            if "rgb" in available
            else (
                "rgba"
                if "rgba" in available
                else None
            )
        )

        # Определя полетата, които ще бъдат прочетени.
        fields: Tuple[str, ...] = (
            (
                "x",
                "y",
                "z",
                color_field,
            )
            if color_field
            else (
                "x",
                "y",
                "z",
            )
        )

        # Прочита точките и пропуска NaN стойностите.
        raw = pc2.read_points(
            msg,
            field_names=fields,
            skip_nans=True,
        )

        # Някои версии връщат директно NumPy масив.
        if isinstance(raw, np.ndarray):
            # Структуриран масив с именувани полета.
            if raw.dtype.names:
                xyz = np.column_stack(
                    [
                        np.asarray(
                            raw["x"],
                            dtype=np.float32,
                        ).reshape(-1),
                        np.asarray(
                            raw["y"],
                            dtype=np.float32,
                        ).reshape(-1),
                        np.asarray(
                            raw["z"],
                            dtype=np.float32,
                        ).reshape(-1),
                    ]
                )

                packed_raw = (
                    np.asarray(
                        raw[color_field]
                    ).reshape(-1)
                    if color_field
                    else None
                )

            else:
                # Обикновен числов масив.
                matrix = np.asarray(raw)

                if matrix.ndim == 1:
                    matrix = matrix.reshape(
                        -1,
                        len(fields),
                    )

                xyz = matrix[
                    :,
                    :3,
                ].astype(
                    np.float32,
                    copy=False,
                )

                packed_raw = (
                    matrix[:, 3]
                    if (
                        color_field
                        and matrix.shape[1] > 3
                    )
                    else None
                )

        else:
            # При генератор го преобразува до списък.
            rows = list(raw)

            if not rows:
                return (
                    np.empty(
                        (0, 3),
                        np.float32,
                    ),
                    np.empty(
                        (0, 3),
                        np.uint8,
                    ),
                )

            matrix = np.asarray(rows)

            xyz = matrix[
                :,
                :3,
            ].astype(
                np.float32,
                copy=False,
            )

            packed_raw = (
                matrix[:, 3]
                if (
                    color_field
                    and matrix.shape[1] > 3
                )
                else None
            )

        # Премахва точки с невалидни координати.
        finite = np.all(
            np.isfinite(xyz),
            axis=1,
        )

        xyz = xyz[finite]

        # При липсващ цвят използва
        # неутрална сива стойност.
        if packed_raw is None:
            rgb = np.full(
                (
                    xyz.shape[0],
                    3,
                ),
                160,
                dtype=np.uint8,
            )

        else:
            # Прилага същата маска
            # върху цветовите данни.
            packed_raw = np.asarray(
                packed_raw
            ).reshape(-1)[finite]

            # Цветът понякога е кодиран
            # като float32 с битово представяне на uint32.
            if np.issubdtype(
                packed_raw.dtype,
                np.floating,
            ):
                packed = packed_raw.astype(
                    np.float32,
                    copy=False,
                ).view(np.uint32)

            else:
                packed = packed_raw.astype(
                    np.uint32,
                    copy=False,
                )

            # Извлича червения, зеления
            # и синия байт.
            rgb = np.column_stack(
                [
                    (
                        (packed >> 16)
                        & 0xFF
                    ),
                    (
                        (packed >> 8)
                        & 0xFF
                    ),
                    (
                        packed
                        & 0xFF
                    ),
                ]
            ).astype(np.uint8)

        return xyz, rgb

    def estimate_table_plane(
        self,
        xyz: np.ndarray,
    ) -> TablePlane:
        """
        Оценява равнината на масата
        чрез хистограма и итеративно линейно напасване.
        """

        cfg = self.config

        # При директно зададена височина
        # използва хоризонтална равнина.
        if cfg.table_z_override_text:
            try:
                z = float(
                    cfg.table_z_override_text
                )

                return TablePlane(
                    0.0,
                    0.0,
                    z,
                    "env_override",
                    xyz.shape[0],
                    0.0,
                )

            except ValueError:
                # При невалидна стойност продължава
                # с автоматично оценяване.
                pass

        # Z координатите на всички точки.
        z = xyz[:, 2]

        # Ограничава анализа до диапазон,
        # в който обичайно се намира масата.
        z_low = float(
            np.percentile(
                z,
                3.0,
            )
        )

        z_high = float(
            np.percentile(
                z,
                80.0,
            )
        )

        # При много малък вертикален диапазон
        # използва медианата.
        if (
            z_high
            <= z_low
            + cfg.table_histogram_bin
        ):
            z_mode = float(
                np.median(z)
            )

        else:
            # Изчислява броя интервали
            # в хистограмата.
            bins = max(
                10,
                int(
                    math.ceil(
                        (
                            z_high
                            - z_low
                        )
                        / cfg.table_histogram_bin
                    )
                ),
            )

            # Изгражда хистограма на височините.
            counts, edges = np.histogram(
                z,
                bins=bins,
                range=(
                    z_low,
                    z_high,
                ),
            )

            # Избира най-гъсто населения интервал.
            best = int(
                np.argmax(counts)
            )

            # Центърът на интервала
            # се използва като начална височина.
            z_mode = float(
                (
                    edges[best]
                    + edges[best + 1]
                )
                * 0.5
            )

        # Начална маска за точки около масата.
        initial = (
            np.abs(
                z - z_mode
            )
            <= cfg.table_initial_band
        )

        table_points = xyz[initial]

        # При твърде малко точки
        # връща хоризонтална равнина.
        if table_points.shape[0] < 30:
            return TablePlane(
                0.0,
                0.0,
                z_mode,
                "histogram_mode",
                int(initial.sum()),
                0.0,
            )

        # Всички начални точки се приемат
        # като вътрешни.
        keep = np.ones(
            table_points.shape[0],
            dtype=bool,
        )

        # Начални коефициенти
        # на хоризонтална равнина.
        coeff = np.array(
            [
                0.0,
                0.0,
                z_mode,
            ],
            dtype=np.float64,
        )

        # Изпълнява до пет итерации
        # за устойчиво отстраняване на отклоняващи точки.
        for _ in range(5):
            pts = table_points[keep]

            if pts.shape[0] < 30:
                break

            # Матрица за линейния модел
            # z = a*x + b*y + c.
            design = np.column_stack(
                [
                    pts[:, 0],
                    pts[:, 1],
                    np.ones(
                        pts.shape[0],
                        dtype=np.float64,
                    ),
                ]
            )

            # Решава задачата по метода
            # на най-малките квадрати.
            coeff, *_ = np.linalg.lstsq(
                design,
                pts[:, 2],
                rcond=None,
            )

            # Отхвърля прекалено наклонена равнина.
            if (
                math.hypot(
                    float(coeff[0]),
                    float(coeff[1]),
                )
                > cfg.table_max_tilt
            ):
                coeff = np.array(
                    [
                        0.0,
                        0.0,
                        z_mode,
                    ],
                    dtype=np.float64,
                )
                break

            # Предсказва височината
            # за всички кандидатни точки.
            predicted = (
                coeff[0]
                * table_points[:, 0]
                + coeff[1]
                * table_points[:, 1]
                + coeff[2]
            )

            # Остатъчна грешка.
            residual = (
                table_points[:, 2]
                - predicted
            )

            # Медиана на абсолютната грешка.
            median_abs = (
                float(
                    np.median(
                        np.abs(
                            residual[keep]
                        )
                    )
                )
                if keep.any()
                else 0.0
            )

            # Адаптивен праг за вътрешни точки.
            threshold = min(
                cfg.table_initial_band,
                max(
                    cfg.table_final_inlier,
                    3.5 * median_abs,
                ),
            )

            # Нова маска на вътрешните точки.
            new_keep = (
                np.abs(residual)
                <= threshold
            )

            # Прекратява при стабилизиране.
            if np.array_equal(
                new_keep,
                keep,
            ):
                keep = new_keep
                break

            keep = new_keep

        # Крайно предсказване.
        predicted = (
            coeff[0]
            * table_points[:, 0]
            + coeff[1]
            * table_points[:, 1]
            + coeff[2]
        )

        residual = (
            table_points[:, 2]
            - predicted
        )

        # Крайна медиана на остатъчната грешка.
        residual_median = (
            float(
                np.median(
                    np.abs(
                        residual[keep]
                    )
                )
            )
            if keep.any()
            else 0.0
        )

        return TablePlane(
            a=float(coeff[0]),
            b=float(coeff[1]),
            c=float(coeff[2]),
            source="iterative_plane_fit",
            inliers=int(keep.sum()),
            residual_median=(
                residual_median
            ),
        )

    def cluster_xy(
        self,
        xyz: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Групира точките в XY равнината
        чрез свързани заети воксели.
        """

        cfg = self.config

        # Преобразува всяка XY позиция
        # до целочислена координата на воксел.
        grid = np.floor(
            xyz[:, :2]
            / cfg.xy_voxel_size
        ).astype(np.int32)

        # Свързва всеки воксел
        # с индексите на точките в него.
        cells: Dict[
            Tuple[int, int],
            List[int],
        ] = defaultdict(list)

        for index, key in enumerate(
            map(
                tuple,
                grid,
            )
        ):
            cells[key].append(index)

        # Множество с вече посетените воксели.
        visited: set[
            Tuple[int, int]
        ] = set()

        # Получените свързани компоненти.
        components: List[
            np.ndarray
        ] = []

        # Всички допустими съседни отмествания.
        offsets = [
            (
                dx,
                dy,
            )
            for dx in range(
                -cfg.xy_neighbour_range,
                cfg.xy_neighbour_range + 1,
            )
            for dy in range(
                -cfg.xy_neighbour_range,
                cfg.xy_neighbour_range + 1,
            )
            if not (
                dx == 0
                and dy == 0
            )
        ]

        # Стартира обхождане от всеки
        # все още непосетен воксел.
        for start in cells:
            if start in visited:
                continue

            visited.add(start)

            # Опашка за обхождане в ширина.
            pending: deque[
                Tuple[int, int]
            ] = deque(
                [start]
            )

            # Вокселите в текущия компонент.
            component_cells: List[
                Tuple[int, int]
            ] = []

            while pending:
                cell = pending.popleft()
                component_cells.append(cell)

                # Проверява всички съседни воксели.
                for dx, dy in offsets:
                    neighbour = (
                        cell[0] + dx,
                        cell[1] + dy,
                    )

                    if (
                        neighbour in cells
                        and neighbour not in visited
                    ):
                        visited.add(
                            neighbour
                        )
                        pending.append(
                            neighbour
                        )

            # Пропуска много малки компоненти.
            if (
                len(component_cells)
                < cfg.min_occupied_voxels
            ):
                continue

            # Обединява индексите на всички точки
            # от компонентните воксели.
            indices = np.fromiter(
                (
                    idx
                    for cell in component_cells
                    for idx in cells[cell]
                ),
                dtype=np.int64,
            )

            # Проверява ограниченията
            # за броя точки.
            if (
                cfg.object_min_points
                <= indices.size
                <= cfg.object_max_points
            ):
                components.append(indices)

        # По-големите компоненти се обработват първи.
        components.sort(
            key=lambda idx: idx.size,
            reverse=True,
        )

        return components

    def object_from_cluster(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        heights: np.ndarray,
        plane: TablePlane,
        component_index: int,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        str,
    ]:
        """
        Преобразува един облачен компонент
        до структуриран предмет.
        """

        cfg = self.config

        # Компонентът няма достатъчно точки.
        if xyz.shape[0] < cfg.object_min_points:
            return None, "TOO_FEW_POINTS"

        # Устойчиви долни и горни граници,
        # които намаляват влиянието на шумни точки.
        low = np.percentile(
            xyz,
            cfg.low_percentile,
            axis=0,
        )

        high = np.percentile(
            xyz,
            cfg.high_percentile,
            axis=0,
        )

        # Сурови размери на ограничителната кутия.
        raw_dims = high - low

        # Размер по X.
        width = float(
            raw_dims[0]
        )

        # Размер по Y.
        depth = float(
            raw_dims[1]
        )

        # Измерена височина над масата.
        measured_height = float(
            np.percentile(
                heights,
                cfg.top_percentile,
            )
        )

        # Проверява за невалидна геометрия.
        if (
            not np.all(
                np.isfinite(raw_dims)
            )
            or not math.isfinite(
                measured_height
            )
        ):
            return None, "NONFINITE_GEOMETRY"

        # Проверява височината.
        if (
            measured_height
            < cfg.object_min_height
            or measured_height
            > cfg.object_max_height
        ):
            return None, "INVALID_HEIGHT"

        # Проверява минималните XY размери.
        if (
            width
            < cfg.object_min_xy_size
            or depth
            < cfg.object_min_xy_size
        ):
            return None, "TOO_THIN_XY"

        # Проверява максималните XY размери.
        if (
            width
            > cfg.object_max_xy_size
            or depth
            > cfg.object_max_xy_size
        ):
            return None, "TOO_LARGE_XY"

        # Център по X и Y
        # според ограничителната кутия.
        center_x = float(
            (
                low[0]
                + high[0]
            )
            * 0.5
        )

        center_y = float(
            (
                low[1]
                + high[1]
            )
            * 0.5
        )

        # Височина на масата
        # под центъра на предмета.
        table_z = float(
            plane.z_at(
                center_x,
                center_y,
            )
        )

        # Център на предмета по Z.
        center_z = (
            table_z
            + measured_height
            * 0.5
        )

        # Горна повърхност.
        top_z = (
            table_z
            + measured_height
        )

        # Разстояние до основата.
        distance_from_base = math.hypot(
            center_x,
            center_y,
        )

        # Допълнителна проверка
        # на центъра спрямо достижимия радиус.
        if (
            cfg.max_base_distance > 0.0
            and distance_from_base
            > cfg.max_base_distance
        ):
            return (
                None,
                "CENTER_BEYOND_MAX_BASE_DISTANCE",
            )

        # Проверява дали центърът се намира
        # в разрешения пространствен обем.
        if not (
            cfg.workspace_x_min
            <= center_x
            <= cfg.workspace_x_max
            and cfg.workspace_y_min
            <= center_y
            <= cfg.workspace_y_max
            and cfg.workspace_z_min
            <= center_z
            <= cfg.workspace_z_max
        ):
            return (
                None,
                "CENTER_OUTSIDE_WORKSPACE",
            )

        # Класифицира цвета
        # на точките от компонента.
        (
            color_name,
            color_confidence,
            representative_rgb,
        ) = self.classify_color(rgb)

        # Централна позиция.
        center = {
            "x": center_x,
            "y": center_y,
            "z": center_z,
        }

        # Размери на предмета.
        dimensions = {
            "x": width,
            "y": depth,
            "z": measured_height,
        }

        # Българско прилагателно за цвета.
        adjective = COLOR_BG.get(
            color_name,
            color_name,
        )

        # Качество на геометрията
        # според броя точки.
        quality = (
            "high"
            if xyz.shape[0] >= 60
            else "medium"
        )

        # Създава пълния документ за предмета.
        obj: Dict[str, Any] = {
            # Индекс на първоначалния компонент.
            "component_index": component_index,

            # Английски етикет.
            "label": (
                f"{color_name} object"
            ),

            # Безопасен машинен етикет.
            "safe_label": safe_label(
                f"{color_name}_object"
            ),

            # Име на български.
            "display_name_bg": (
                f"{adjective} обект"
            ),

            # Роля в системата.
            "role": "movable_object",

            # Списък с открити цветове.
            "colors": [
                color_name
            ],

            # Общ геометричен тип.
            "shape": "object",

            # Конкретната семантична форма
            # остава неизвестна.
            "semantic_shape": "unknown",

            # За планиране се използва
            # кръгов хоризонтален заместител.
            "geometry_proxy": "circle",

            # Геометричният модул не определя
            # дали обектът е куб, топка и др.
            "geometry_can_classify_shape": False,

            # Увереност и представителен цвят.
            "color_confidence": (
                color_confidence
            ),
            "representative_rgb": (
                representative_rgb
            ),

            # Координатна система.
            "frame": cfg.target_frame,

            # Директни координати на центъра.
            "x": center_x,
            "y": center_y,
            "z": center_z,

            # Информация за достижимостта.
            "distance_from_base_m": (
                distance_from_base
            ),
            "max_base_distance_m": (
                cfg.max_base_distance
            ),
            "within_reach": True,

            # Център и основна позиция.
            "center": center,
            "object_position": dict(
                center
            ),

            # Горна повърхност.
            "top_position": {
                "x": center_x,
                "y": center_y,
                "z": top_z,
            },

            # Позиция за хващане.
            "grasp_position": {
                "x": center_x,
                "y": center_y,
                "z": (
                    top_z
                    + cfg.point_grasp_height_offset
                ),
            },

            # Позиция за подход.
            "approach_position": {
                "x": center_x,
                "y": center_y,
                "z": (
                    top_z
                    + cfg.point_approach_height
                ),
            },

            # Позиция за поставяне.
            "place_position": {
                "x": center_x,
                "y": center_y,
                "z": (
                    center_z
                    + cfg.point_place_height_offset
                ),
            },

            # Позиция за използване като крайна цел.
            "destination_position": {
                "x": center_x,
                "y": center_y,
                "z": (
                    center_z
                    + cfg.point_place_height_offset
                ),
            },

            # Размери.
            "dimensions": dimensions,
            "raw_dimensions": dict(
                dimensions
            ),
            "width": width,
            "depth": depth,
            "height": measured_height,

            # Тримерна ограничителна кутия.
            "bbox_3d": {
                "x_min": float(
                    low[0]
                ),
                "x_max": float(
                    high[0]
                ),
                "y_min": float(
                    low[1]
                ),
                "y_max": float(
                    high[1]
                ),
                "z_min": table_z,
                "z_max": top_z,
            },

            # Няма двумерна RGB ограничителна кутия.
            "bbox_pixels_rgb": None,

            # Начална ориентация.
            "orientation": 0.0,

            # Единичен кватернион.
            "quaternion": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "w": 1.0,
            },

            # Пространствен тип.
            "spatial_type": "object",

            # Предметът може да бъде хващан.
            "is_graspable": True,

            # Самият предмет не е повърхност
            # за поставяне.
            "is_placeable": False,

            # Разрешен е за действия.
            "actionable": True,

            # Не е защитно препятствие.
            "safety_blocker": False,

            # Брой точки.
            "points": int(
                xyz.shape[0]
            ),

            # Използван метод за центъра.
            "center_method": (
                "trimmed_3d_bounds_midpoint_"
                "table_anchored"
            ),

            # Източник на геометрията.
            "geometry_source": (
                "current_cloud_table_removed_"
                "xy_cluster"
            ),

            # Оценка на качеството.
            "geometry_quality": quality,

            # Показва, че стойностите идват
            # от текущото сканиране.
            "fresh_measurement": True,
            "tracking_state": (
                "fresh_no_history"
            ),

            # Параметри на масата.
            "table_plane": (
                self.table_plane_dict(
                    plane
                )
            ),

            # Време на създаване.
            "timestamp": time.time(),
        }

        return obj, "OK"

    def classify_color(
        self,
        rgb: np.ndarray,
    ) -> Tuple[
        str,
        float,
        Dict[str, int],
    ]:
        """
        Класифицира представителния цвят
        на облака от точки.
        """

        # При липсващи цветове връща
        # неутрална сива стойност.
        if rgb.shape[0] == 0:
            return (
                "unknown",
                0.0,
                {
                    "r": 160,
                    "g": 160,
                    "b": 160,
                },
            )

        # При много голям облак използва
        # равномерна извадка до около 3000 точки.
        sample = (
            rgb[
                ::
                max(
                    1,
                    rgb.shape[0] // 3000,
                )
            ]
            if rgb.shape[0] > 3000
            else rgb
        )

        # Нормализира RGB до диапазона 0–1.
        sample_float = (
            sample.astype(
                np.float32
            )
            / 255.0
        )

        # Максимален и минимален цветови канал.
        maximum = sample_float.max(
            axis=1
        )
        minimum = sample_float.min(
            axis=1
        )

        # Цветова разлика.
        delta = (
            maximum
            - minimum
        )

        # Наситеност на всяка точка.
        saturation = np.divide(
            delta,
            maximum,
            out=np.zeros_like(delta),
            where=maximum > 1e-6,
        )

        # Маска за достатъчно цветни точки.
        chromatic = (
            (saturation >= 0.18)
            & (maximum >= 0.15)
        )

        # Предпочита наситените точки,
        # когато са достатъчно на брой.
        selected = (
            sample_float[chromatic]
            if int(chromatic.sum()) >= 8
            else sample_float
        )

        # Медианният цвят е устойчив
        # на единични шумни пиксели.
        representative = np.median(
            selected,
            axis=0,
        )

        r, g, b = map(
            float,
            representative,
        )

        mx = max(
            r,
            g,
            b,
        )
        mn = min(
            r,
            g,
            b,
        )
        d = mx - mn

        # Наситеност на представителния цвят.
        sat = (
            0.0
            if mx <= 1e-6
            else d / mx
        )

        # Много тъмните стойности
        # се класифицират като черни.
        if mx < 0.17:
            name = "black"
            confidence = 1.0 - mx

        # Ниска наситеност означава
        # бяло или сиво.
        elif sat < 0.13:
            if mx > 0.78:
                name = "white"
                confidence = min(
                    1.0,
                    (
                        mx - 0.65
                    )
                    / 0.35,
                )
            else:
                name = "gray"
                confidence = min(
                    1.0,
                    1.0
                    - sat
                    / 0.13,
                )

        else:
            # Изчислява HSV нюанса
            # в градуси.
            if d <= 1e-9:
                hue = 0.0

            elif mx == r:
                hue = (
                    60.0
                    * (
                        (g - b)
                        / d
                    )
                    + 360.0
                ) % 360.0

            elif mx == g:
                hue = 60.0 * (
                    (
                        b - r
                    )
                    / d
                    + 2.0
                )

            else:
                hue = 60.0 * (
                    (
                        r - g
                    )
                    / d
                    + 4.0
                )

            # Класифицира цвета
            # според диапазона на нюанса.
            if (
                hue < 15.0
                or hue >= 345.0
            ):
                name = "red"

            elif hue < 40.0:
                name = "orange"

            elif hue < 70.0:
                name = "yellow"

            elif hue < 170.0:
                name = "green"

            elif hue < 200.0:
                name = "cyan"

            elif hue < 255.0:
                name = "blue"

            elif hue < 300.0:
                name = "purple"

            else:
                name = "pink"

            # Наситеността се използва
            # като увереност.
            confidence = min(
                1.0,
                sat,
            )

        # Преобразува представителния цвят
        # обратно до 8-битови RGB стойности.
        rep = {
            "r": int(
                np.clip(
                    round(
                        r * 255.0
                    ),
                    0,
                    255,
                )
            ),
            "g": int(
                np.clip(
                    round(
                        g * 255.0
                    ),
                    0,
                    255,
                )
            ),
            "b": int(
                np.clip(
                    round(
                        b * 255.0
                    ),
                    0,
                    255,
                )
            ),
        }

        return (
            name,
            float(confidence),
            rep,
        )

    def assign_names_and_aliases(
        self,
        objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Назначава номера, идентификатори
        и естественоезикови псевдоними.
        """

        # Групира предметите по безопасния им етикет.
        groups: Dict[
            str,
            List[Dict[str, Any]],
        ] = defaultdict(list)

        for obj in objects:
            groups[
                obj["safe_label"]
            ].append(obj)

        # Общ пореден номер
        # за всички открити предмети.
        global_index = 1

        # Обработва всяка семантична група.
        for safe_name, same in groups.items():
            # Най-близкият предмет получава
            # най-малък instance_index.
            same.sort(
                key=lambda item: math.hypot(
                    item["x"],
                    item["y"],
                )
            )

            for instance_index, obj in enumerate(
                same,
                start=1,
            ):
                # Пореден номер в групата.
                obj["instance_index"] = (
                    instance_index
                )

                # Глобален пореден номер.
                obj["global_index"] = (
                    global_index
                )

                # Общ идентификатор на предмета.
                obj["object_id"] = (
                    f"cloud_object_{global_index}"
                )

                # Име на точката за движение.
                obj["point_label"] = (
                    f"point_{safe_name}_"
                    f"{instance_index}"
                )

                # Основният цвят.
                color = obj["colors"][0]

                # Българско прилагателно.
                adjective = COLOR_BG.get(
                    color,
                    color,
                )

                # Псевдоними за избор
                # на предмета чрез естествен език.
                aliases = {
                    obj["point_label"],
                    safe_name,
                    safe_name.replace(
                        "_",
                        " ",
                    ),
                    obj["label"],
                    obj["display_name_bg"],
                    color,
                    "object",
                    "обект",
                    "предмет",
                    "circle",
                    "кръг",
                    "cube",
                    "cuboid",
                    "block",
                    "ball",
                    "sphere",
                    "куб",
                    "кубче",
                    "топка",
                    "топче",
                    f"{color}_object",
                    f"{color}_cube",
                    f"{color}_ball",
                    f"{color} object",
                    f"{color} cube",
                    f"{color} ball",
                    f"{adjective} обект",
                    f"{adjective} куб",
                    f"{adjective} кубче",
                    (
                        f"{obj['display_name_bg']} "
                        f"{instance_index}"
                    ),
                }

                # Първият предмет в групата
                # може да бъде избран като най-близък.
                if instance_index == 1:
                    aliases.add(
                        f"nearest_{safe_name}"
                    )

                # Сортира и премахва празните псевдоними.
                obj["aliases"] = sorted(
                    alias
                    for alias in aliases
                    if alias
                )

                global_index += 1

        # Добавя глобални пространствени псевдоними.
        if objects:
            # Най-близък до основата.
            min(
                objects,
                key=lambda item: math.hypot(
                    item["x"],
                    item["y"],
                ),
            )["aliases"].extend(
                [
                    "nearest_object",
                    "най-близкия обект",
                ]
            )

            # Най-ляв според максималната Y координата.
            max(
                objects,
                key=lambda item: item["y"],
            )["aliases"].extend(
                [
                    "left_object",
                    "левия обект",
                ]
            )

            # Най-десен според минималната Y координата.
            min(
                objects,
                key=lambda item: item["y"],
            )["aliases"].extend(
                [
                    "right_object",
                    "десния обект",
                ]
            )

        # Премахва повторенията
        # в списъците с псевдоними.
        for obj in objects:
            obj["aliases"] = sorted(
                set(
                    obj["aliases"]
                )
            )

        # Връща предметите
        # в глобалния им ред.
        objects.sort(
            key=lambda item: item[
                "global_index"
            ]
        )

        return objects

    def build_motion_points(
        self,
        objects: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Създава документа с точки за движение,
        използван от планиращия агент.
        """

        cfg = self.config

        # Изходният списък с точки.
        points: List[
            Dict[str, Any]
        ] = []

        # Преобразува всеки предмет
        # до точка за движение.
        for obj in objects:
            points.append(
                {
                    # Уникално име на точката.
                    "point": obj[
                        "point_label"
                    ],

                    # Тип на точката.
                    "point_type": (
                        "object_center"
                    ),

                    # Идентификатор на предмета.
                    "object_id": obj[
                        "object_id"
                    ],

                    # Семантични имена.
                    "label": obj["label"],
                    "safe_label": obj[
                        "safe_label"
                    ],
                    "display_name_bg": obj[
                        "display_name_bg"
                    ],

                    # Роля и свойства.
                    "role": obj["role"],
                    "colors": obj["colors"],
                    "shape": obj["shape"],
                    "semantic_shape": obj[
                        "semantic_shape"
                    ],
                    "geometry_proxy": obj[
                        "geometry_proxy"
                    ],
                    "center_method": obj[
                        "center_method"
                    ],

                    # Номера и псевдоними.
                    "instance_index": obj[
                        "instance_index"
                    ],
                    "global_index": obj[
                        "global_index"
                    ],
                    "aliases": obj[
                        "aliases"
                    ],

                    # Координатна система.
                    "frame": cfg.target_frame,

                    # Директни координати.
                    "x": obj["x"],
                    "y": obj["y"],
                    "z": obj["z"],

                    # Достижимост.
                    "distance_from_base_m": obj[
                        "distance_from_base_m"
                    ],
                    "max_base_distance_m": obj[
                        "max_base_distance_m"
                    ],
                    "within_reach": obj[
                        "within_reach"
                    ],

                    # Позиции, използвани
                    # от различните движения.
                    "position": obj["center"],
                    "center": obj["center"],
                    "object_position": obj[
                        "object_position"
                    ],
                    "top_position": obj[
                        "top_position"
                    ],
                    "grasp_position": obj[
                        "grasp_position"
                    ],
                    "approach_position": obj[
                        "approach_position"
                    ],
                    "place_position": obj[
                        "place_position"
                    ],
                    "destination_position": obj[
                        "destination_position"
                    ],

                    # Геометрични размери.
                    "dimensions": obj[
                        "dimensions"
                    ],
                    "raw_dimensions": obj[
                        "raw_dimensions"
                    ],
                    "width": obj["width"],
                    "height": obj["height"],
                    "depth": obj["depth"],
                    "bbox_3d": obj[
                        "bbox_3d"
                    ],

                    # Няма двумерна ограничителна кутия.
                    "bbox_pixels_rgb": None,

                    # Качество и източник
                    # на геометричните данни.
                    "points": obj["points"],
                    "geometry_source": obj[
                        "geometry_source"
                    ],
                    "geometry_quality": obj[
                        "geometry_quality"
                    ],

                    # Данните са от текущото сканиране.
                    "fresh_measurement": True,
                    "tracking_state": (
                        "fresh_no_history"
                    ),

                    # Разрешени действия.
                    "actionable": True,
                    "is_graspable": True,
                    "is_placeable": False,
                    "safety_blocker": False,

                    # Източник и време.
                    "source": SOURCE_NAME,
                    "timestamp": obj[
                        "timestamp"
                    ],
                }
            )

        # Връща пълния документ.
        return {
            "source": SOURCE_NAME,
            "scan_state": "OK",
            "frame": cfg.target_frame,
            "max_base_distance_m": (
                cfg.max_base_distance
            ),
            "motion_points": points,
            "timestamp": time.time(),
        }

    @staticmethod
    def table_plane_dict(
        plane: TablePlane,
    ) -> Dict[str, Any]:
        """
        Преобразува равнината на масата
        до JSON-съвместим речник.
        """

        return {
            # Текстово представяне на уравнението.
            "equation": (
                "z = a*x + b*y + c"
            ),

            # Коефициенти на равнината.
            "a": plane.a,
            "b": plane.b,
            "c": plane.c,

            # Метод на изчисляване.
            "source": plane.source,

            # Брой използвани вътрешни точки.
            "inliers": plane.inliers,

            # Медиана на остатъчната грешка.
            "median_residual_m": (
                plane.residual_median
            ),
        }