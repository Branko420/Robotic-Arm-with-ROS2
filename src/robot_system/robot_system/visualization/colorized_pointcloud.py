#!/usr/bin/env python3

# Математически функции за работа с ъгли,
# тригонометрия и преобразуване между градуси и радиани.
import math

# Работа с времеви стойности за ограничаване
# на честотата на диагностичните съобщения и FPS статистиката.
import time

# Типови анотации за речници, незадължителни стойности
# и кортежи с координати или кватерниони.
from typing import Dict, Optional, Tuple

# NumPy се използва за ефективно обработване
# на облаците от точки, цветовете и матриците за трансформация.
import numpy as np

# Основна ROS 2 библиотека за Python.
import rclpy

# Базов клас за създаване на ROS 2 възел.
from rclpy.node import Node

# QoS профил, предназначен за потоци от сензорни данни,
# при които ниската латентност е по-важна от гарантираната доставка.
from rclpy.qos import qos_profile_sensor_data

# Преобразува ROS изображения до OpenCV/NumPy представяне.
from cv_bridge import CvBridge

# Стандартен ROS 2 заглавен блок с времеви печат
# и координатна система.
from std_msgs.msg import Header

# Съобщения за облак от точки, описание на неговите полета,
# RGB изображение и вътрешни параметри на камерата.
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo

# Съобщение за пространствена трансформация между две координатни системи.
from geometry_msgs.msg import TransformStamped

# Издател на статични TF трансформации.
from tf2_ros import StaticTransformBroadcaster

# Помощни функции за създаване и прочитане
# на ROS 2 съобщения PointCloud2.
import sensor_msgs_py.point_cloud2 as pc2


# ROS 2 възел, който:
# - получава облак от точки от дълбочинната камера;
# - проектира точките върху RGB изображението;
# - добавя цвят към всяка валидна тримерна точка;
# - преобразува точките към координатната система base_link;
# - филтрира работното пространство;
# - намалява броя на точките чрез вокселизация;
# - публикува цветен облак за цифровия модел в RViz.
class ColorizedPointCloud(Node):
    def __init__(self):
        # Инициализира ROS 2 възела с име colorized_pointcloud.
        super().__init__("colorized_pointcloud")

        # ---------------------------------------------------------
        # Topics
        # ---------------------------------------------------------

        # Входна тема с оригиналния облак от точки
        # на дълбочинната камера.
        self.declare_parameter("pointcloud_topic", "/aurora/points2")

        # Входна тема с RGB изображението.
        self.declare_parameter("rgb_topic", "/aurora/rgb/image_raw")

        # Входна тема с вътрешните параметри на RGB камерата.
        self.declare_parameter("camera_info_topic", "/aurora/rgb/camera_info")

        # Изходна тема с цветния облак от точки,
        # преобразуван към координатната система на робота.
        self.declare_parameter("colorized_cloud_topic", "/digital_twin/live_cloud_rgb")

        # ---------------------------------------------------------
        # Frames
        # ---------------------------------------------------------

        # Основна координатна система на стойката или робота.
        self.declare_parameter("stand_frame", "base_link")

        # Координатна система на оптичния сензор.
        self.declare_parameter("camera_frame", "depth_camera_link")

        # Camera position relative to stand_frame/base_link.

        # Позиция на камерата спрямо base_link по оста X.
        self.declare_parameter("camera_x", -0.188)

        # Позиция на камерата спрямо base_link по оста Y.
        self.declare_parameter("camera_y", 0.0)

        # Височина на камерата спрямо base_link.
        self.declare_parameter("camera_z", 0.48)

        # Camera angle.

        # Наклон на камерата надолу към работната повърхност.
        self.declare_parameter("camera_tilt_degrees", 45.0)

        # NEW: rotate whole map around Z axis of base_link/world.
        # Try 0, 90, -90, 180 until mapping lines up with robot.

        # Допълнително завъртане на цялото картографиране
        # около оста Z на base_link.
        self.declare_parameter("camera_yaw_degrees", 0.0)

        # Завъртане на изображението около оптичната ос на камерата.
        self.declare_parameter("camera_roll_degrees", 0.0)

        # Определя дали дясната страна на RGB изображението
        # съответства на отрицателната Y посока на робота.
        self.declare_parameter("image_right_is_negative_y", True)

        # Определя дали възелът да публикува статична TF трансформация
        # между base_link и координатната система на камерата.
        self.declare_parameter("publish_camera_static_tf", True)

        # ---------------------------------------------------------
        # Projection tuning
        # ---------------------------------------------------------

        # Допълнително хоризонтално отместване
        # при проектирането върху RGB изображението.
        self.declare_parameter("projection_u_offset", 0.0)

        # Допълнително вертикално отместване
        # при проектирането върху RGB изображението.
        self.declare_parameter("projection_v_offset", 0.0)

        # Позволява огледално обръщане на проекцията по X.
        self.declare_parameter("flip_projection_x", False)

        # Позволява огледално обръщане на проекцията по Y.
        self.declare_parameter("flip_projection_y", False)

        # ---------------------------------------------------------
        # Filtering
        # ---------------------------------------------------------

        # Минимална допустима дълбочина от камерата.
        self.declare_parameter("min_camera_depth", 0.05)

        # Максимална допустима дълбочина от камерата.
        self.declare_parameter("max_camera_depth", 2.50)

        # Пространствени граници на работната област спрямо base_link.
        self.declare_parameter("workspace_x_min", -0.80)
        self.declare_parameter("workspace_x_max", 1.50)
        self.declare_parameter("workspace_y_min", -1.00)
        self.declare_parameter("workspace_y_max", 1.00)
        self.declare_parameter("workspace_z_min", -0.30)
        self.declare_parameter("workspace_z_max", 1.30)

        # Bigger = faster, less detail.

        # Размер на воксела при намаляване на облака.
        # По-голяма стойност означава по-малко точки и по-висока скорост.
        self.declare_parameter("voxel_size", 0.008)

        # Максимален брой точки в публикувания облак.
        self.declare_parameter("max_points_publish", 80000)

        # Позволява обработване само на всеки N-ти входен кадър.
        self.declare_parameter("process_every_n_frames", 1)

        # ---------------------------------------------------------
        # Read params
        # ---------------------------------------------------------

        # Прочита имената на входните и изходните ROS 2 теми.
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.colorized_cloud_topic = self.get_parameter("colorized_cloud_topic").value

        # Прочита координатните системи.
        self.stand_frame = self.get_parameter("stand_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        # Прочита позицията на камерата спрямо основата.
        self.camera_x = float(self.get_parameter("camera_x").value)
        self.camera_y = float(self.get_parameter("camera_y").value)
        self.camera_z = float(self.get_parameter("camera_z").value)

        # Прочита ъглите на пространственото ориентиране на камерата.
        self.camera_tilt_degrees = float(
            self.get_parameter("camera_tilt_degrees").value
        )
        self.camera_yaw_degrees = float(
            self.get_parameter("camera_yaw_degrees").value
        )
        self.camera_roll_degrees = float(
            self.get_parameter("camera_roll_degrees").value
        )

        # Прочита настройката за посоката на оста Y
        # и публикуването на статична TF трансформация.
        self.image_right_is_negative_y = bool(
            self.get_parameter("image_right_is_negative_y").value
        )
        self.publish_camera_static_tf_enabled = bool(
            self.get_parameter("publish_camera_static_tf").value
        )

        # Прочита ръчните корекции на RGB проекцията.
        self.projection_u_offset = float(
            self.get_parameter("projection_u_offset").value
        )
        self.projection_v_offset = float(
            self.get_parameter("projection_v_offset").value
        )
        self.flip_projection_x = bool(self.get_parameter("flip_projection_x").value)
        self.flip_projection_y = bool(self.get_parameter("flip_projection_y").value)

        # Прочита ограниченията на дълбочината.
        self.min_camera_depth = float(self.get_parameter("min_camera_depth").value)
        self.max_camera_depth = float(self.get_parameter("max_camera_depth").value)

        # Прочита границите на работното пространство.
        self.workspace_x_min = float(self.get_parameter("workspace_x_min").value)
        self.workspace_x_max = float(self.get_parameter("workspace_x_max").value)
        self.workspace_y_min = float(self.get_parameter("workspace_y_min").value)
        self.workspace_y_max = float(self.get_parameter("workspace_y_max").value)
        self.workspace_z_min = float(self.get_parameter("workspace_z_min").value)
        self.workspace_z_max = float(self.get_parameter("workspace_z_max").value)

        # Прочита настройките за вокселизация,
        # ограничаване на точките и пропускане на кадри.
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.max_points_publish = int(self.get_parameter("max_points_publish").value)
        self.process_every_n_frames = max(
            1,
            int(self.get_parameter("process_every_n_frames").value),
        )

        # ---------------------------------------------------------
        # ROS
        # ---------------------------------------------------------

        # Създава преобразувател между ROS изображения
        # и OpenCV/NumPy масиви.
        self.bridge = CvBridge()

        # Абонира възела за входния облак от точки.
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        # Абонира възела за RGB изображенията.
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            qos_profile_sensor_data,
        )

        # Абонира възела за вътрешните параметри
        # на RGB камерата.
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        # Създава издател за готовия цветен облак.
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.colorized_cloud_topic,
            5,
        )

        # Създава издател за статичната TF трансформация.
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------

        # Последното успешно преобразувано RGB изображение.
        self.latest_rgb: Optional[np.ndarray] = None

        # Вътрешни параметри на камерата:
        # фокусни разстояния и координати на оптичния център.
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # Брояч на входните облаци от точки.
        self.frame_count = 0

        # Последно време на извеждане
        # за всяко ограничено предупреждение.
        self.last_warn_times: Dict[str, float] = {}

        # Стойности за изчисляване
        # на честотата на обработване.
        self.last_fps_time = time.time()
        self.processed_frames = 0

        # Матрица, която преобразува координати
        # от оптичната система на камерата към base_link.
        self.camera_rotation_base_from_optical = self.make_camera_rotation_matrix()

        # Транслационен вектор на камерата
        # спрямо base_link.
        self.camera_translation_base = np.array(
            [self.camera_x, self.camera_y, self.camera_z],
            dtype=np.float32,
        )

        # Запомня за коя камера вече е публикувана
        # статична трансформация.
        self.static_tf_published_for_frame: Optional[str] = None

        # Публикува първоначалната статична трансформация,
        # когато настройката е активирана.
        if self.publish_camera_static_tf_enabled:
            self.publish_camera_static_tf(force=True)

        # Извежда началната конфигурация
        # на възела в ROS 2 журнала.
        self.get_logger().info("✅ Colorized PointCloud node started")
        self.get_logger().info(f"PointCloud input: {self.pointcloud_topic}")
        self.get_logger().info(f"RGB input:        {self.rgb_topic}")
        self.get_logger().info(f"CameraInfo input: {self.camera_info_topic}")
        self.get_logger().info(f"Output cloud:     {self.colorized_cloud_topic}")
        self.get_logger().info(f"RViz Fixed Frame: {self.stand_frame}")
        self.get_logger().info(
            f"Camera transform: x={self.camera_x:.3f}, y={self.camera_y:.3f}, "
            f"z={self.camera_z:.3f}, tilt={self.camera_tilt_degrees:.1f}, "
            f"yaw={self.camera_yaw_degrees:.1f}, roll={self.camera_roll_degrees:.1f}"
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    # Извежда предупреждение най-много веднъж
    # за зададения период за всеки отделен ключ.
    def warn_throttled(self, key: str, message: str, period_sec: float = 3.0):
        now = time.time()
        last = self.last_warn_times.get(key, 0.0)

        if now - last >= period_sec:
            self.get_logger().warn(message)
            self.last_warn_times[key] = now

    # Изчислява и извежда средната честота
    # на обработените цветни облаци.
    def log_fps(self):
        self.processed_frames += 1
        now = time.time()

        if now - self.last_fps_time >= 2.0:
            fps = self.processed_frames / (now - self.last_fps_time)
            self.get_logger().info(f"Colorized cloud FPS: {fps:.1f}")
            self.last_fps_time = now
            self.processed_frames = 0

    # -------------------------------------------------------------------------
    # Camera transform
    # -------------------------------------------------------------------------

    def make_camera_rotation_matrix(self) -> np.ndarray:
        """
        stand_frame/base_link:
          +X forward from robot
          +Y left from robot
          +Z up

        camera optical:
          +X image right
          +Y image down
          +Z depth forward

        yaw rotates the whole mapping around base_link Z.
        roll rotates the image X/Y axes around the camera optical Z.
        """

        # Преобразува конфигурираните ъгли
        # от градуси в радиани.
        tilt = math.radians(self.camera_tilt_degrees)
        yaw = math.radians(self.camera_yaw_degrees)
        roll = math.radians(self.camera_roll_degrees)

        # Оптичната ос Z сочи напред от камерата
        # и се накланя надолу според camera_tilt_degrees.
        optical_z = np.array(
            [math.cos(tilt), 0.0, -math.sin(tilt)],
            dtype=np.float64,
        )

        # Оптичната ос X съответства на хоризонталната
        # посока на изображението.
        optical_x = np.array(
            [0.0, -1.0 if self.image_right_is_negative_y else 1.0, 0.0],
            dtype=np.float64,
        )

        # Изчислява оптичната ос Y чрез векторно произведение,
        # така че координатната система да остане дясно ориентирана.
        optical_y = np.cross(optical_z, optical_x)

        # Нормализира трите базисни вектора.
        optical_x /= max(np.linalg.norm(optical_x), 1e-12)
        optical_y /= max(np.linalg.norm(optical_y), 1e-12)
        optical_z /= max(np.linalg.norm(optical_z), 1e-12)

        # Right-handed roll around optical Z.

        # Прилага завъртане около оптичната ос Z.
        rolled_x = math.cos(roll) * optical_x + math.sin(roll) * optical_y
        rolled_y = -math.sin(roll) * optical_x + math.cos(roll) * optical_y

        # Подрежда базисните вектори като колони
        # в матрицата на въртене.
        rotation_no_yaw = np.column_stack((rolled_x, rolled_y, optical_z))

        # Създава матрица за завъртане около оста Z
        # на base_link.
        yaw_rotation = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw),  math.cos(yaw), 0.0],
                [0.0,            0.0,           1.0],
            ],
            dtype=np.float64,
        )

        # Комбинира завъртането около Z
        # с ориентацията на оптичните оси.
        return (yaw_rotation @ rotation_no_yaw).astype(np.float32)

    # Преобразува матрица на въртене
    # до кватернион във формат x, y, z, w.
    def rotation_matrix_to_quaternion(
        self,
        rotation: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        m = rotation

        # Следата на матрицата се използва за избор
        # на числено стабилния клон на преобразуването.
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

        return float(qx), float(qy), float(qz), float(qw)

    # Публикува статична TF трансформация
    # между base_link и координатната система на камерата.
    def publish_camera_static_tf(self, force: bool = False):
        # Не публикува повторно същата трансформация,
        # освен когато force=True.
        if not force and self.static_tf_published_for_frame == self.camera_frame:
            return

        # Преобразува матрицата на въртене
        # до кватернион за ROS TF.
        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(
            self.camera_rotation_base_from_optical
        )

        # Създава TF съобщението.
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.stand_frame
        tf_msg.child_frame_id = self.camera_frame

        # Задава транслацията на камерата.
        tf_msg.transform.translation.x = float(self.camera_x)
        tf_msg.transform.translation.y = float(self.camera_y)
        tf_msg.transform.translation.z = float(self.camera_z)

        # Задава ориентацията на камерата.
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        # Публикува статичната трансформация.
        self.static_tf_broadcaster.sendTransform(tf_msg)

        # Запомня за коя координатна система
        # е публикувана трансформацията.
        self.static_tf_published_for_frame = self.camera_frame

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------

    # Получава RGB изображение и го преобразува
    # до BGR NumPy масив, използван от OpenCV.
    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.warn_throttled("rgb_error", f"RGB conversion failed: {e}")

    # Получава вътрешните параметри на RGB камерата.
    def camera_info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    # Основен callback за обработване
    # на нов облак от точки.
    def pointcloud_callback(self, msg: PointCloud2):
        self.frame_count += 1

        # Пропуска кадъра, когато е зададено
        # обработване само на всеки N-ти кадър.
        if self.frame_count % self.process_every_n_frames != 0:
            return

        # Изчаква първото RGB изображение.
        if self.latest_rgb is None:
            self.warn_throttled("no_rgb", "Waiting for RGB image...")
            return

        # Изчаква параметрите за проекция
        # на RGB камерата.
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.warn_throttled("no_camera_info", "Waiting for RGB CameraInfo...")
            return

        # При различна координатна система в облака
        # обновява camera_frame и статичната TF трансформация.
        if msg.header.frame_id and msg.header.frame_id != self.camera_frame:
            self.camera_frame = msg.header.frame_id

            if self.publish_camera_static_tf_enabled:
                self.publish_camera_static_tf(force=True)

        # Извлича валидните XYZ точки от PointCloud2.
        points_cam = self.read_xyz_points(msg)

        # При празен или невалиден облак
        # публикува празен резултат.
        if points_cam is None or points_cam.shape[0] == 0:
            self.publish_empty_cloud()
            return

        # Проектира точките върху RGB изображението
        # и свързва всяка точка със съответния цвят.
        points_cam_colored, colors_rgb = self.colorize_points_from_rgb(points_cam)

        # Няма точки, които попадат в границите
        # на RGB изображението.
        if points_cam_colored.shape[0] == 0:
            self.publish_empty_cloud()
            return

        # Преобразува точките от оптичната система
        # на камерата към base_link.
        points_base = self.transform_points_to_base(points_cam_colored)

        # Оставя само точките в зададеното
        # работно пространство.
        points_base, colors_rgb = self.filter_workspace(points_base, colors_rgb)

        # Няма точки след пространственото филтриране.
        if points_base.shape[0] == 0:
            self.publish_empty_cloud()
            return

        # Намалява плътността на облака,
        # запазвайки съответствието между точките и цветовете.
        points_base, colors_rgb = self.fast_voxel_downsample_with_colors(
            points_base,
            colors_rgb,
            self.voxel_size,
        )

        # Публикува крайния цветен облак.
        self.publish_colorized_cloud(points_base, colors_rgb)

        # Обновява статистиката за честотата на обработване.
        self.log_fps()

    # -------------------------------------------------------------------------
    # Read PointCloud2
    # -------------------------------------------------------------------------

    # Извлича XYZ координатите директно от бинарното
    # съдържание на PointCloud2 чрез NumPy dtype.
    def read_xyz_points(self, msg: PointCloud2) -> Optional[np.ndarray]:
        # Проверява дали съобщението съдържа точки.
        if msg.width == 0 or msg.height == 0:
            return None

        # Създава речник с описанието
        # на всяко поле от PointCloud2.
        field_map = {field.name: field for field in msg.fields}

        # XYZ полетата са задължителни.
        if "x" not in field_map or "y" not in field_map or "z" not in field_map:
            self.warn_throttled(
                "missing_xyz",
                f"Cloud fields missing x/y/z: {[f.name for f in msg.fields]}",
            )
            return None

        # Получава описанията на трите координатни полета.
        x_field = field_map["x"]
        y_field = field_map["y"]
        z_field = field_map["z"]

        # Текущата бърза NumPy реализация
        # очаква координатите да бъдат FLOAT32.
        if (
            x_field.datatype != PointField.FLOAT32
            or y_field.datatype != PointField.FLOAT32
            or z_field.datatype != PointField.FLOAT32
        ):
            self.warn_throttled(
                "bad_xyz_type",
                "PointCloud2 x/y/z fields are not FLOAT32.",
            )
            return None

        # Определя реда на байтовете.
        endian = ">" if msg.is_bigendian else "<"

        # Създава структуриран NumPy тип,
        # съобразен с offset стойностите и point_step.
        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [endian + "f4", endian + "f4", endian + "f4"],
            "offsets": [x_field.offset, y_field.offset, z_field.offset],
            "itemsize": msg.point_step,
        })

        try:
            # Интерпретира бинарното съдържание
            # без допълнително копиране.
            raw = np.frombuffer(msg.data, dtype=dtype)

            # Ограничава резултата до декларирания
            # брой точки в съобщението.
            expected = msg.width * msg.height
            raw = raw[:expected]

            # Създава стандартен масив с форма N x 3.
            points = np.empty((raw.shape[0], 3), dtype=np.float32)
            points[:, 0] = raw["x"]
            points[:, 1] = raw["y"]
            points[:, 2] = raw["z"]

            # Премахва NaN, безкрайни стойности
            # и точки извън разрешения диапазон на дълбочина.
            valid = (
                np.isfinite(points[:, 0])
                & np.isfinite(points[:, 1])
                & np.isfinite(points[:, 2])
                & (points[:, 2] > self.min_camera_depth)
                & (points[:, 2] < self.max_camera_depth)
            )

            return points[valid]

        except Exception as e:
            self.warn_throttled("read_failed", f"PointCloud read failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Colorization
    # -------------------------------------------------------------------------

    # Проектира тримерните точки върху RGB изображението
    # чрез модела на перспективна камера.
    def colorize_points_from_rgb(
        self,
        points_cam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Използва последното налично RGB изображение.
        rgb = self.latest_rgb

        # Размери на RGB изображението.
        img_h, img_w = rgb.shape[:2]

        # XYZ координати в оптичната система на камерата.
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]

        # Проектира точките до пикселни координати.
        u = (self.fx * x / z) + self.cx + self.projection_u_offset
        v = (self.fy * y / z) + self.cy + self.projection_v_offset

        # По желание обръща хоризонталната проекция.
        if self.flip_projection_x:
            u = (img_w - 1) - u

        # По желание обръща вертикалната проекция.
        if self.flip_projection_y:
            v = (img_h - 1) - v

        # Закръгля координатите до цели пикселни индекси.
        u_i = np.round(u).astype(np.int32)
        v_i = np.round(v).astype(np.int32)

        # Валидни са само проекциите,
        # които попадат вътре в изображението.
        valid = (
            (u_i >= 0)
            & (u_i < img_w)
            & (v_i >= 0)
            & (v_i < img_h)
            & np.isfinite(u)
            & np.isfinite(v)
        )

        # Връща празни масиви, когато нито една точка
        # не може да бъде оцветена.
        if not np.any(valid):
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )

        # Запазва само точките с валидна проекция.
        points_valid = points_cam[valid]
        u_valid = u_i[valid]
        v_valid = v_i[valid]

        # OpenCV image is BGR. Convert to RGB.

        # Извлича BGR цвета от съответния пиксел.
        bgr = rgb[v_valid, u_valid]

        # Обръща каналите от BGR към RGB.
        colors_rgb = bgr[:, ::-1].astype(np.uint8)

        return points_valid, colors_rgb

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    # Прилага въртенето и транслацията
    # от координатната система на камерата към base_link.
    def transform_points_to_base(self, points_cam: np.ndarray) -> np.ndarray:
        points_base = points_cam @ self.camera_rotation_base_from_optical.T
        points_base = points_base + self.camera_translation_base
        return points_base.astype(np.float32)

    # Оставя само точките, които попадат
    # в предварително зададените граници на работното пространство.
    def filter_workspace(
        self,
        points_base: np.ndarray,
        colors_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Разделя координатите по оси.
        x = points_base[:, 0]
        y = points_base[:, 1]
        z = points_base[:, 2]

        # Създава обща маска за разрешения тримерен обем.
        mask = (
            (x >= self.workspace_x_min)
            & (x <= self.workspace_x_max)
            & (y >= self.workspace_y_min)
            & (y <= self.workspace_y_max)
            & (z >= self.workspace_z_min)
            & (z <= self.workspace_z_max)
        )

        # Прилага еднаква маска върху точките и цветовете.
        return points_base[mask], colors_rgb[mask]

    # Намалява броя на точките, като запазва
    # по една представителна точка за всеки воксел.
    def fast_voxel_downsample_with_colors(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Не извършва обработка при празен облак
        # или невалиден размер на воксела.
        if points.shape[0] == 0 or voxel_size <= 0:
            return points, colors

        # Преобразува координатите до целочислени
        # индекси на воксели.
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Добавя положително отместване,
        # за да намали проблемите при отрицателни координати.
        ix = voxel_indices[:, 0].astype(np.int64) + 100000
        iy = voxel_indices[:, 1].astype(np.int64) + 100000
        iz = voxel_indices[:, 2].astype(np.int64) + 100000

        # Създава хеш ключ за всеки тримерен воксел.
        keys = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791

        # Получава индекса на първата точка
        # във всеки уникален воксел.
        _, unique_indices = np.unique(keys, return_index=True)

        # Избира точките и съответните им цветове.
        points_ds = points[unique_indices]
        colors_ds = colors[unique_indices]

        # При прекалено голям резултат
        # избира равномерно разпределено подмножество.
        if points_ds.shape[0] > self.max_points_publish:
            indices = np.linspace(
                0,
                points_ds.shape[0] - 1,
                self.max_points_publish,
                dtype=np.int32,
            )
            points_ds = points_ds[indices]
            colors_ds = colors_ds[indices]

        return points_ds, colors_ds

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    # Кодира отделните RGB канали
    # в една 32-битова стойност 0x00RRGGBB.
    def pack_rgb_uint32(self, colors_rgb: np.ndarray) -> np.ndarray:
        r = colors_rgb[:, 0].astype(np.uint32)
        g = colors_rgb[:, 1].astype(np.uint32)
        b = colors_rgb[:, 2].astype(np.uint32)

        rgb_uint32 = (r << 16) | (g << 8) | b
        return rgb_uint32

    # Създава и публикува PointCloud2
    # с полета X, Y, Z и RGB.
    def publish_colorized_cloud(
        self,
        points_base: np.ndarray,
        colors_rgb: np.ndarray,
    ):
        # При празен резултат публикува
        # валидно празно PointCloud2 съобщение.
        if points_base.shape[0] == 0:
            self.publish_empty_cloud()
            return

        # Кодира RGB каналите до uint32.
        rgb_uint32 = self.pack_rgb_uint32(colors_rgb)

        # Създава структуриран NumPy масив,
        # съвместим с описанието на PointCloud2 полетата.
        cloud_points = np.zeros(
            points_base.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ],
        )

        # Попълва геометричните координати.
        cloud_points["x"] = points_base[:, 0]
        cloud_points["y"] = points_base[:, 1]
        cloud_points["z"] = points_base[:, 2]

        # Попълва пакетирания цвят.
        cloud_points["rgb"] = rgb_uint32

        # Създава заглавната част на съобщението.
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.stand_frame

        # Описва бинарното разположение
        # на четирите полета в една точка.
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Създава и публикува ROS 2 съобщението.
        msg = pc2.create_cloud(header, fields, cloud_points)
        self.cloud_pub.publish(msg)

    # Публикува празен цветен облак,
    # запазвайки правилната структура на полетата.
    def publish_empty_cloud(self):
        # Създава заглавна част в координатната система base_link.
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.stand_frame

        # Използва същите полета като нормалния цветен облак.
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Създава празен структуриран масив.
        empty = np.zeros(
            0,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ],
        )

        # Създава и публикува празното PointCloud2 съобщение.
        msg = pc2.create_cloud(header, fields, empty)
        self.cloud_pub.publish(msg)


# Основна входна точка на ROS 2 възела.
def main(args=None):
    # Инициализира ROS 2 комуникацията.
    rclpy.init(args=args)

    # Създава възела за цветния облак.
    node = ColorizedPointCloud()

    try:
        # Обработва ROS 2 callback функциите,
        # докато възелът не бъде спрян.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Позволява чисто спиране чрез Ctrl+C.
        pass

    # Унищожава възела и освобождава неговите ресурси.
    node.destroy_node()

    try:
        # Спира ROS 2 контекста.
        rclpy.shutdown()
    except Exception:
        # Игнорира повторно или неуспешно спиране.
        pass


# Стартира възела само когато файлът
# се изпълнява директно.
if __name__ == "__main__":
    main()