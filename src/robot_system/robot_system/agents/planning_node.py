#!/usr/bin/env python3
from __future__ import annotations

# Стандартни библиотеки за опашка от заявки, нишки, файлови пътища
# и типови анотации.
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Външни зависимости за Redis и ROS 2.
import redis
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray

# Основни класове от подсистемата за планиране.
from robot_system.planning import (
    GeometricIKSolver,
    MotionBridge,
    ObjectInstanceRegistry,
    ObjectTracebackStore,
    PoseLibrary,
    WorkspaceTargetTools,
)

# Общи помощни функции за декодиране и зареждане
# на променливи от средата.
from robot_system.planning.common import decode, env, env_bool, env_float

# Mixin класове, които разделят поведението на планиращия агент
# по отделни функционални отговорности.
from robot_system.planning.action_router_mixin import PlanningActionRouterMixin
from robot_system.planning.cartesian_solver_mixin import (
    PlanningCartesianSolverMixin,
)
from robot_system.planning.held_object_flow_mixin import (
    PlanningHeldObjectFlowMixin,
)
from robot_system.planning.pick_flow_mixin import PlanningPickFlowMixin
from robot_system.planning.pose_flow_mixin import PlanningPoseFlowMixin
from robot_system.planning.request_stream_mixin import (
    PlanningRequestStreamMixin,
)
from robot_system.planning.return_flow_mixin import PlanningReturnFlowMixin
from robot_system.planning.transfer_flow_mixin import (
    PlanningTransferFlowMixin,
)

# Константи, Redis ключове, имена на потоци и конфигурационни пътища.
from robot_system.planning.settings import (
    LAST_TARGET_KEY,
    MOTION_POINTS_KEY,
    OBJECT_INSTANCE_REGISTRY_KEY,
    OBJECT_TRACEBACK_KEY,
    PACKAGE_CONFIG_DIR,
    PLANNING_REQUEST_STREAM,
    PLANNING_RESULT_STREAM,
    REDIS_URL,
    TARGET_OFFSETS_KEY,
    TRACEBACK_MARKER_TOPIC,
)

# Mixin класове за състоянието и визуализацията.
from robot_system.planning.state_mixin import PlanningStateMixin
from robot_system.planning.visualization_mixin import (
    PlanningVisualizationMixin,
)


class PlanningAgent(
    PlanningCartesianSolverMixin,
    PlanningStateMixin,
    PlanningRequestStreamMixin,
    PlanningActionRouterMixin,
    PlanningPoseFlowMixin,
    PlanningPickFlowMixin,
    PlanningTransferFlowMixin,
    PlanningHeldObjectFlowMixin,
    PlanningReturnFlowMixin,
    PlanningVisualizationMixin,
    Node,
):
    """
    Основен ROS 2 възел на планиращия агент.

    Класът обединява специализираните mixin модули за:
    - обработване на заявки;
    - избор на цели;
    - обратна кинематика;
    - именувани позиции;
    - хващане и поставяне;
    - връщане на предмети;
    - визуализация;
    - управление на цифровия и физическия манипулатор.
    """

    def __init__(self) -> None:
        # Инициализира ROS 2 възела с име planning_agent.
        super().__init__("planning_agent")

        # Създава връзка към Redis чрез зададения URL.
        # decode_responses=True връща текстови стойности вместо байтове.
        self.redis = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
        )

        # Запазва идентификатора на последната съществуваща заявка,
        # за да не се обработват повторно стари записи при стартиране.
        self.last_request_id = self.stream_tail_id(
            PLANNING_REQUEST_STREAM
        )

        # Последното получено съобщение със ставните положения.
        self.latest_joint_state: Optional[JointState] = None

        # Събитие за прекратяване на текущо движение или операция.
        self.cancel_event = threading.Event()

        # Събитие за окончателно спиране на работната нишка.
        self.stop_event = threading.Event()

        # Опашка за последователно обработване на постъпилите заявки.
        self.request_queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = (
            queue.Queue()
        )

        # Геометричен решател за обратна кинематика.
        self.ik = GeometricIKSolver()

        # Инструменти за достъп до откритите предмети,
        # техните точки за движение и приложените корекции.
        self.targets = WorkspaceTargetTools(
            self.redis,
            MOTION_POINTS_KEY,
            TARGET_OFFSETS_KEY,
            LAST_TARGET_KEY,
        )

        # Хранилище за историята на движенията на предметите
        # и началните им позиции.
        self.traceback = ObjectTracebackStore(
            self.redis,
            OBJECT_TRACEBACK_KEY,
            MOTION_POINTS_KEY,
        )

        # Регистър за устойчиви идентификатори на еднакви
        # или сходни предмети.
        self.instances = ObjectInstanceRegistry(
            self.redis,
            OBJECT_INSTANCE_REGISTRY_KEY,
        )

        # Абонира се за текущите ставни положения.
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            20,
        )

        # Публикува изчислените ставни положения директно
        # към цифровия модел в RViz.
        self.joint_state_pub = self.create_publisher(
            JointState,
            "/joint_states",
            20,
        )

        # Публикува маркери, свързани с планирането,
        # целевите точки и траекториите.
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/digital_twin/planner_markers",
            10,
        )

        # Публикува маркери за началните и предишните
        # позиции на преместваните предмети.
        self.traceback_marker_pub = self.create_publisher(
            MarkerArray,
            TRACEBACK_MARKER_TOPIC,
            10,
        )

        # Създава мост за изпращане на движенията към:
        # - цифровия модел чрез /joint_states;
        # - физическия манипулатор чрез UDP.
        self.motion = MotionBridge(
            self.joint_state_pub,
            self.get_clock(),
            self.get_logger(),
            cancel_check=self.cancel_event.is_set,
        )

        # Определя пътя до файла с именуваните позиции.
        # При липса на променлива от средата се използва
        # permanent_points.json от конфигурационната директория.
        pose_path = Path(
            env(
                "PLANNER_POSES_FILE",
                str(PACKAGE_CONFIG_DIR / "permanent_points.json"),
            )
        )

        # Зарежда именуваните позиции и ги свързва
        # с калибрационните преобразувания на физическата ръка.
        self.poses = PoseLibrary(
            pose_path,
            self.motion.real.calibrators,
        )

        # Определя дали по подразбиране движенията трябва
        # да се изпълняват само като визуализация в RViz.
        self.preview_only_default = env_bool(
            "PLANNER_PREVIEW_ONLY",
            False,
        )

        # Допълнително вертикално разстояние при поставяне на предмет.
        self.place_clearance = env_float(
            "PLANNER_PLACE_CLEARANCE",
            0.012,
        )

        # Височина на подход над крайната точка за поставяне.
        self.place_approach_height = env_float(
            "PLANNER_PLACE_APPROACH_HEIGHT",
            0.10,
        )

        # Стандартна височина над открит предмет.
        self.above_object_clearance = env_float(
            "PLANNER_ABOVE_OBJECT_CLEARANCE_M",
            0.10,
        )

        # Минимално допустимо отстояние над предмета.
        self.minimum_above_clearance = env_float(
            "PLANNER_MIN_ABOVE_CLEARANCE_M",
            0.06,
        )

        # Допълнителна безопасна височина при преместване
        # между две точки.
        self.transit_clearance = max(
            0.0,
            env_float("PLANNER_TRANSIT_CLEARANCE_M", 0.04),
        )

        # Страничен запас около коридора на движение.
        self.transit_corridor_margin = max(
            0.0,
            env_float("PLANNER_TRANSIT_CORRIDOR_MARGIN_M", 0.025),
        )

        # Максимално разрешена височина при транзитно движение.
        self.maximum_transit_z = max(
            0.10,
            env_float("PLANNER_MAX_TRANSIT_Z_M", 0.34),
        )

        # Периодично проверява Redis за нови заявки
        # на всеки 0.05 секунди.
        self.create_timer(0.05, self.poll_requests)

        # Отделна работна нишка обработва заявките,
        # без да блокира основния ROS 2 цикъл.
        self.worker = threading.Thread(
            target=self.request_worker,
            name="object-planning-worker",
            daemon=True,
        )
        self.worker.start()

        # Информационни съобщения при успешно стартиране.
        self.get_logger().info("✅ planning_agent ready")

        self.get_logger().info(
            "Planning solutions update /joint_states directly for RViz; "
            "the physical arm remains a separate slower UDP target."
        )

        # Показва дали изпращането към физическия манипулатор
        # е активно и към кой адрес се изпращат пакетите.
        self.get_logger().info(
            f"send_to_real={self.motion.real.enabled} "
            f"pi={self.motion.real.ip}:{self.motion.real.port}"
        )

        # Показва заредената начална позиция.
        self.get_logger().info(
            f"home/scan pose={self.poses.describe('home')}"
        )

    def destroy_node(self) -> bool:
        """
        Безопасно прекратява планиращия агент.

        Спира работната нишка, отменя текущото движение,
        затваря връзките на MotionBridge и унищожава ROS 2 възела.
        """

        # Сигнализира на работната нишка да приключи.
        self.stop_event.set()

        # Прекратява текущото движение или активна заявка.
        self.cancel_event.set()

        # Изчаква работната нишка да приключи,
        # но не повече от две секунди.
        worker = getattr(self, "worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

        # Затваря използваните комуникационни ресурси.
        self.motion.close()

        # Извиква стандартното унищожаване на ROS 2 възела.
        return super().destroy_node()

    def stream_tail_id(self, stream: str) -> str:
        """
        Връща идентификатора на последния запис в Redis поток.

        Ако потокът е празен или възникне грешка,
        се връща началният идентификатор 0-0.
        """

        try:
            # Прочита последния запис от потока.
            rows = self.redis.xrevrange(stream, count=1)

            # При наличен запис връща декодиран идентификатор.
            if rows:
                return decode(rows[0][0])
        except Exception:
            # При грешка възелът продължава работа
            # и започва от началния идентификатор.
            pass

        return "0-0"

    def joint_state_callback(self, msg: JointState) -> None:
        """
        Обработва ново съобщение със ставните положения.

        Запазва последното състояние и изчислява текущата
        декартова позиция чрез права кинематика.
        """

        # Запазва последното получено съобщение.
        self.latest_joint_state = msg

        try:
            # Актуализира текущата декартова поза,
            # използвана при следващи движения и проверки.
            self.persist_current_pose_from_joint_state(msg)
        except Exception as exc:
            # Грешката не прекратява възела, а се записва
            # като диагностично съобщение.
            self.get_logger().debug(
                f"Could not update current Cartesian pose: {exc}"
            )


def main(args=None) -> None:
    """
    Главна функция за стартиране на planning_agent.
    """

    # Инициализира ROS 2 комуникацията.
    rclpy.init(args=args)

    # Създава екземпляр на планиращия агент.
    node = PlanningAgent()

    try:
        # Стартира основния ROS 2 цикъл за обработване
        # на съобщения, таймери и обратни извиквания.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Позволява прекратяване чрез Ctrl+C.
        pass
    finally:
        # Освобождава ресурсите и спира ROS 2.
        node.destroy_node()
        rclpy.shutdown()


# Стартира main() само когато файлът се изпълнява директно.
if __name__ == "__main__":
    main()