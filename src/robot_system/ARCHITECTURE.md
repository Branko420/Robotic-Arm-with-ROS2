# Architecture

<!--
Пакетът е структуриран според отговорността на отделните компоненти,
а не според последователността, в която са били разработвани.
-->

The package is organized by responsibility instead of by historical implementation stage.

<!--
Следващата структура показва основните директории на пакета robot_system
и предназначението на всяка от тях.
-->

```text
robot_system/
├── agents/          ROS nodes and the OpenAI conversation-agent definition
├── interfaces/      Function tools used by the conversation agent
├── planning/        Planning domain, object geometry, IK and arm bridge
├── scanning/        Point-cloud processing used by workspace scanning
├── realtime/        Audio/realtime session runtime
├── storage/         Redis adapter shared by non-planning legacy-independent code
├── visualization/   RViz point-cloud publishing
└── config/          Versioned calibration and named servo poses
```

<!--
agents/ съдържа ROS възлите и дефиницията на разговорния агент.

interfaces/ съдържа функционалните инструменти, използвани
от разговорния агент за изпращане на структурирани заявки.

planning/ съдържа логиката за планиране, геометрията на предметите,
обратната кинематика и връзката с физическия манипулатор.

scanning/ съдържа обработването на облака от точки,
използвано при сканиране на работното пространство.

realtime/ съдържа компонентите за аудиообработка
и изпълнение на сесията в реално време.

storage/ съдържа общия Redis адаптер,
използван от компонентите извън планиращия модул.

visualization/ съдържа публикуването на облака от точки
и останалите визуални данни към RViz.

config/ съдържа версионираните калибрационни файлове
и именуваните сервомоторни позиции.
-->

## Dependency direction

<!--
Следващата схема представя посоката на зависимостите
и обмена на заявки между основните компоненти.
-->

```text
realtime -> agents/realtime_agent -> interfaces -> Redis streams
                                             |
                                             v
                                    agents/planning_node
                                             |
                                             v
                                         planning

workspace_scan_node -> scanning
visualization is independent of planning
```

<!--
Модулът realtime предава аудиоданните към realtime_agent.

realtime_agent използва функционалните инструменти от interfaces.

Функционалните инструменти публикуват структурирани заявки
в Redis потоците.

planning_node получава заявките от Redis и използва
модулите от planning за тяхното обработване.

workspace_scan_node използва модулите от scanning
за обработване на данните от RGB-D камерата.

Модулът за визуализация е независим от планиращия модул.
-->

The planning domain does not import the realtime agent. Function tools do not contain IK or geometry algorithms; they only validate user-facing requests and send structured requests to the planning node.

<!--
Планиращият модул не импортира разговорния агент,
което предотвратява пряка зависимост между езиковото взаимодействие
и алгоритмите за движение.

Функционалните инструменти не съдържат обратна кинематика
или алгоритми за геометрична обработка.

Те проверяват подадените от потребителя параметри
и създават структурирани заявки към planning_node.
-->

## Planning modules

<!--
Следва описание на модулите в подсистемата за планиране.
Всеки модул има отделна и ясно определена отговорност.
-->

- `common.py`: environment loading and small shared helpers.
- `constants.py`: canonical joint/servo names and safe ranges.
- `calibration.py`: piecewise RViz-to-servo calibration and named pose loading.
- `arm_client.py`: UDP communication with the Raspberry Pi executor.
- `motion_bridge.py`: sends planner solutions directly to the RViz model and to the physical arm.
- `targets.py`: target resolution and normalized object geometry.
- `instances.py`: stable identities for repeated/similar objects.
- `traceback.py`: object movement history and return locations.
- `ik.py`: geometric IK. The equations were not changed by the refactor.
- `state_mixin.py`: current joints, held-object state and height-aware transit calculation.
- `request_stream_mixin.py`: Redis stream polling, result caching and cancellation.
- `action_router_mixin.py`: validates actions and routes them to focused flows.
- `pose_flow_mixin.py`: named poses such as HOME and SCAN_POSE.
- `pick_flow_mixin.py`: standalone pick sequence.
- `transfer_flow_mixin.py`: pick-and-place sequence.
- `held_object_flow_mixin.py`: placement of an already-held object.
- `return_flow_mixin.py`: return-to-origin sequence and history updates.
- `visualization_mixin.py`: RViz markers and result serialization.
- `robot_model_state.py`: direct interpolated `/joint_states` output for `robot_state_publisher`.

<!--
common.py зарежда променливите на средата
и предоставя малки общи помощни функции.

constants.py съдържа основните имена на ставите,
сервомоторите и техните безопасни диапазони.

calibration.py извършва участъковото калибрационно преобразуване
между ставните стойности в RViz и физическите сервомоторни стойности.
Модулът зарежда и именуваните позиции на манипулатора.

arm_client.py изолира UDP комуникацията
с изпълнителния възел на Raspberry Pi.

motion_bridge.py изпраща изчислените решения
към цифровия модел в RViz и към физическия манипулатор.

targets.py определя целевите точки и нормализира
геометричната информация за откритите предмети.

instances.py поддържа устойчиви идентификатори
при наличие на повтарящи се или сходни предмети.

traceback.py съхранява историята на преместванията
и началните позиции на предметите.

ik.py съдържа геометричния решател за обратна кинематика.
Формулите в решателя не са променяни при преструктурирането.

state_mixin.py поддържа текущите ставни положения,
състоянието на държания предмет и изчисляването
на безопасна височина при преместване.

request_stream_mixin.py извършва прочитането на Redis потоците,
кеширането на резултатите и прекратяването на заявки.

action_router_mixin.py проверява действията
и ги насочва към подходящата специализирана последователност.

pose_flow_mixin.py съдържа операциите
с именувани позиции като HOME и SCAN_POSE.

pick_flow_mixin.py съдържа самостоятелната последователност
за хващане на предмет.

transfer_flow_mixin.py съдържа последователността
за хващане, преместване и поставяне на предмет.

held_object_flow_mixin.py се използва за поставяне
на предмет, който вече се държи от хващача.

return_flow_mixin.py връща предмета в началната му позиция
и актуализира историята на преместванията.

visualization_mixin.py създава RViz маркерите
и преобразува резултатите в структуриран формат.

robot_model_state.py публикува директно интерполирани
съобщения в /joint_states за robot_state_publisher.
-->

## Extension rules

<!--
Следващите правила определят начина за добавяне
на нова функционалност, без да се нарушава модулната архитектура.
-->

1. New conversational capability: add a function tool under `interfaces/`, then register it in `agents/realtime_agent.py`.
2. New planner action: route it in `action_router_mixin.py`, then implement the motion sequence in the appropriate focused flow mixin.
3. New geometry rule: put it in `targets.py`; do not duplicate Z calculations in tools or prompts.
4. New hardware protocol: implement it behind `arm_client.py`; keep the planner independent of UDP packet details.
5. New persistent state: define one key in `settings.py` and keep read/write logic in `state_mixin.py` or `traceback.py`.

<!--
При добавяне на нова възможност за разговорния агент
трябва да бъде създаден нов функционален инструмент в interfaces/,
след което инструментът трябва да бъде регистриран
в agents/realtime_agent.py.

При добавяне на ново действие към планиращия агент
то трябва да бъде насочено чрез action_router_mixin.py,
а последователността на движение трябва да бъде реализирана
в подходящия специализиран flow mixin.

Новите геометрични правила трябва да бъдат добавяни в targets.py.
Изчисленията по оста Z не трябва да се дублират
във функционалните инструменти или в инструкциите към езиковия модел.

Новите хардуерни протоколи трябва да бъдат реализирани
зад интерфейса на arm_client.py.

Това запазва планиращия модул независим
от конкретния формат на UDP пакетите.

При добавяне на ново постоянно състояние
трябва да бъде създаден един ключ в settings.py.

Логиката за четене и записване на състоянието
трябва да остане в state_mixin.py или traceback.py.
-->

## Direct RobotModel movement

<!--
Следва описание на директното управление
на цифровия модел на манипулатора в RViz.
-->

The planning agent now publishes interpolated `sensor_msgs/JointState` messages
directly on `/joint_states`. `robot_state_publisher` consumes those states and
updates the URDF transforms used by RViz. There is no preview bridge, no
`JointTrajectory` controller topic and no MoveIt dependency in this path.

<!--
Планиращият агент публикува интерполирани съобщения
от тип sensor_msgs/JointState директно в /joint_states.

robot_state_publisher получава ставните положения
и актуализира координатните трансформации на URDF модела.

RViz използва тези трансформации,
за да визуализира актуалното положение на цифровия манипулатор.

В този път за управление не се използват:
- междинен възел за предварителен преглед;
- тема за контролер от тип JointTrajectory;
- MoveIt;
- MoveIt 2.
-->

Disable `PLANNER_PUBLISH_MODEL_JOINT_STATES` only when a real
`joint_state_broadcaster` is already publishing `/joint_states`.

<!--
Променливата PLANNER_PUBLISH_MODEL_JOINT_STATES
трябва да бъде изключена само когато реален
joint_state_broadcaster вече публикува в /joint_states.

Едновременното публикуване от planning_agent
и joint_state_broadcaster може да доведе
до конфликтни ставни стойности и неправилна визуализация.
-->

## Planning action routing

<!--
Маршрутизаторът на заявките само определя
кой специализиран модул трябва да обработи даденото действие.

Той не съдържа реализацията на самите движения.
-->

The request router only dispatches actions. Implementations are separated into:

- `status_actions_mixin.py`
- `pose_actions_mixin.py`
- `workspace_actions_mixin.py`
- `held_actions_mixin.py`
- `object_actions_mixin.py`

<!--
status_actions_mixin.py съдържа действията
за проверка и получаване на състоянието на системата.

pose_actions_mixin.py съдържа действията,
свързани с именуваните позиции на манипулатора.

workspace_actions_mixin.py съдържа действията,
свързани с работното пространство и откритите цели.

held_actions_mixin.py съдържа действията,
които се изпълняват, когато хващачът вече държи предмет.

object_actions_mixin.py съдържа действията,
които се изпълняват върху откритите предмети.
-->

The planner-side UDP client no longer carries motor release fields.

<!--
UDP клиентът от страната на планиращия агент
вече не съдържа полета за освобождаване на сервомоторите.

Клиентът предава само необходимите параметри
за изпълнение на конкретните движения.
-->