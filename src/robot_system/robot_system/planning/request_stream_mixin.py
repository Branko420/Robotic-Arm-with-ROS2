#!/usr/bin/env python3

# Позволява отложено оценяване на типовите анотации.
from __future__ import annotations

# Стандартни библиотеки за:
# - JSON сериализация и десериализация;
# - опашка за заявки;
# - работа с времеви стойности;
# - типови анотации.
import json
import queue
import time
from typing import Any, Dict, Optional

# Изключения, свързани с недостъпен физически
# манипулатор или неуспешна физическа команда.
from .arm_client import (
    PhysicalArmCommandError,
    PhysicalArmUnavailableError,
)

# Помощна функция за декодиране на стойности,
# върнати от Redis.
from .common import decode

# Конфигурационни стойности за:
# - Redis потока с входящи заявки;
# - Redis потока с резултати;
# - ключовете за кеширани резултати;
# - времето за съхранение на кеша.
from .settings import (
    PLANNING_REQUEST_STREAM,
    PLANNING_RESULT_STREAM,
    REQUEST_RESULT_PREFIX,
    REQUEST_RESULT_TTL_SEC,
)

# Изключение при откриване на повече от една
# възможна цел за потребителската заявка.
from .targets import AmbiguousTargetError


class PlanningRequestStreamMixin:
    """
    Съдържа логиката за приемане, изпълнение,
    кеширане и публикуване на заявки към планиращия агент.

    Заявките се получават чрез Redis Stream, след което
    се поставят във вътрешна опашка и се обработват
    последователно от отделен работен поток.
    """

    def poll_requests(self) -> None:
        """
        Проверява Redis потока за нови заявки
        към планиращия агент.
        """

        try:
            # Прочита до 20 нови записа след
            # последно обработения Redis идентификатор.
            messages = self.redis.xread(
                {
                    PLANNING_REQUEST_STREAM:
                    self.last_request_id
                },
                count=20,
            )

        except Exception as exc:
            # При проблем с Redis записва грешката
            # и прекратява текущото извикване.
            self.get_logger().error(
                f"Redis xread failed: {exc}"
            )
            return

        # Обхожда върнатите Redis потоци.
        for _stream, entries in messages:
            # Обхожда отделните записи от потока.
            for entry_id, fields in entries:
                # Запазва идентификатора на последно
                # прочетената заявка.
                self.last_request_id = decode(
                    entry_id
                )

                try:
                    # Декодира JSON съдържанието
                    # от полето payload.
                    request = json.loads(
                        fields.get(
                            "payload",
                            "{}",
                        )
                    )

                except Exception:
                    # При невалиден JSON използва
                    # празна заявка.
                    request = {}

                # Получава идентификатора на заявката.
                #
                # Когато липсва, се генерира уникален
                # идентификатор чрез текущото време.
                request_id = str(
                    request.get("request_id")
                    or f"plan-{time.time_ns()}"
                )

                # Получава и нормализира исканото действие.
                action = str(
                    request.get(
                        "action",
                        "move",
                    )
                ).strip().lower()

                # Записва диагностична информация
                # за получената заявка.
                self.get_logger().info(
                    f"📥 planning request: {request_id} "
                    f"action={action} "
                    f"target={request.get('target')}"
                )

                # Командата stop се обработва незабавно,
                # без да се добавя към работната опашка.
                if action == "stop":
                    # Задава събитието за прекратяване
                    # на текущото движение.
                    self.cancel_event.set()

                    # Отменя всички заявки,
                    # които още не са започнали.
                    cancelled_pending = (
                        self.cancel_pending_requests()
                    )

                    # Изпраща команда за задържане
                    # на текущата физическа позиция.
                    physical_hold_sent = (
                        self.publish_hold_position()
                    )

                    # Публикува резултат за успешно
                    # обработена команда stop.
                    self.publish_result(
                        self.result(
                            request_id,
                            True,
                            "STOPPED",
                            cancelled_pending=(
                                cancelled_pending
                            ),
                            physical_hold_sent=(
                                physical_hold_sent
                            ),
                        )
                    )

                    # Преминава към следващия
                    # запис от Redis потока.
                    continue

                # Добавя заявката към вътрешната опашка
                # за последователна обработка.
                self.request_queue.put(
                    (
                        request_id,
                        request,
                    )
                )

    def request_worker(self) -> None:
        """
        Обработва заявките от вътрешната опашка.

        Методът обикновено се изпълнява в отделен поток,
        докато не бъде зададено събитието stop_event.
        """

        # Работи до спиране на планиращия агент.
        while not self.stop_event.is_set():
            try:
                # Изчаква нова заявка до 0.25 секунди.
                request_id, request = (
                    self.request_queue.get(
                        timeout=0.25
                    )
                )

            except queue.Empty:
                # При празна опашка започва
                # следваща итерация.
                continue

            # Проверява дали заявката вече е била
            # обработена и резултатът е кеширан.
            cached = self.cached_result(
                request_id
            )

            if cached is not None:
                # Отбелязва, че резултатът
                # е повторно публикуван от кеша.
                cached["replayed"] = True

                # Публикува резултата, без да го
                # записва повторно в кеша.
                self.publish_result(
                    cached,
                    cache=False,
                )

                # Маркира задачата като завършена.
                self.request_queue.task_done()
                continue

            # Изчиства евентуален стар сигнал
            # за прекратяване преди новата заявка.
            self.cancel_event.clear()

            try:
                # Предава заявката към основната
                # логика на планиращия агент.
                result = self.handle_request(
                    request_id,
                    request,
                )

            except AmbiguousTargetError as exc:
                # Обработва случай, при който повече
                # от един предмет отговаря на описанието.
                result = self.result(
                    request_id,
                    False,
                    "AMBIGUOUS_TARGET",
                    query=exc.query,
                    candidates=exc.candidates,
                    hint=(
                        "Specify instance number, left/right, "
                        "nearest/farthest, or an instance_id."
                    ),
                )

            except PhysicalArmUnavailableError as exc:
                # Обработва недостъпност
                # на физическия манипулатор.
                result = self.result(
                    request_id,
                    False,
                    "PHYSICAL_ARM_UNAVAILABLE",
                    error=str(exc),

                    # Изрично отбелязва, че не е изпратено
                    # физическо движение.
                    physical_movement_sent=False,
                )

            except PhysicalArmCommandError as exc:
                # Обработва неуспешно изпълнение
                # на команда към Raspberry Pi.
                result = self.result(
                    request_id,
                    False,
                    "PHYSICAL_ARM_COMMAND_FAILED",
                    error=str(exc),
                    physical_movement_sent=False,
                )

            except Exception as exc:
                # Записва пълната информация
                # за непредвидена грешка.
                self.get_logger().exception(
                    f"planning request crashed: {exc}"
                )

                # Създава структуриран резултат
                # за срив на планирането.
                result = self.result(
                    request_id,
                    False,
                    "PLANNING_CRASHED",
                    error=str(exc),
                )

            # Публикува крайния резултат
            # в Redis потока.
            self.publish_result(result)

            # Маркира текущата задача
            # от опашката като приключена.
            self.request_queue.task_done()

    def request_result_key(
        self,
        request_id: str,
    ) -> str:
        """
        Създава Redis ключ за кеширания
        резултат на конкретна заявка.
        """

        return (
            f"{REQUEST_RESULT_PREFIX}:"
            f"{request_id}"
        )

    def cached_result(
        self,
        request_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Зарежда кеширан резултат за заявка.

        Връща None, когато резултатът липсва,
        е невалиден или не може да бъде прочетен.
        """

        try:
            # Прочита стойността от Redis.
            raw = self.redis.get(
                self.request_result_key(
                    request_id
                )
            )

            # При липсващ кеш връща None.
            if not raw:
                return None

            # Декодира JSON резултата.
            value = json.loads(raw)

            # Приема само речников резултат.
            return (
                value
                if isinstance(value, dict)
                else None
            )

        except Exception:
            # При Redis или JSON грешка
            # приема, че кеш няма.
            return None

    def publish_result(
        self,
        result: Dict[str, Any],
        *,
        cache: bool = True,
    ) -> None:
        """
        Публикува резултат от планирането
        в Redis Stream.

        По подразбиране резултатът се кешира
        временно спрямо request_id.
        """

        try:
            # Сериализира резултата като JSON,
            # като запазва символите на кирилица.
            payload = json.dumps(
                result,
                ensure_ascii=False,
            )

            # Кешира резултата само когато:
            # - кеширането е разрешено;
            # - резултатът съдържа request_id.
            if (
                cache
                and result.get("request_id")
            ):
                self.redis.setex(
                    self.request_result_key(
                        str(result["request_id"])
                    ),
                    REQUEST_RESULT_TTL_SEC,
                    payload,
                )

            # Добавя резултата към Redis потока
            # с резултати от планирането.
            self.redis.xadd(
                PLANNING_RESULT_STREAM,
                {
                    "payload": payload,
                },
            )

        except Exception as exc:
            # Записва грешка, когато резултатът
            # не може да бъде публикуван.
            self.get_logger().error(
                "Could not publish planning result: "
                f"{exc}"
            )

    def cancel_pending_requests(self) -> int:
        """
        Отменя всички заявки, които чакат
        във вътрешната опашка.

        Връща броя на отменените заявки.
        """

        # Брой успешно премахнати заявки.
        cancelled = 0

        while True:
            try:
                # Взема незабавно следващата заявка,
                # без блокиращо изчакване.
                request_id, _request = (
                    self.request_queue.get_nowait()
                )

            except queue.Empty:
                # При празна опашка приключва.
                break

            # Публикува отделен резултат
            # за всяка отменена заявка.
            self.publish_result(
                self.result(
                    request_id,
                    False,
                    "CANCELLED_BEFORE_START",
                )
            )

            # Маркира задачата като приключена.
            self.request_queue.task_done()

            # Увеличава броя на отменените заявки.
            cancelled += 1

        return cancelled