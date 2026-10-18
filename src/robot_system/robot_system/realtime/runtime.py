#!/usr/bin/env python3

# Асинхронна среда за изпълнение на комуникацията
# с OpenAI Realtime и паралелното обработване на звука.
import asyncio

# Сериализация и десериализация на съобщения,
# изпращани към Redis.
import json

# Достъп до променливите на средата.
import os

# Нишково безопасна опашка за аудиопакетите
# от микрофона.
import queue

# Регулярни изрази за разпознаване
# на директни гласови команди.
import re

# Достъп до стандартния поток за грешки.
import sys

# Управление на заключвания и състояние
# между PortAudio нишките.
import threading

# Работа с времеви стойности
# и уникални времеви идентификатори.
import time

# Работа с файлови пътища.
from pathlib import Path

# Обща типова анотация.
from typing import Any

# NumPy се използва за обработване,
# преобразуване и интерполация на PCM16 звук.
import numpy as np

# Интерфейс към PortAudio за достъп
# до микрофона и високоговорителите.
import sounddevice as sd

# Изпълнителят на OpenAI Realtime агентите.
from agents.realtime import RealtimeRunner

# Зарежда променливите на средата от .env файл.
from dotenv import load_dotenv

# Основният разговорен агент на системата.
from robot_system.agents.realtime_agent import (
    conversation_agent,
)

# Общото Redis хранилище на системата.
from robot_system.storage.redis_store import store


# ============================================================================
# Конфигурация
# ============================================================================

# Основният .env файл в Docker средата.
ENV_FILE = Path("/workspace/.env")

# Аудиото е моно.
CHANNELS = 1

# Един PCM16 аудиосемпъл заема два байта.
SAMPLE_WIDTH_BYTES = 2  # int16 PCM

# OpenAI Realtime използва PCM16 звук
# с честота 24 kHz.
OPENAI_AUDIO_SAMPLE_RATE = 24_000

# Физическото аудиоустройство може да работи
# на 24, 44.1, 48 kHz или друга честота.
#
# Повечето USB слушалки и конферентни устройства
# използват 48 kHz.
AUDIO_HARDWARE_SAMPLE_RATE = 48_000

# Продължителност на един аудиоблок
# в милисекунди.
BLOCK_DURATION_MS = 20

# Максимален брой аудиоблокове,
# които могат да чакат в опашката на микрофона.
MAX_INPUT_QUEUE_CHUNKS = 250


# ============================================================================
# Помощни функции
# ============================================================================

def env_bool(
    name: str,
    default: bool,
) -> bool:
    """
    Зарежда булева стойност от променлива на средата.
    """

    # Получава необработената текстова стойност.
    raw = os.getenv(name)

    # При липсваща променлива връща
    # предварително зададената стойност.
    if raw is None:
        return default

    # Разпознава най-често използваните
    # текстови представяния на True.
    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_float(
    name: str,
    default: float,
) -> float:
    """
    Зарежда число с плаваща запетая
    от променлива на средата.
    """

    raw = os.getenv(name)

    # При липсваща или празна стойност
    # използва стойността по подразбиране.
    if raw is None or not raw.strip():
        return default

    try:
        return float(raw)

    except Exception:
        # При невалидна стойност не прекратява
        # стартирането на системата.
        return default


def env_int(
    name: str,
    default: int,
) -> int:
    """
    Зарежда цяло число от променлива на средата.
    """

    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        return int(raw)

    except Exception:
        return default


def resample_int16_linear(
    samples: np.ndarray,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    """
    Small dependency-free resampler.

    Good enough for realtime voice:
    - mic 48k -> OpenAI 24k
    - OpenAI 24k -> speaker 48k

    Input/output int16 mono.
    """

    # Помощен линеен преобразувател на честотата
    # на дискретизация, който не изисква допълнителна
    # външна библиотека.
    #
    # Използва се в двете посоки:
    # - от хардуна
    # външна библиотека.
    #
    # Изерната честота към 24 kHz за OpenAI;
    # - от 24 kHz към хардуерната честота за възпроизвеждане.

    # При еднакви честоти не е необходимо
    # преизчисляване на семплите.
    if src_rate == dst_rate:
        return np.asarray(
            samples,
            dtype=np.int16,
        )

    # Гарантира PCM16 представяне.
    samples = np.asarray(
        samples,
        dtype=np.int16,
    )

    # При празен вход връща празния масив.
    if samples.size == 0:
        return samples

    # Преобразува семплите до float32
    # за интерполацията.
    src = samples.astype(np.float32)

    # Изчислява продължителността на входния блок.
    duration = samples.size / float(src_rate)

    # Изчислява необходимия брой изходни семпли.
    dst_size = max(
        1,
        int(round(duration * dst_rate)),
    )

    # Координатна ос на оригиналните семпли.
    src_x = np.linspace(
        0.0,
        samples.size - 1,
        num=samples.size,
        dtype=np.float32,
    )

    # Координатна ос на новите семпли.
    dst_x = np.linspace(
        0.0,
        samples.size - 1,
        num=dst_size,
        dtype=np.float32,
    )

    # Извършва линейна интерполация.
    dst = np.interp(
        dst_x,
        src_x,
        src,
    )

    # Ограничава стойностите до валидния
    # диапазон на int16.
    dst = np.clip(
        dst,
        -32768,
        32767,
    )

    # Връща PCM16 масив.
    return dst.astype(np.int16)


def decode_redis(value: Any) -> Any:
    """
    Декодира Redis bytes стойност до UTF-8 текст.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return value


def xadd_json(
    stream: str,
    payload: dict[str, Any],
) -> str:
    """
    Добавя JSON съобщение към Redis Stream
    и връща идентификатора на създаденото събитие.
    """

    # Сериализира съдържанието като JSON,
    # запазвайки символите на кирилица.
    event_id = store.client.xadd(
        stream,
        {
            "payload": json.dumps(
                payload,
                ensure_ascii=False,
            )
        },
    )

    # Нормализира Redis идентификатора до текст.
    return str(
        decode_redis(event_id)
    )


# ============================================================================
# Основна среда за изпълнение
# ============================================================================

class RealtimeAudioRuntime:
    """
    Управлява двупосочния аудиопоток между потребителя,
    OpenAI Realtime агента и локалните аудиоустройства.

    Класът отговаря за:
    - избора на входно и изходно аудиоустройство;
    - автоматичния избор на хардуерна честота;
    - преобразуването между хардуерната честота и 24 kHz;
    - изпращането на звук към OpenAI;
    - възпроизвеждането на отговора;
    - обработването на Realtime събития;
    - прекъсването на асистента от потребителя;
    - директното маршрутизиране на критични команди;
    - диагностиката и контролираното спиране.
    """

    def __init__(self) -> None:
        """
        Зарежда конфигурацията и създава
        необходимото състояние за аудиосесията.
        """

        # Зарежда променливите от /workspace/.env.
        load_dotenv(ENV_FILE)

        # OpenAI ключът е задължителен
        # за създаване на Realtime сесия.
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is missing from /workspace/.env"
            )

        # Глобалните стойности се обновяват
        # според конфигурацията.
        global OPENAI_AUDIO_SAMPLE_RATE
        global AUDIO_HARDWARE_SAMPLE_RATE

        # Честотата на аудиото за OpenAI.
        OPENAI_AUDIO_SAMPLE_RATE = env_int(
            "OPENAI_AUDIO_SAMPLE_RATE",
            24_000,
        )

        # Запазва оригиналната конфигурационна стойност
        # за хардуерната честота.
        self.requested_hardware_rate_raw = os.getenv(
            "AUDIO_SAMPLE_RATE",
            "auto",
        ).strip()

        # Използван Realtime модел.
        self.model = os.getenv(
            "OPENAI_REALTIME_MODEL",
            "gpt-realtime-2",
        )

        # Глас на асистента.
        self.voice = os.getenv(
            "OPENAI_REALTIME_VOICE",
            "ash",
        )

        # Скорост на синтезираната реч.
        self.speech_speed = env_float(
            "OPENAI_REALTIME_SPEED",
            0.90,
        )

        # Определя входното аудиоустройство.
        self.input_device = self.resolve_audio_device(
            env_key="AUDIO_INPUT_DEVICE",
            kind="input",
        )

        # Определя изходното аудиоустройство.
        self.output_device = self.resolve_audio_device(
            env_key="AUDIO_OUTPUT_DEVICE",
            kind="output",
        )

        # Намира честота, поддържана едновременно
        # от входното и изходното устройство.
        self.hardware_sample_rate = (
            self.resolve_hardware_sample_rate(
                input_device=self.input_device,
                output_device=self.output_device,
            )
        )

        # Обновява глобалната хардуерна честота.
        AUDIO_HARDWARE_SAMPLE_RATE = (
            self.hardware_sample_rate
        )

        # Брой OpenAI семпли в един 20 ms блок.
        self.openai_block_size = (
            OPENAI_AUDIO_SAMPLE_RATE
            * BLOCK_DURATION_MS
            // 1000
        )

        # Брой хардуерни семпли в един 20 ms блок.
        self.hardware_block_size = (
            self.hardware_sample_rate
            * BLOCK_DURATION_MS
            // 1000
        )

        # Нишково безопасна опашка,
        # в която callback функцията на микрофона
        # поставя PCM16 аудиоблокове.
        self.input_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=MAX_INPUT_QUEUE_CHUNKS
        )

        # Буфер с PCM16 звук на хардуерната честота,
        # който чака за възпроизвеждане.
        self.output_buffer = bytearray()

        # Заключване за безопасен паралелен достъп
        # до изходния аудиобуфер.
        self.output_lock = threading.RLock()

        # Заключване за общото състояние
        # на възпроизвеждането.
        self.state_lock = threading.RLock()

        # Показва дали в момента има активен
        # аудиоотговор от асистента.
        self.assistant_output_active = False

        # Показва дали OpenAI е приключил
        # генерирането на текущия аудиоотговор.
        self.assistant_generation_finished = True

        # По желание заглушава микрофона,
        # докато асистентът говори.
        self.mute_input_while_assistant_speaks = env_bool(
            "MUTE_INPUT_WHILE_ASSISTANT_SPEAKS",
            False,
        )

        # Основен флаг за работа на средата.
        self.running = True

        # Показва, че е започнало контролирано спиране.
        self.shutting_down = False

        # Идентификатори на потребителски съобщения,
        # които вече са маршрутизирани директно към Redis.
        self.directly_routed_user_items: set[str] = set()

        # Идентификатори на потребителски реплики,
        # които вече са обработени като прекъсване.
        self.barge_in_user_items: set[str] = set()

        # Брой инструменти, които в момента се изпълняват.
        self.active_tool_count = 0

        # Разрешава прекъсване на асистента
        # чрез нова потребителска реч.
        self.barge_in_enabled = env_bool(
            "REALTIME_BARGE_IN_ENABLED",
            True,
        )

        # Определя дали прекъсването на речта
        # трябва да изпраща и команда stop към планиращия агент.
        self.cancel_motion_on_barge_in = env_bool(
            "REALTIME_CANCEL_MOTION_ON_BARGE_IN",
            True,
        )

        # Redis Stream за заявки към планиращия агент.
        self.planning_request_stream = os.getenv(
            "PLANNING_REQUEST_STREAM",
            "stream:planning_requests",
        )

        # Диагностични броячи за аудиопотока.
        self.input_overflows = 0
        self.input_drops = 0
        self.muted_input_chunks = 0
        self.output_underflows = 0
        self.output_overflows = 0
        self.sent_audio_chunks = 0
        self.received_audio_chunks = 0
        self.mic_resampled_chunks = 0
        self.speaker_resampled_chunks = 0

        # Последно отпечатаният текст за всеки
        # елемент от историята.
        self.last_printed_text: dict[str, str] = {}

        # Създава OpenAI Realtime изпълнителя
        # с разговорния агент като начална точка.
        self.runner = RealtimeRunner(
            starting_agent=conversation_agent,
            config={
                "model_settings": {
                    # Име на Realtime модела.
                    "model_name": self.model,

                    # Избран синтезиран глас.
                    "voice": self.voice,

                    # Скорост на речта.
                    "speed": self.speech_speed,

                    # Входният и изходният звук
                    # използват PCM16 формат.
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",

                    # Конфигурация на автоматичната
                    # транскрипция на микрофонния звук.
                    "input_audio_transcription": {
                        "model": "gpt-4o-mini-transcribe",
                        "language": "bg",
                        "prompt": (
                            "Разговорът е изцяло на български език. "
                            "Разпознавай точно естествена българска реч. "
                            "Контекстът включва роботизирана ръка, "
                            "работно пространство, камера, кубове, предмети, "
                            "цветове, позиции и команди като спри, "
                            "продължи, вземи, върни, планирай и постави."
                        ),
                    },

                    # Потискане на фонов шум,
                    # подходящо за отдалечен микрофон.
                    "input_audio_noise_reduction": {
                        "type": "far_field",
                    },

                    # Семантично откриване на край
                    # на потребителската реплика.
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": "low",
                        "create_response": True,
                        "interrupt_response": True,
                    },

                    # Моделът автоматично решава
                    # кога да използва инструмент.
                    "tool_choice": "auto",
                }
            },
        )

    # =========================================================================
    # Избор на аудиоустройства
    # =========================================================================

    def resolve_audio_device(
        self,
        env_key: str,
        kind: str,
    ) -> int | None:
        """
        kind:
            input
            output

        Supports:
            AUDIO_INPUT_DEVICE=4
            AUDIO_INPUT_DEVICE=auto
            AUDIO_INPUT_DEVICE=EPOS
            AUDIO_DEVICE_NAME=EPOS IMPACT 30
        """

        # Поддържа избор чрез:
        # - числов индекс;
        # - част от името;
        # - общото AUDIO_DEVICE_NAME;
        # - автоматичен избор;
        # - системното устройство по подразбиране.

        # Чете настройката за конкретния тип устройство.
        raw = os.getenv(
            env_key,
            "auto",
        ).strip()

        # Незадължително общо име,
        # приложимо за входа и изхода.
        wanted_name = os.getenv(
            "AUDIO_DEVICE_NAME",
            "",
        ).strip().lower()

        # Зарежда всички налични PortAudio устройства.
        devices = sd.query_devices()

        def has_channels(
            device: dict[str, Any],
        ) -> bool:
            """
            Проверява дали устройството има канали
            за желания входен или изходен режим.
            """

            if kind == "input":
                return (
                    int(
                        device.get(
                            "max_input_channels",
                            0,
                        )
                    )
                    > 0
                )

            return (
                int(
                    device.get(
                        "max_output_channels",
                        0,
                    )
                )
                > 0
            )

        # Проверява изрично зададено устройство.
        if raw and raw.lower() not in {
            "auto",
            "default",
        }:
            # При числова стойност я приема
            # като индекс на PortAudio устройство.
            if raw.isdigit():
                index = int(raw)

                print(
                    f"Selected {kind} device by index "
                    f"{env_key}: [{index}]"
                )

                return index

            # При текстова стойност търси
            # частично съвпадение в името.
            raw_lower = raw.lower()

            for index, device in enumerate(devices):
                name = str(
                    device.get("name", "")
                ).lower()

                if (
                    raw_lower in name
                    and has_channels(device)
                ):
                    print(
                        f"Selected {kind} device by {env_key}: "
                        f"[{index}] {device['name']}"
                    )

                    return index

            print(
                f"WARNING: Could not find {kind} "
                f"device matching {raw!r}"
            )

        # Проверява общото име на устройството.
        if wanted_name:
            for index, device in enumerate(devices):
                name = str(
                    device.get("name", "")
                ).lower()

                if (
                    wanted_name in name
                    and has_channels(device)
                ):
                    print(
                        f"Selected {kind} device by "
                        "AUDIO_DEVICE_NAME: "
                        f"[{index}] {device['name']}"
                    )

                    return index

            print(
                "WARNING: Could not find "
                f"{kind} device matching "
                f"AUDIO_DEVICE_NAME={wanted_name!r}"
            )

        # Предпочитани думи в имената
        # на USB слушалки и конферентни устройства.
        preferred_tokens = [
            "epos",
            "impact",
            "emeet",
            "usb",
            "headset",
            "speakerphone",
            "audio",
        ]

        # Автоматично избира първото подходящо
        # устройство с предпочитан маркер в името.
        for index, device in enumerate(devices):
            name = str(
                device.get("name", "")
            ).lower()

            if not has_channels(device):
                continue

            if any(
                token in name
                for token in preferred_tokens
            ):
                print(
                    f"Selected {kind} USB/headset "
                    "device automatically: "
                    f"[{index}] {device['name']}"
                )

                return index

        # None указва използване
        # на системното устройство по подразбиране.
        print(
            f"Using default {kind} device"
        )

        return None

    def resolve_hardware_sample_rate(
        self,
        input_device: int | None,
        output_device: int | None,
    ) -> int:
        """
        Select a sample rate supported by BOTH input and output devices.

        If AUDIO_SAMPLE_RATE=auto:
            try 24000 first, then 48000, then 44100, 32000, 16000.

        For most USB headsets:
            48000 will be selected.

        OpenAI audio stays 24000 either way.
        """

        # Избира честота, поддържана едновременно
        # от микрофона и високоговорителя.
        #
        # OpenAI продължава да получава и връща 24 kHz,
        # независимо от хардуерната честота.

        requested = (
            self.requested_hardware_rate_raw.lower()
        )

        # Първо проверява изрично зададената честота.
        if requested and requested not in {
            "auto",
            "default",
        }:
            try:
                rate = int(
                    float(requested)
                )

                # Проверява поддръжката
                # от двете устройства.
                self.check_device_rate(
                    input_device=input_device,
                    output_device=output_device,
                    rate=rate,
                )

                return rate

            except Exception as exc:
                print(
                    f"WARNING: AUDIO_SAMPLE_RATE="
                    f"{requested!r} failed: {exc}"
                )

        # Основен списък с честоти
        # за автоматично тестване.
        candidates = [
            OPENAI_AUDIO_SAMPLE_RATE,
            48_000,
            44_100,
            32_000,
            16_000,
        ]

        # Добавя честотите по подразбиране
        # на избраните устройства в началото на списъка.
        for device_id, kind in [
            (input_device, "input"),
            (output_device, "output"),
        ]:
            try:
                info = sd.query_devices(
                    device_id,
                    kind,
                )

                default_rate = int(
                    round(
                        float(
                            info["default_samplerate"]
                        )
                    )
                )

                if default_rate not in candidates:
                    candidates.insert(
                        0,
                        default_rate,
                    )

            except Exception:
                # Липсващата информация не прекратява
                # автоматичния избор.
                pass

        # Последната грешка се пази
        # за диагностичното съобщение.
        last_error = None

        # Проверява кандидатите последователно.
        for rate in candidates:
            try:
                self.check_device_rate(
                    input_device=input_device,
                    output_device=output_device,
                    rate=rate,
                )

                # При различна хардуерна честота
                # уведомява за автоматичното преобразуване.
                if rate != OPENAI_AUDIO_SAMPLE_RATE:
                    print(
                        f"Hardware audio will run at {rate} Hz. "
                        f"OpenAI audio stays "
                        f"{OPENAI_AUDIO_SAMPLE_RATE} Hz. "
                        "Runtime will resample automatically."
                    )

                else:
                    print(
                        "Hardware and OpenAI audio "
                        f"both use {rate} Hz."
                    )

                return int(rate)

            except Exception as exc:
                last_error = exc

        # Няма честота, която се поддържа
        # едновременно от входа и изхода.
        raise RuntimeError(
            "Could not find sample rate supported by both input/output "
            f"devices. Last error: {last_error}"
        )

    def check_device_rate(
        self,
        input_device: int | None,
        output_device: int | None,
        rate: int,
    ) -> None:
        """
        Проверява дали входното и изходното устройство
        поддържат зададената PCM16 честота.
        """

        # Проверява входната конфигурация.
        sd.check_input_settings(
            device=input_device,
            channels=CHANNELS,
            dtype="int16",
            samplerate=rate,
        )

        # Проверява изходната конфигурация.
        sd.check_output_settings(
            device=output_device,
            channels=CHANNELS,
            dtype="int16",
            samplerate=rate,
        )

    def validate_audio_devices(self) -> None:
        """
        Извежда информация за избраните устройства
        и проверява окончателната аудиоконфигурация.
        """

        # Данни за микрофона.
        input_info = sd.query_devices(
            self.input_device,
            "input",
        )

        # Данни за високоговорителя.
        output_info = sd.query_devices(
            self.output_device,
            "output",
        )

        print("Selected input device:")
        print(input_info)
        print()

        print("Selected output device:")
        print(output_info)
        print()

        # Последна проверка преди създаване
        # на PortAudio потоците.
        self.check_device_rate(
            input_device=self.input_device,
            output_device=self.output_device,
            rate=self.hardware_sample_rate,
        )

    # =========================================================================
    # Споделено аудиосъстояние
    # =========================================================================

    def is_assistant_playing(self) -> bool:
        """
        Проверява дали в момента се възпроизвежда
        аудио от асистента.
        """

        with self.state_lock:
            return self.assistant_output_active

    def mark_assistant_audio_started(self) -> None:
        """
        Отбелязва началото на нов аудиоотговор.
        """

        with self.state_lock:
            self.assistant_output_active = True
            self.assistant_generation_finished = False

    def mark_assistant_generation_finished(self) -> None:
        """
        Отбелязва, че моделът е приключил генерирането,
        но в локалния буфер може още да има звук.
        """

        with self.state_lock:
            self.assistant_generation_finished = True

            # Асистентът се счита за напълно приключил
            # само когато и буферът е празен.
            with self.output_lock:
                if not self.output_buffer:
                    self.assistant_output_active = False

    def update_playback_finished_if_needed(self) -> None:
        """
        Проверява дали генерирането и локалното
        възпроизвеждане са приключили напълно.
        """

        with self.state_lock:
            if not self.assistant_generation_finished:
                return

            with self.output_lock:
                if not self.output_buffer:
                    self.assistant_output_active = False

    # =========================================================================
    # PortAudio callback функции
    # =========================================================================

    def microphone_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Получава нов аудиоблок от микрофона,
        преобразува го до 24 kHz PCM16 и го поставя
        в опашката за изпращане към OpenAI.
        """

        # Тези параметри са задължителни
        # за callback интерфейса, но не се използват.
        del frames
        del time_info

        # Брои входните препълвания на PortAudio.
        if status.input_overflow:
            self.input_overflows += 1

        # По конфигурация може да игнорира микрофона,
        # докато асистентът говори.
        if (
            self.mute_input_while_assistant_speaks
            and self.is_assistant_playing()
        ):
            self.muted_input_chunks += 1
            return

        # Извлича първия и единствен аудиоканал.
        hardware_samples = np.asarray(
            indata[:, 0],
            dtype=np.int16,
        )

        # Преобразува хардуерния звук
        # до изискваните от OpenAI 24 kHz.
        openai_samples = resample_int16_linear(
            samples=hardware_samples,
            src_rate=self.hardware_sample_rate,
            dst_rate=OPENAI_AUDIO_SAMPLE_RATE,
        )

        # Брои реално преобразуваните блокове.
        if (
            self.hardware_sample_rate
            != OPENAI_AUDIO_SAMPLE_RATE
        ):
            self.mic_resampled_chunks += 1

        # Преобразува NumPy масива до PCM bytes.
        pcm = openai_samples.tobytes()

        try:
            # Добавя блока без блокиращо изчакване.
            self.input_queue.put_nowait(pcm)
            return

        except queue.Full:
            # Опашката е пълна и трябва
            # да се освободи най-старият блок.
            self.input_drops += 1

        try:
            # Премахва най-стария аудиоблок,
            # за да запази по-актуалния звук.
            self.input_queue.get_nowait()

        except queue.Empty:
            pass

        try:
            # Опитва повторно да добави новия блок.
            self.input_queue.put_nowait(pcm)

        except queue.Full:
            self.input_drops += 1

    def speaker_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Извлича звук от изходния буфер
        и го подава към високоговорителя.
        """

        del time_info

        # Брои случаите, когато PortAudio
        # не е получил достатъчно звук навреме.
        if status.output_underflow:
            self.output_underflows += 1

        # Брои изходните препълвания.
        if status.output_overflow:
            self.output_overflows += 1

        # Изчислява броя байтове,
        # необходим за текущия callback блок.
        required_bytes = (
            frames
            * CHANNELS
            * SAMPLE_WIDTH_BYTES
        )

        # Извлича наличния звук
        # под защита на заключването.
        with self.output_lock:
            available_bytes = min(
                required_bytes,
                len(self.output_buffer),
            )

            audio = bytes(
                self.output_buffer[:available_bytes]
            )

            # Премахва вече използваните байтове.
            del self.output_buffer[:available_bytes]

        # При недостатъчно звук допълва блока
        # с тишина.
        if available_bytes < required_bytes:
            audio += b"\x00" * (
                required_bytes - available_bytes
            )

        # Преобразува PCM bytes до int16 семпли.
        samples = np.frombuffer(
            audio,
            dtype=np.int16,
        )

        # Първоначално запълва изхода с тишина.
        outdata.fill(0)

        # Избира броя семпли,
        # които могат да бъдат копирани.
        sample_count = min(
            frames,
            samples.size,
        )

        if sample_count:
            outdata[
                :sample_count,
                0,
            ] = samples[:sample_count]

        # Проверява дали възпроизвеждането
        # на текущия отговор е приключило.
        self.update_playback_finished_if_needed()

    # =========================================================================
    # Изпращане на микрофонния звук
    # =========================================================================

    async def send_microphone_audio(
        self,
        session,
    ) -> None:
        """
        Изпраща аудиоблоковете от микрофонната опашка
        към активната OpenAI Realtime сесия.
        """

        while self.running:
            # queue.Queue е блокираща, затова операцията
            # се изпълнява в помощна нишка.
            pcm = await asyncio.to_thread(
                self.input_queue.get
            )

            # Прекратява след спиране на средата.
            if not self.running:
                return

            # Празният блок се използва
            # като сигнал при спиране.
            if not pcm:
                continue

            try:
                # Към OpenAI винаги се изпраща
                # 24 kHz PCM16 моно звук.
                await session.send_audio(pcm)

                self.sent_audio_chunks += 1

            except asyncio.CancelledError:
                # Позволява коректното прекратяване
                # на асинхронната задача.
                raise

            except Exception as exc:
                # При контролирано спиране
                # не извежда излишна грешка.
                if self.shutting_down:
                    return

                print(
                    f"\n[send audio error] {exc}",
                    file=sys.stderr,
                )

                # Кратко изчакване преди следващия опит.
                await asyncio.sleep(0.1)

    # =========================================================================
    # Обработване на Realtime събития
    # =========================================================================

    async def receive_events(
        self,
        session,
    ) -> None:
        """
        Получава и маршрутизира събитията
        от OpenAI Realtime сесията.
        """

        async for event in session:
            # Получава типа на събитието.
            event_type = getattr(
                event,
                "type",
                None,
            )

            # Нов аудиофрагмент от асистента.
            if event_type == "audio":
                self.handle_audio_event(event)

            # Моделът е приключил генерирането
            # на аудио за текущия отговор.
            elif event_type == "audio_end":
                self.mark_assistant_generation_finished()

                print(
                    "\n[assistant finished generating audio]"
                )

            # Аудиоотговорът е прекъснат.
            elif event_type == "audio_interrupted":
                # Премахва останалия звук,
                # за да не бъде възпроизведен след прекъсването.
                self.clear_output_audio()

                with self.state_lock:
                    self.assistant_generation_finished = True
                    self.assistant_output_active = False

                print(
                    "\n[assistant audio interrupted]"
                )

            # Нов елемент е добавен
            # към историята на разговора.
            elif event_type == "history_added":
                item = getattr(
                    event,
                    "item",
                    None,
                )

                self.print_history_item(
                    item=item,
                    source_event="history_added",
                )

            # Историята на разговора е обновена.
            elif event_type == "history_updated":
                history = getattr(
                    event,
                    "history",
                    None,
                )

                if history is None:
                    history = []

                for item in history:
                    self.print_history_item(
                        item=item,
                        source_event="history_updated",
                    )

            # Започнало е изпълнение на агент.
            elif event_type == "agent_start":
                agent = getattr(
                    event,
                    "agent",
                    None,
                )

                print(
                    "\n[agent started: "
                    f"{getattr(agent, 'name', 'unknown')}]"
                )

            # Приключило е изпълнение на агент.
            elif event_type == "agent_end":
                agent = getattr(
                    event,
                    "agent",
                    None,
                )

                print(
                    "\n[agent ended: "
                    f"{getattr(agent, 'name', 'unknown')}]"
                )

            # Засечено е начало на потребителска реч.
            elif event_type in {
                "speech_started",
                "input_audio_buffer.speech_started",
                "input_audio_buffer_speech_started",
            }:
                self.interrupt_assistant_output(
                    reason="user_speech_started"
                )

            # Започнало е изпълнение на инструмент.
            elif event_type == "tool_start":
                self.active_tool_count += 1

                tool = getattr(
                    event,
                    "tool",
                    None,
                )

                print(
                    "\n[tool started: "
                    f"{getattr(tool, 'name', str(tool))}]"
                )

            # Инструментът е приключил.
            elif event_type == "tool_end":
                self.active_tool_count = max(
                    0,
                    self.active_tool_count - 1,
                )

                tool = getattr(
                    event,
                    "tool",
                    None,
                )

                print(
                    "\n[tool ended: "
                    f"{getattr(tool, 'name', str(tool))}]"
                )

            # Изпълнен е преход между агенти.
            elif event_type == "handoff":
                from_agent = getattr(
                    getattr(
                        event,
                        "from_agent",
                        None,
                    ),
                    "name",
                    "unknown",
                )

                to_agent = getattr(
                    getattr(
                        event,
                        "to_agent",
                        None,
                    ),
                    "name",
                    "unknown",
                )

                print(
                    f"\n[handoff: {from_agent} -> {to_agent}]"
                )

            # Активирано е защитно ограничение.
            elif event_type == "guardrail_tripped":
                print(
                    "\n[guardrail triggered]",
                    file=sys.stderr,
                )

            # Получена е грешка от Realtime сесията.
            elif event_type == "error":
                error = getattr(
                    event,
                    "error",
                    event,
                )

                print(
                    f"\n[realtime error] {error}",
                    file=sys.stderr,
                )

    def handle_audio_event(
        self,
        event,
    ) -> None:
        """
        Обработва аудиофрагмент от OpenAI,
        преобразува го до хардуерната честота
        и го добавя в буфера за възпроизвеждане.
        """

        # Получава вложения аудиообект.
        audio_event = getattr(
            event,
            "audio",
            None,
        )

        # Получава PCM16 съдържанието.
        audio_data = getattr(
            audio_event,
            "data",
            None,
        )

        if not audio_data:
            return

        # Гарантира bytes представяне.
        if not isinstance(
            audio_data,
            (
                bytes,
                bytearray,
            ),
        ):
            audio_data = bytes(audio_data)

        # Интерпретира входа като PCM16.
        openai_samples = np.frombuffer(
            audio_data,
            dtype=np.int16,
        )

        # Преобразува от OpenAI 24 kHz
        # до честотата на високоговорителя.
        hardware_samples = resample_int16_linear(
            samples=openai_samples,
            src_rate=OPENAI_AUDIO_SAMPLE_RATE,
            dst_rate=self.hardware_sample_rate,
        )

        if (
            self.hardware_sample_rate
            != OPENAI_AUDIO_SAMPLE_RATE
        ):
            self.speaker_resampled_chunks += 1

        # Отбелязва, че има активен аудиоотговор.
        self.mark_assistant_audio_started()

        # Добавя преобразувания звук
        # към нишково защитения изходен буфер.
        with self.output_lock:
            self.output_buffer.extend(
                hardware_samples.tobytes()
            )

        self.received_audio_chunks += 1

    # =========================================================================
    # Директно маршрутизиране на команди към робота
    # =========================================================================

    def route_direct_user_command(
        self,
        item_id: str,
        full_text: str,
    ) -> None:
        """
        Safety net.

        Even if the LLM tool call does not happen, critical commands like:
        - stop
        - go home
        - point 0
        can still reach Redis.
        """

        # Методът представлява защитен резервен механизъм.
        #
        # Дори когато езиковият модел не извика инструмент,
        # критични команди като stop или go home могат
        # да бъдат изпратени директно към Redis.

        # Не маршрутизира повторно
        # вече обработена потребителска реплика.
        if item_id in self.directly_routed_user_items:
            return

        # Нормализира текста.
        text = full_text.lower().strip()

        if not text:
            return

        # Български и английски варианти
        # на командата за незабавно спиране.
        stop_patterns = [
            r"\bstop\b",
            r"\bstop now\b",
            r"\bemergency stop\b",
            r"\bспри\b",
            r"\bспирай\b",
            r"\bстой\b",
            r"\bстоп\b",
            r"моля те спри",
        ]

        # Български и английски варианти
        # на командата за връщане до начална позиция.
        go_home_patterns = [
            r"\bhome\b",
            r"\bgo home\b",
            r"\bpoint zero\b",
            r"\bpoint 0\b",
            r"\bpoint_0\b",
            r"\bstart position\b",
            r"\bначална позиция\b",
            r"\bнулева точка\b",
            r"\bточка 0\b",
            r"\bточка нула\b",
            r"\bприбери\b",
            r"\bвърни се\b",
        ]

        # Проверява за команда stop.
        if any(
            re.search(pattern, text)
            for pattern in stop_patterns
        ):
            # Създава събитие за незабавно спиране.
            payload = {
                "action": "stop",
                "source": "realtime_direct_router",
                "text": full_text,
                "created_at": time.time(),
            }

            # Публикува командата към потока
            # за директни команди за движение.
            event_id = xadd_json(
                "stream:motion_commands",
                payload,
            )

            # Обновява общото състояние на робота.
            store.client.set(
                "robot:state",
                json.dumps(
                    {
                        "state": "STOP_REQUESTED",
                        "source": "realtime_direct_router",
                        "text": full_text,
                        "event_id": event_id,
                        "updated_at": time.time(),
                    },
                    ensure_ascii=False,
                ),
            )

            # Маркира репликата като обработена.
            self.directly_routed_user_items.add(
                item_id
            )

            # Извежда диагностична информация.
            print()
            print(
                "REALTIME DIRECT ROUTER SENT ROBOT EVENT"
            )
            print(
                "Redis stream: stream:motion_commands"
            )
            print(
                f"event_id: {event_id}"
            )
            print(payload)
            print()

            return

        # Проверява за команда за начална позиция.
        if any(
            re.search(pattern, text)
            for pattern in go_home_patterns
        ):
            # Създава заявка за движение до point_0.
            payload = {
                "action": "go_to_point",
                "point": "point_0",
                "duration": 2.0,
                "source": "realtime_direct_router",
                "text": full_text,
                "created_at": time.time(),
            }

            # Публикува заявката в потока
            # за движения до точки.
            event_id = xadd_json(
                "stream:motion_requests",
                payload,
            )

            # Обновява общото състояние на робота.
            store.client.set(
                "robot:state",
                json.dumps(
                    {
                        "state": "POINT_0_REQUESTED",
                        "source": "realtime_direct_router",
                        "text": full_text,
                        "event_id": event_id,
                        "updated_at": time.time(),
                    },
                    ensure_ascii=False,
                ),
            )

            # Маркира репликата като обработена.
            self.directly_routed_user_items.add(
                item_id
            )

            print()
            print(
                "REALTIME DIRECT ROUTER SENT ROBOT EVENT"
            )
            print(
                "Redis stream: stream:motion_requests"
            )
            print(
                f"event_id: {event_id}"
            )
            print(payload)
            print()

            return

    def interrupt_assistant_output(
        self,
        reason: str,
    ) -> None:
        """
        Прекъсва текущия аудиоотговор на асистента.

        При активирана настройка изпраща и команда stop
        към планиращия агент, когато има активен отговор
        или изпълняващ се инструмент.
        """

        # При изключено barge-in поведение
        # не извършва прекъсване.
        if not self.barge_in_enabled:
            return

        # Проверява дали има активен аудиоотговор
        # или активен инструмент.
        was_active = (
            self.is_assistant_playing()
            or self.active_tool_count > 0
        )

        # Премахва неизсвирения аудиобуфер.
        self.clear_output_audio()

        # Обновява състоянието на асистента.
        with self.state_lock:
            self.assistant_output_active = False
            self.assistant_generation_finished = True

        # По конфигурация прекъсва и движението
        # на манипулатора.
        if (
            was_active
            and self.cancel_motion_on_barge_in
        ):
            try:
                xadd_json(
                    self.planning_request_stream,
                    {
                        "request_id": (
                            f"barge-in-stop-{time.time_ns()}"
                        ),
                        "action": "stop",
                        "source": "realtime_barge_in",
                        "reason": reason,
                        "created_at": time.time(),
                    },
                )

            except Exception as exc:
                print(
                    "[barge-in motion cancel failed] "
                    f"{exc}",
                    file=sys.stderr,
                )

    def handle_user_barge_in(
        self,
        item_id: str,
    ) -> None:
        """
        Обработва нова потребителска реплика
        като заявка за прекъсване.
        """

        # Не обработва една и съща реплика повторно.
        if item_id in self.barge_in_user_items:
            return

        # Запомня репликата.
        self.barge_in_user_items.add(item_id)

        # Прекъсва текущия отговор.
        self.interrupt_assistant_output(
            reason="new_user_utterance"
        )

    # =========================================================================
    # Извличане и отпечатване на транскрипцията
    # =========================================================================

    @staticmethod
    def extract_text_from_item(
        item: Any,
    ) -> list[str]:
        """
        Извлича текстовите и транскрибирани части
        от елемент на Realtime историята.
        """

        # Списък с откритите текстови части.
        texts: list[str] = []

        if item is None:
            return texts

        # Получава съдържанието на елемента.
        content = getattr(
            item,
            "content",
            None,
        )

        if not content:
            return texts

        # Обхожда всички части на съобщението.
        for part in content:
            # Транскрипция на аудиочаст.
            transcript = getattr(
                part,
                "transcript",
                None,
            )

            # Директна текстова част.
            text = getattr(
                part,
                "text",
                None,
            )

            if transcript:
                cleaned = str(
                    transcript
                ).strip()

                if cleaned:
                    texts.append(cleaned)

            elif text:
                cleaned = str(
                    text
                ).strip()

                if cleaned:
                    texts.append(cleaned)

        return texts

    def print_history_item(
        self,
        item: Any,
        source_event: str,
    ) -> None:
        """
        Отпечатва нов или обновен елемент
        от историята на Realtime разговора.
        """

        if item is None:
            return

        # Роля на елемента: user, assistant или друга.
        role = getattr(
            item,
            "role",
            "unknown",
        )

        # Текущо състояние на елемента.
        status = getattr(
            item,
            "status",
            "unknown",
        )

        # Определя устойчив идентификатор.
        item_id = (
            getattr(item, "item_id", None)
            or getattr(item, "id", None)
            or f"python-{id(item)}"
        )

        # Извлича текстовото съдържание.
        texts = self.extract_text_from_item(
            item
        )

        if not texts:
            return

        # Обединява отделните части.
        full_text = " ".join(
            texts
        ).strip()

        if not full_text:
            return

        # Проверява последно отпечатаната версия
        # на същия елемент.
        previous_text = self.last_printed_text.get(
            item_id
        )

        # Не отпечатва повторно идентично съдържание.
        if previous_text == full_text:
            return

        # Запомня новата версия.
        self.last_printed_text[
            item_id
        ] = full_text

        print()
        print("=" * 76)

        if role == "user":
            print("MODEL HEARD:")

            # Новата потребителска реплика
            # може да прекъсне текущия отговор.
            self.handle_user_barge_in(
                item_id
            )

            # Проверява репликата за критична
            # директна команда към робота.
            self.route_direct_user_command(
                item_id=item_id,
                full_text=full_text,
            )

        elif role == "assistant":
            print("MODEL INTENDED TO SAY:")

        else:
            print(
                f"HISTORY ITEM — {role}:"
            )

        # Отпечатва самия текст.
        print(full_text)

        # Добавя диагностични метаданни.
        print(
            f"[event={source_event}, "
            f"status={status}, "
            f"id={item_id}]"
        )

        print("=" * 76)

    # =========================================================================
    # Диагностика
    # =========================================================================

    async def print_audio_stats(self) -> None:
        """
        Периодично извежда диагностична информация
        за аудиопотоците и вътрешните буфери.
        """

        while self.running:
            # Статистиката се извежда
            # веднъж на пет секунди.
            await asyncio.sleep(5.0)

            # Получава размера на изходния буфер.
            with self.output_lock:
                output_bytes = len(
                    self.output_buffer
                )

            # Преобразува размера на буфера
            # до оставащи секунди звук.
            output_seconds = (
                output_bytes
                / (
                    self.hardware_sample_rate
                    * CHANNELS
                    * SAMPLE_WIDTH_BYTES
                )
            )

            print(
                "\n[audio stats]"
                f" input_queue={self.input_queue.qsize()}"
                f" input_overflows={self.input_overflows}"
                f" input_drops={self.input_drops}"
                f" input_muted={self.muted_input_chunks}"
                f" output_buffer={output_seconds:.2f}s"
                f" output_underflows={self.output_underflows}"
                f" portaudio_output_overflows={self.output_overflows}"
                f" sent_chunks={self.sent_audio_chunks}"
                f" received_chunks={self.received_audio_chunks}"
                f" mic_resampled={self.mic_resampled_chunks}"
                f" speaker_resampled={self.speaker_resampled_chunks}"
                f" assistant_playing={self.is_assistant_playing()}"
            )

    # =========================================================================
    # Жизнен цикъл
    # =========================================================================

    def clear_output_audio(self) -> None:
        """
        Изчиства неизсвиреното аудио
        от асистента.
        """

        with self.output_lock:
            self.output_buffer.clear()

    def stop(self) -> None:
        """
        Започва контролирано спиране
        на Realtime средата.
        """

        # Не изпълнява спирането повторно.
        if self.shutting_down:
            return

        self.shutting_down = True
        self.running = False

        # Премахва чакащия изходен звук.
        self.clear_output_audio()

        try:
            # Празният блок освобождава
            # чакащата задача send_microphone_audio.
            self.input_queue.put_nowait(b"")

        except queue.Full:
            # При пълна опашка флагът running
            # отново ще прекрати задачата.
            pass

    async def run(self) -> None:
        """
        Стартира аудиоустройствата,
        OpenAI Realtime сесията и асинхронните задачи.
        """

        # Проверява устройствата
        # преди създаване на потоците.
        self.validate_audio_devices()

        # Създава Realtime сесията.
        session = await self.runner.run()

        # Създава входния PortAudio поток.
        input_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.hardware_sample_rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self.hardware_block_size,
            latency="high",
            callback=self.microphone_callback,
        )

        # Създава изходния PortAudio поток.
        output_stream = sd.OutputStream(
            device=self.output_device,
            samplerate=self.hardware_sample_rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self.hardware_block_size,
            latency="high",
            callback=self.speaker_callback,
        )

        # Извежда окончателната конфигурация.
        print()
        print(
            "Bulgarian Realtime robotic-arm assistant"
        )
        print(
            "-----------------------------------------"
        )
        print(
            f"Model: {self.model}"
        )
        print(
            f"Voice: {self.voice}"
        )
        print(
            f"Speech speed: {self.speech_speed}"
        )
        print(
            f"Input device: {self.input_device}"
        )
        print(
            f"Output device: {self.output_device}"
        )
        print(
            "Hardware audio: "
            f"{self.hardware_sample_rate} Hz PCM16 mono"
        )
        print(
            "OpenAI audio: "
            f"{OPENAI_AUDIO_SAMPLE_RATE} Hz PCM16 mono"
        )
        print(
            f"Block: {BLOCK_DURATION_MS} ms"
        )
        print(
            "Hardware block size: "
            f"{self.hardware_block_size}"
        )
        print(
            "OpenAI block size: "
            f"{self.openai_block_size}"
        )
        print(
            "Microphone muted during assistant playback: "
            f"{self.mute_input_while_assistant_speaks}"
        )
        print(
            "Press Ctrl+C to stop."
        )
        print()

        try:
            # Realtime сесията се управлява
            # чрез асинхронен контекст.
            async with session:
                # PortAudio потоците се стартират
                # чрез синхронен контекст.
                with input_stream, output_stream:
                    print(
                        "Actual input rate:",
                        input_stream.samplerate,
                    )

                    print(
                        "Actual output rate:",
                        output_stream.samplerate,
                    )

                    # Стартира паралелните задачи за:
                    # - изпращане на микрофонен звук;
                    # - получаване на Realtime събития;
                    # - периодична диагностика.
                    tasks = [
                        asyncio.create_task(
                            self.send_microphone_audio(
                                session
                            )
                        ),
                        asyncio.create_task(
                            self.receive_events(
                                session
                            )
                        ),
                        asyncio.create_task(
                            self.print_audio_stats()
                        ),
                    ]

                    # Изчаква първата задача,
                    # която приключи с изключение.
                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=(
                            asyncio.FIRST_EXCEPTION
                        ),
                    )

                    # Препредава откритото изключение
                    # към основния жизнен цикъл.
                    for task in done:
                        exception = task.exception()

                        if exception is not None:
                            raise exception

                    # Отменя останалите задачи.
                    for task in pending:
                        task.cancel()

                    # Изчаква контролираното им приключване.
                    await asyncio.gather(
                        *pending,
                        return_exceptions=True,
                    )

        finally:
            # Гарантира спиране на средата
            # при нормален край или грешка.
            self.stop()


# ============================================================================
# Входна точка
# ============================================================================

async def main_async() -> None:
    """
    Създава и стартира Realtime аудиосредата.
    """

    runtime = RealtimeAudioRuntime()

    try:
        await runtime.run()

    finally:
        runtime.stop()


def main() -> None:
    """
    Синхронна входна точка на програмата.
    """

    try:
        # Стартира асинхронния жизнен цикъл.
        asyncio.run(main_async())

    except KeyboardInterrupt:
        # Обработва Ctrl+C без излишен traceback.
        print(
            "\nRealtime assistant stopped."
        )


# Стартира main() само когато файлът
# се изпълнява директно.
if __name__ == "__main__":
    main()