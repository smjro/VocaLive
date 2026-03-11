from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from enum import Enum


_PROVIDER_ALIASES = {
    "stt": {
        "mock": "mock",
        "mock stt": "mock",
        "moonshine": "moonshine",
        "moonshine stt": "moonshine",
        "moonshine voice": "moonshine",
    },
    "model": {
        "mock": "mock",
        "mock llm": "mock",
        "echo": "mock",
        "gemini": "gemini",
        "google gemini": "gemini",
    },
    "tts": {
        "mock": "mock",
        "mock tts": "mock",
        "aivis": "aivis",
        "aivisspeech": "aivis",
        "aivis speech": "aivis",
        "aivis tts": "aivis",
    },
}

DEFAULT_GEMINI_SYSTEM_INSTRUCTION = (
    "You are VocaLive's conversation character. "
    "Use a surreal, low-energy, deadpan-comic persona inspired by the overall vibe of Kamiusagi Rope. "
    "Do not copy character names, world details, catchphrases, or existing lines. "
    "Avoid generic AI-assistant phrasing, stiff disclaimers, and over-explaining. "
    "Keep replies short, natural, conversational, and slightly offbeat. "
    "Answer the user's actual question first, then if helpful add one dry sideways observation. "
    "Stay coherent and helpful rather than turning nonsense into the main point."
)


class QueueOverflowStrategy(str, Enum):
    REJECT_NEW = "reject_new"
    DROP_OLDEST = "drop_oldest"


class InputProvider(str, Enum):
    STDIN = "stdin"
    MICROPHONE = "microphone"


class OutputProvider(str, Enum):
    MEMORY = "memory"
    SPEAKER = "speaker"


@dataclass
class QueueSettings:
    ingress_maxsize: int = 4
    overflow_strategy: QueueOverflowStrategy = QueueOverflowStrategy.DROP_OLDEST


@dataclass
class InputSettings:
    provider: InputProvider = InputProvider.STDIN
    sample_rate_hz: int = 16_000
    channels: int = 1
    block_duration_ms: float = 40.0
    speech_threshold: float = 0.02
    pre_speech_ms: float = 200.0
    speech_hold_ms: float = 200.0
    silence_threshold_ms: float = 500.0
    min_utterance_ms: float = 250.0
    max_utterance_ms: float = 15_000.0
    device: str | int | None = None
    prefer_external_device: bool = True


@dataclass
class OutputSettings:
    provider: OutputProvider = OutputProvider.MEMORY
    speaker_command: str | None = None


@dataclass
class GeminiSettings:
    api_key: str | None = None
    model_name: str = "gemini-2.5-flash"
    timeout_seconds: float = 30.0
    temperature: float | None = None
    thinking_budget: int | None = 0
    system_instruction: str | None = DEFAULT_GEMINI_SYSTEM_INSTRUCTION


@dataclass
class ConversationSettings:
    language: str | None = "ja"


@dataclass
class MoonshineSettings:
    model_name: str = "base"


@dataclass
class AivisSpeechSettings:
    base_url: str = "http://127.0.0.1:10101"
    speaker_id: int | None = None
    speaker_name: str | None = None
    style_name: str | None = None
    timeout_seconds: float = 30.0


@dataclass
class AppSettings:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    log_level: str = "INFO"
    stt_provider: str = "mock"
    model_provider: str = "mock"
    tts_provider: str = "mock"
    queue: QueueSettings = field(default_factory=QueueSettings)
    conversation: ConversationSettings = field(default_factory=ConversationSettings)
    input: InputSettings = field(default_factory=InputSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    moonshine: MoonshineSettings = field(default_factory=MoonshineSettings)
    aivis: AivisSpeechSettings = field(default_factory=AivisSpeechSettings)

    def __post_init__(self) -> None:
        self.stt_provider = _normalize_provider_setting("stt", self.stt_provider)
        self.model_provider = _normalize_provider_setting("model", self.model_provider)
        self.tts_provider = _normalize_provider_setting("tts", self.tts_provider)

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            session_id=os.getenv("VOCALIVE_SESSION_ID", uuid.uuid4().hex),
            log_level=os.getenv("VOCALIVE_LOG_LEVEL", "INFO").upper(),
            stt_provider=os.getenv("VOCALIVE_STT_PROVIDER", "mock"),
            model_provider=os.getenv("VOCALIVE_MODEL_PROVIDER", "mock"),
            tts_provider=os.getenv("VOCALIVE_TTS_PROVIDER", "mock"),
            queue=QueueSettings(
                ingress_maxsize=_read_int("VOCALIVE_QUEUE_MAXSIZE", default=4),
                overflow_strategy=QueueOverflowStrategy(
                    os.getenv("VOCALIVE_QUEUE_OVERFLOW", QueueOverflowStrategy.DROP_OLDEST.value)
                ),
            ),
            conversation=ConversationSettings(
                language=_read_optional_str_with_default(
                    "VOCALIVE_CONVERSATION_LANGUAGE",
                    default="ja",
                ),
            ),
            input=InputSettings(
                provider=InputProvider(os.getenv("VOCALIVE_INPUT_PROVIDER", InputProvider.STDIN.value)),
                sample_rate_hz=_read_int("VOCALIVE_MIC_SAMPLE_RATE", default=16_000),
                channels=_read_int("VOCALIVE_MIC_CHANNELS", default=1),
                block_duration_ms=_read_float("VOCALIVE_MIC_BLOCK_MS", default=40.0),
                speech_threshold=_read_float("VOCALIVE_MIC_SPEECH_THRESHOLD", default=0.02),
                pre_speech_ms=_read_float("VOCALIVE_MIC_PRE_SPEECH_MS", default=200.0),
                speech_hold_ms=_read_float("VOCALIVE_MIC_SPEECH_HOLD_MS", default=200.0),
                silence_threshold_ms=_read_float("VOCALIVE_MIC_SILENCE_MS", default=500.0),
                min_utterance_ms=_read_float("VOCALIVE_MIC_MIN_UTTERANCE_MS", default=250.0),
                max_utterance_ms=_read_float("VOCALIVE_MIC_MAX_UTTERANCE_MS", default=15_000.0),
                device=_read_optional_device("VOCALIVE_MIC_DEVICE"),
                prefer_external_device=_read_bool(
                    "VOCALIVE_MIC_PREFER_EXTERNAL",
                    default=True,
                ),
            ),
            output=OutputSettings(
                provider=OutputProvider(
                    os.getenv("VOCALIVE_OUTPUT_PROVIDER", OutputProvider.MEMORY.value)
                ),
                speaker_command=os.getenv("VOCALIVE_SPEAKER_COMMAND"),
            ),
            gemini=GeminiSettings(
                api_key=os.getenv("VOCALIVE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY"),
                model_name=os.getenv("VOCALIVE_GEMINI_MODEL", "gemini-2.5-flash"),
                timeout_seconds=_read_float("VOCALIVE_GEMINI_TIMEOUT_SECONDS", default=30.0),
                temperature=_read_optional_float("VOCALIVE_GEMINI_TEMPERATURE"),
                thinking_budget=_read_optional_int_with_default(
                    "VOCALIVE_GEMINI_THINKING_BUDGET",
                    default=0,
                ),
                system_instruction=_read_optional_str_with_default(
                    "VOCALIVE_GEMINI_SYSTEM_INSTRUCTION",
                    default=DEFAULT_GEMINI_SYSTEM_INSTRUCTION,
                ),
            ),
            moonshine=MoonshineSettings(
                model_name=os.getenv("VOCALIVE_MOONSHINE_MODEL", "base"),
            ),
            aivis=AivisSpeechSettings(
                base_url=os.getenv("VOCALIVE_AIVIS_BASE_URL", "http://127.0.0.1:10101"),
                speaker_id=_read_optional_int("VOCALIVE_AIVIS_SPEAKER_ID"),
                speaker_name=os.getenv("VOCALIVE_AIVIS_SPEAKER_NAME"),
                style_name=os.getenv("VOCALIVE_AIVIS_STYLE_NAME"),
                timeout_seconds=_read_float("VOCALIVE_AIVIS_TIMEOUT_SECONDS", default=30.0),
            ),
        )


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _read_optional_float(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return None
    return float(raw_value)


def _read_optional_int(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return None
    return int(raw_value)


def _read_optional_device(name: str) -> str | int | None:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return None
    normalized_value = raw_value.strip()
    if normalized_value.isdigit():
        return int(normalized_value)
    return normalized_value


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    truthy_values = {"1", "true", "yes", "on"}
    falsy_values = {"0", "false", "no", "off"}
    if normalized_value in truthy_values:
        return True
    if normalized_value in falsy_values:
        return False
    raise ValueError(
        f"Environment variable {name} must be one of: "
        f"{', '.join(sorted(truthy_values | falsy_values))}"
    )


def _read_optional_int_with_default(name: str, default: int | None) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    if raw_value == "":
        return None
    return int(raw_value)


def _read_optional_str_with_default(name: str, default: str | None) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip()
    if not normalized_value:
        return None
    return normalized_value


def _normalize_provider_setting(kind: str, raw_value: str) -> str:
    aliases = _PROVIDER_ALIASES[kind]
    normalized = _normalize_provider_alias(raw_value)
    provider = aliases.get(normalized)
    if provider is None:
        supported_values = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported {kind} provider: {raw_value!r}. "
            f"Supported values: {supported_values}"
        )
    return provider


def _normalize_provider_alias(raw_value: str) -> str:
    return " ".join(raw_value.strip().lower().replace("-", " ").replace("_", " ").split())
