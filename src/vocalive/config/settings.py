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
DEFAULT_SCREEN_TRIGGER_PHRASES = (
    "画面みて",
    "画面見て",
    "画面をみて",
    "画面を見て",
    "スクショみて",
    "スクショ見て",
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


class ApplicationAudioMode(str, Enum):
    CONTEXT_ONLY = "context_only"
    RESPOND = "respond"


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
class ApplicationAudioSettings:
    enabled: bool = False
    mode: ApplicationAudioMode = ApplicationAudioMode.CONTEXT_ONLY
    target: str | None = None
    sample_rate_hz: int = 16_000
    channels: int = 1
    block_duration_ms: float = 40.0
    speech_threshold: float = 0.02
    pre_speech_ms: float = 200.0
    speech_hold_ms: float = 320.0
    silence_threshold_ms: float = 650.0
    min_utterance_ms: float = 250.0
    max_utterance_ms: float = 15_000.0
    timeout_seconds: float = 10.0
    adaptive_vad_enabled: bool = True
    stt_enhancement_enabled: bool = True


@dataclass
class OutputSettings:
    provider: OutputProvider = OutputProvider.MEMORY
    speaker_command: str | None = None


@dataclass
class OverlaySettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    auto_open: bool = True
    title: str = "VocaLive Overlay"
    character_name: str = "Tora"


@dataclass
class ReplySettings:
    debounce_ms: float = 1000.0
    policy_enabled: bool = True
    min_gap_ms: float = 6000.0
    short_utterance_max_chars: int = 12


@dataclass
class GeminiSettings:
    api_key: str | None = None
    model_name: str = "gemini-2.5-flash"
    timeout_seconds: float = 30.0
    temperature: float | None = None
    thinking_budget: int | None = 0
    system_instruction: str | None = DEFAULT_GEMINI_SYSTEM_INSTRUCTION


@dataclass
class ScreenCaptureSettings:
    enabled: bool = False
    window_name: str | None = None
    trigger_phrases: tuple[str, ...] = DEFAULT_SCREEN_TRIGGER_PHRASES
    timeout_seconds: float = 5.0
    resize_max_edge_px: int | None = 1280


@dataclass
class ConversationSettings:
    language: str | None = "ja"


@dataclass
class ContextSettings:
    recent_message_count: int = 8
    conversation_summary_max_chars: int = 1200
    application_recent_message_count: int = 4
    application_summary_max_chars: int = 900
    application_summary_min_message_chars: int = 8


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
    context: ContextSettings = field(default_factory=ContextSettings)
    input: InputSettings = field(default_factory=InputSettings)
    application_audio: ApplicationAudioSettings = field(default_factory=ApplicationAudioSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    reply: ReplySettings = field(default_factory=ReplySettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    screen_capture: ScreenCaptureSettings = field(default_factory=ScreenCaptureSettings)
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
            context=ContextSettings(
                recent_message_count=_read_int(
                    "VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT",
                    default=8,
                ),
                conversation_summary_max_chars=_read_int(
                    "VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS",
                    default=1200,
                ),
                application_recent_message_count=_read_int(
                    "VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT",
                    default=4,
                ),
                application_summary_max_chars=_read_int(
                    "VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS",
                    default=900,
                ),
                application_summary_min_message_chars=_read_int(
                    "VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS",
                    default=8,
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
            application_audio=ApplicationAudioSettings(
                enabled=_read_bool("VOCALIVE_APP_AUDIO_ENABLED", default=False),
                mode=_read_application_audio_mode(
                    "VOCALIVE_APP_AUDIO_MODE",
                    default=ApplicationAudioMode.CONTEXT_ONLY,
                ),
                target=_read_optional_str_with_default(
                    "VOCALIVE_APP_AUDIO_TARGET",
                    default=None,
                ),
                sample_rate_hz=_read_int("VOCALIVE_APP_AUDIO_SAMPLE_RATE", default=16_000),
                channels=_read_int("VOCALIVE_APP_AUDIO_CHANNELS", default=1),
                block_duration_ms=_read_float("VOCALIVE_APP_AUDIO_BLOCK_MS", default=40.0),
                speech_threshold=_read_float("VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD", default=0.02),
                pre_speech_ms=_read_float("VOCALIVE_APP_AUDIO_PRE_SPEECH_MS", default=200.0),
                speech_hold_ms=_read_float("VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS", default=320.0),
                silence_threshold_ms=_read_float("VOCALIVE_APP_AUDIO_SILENCE_MS", default=650.0),
                min_utterance_ms=_read_float("VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS", default=250.0),
                max_utterance_ms=_read_float("VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS", default=15_000.0),
                timeout_seconds=_read_float("VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS", default=10.0),
                adaptive_vad_enabled=_read_bool(
                    "VOCALIVE_APP_AUDIO_ADAPTIVE_VAD",
                    default=True,
                ),
                stt_enhancement_enabled=_read_bool(
                    "VOCALIVE_APP_AUDIO_STT_ENHANCEMENT",
                    default=True,
                ),
            ),
            output=OutputSettings(
                provider=OutputProvider(
                    os.getenv("VOCALIVE_OUTPUT_PROVIDER", OutputProvider.MEMORY.value)
                ),
                speaker_command=os.getenv("VOCALIVE_SPEAKER_COMMAND"),
            ),
            overlay=OverlaySettings(
                enabled=_read_bool("VOCALIVE_OVERLAY_ENABLED", default=False),
                host=os.getenv("VOCALIVE_OVERLAY_HOST", "127.0.0.1"),
                port=_read_int("VOCALIVE_OVERLAY_PORT", default=8765),
                auto_open=_read_bool("VOCALIVE_OVERLAY_AUTO_OPEN", default=True),
                title=os.getenv("VOCALIVE_OVERLAY_TITLE", "VocaLive Overlay"),
                character_name=os.getenv("VOCALIVE_OVERLAY_CHARACTER_NAME", "Tora"),
            ),
            reply=ReplySettings(
                debounce_ms=_read_float("VOCALIVE_REPLY_DEBOUNCE_MS", default=1000.0),
                policy_enabled=_read_bool("VOCALIVE_REPLY_POLICY_ENABLED", default=True),
                min_gap_ms=_read_float("VOCALIVE_REPLY_MIN_GAP_MS", default=6000.0),
                short_utterance_max_chars=_read_int(
                    "VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS",
                    default=12,
                ),
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
            screen_capture=ScreenCaptureSettings(
                enabled=_read_bool("VOCALIVE_SCREEN_CAPTURE_ENABLED", default=False),
                window_name=_read_optional_str_with_default(
                    "VOCALIVE_SCREEN_WINDOW_NAME",
                    default=None,
                ),
                trigger_phrases=_read_str_tuple(
                    "VOCALIVE_SCREEN_TRIGGER_PHRASES",
                    default=DEFAULT_SCREEN_TRIGGER_PHRASES,
                ),
                timeout_seconds=_read_float("VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS", default=5.0),
                resize_max_edge_px=_read_optional_int_with_default(
                    "VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX",
                    default=1280,
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


def _read_str_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _read_application_audio_mode(
    name: str,
    default: ApplicationAudioMode,
) -> ApplicationAudioMode:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized_value = "_".join(raw_value.strip().lower().replace("-", " ").split())
    aliases = {
        "context_only": ApplicationAudioMode.CONTEXT_ONLY,
        "respond": ApplicationAudioMode.RESPOND,
    }
    mode = aliases.get(normalized_value)
    if mode is None:
        supported_values = ", ".join(mode.value for mode in ApplicationAudioMode)
        raise ValueError(
            f"Unsupported application audio mode: {raw_value!r}. "
            f"Supported values: {supported_values}"
        )
    return mode


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
