from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum


_PROVIDER_ALIASES = {
    "stt": {
        "mock": "mock",
        "mock stt": "mock",
        "moonshine": "moonshine",
        "moonshine stt": "moonshine",
        "moonshine voice": "moonshine",
        "openai": "openai",
        "openai stt": "openai",
        "openai transcription": "openai",
        "gpt 4o mini transcribe": "openai",
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
    "You are VocaLive's conversation character, and your name is コハク. "
    "Use an understated, low-energy, deadpan-comic conversation style inspired by the overall vibe of Kamiusagi Rope. "
    "Do not copy character names, world details, catchphrases, running bits, or existing lines. "
    "Speak like a familiar companion chatting beside the user: casual, dry, slightly blunt in a friendly way, and never theatrical. "
    "Prefer compact spoken Japanese with everyday wording over formal written prose. "
    "Keep replies short, usually one to three brief sentences, unless the user clearly needs more. "
    "Answer or react to the user's actual point first, then if it fits add one small sideways observation or mild tsukkomi. "
    "It is fine to be a little surreal or offbeat, but do not force jokes every turn. "
    "Avoid generic AI-assistant phrasing, stiff disclaimers, praise-heavy cheerleading, and over-explaining. "
    "Do not start replies by addressing the user by name unless they clearly ask for that or it is needed for clarity. "
    "Stay coherent, helpful, and grounded even when the tone is playful."
)
DEFAULT_SCREEN_TRIGGER_PHRASES = (
    "画面みて",
    "画面見て",
    "画面をみて",
    "画面を見て",
    "スクショみて",
    "スクショ見て",
)
DEFAULT_SCREEN_PASSIVE_TRIGGER_PHRASES = (
    "この画面",
    "今の画面",
    "いまの画面",
    "見えてる",
    "見えてます",
)

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSY_VALUES = {"0", "false", "no", "off"}


class QueueOverflowStrategy(str, Enum):
    REJECT_NEW = "reject_new"
    DROP_OLDEST = "drop_oldest"


class InputProvider(str, Enum):
    STDIN = "stdin"
    MICROPHONE = "microphone"


class MicrophoneInterruptMode(str, Enum):
    ALWAYS = "always"
    EXPLICIT = "explicit"
    DISABLED = "disabled"


class ConversationWindowResetPolicy(str, Enum):
    CLEAR = "clear"
    RESUME_SUMMARY = "resume_summary"


class OutputProvider(str, Enum):
    MEMORY = "memory"
    SPEAKER = "speaker"


class ApplicationAudioMode(str, Enum):
    CONTEXT_ONLY = "context_only"
    RESPOND = "respond"


class AivisEngineMode(str, Enum):
    EXTERNAL = "external"
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class SettingDefinition:
    env_name: str
    group: str
    kind: str
    default_raw: str | None
    nullable: bool = False
    secret: bool = False
    multiline: bool = False
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class SettingDocumentation:
    description: str
    default_label: str | None = None


def _enum_values(enum_class: type[Enum]) -> tuple[str, ...]:
    return tuple(str(member.value) for member in enum_class)


CONTROLLER_SETTING_DEFINITIONS = (
    SettingDefinition("VOCALIVE_SESSION_ID", "general", "string", None, nullable=True),
    SettingDefinition(
        "VOCALIVE_LOG_LEVEL",
        "general",
        "enum",
        "INFO",
        options=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    ),
    SettingDefinition(
        "VOCALIVE_STT_PROVIDER",
        "providers",
        "enum",
        "mock",
        options=("mock", "moonshine", "openai"),
    ),
    SettingDefinition(
        "VOCALIVE_MODEL_PROVIDER",
        "providers",
        "enum",
        "mock",
        options=("mock", "gemini"),
    ),
    SettingDefinition(
        "VOCALIVE_TTS_PROVIDER",
        "providers",
        "enum",
        "mock",
        options=("mock", "aivis"),
    ),
    SettingDefinition("VOCALIVE_QUEUE_MAXSIZE", "queue", "int", "4"),
    SettingDefinition(
        "VOCALIVE_QUEUE_OVERFLOW",
        "queue",
        "enum",
        QueueOverflowStrategy.DROP_OLDEST.value,
        options=_enum_values(QueueOverflowStrategy),
    ),
    SettingDefinition(
        "VOCALIVE_CONVERSATION_LANGUAGE",
        "conversation",
        "string",
        "ja",
        nullable=True,
    ),
    SettingDefinition("VOCALIVE_USER_NAME", "conversation", "string", None, nullable=True),
    SettingDefinition("VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT", "context", "int", "8"),
    SettingDefinition(
        "VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS",
        "context",
        "float",
        "90.0",
    ),
    SettingDefinition(
        "VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS",
        "context",
        "int",
        "1200",
    ),
    SettingDefinition(
        "VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT",
        "context",
        "int",
        "4",
    ),
    SettingDefinition(
        "VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS",
        "context",
        "int",
        "900",
    ),
    SettingDefinition(
        "VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS",
        "context",
        "int",
        "8",
    ),
    SettingDefinition(
        "VOCALIVE_INPUT_PROVIDER",
        "input",
        "enum",
        InputProvider.STDIN.value,
        options=_enum_values(InputProvider),
    ),
    SettingDefinition("VOCALIVE_MIC_SAMPLE_RATE", "input", "int", "16000"),
    SettingDefinition("VOCALIVE_MIC_CHANNELS", "input", "int", "1"),
    SettingDefinition("VOCALIVE_MIC_BLOCK_MS", "input", "float", "40.0"),
    SettingDefinition("VOCALIVE_MIC_SPEECH_THRESHOLD", "input", "float", "0.02"),
    SettingDefinition("VOCALIVE_MIC_PRE_SPEECH_MS", "input", "float", "200.0"),
    SettingDefinition("VOCALIVE_MIC_SPEECH_HOLD_MS", "input", "float", "200.0"),
    SettingDefinition("VOCALIVE_MIC_SILENCE_MS", "input", "float", "300.0"),
    SettingDefinition("VOCALIVE_MIC_MIN_UTTERANCE_MS", "input", "float", "250.0"),
    SettingDefinition("VOCALIVE_MIC_MAX_UTTERANCE_MS", "input", "float", "15000.0"),
    SettingDefinition("VOCALIVE_MIC_DEVICE", "input", "string", None, nullable=True),
    SettingDefinition("VOCALIVE_MIC_PREFER_EXTERNAL", "input", "bool", "true"),
    SettingDefinition(
        "VOCALIVE_MIC_INTERRUPT_MODE",
        "input",
        "enum",
        MicrophoneInterruptMode.ALWAYS.value,
        options=_enum_values(MicrophoneInterruptMode),
    ),
    SettingDefinition("VOCALIVE_CONVERSATION_WINDOW_ENABLED", "input", "bool", "false"),
    SettingDefinition(
        "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS",
        "input",
        "float",
        "20.0",
    ),
    SettingDefinition(
        "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS",
        "input",
        "float",
        "180.0",
    ),
    SettingDefinition(
        "VOCALIVE_CONVERSATION_WINDOW_START_OPEN",
        "input",
        "bool",
        "true",
    ),
    SettingDefinition(
        "VOCALIVE_CONVERSATION_WINDOW_APPLY_TO_APP_AUDIO",
        "input",
        "bool",
        "false",
    ),
    SettingDefinition(
        "VOCALIVE_CONVERSATION_WINDOW_RESET_POLICY",
        "input",
        "enum",
        ConversationWindowResetPolicy.CLEAR.value,
        options=_enum_values(ConversationWindowResetPolicy),
    ),
    SettingDefinition("VOCALIVE_APP_AUDIO_ENABLED", "application_audio", "bool", "false"),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_MODE",
        "application_audio",
        "enum",
        ApplicationAudioMode.CONTEXT_ONLY.value,
        options=_enum_values(ApplicationAudioMode),
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_TARGET",
        "application_audio",
        "string",
        None,
        nullable=True,
    ),
    SettingDefinition("VOCALIVE_APP_AUDIO_SAMPLE_RATE", "application_audio", "int", "16000"),
    SettingDefinition("VOCALIVE_APP_AUDIO_CHANNELS", "application_audio", "int", "1"),
    SettingDefinition("VOCALIVE_APP_AUDIO_BLOCK_MS", "application_audio", "float", "40.0"),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD",
        "application_audio",
        "float",
        "0.02",
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_PRE_SPEECH_MS",
        "application_audio",
        "float",
        "200.0",
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS",
        "application_audio",
        "float",
        "320.0",
    ),
    SettingDefinition("VOCALIVE_APP_AUDIO_SILENCE_MS", "application_audio", "float", "650.0"),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS",
        "application_audio",
        "float",
        "250.0",
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS",
        "application_audio",
        "float",
        "15000.0",
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS",
        "application_audio",
        "float",
        "10.0",
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS",
        "application_audio",
        "float",
        "0.0",
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS",
        "application_audio",
        "float",
        "0.0",
    ),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS",
        "application_audio",
        "float",
        "0.0",
    ),
    SettingDefinition("VOCALIVE_APP_AUDIO_ADAPTIVE_VAD", "application_audio", "bool", "true"),
    SettingDefinition(
        "VOCALIVE_APP_AUDIO_STT_ENHANCEMENT",
        "application_audio",
        "bool",
        "true",
    ),
    SettingDefinition(
        "VOCALIVE_OUTPUT_PROVIDER",
        "output",
        "enum",
        OutputProvider.MEMORY.value,
        options=_enum_values(OutputProvider),
    ),
    SettingDefinition("VOCALIVE_SPEAKER_COMMAND", "output", "string", None, nullable=True),
    SettingDefinition("VOCALIVE_OVERLAY_ENABLED", "overlay", "bool", "false"),
    SettingDefinition("VOCALIVE_OVERLAY_HOST", "overlay", "string", "127.0.0.1"),
    SettingDefinition("VOCALIVE_OVERLAY_PORT", "overlay", "int", "8765"),
    SettingDefinition("VOCALIVE_OVERLAY_AUTO_OPEN", "overlay", "bool", "true"),
    SettingDefinition("VOCALIVE_OVERLAY_TITLE", "overlay", "string", "VocaLive Overlay"),
    SettingDefinition("VOCALIVE_OVERLAY_CHARACTER_NAME", "overlay", "string", "Tora"),
    SettingDefinition("VOCALIVE_REPLY_DEBOUNCE_MS", "reply", "float", "200.0"),
    SettingDefinition("VOCALIVE_REPLY_POLICY_ENABLED", "reply", "bool", "true"),
    SettingDefinition("VOCALIVE_REPLY_MIN_GAP_MS", "reply", "float", "6000.0"),
    SettingDefinition("VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS", "reply", "int", "12"),
    SettingDefinition(
        "VOCALIVE_REPLY_REQUIRE_EXPLICIT_TRIGGER",
        "reply",
        "bool",
        "false",
    ),
    SettingDefinition("VOCALIVE_PROACTIVE_ENABLED", "proactive", "bool", "false"),
    SettingDefinition("VOCALIVE_PROACTIVE_MICROPHONE_ENABLED", "proactive", "bool", "true"),
    SettingDefinition(
        "VOCALIVE_PROACTIVE_APPLICATION_AUDIO_ENABLED",
        "proactive",
        "bool",
        "true",
    ),
    SettingDefinition("VOCALIVE_PROACTIVE_SCREEN_ENABLED", "proactive", "bool", "true"),
    SettingDefinition("VOCALIVE_PROACTIVE_IDLE_SECONDS", "proactive", "float", "20.0"),
    SettingDefinition("VOCALIVE_PROACTIVE_COOLDOWN_SECONDS", "proactive", "float", "45.0"),
    SettingDefinition("VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS", "proactive", "float", "10.0"),
    SettingDefinition(
        "VOCALIVE_GEMINI_API_KEY",
        "gemini",
        "string",
        None,
        nullable=True,
        secret=True,
    ),
    SettingDefinition("VOCALIVE_GEMINI_MODEL", "gemini", "string", "gemini-2.5-flash"),
    SettingDefinition("VOCALIVE_GEMINI_TIMEOUT_SECONDS", "gemini", "float", "30.0"),
    SettingDefinition(
        "VOCALIVE_GEMINI_TEMPERATURE",
        "gemini",
        "float",
        None,
        nullable=True,
    ),
    SettingDefinition(
        "VOCALIVE_GEMINI_THINKING_BUDGET",
        "gemini",
        "int",
        "0",
        nullable=True,
    ),
    SettingDefinition(
        "VOCALIVE_GEMINI_SYSTEM_INSTRUCTION",
        "gemini",
        "string",
        DEFAULT_GEMINI_SYSTEM_INSTRUCTION,
        nullable=True,
        multiline=True,
    ),
    SettingDefinition("VOCALIVE_SCREEN_CAPTURE_ENABLED", "screen_capture", "bool", "false"),
    SettingDefinition(
        "VOCALIVE_SCREEN_WINDOW_NAME",
        "screen_capture",
        "string",
        None,
        nullable=True,
    ),
    SettingDefinition(
        "VOCALIVE_SCREEN_TRIGGER_PHRASES",
        "screen_capture",
        "tuple",
        ",".join(DEFAULT_SCREEN_TRIGGER_PHRASES),
        nullable=True,
        multiline=True,
    ),
    SettingDefinition("VOCALIVE_SCREEN_PASSIVE_ENABLED", "screen_capture", "bool", "false"),
    SettingDefinition(
        "VOCALIVE_SCREEN_PASSIVE_TRIGGER_PHRASES",
        "screen_capture",
        "tuple",
        ",".join(DEFAULT_SCREEN_PASSIVE_TRIGGER_PHRASES),
        nullable=True,
        multiline=True,
    ),
    SettingDefinition(
        "VOCALIVE_SCREEN_PASSIVE_COOLDOWN_SECONDS",
        "screen_capture",
        "float",
        "30.0",
    ),
    SettingDefinition(
        "VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS",
        "screen_capture",
        "float",
        "5.0",
    ),
    SettingDefinition(
        "VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX",
        "screen_capture",
        "int",
        "1280",
        nullable=True,
    ),
    SettingDefinition("VOCALIVE_MOONSHINE_MODEL", "moonshine", "string", "base"),
    SettingDefinition(
        "VOCALIVE_OPENAI_API_KEY",
        "openai",
        "string",
        None,
        nullable=True,
        secret=True,
    ),
    SettingDefinition(
        "VOCALIVE_OPENAI_MODEL",
        "openai",
        "string",
        "gpt-4o-mini-transcribe",
    ),
    SettingDefinition(
        "VOCALIVE_OPENAI_BASE_URL",
        "openai",
        "string",
        "https://api.openai.com/v1",
    ),
    SettingDefinition(
        "VOCALIVE_OPENAI_TIMEOUT_SECONDS",
        "openai",
        "float",
        "30.0",
    ),
    SettingDefinition("VOCALIVE_AIVIS_BASE_URL", "aivis", "string", "http://127.0.0.1:10101"),
    SettingDefinition(
        "VOCALIVE_AIVIS_ENGINE_MODE",
        "aivis",
        "enum",
        AivisEngineMode.EXTERNAL.value,
        options=_enum_values(AivisEngineMode),
    ),
    SettingDefinition("VOCALIVE_AIVIS_ENGINE_PATH", "aivis", "string", None, nullable=True),
    SettingDefinition(
        "VOCALIVE_AIVIS_CPU_NUM_THREADS",
        "aivis",
        "int",
        None,
        nullable=True,
    ),
    SettingDefinition("VOCALIVE_AIVIS_STARTUP_TIMEOUT_SECONDS", "aivis", "float", "60.0"),
    SettingDefinition("VOCALIVE_AIVIS_SPEAKER_ID", "aivis", "int", None, nullable=True),
    SettingDefinition("VOCALIVE_AIVIS_SPEAKER_NAME", "aivis", "string", None, nullable=True),
    SettingDefinition("VOCALIVE_AIVIS_STYLE_NAME", "aivis", "string", None, nullable=True),
    SettingDefinition("VOCALIVE_AIVIS_TIMEOUT_SECONDS", "aivis", "float", "30.0"),
)

_CONTROLLER_SETTING_INDEX = {
    definition.env_name: definition for definition in CONTROLLER_SETTING_DEFINITIONS
}
_CONTROLLER_SECRET_ENV_NAMES = tuple(
    definition.env_name
    for definition in CONTROLLER_SETTING_DEFINITIONS
    if definition.secret
)

_CONTROLLER_SETTING_DOCUMENTATION = {
    "VOCALIVE_SESSION_ID": SettingDocumentation(
        description="Stable identifier for one conversation session",
        default_label="random UUID",
    ),
    "VOCALIVE_LOG_LEVEL": SettingDocumentation(description="Python logging level"),
    "VOCALIVE_INPUT_PROVIDER": SettingDocumentation(description="`stdin` or `microphone`"),
    "VOCALIVE_MIC_SAMPLE_RATE": SettingDocumentation(description="Microphone capture sample rate"),
    "VOCALIVE_MIC_CHANNELS": SettingDocumentation(description="Captured microphone channel count"),
    "VOCALIVE_MIC_BLOCK_MS": SettingDocumentation(description="Duration of each captured PCM block"),
    "VOCALIVE_MIC_DEVICE": SettingDocumentation(
        description="Optional input device id, device name, `default`, or `external`"
    ),
    "VOCALIVE_MIC_PREFER_EXTERNAL": SettingDocumentation(
        description=(
            "Prefer a connected higher-fidelity external mic when the default input looks "
            "built-in; auto-selection avoids Bluetooth hands-free inputs"
        )
    ),
    "VOCALIVE_MIC_INTERRUPT_MODE": SettingDocumentation(
        description=(
            "Microphone barge-in policy: `always` interrupts active assistant speech on new "
            "user speech, `explicit` waits for a finalized utterance that directly calls the "
            "assistant or clearly asks a question / makes a request, and `disabled` never "
            "interrupts early"
        )
    ),
    "VOCALIVE_MIC_SPEECH_THRESHOLD": SettingDocumentation(
        description="RMS threshold for treating a block as speech"
    ),
    "VOCALIVE_MIC_PRE_SPEECH_MS": SettingDocumentation(
        description="Audio kept before speech starts so utterance onsets are not clipped"
    ),
    "VOCALIVE_MIC_SPEECH_HOLD_MS": SettingDocumentation(
        description="Keep an utterance in the speech state briefly after the threshold drops"
    ),
    "VOCALIVE_MIC_SILENCE_MS": SettingDocumentation(
        description="Silence required before emitting the buffered utterance"
    ),
    "VOCALIVE_MIC_MIN_UTTERANCE_MS": SettingDocumentation(
        description="Minimum buffered audio before end-of-turn detection may emit"
    ),
    "VOCALIVE_MIC_MAX_UTTERANCE_MS": SettingDocumentation(
        description="Hard cap for one buffered utterance"
    ),
    "VOCALIVE_CONVERSATION_WINDOW_ENABLED": SettingDocumentation(
        description=(
            "When enabled, live audio is only forwarded to STT during conversation windows "
            "that reopen on user speech after each closed interval"
        )
    ),
    "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS": SettingDocumentation(
        description="How long each conversation window stays open after user speech reopens it"
    ),
    "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS": SettingDocumentation(
        description="How long live audio stays skipped before the next user speech may reopen the window"
    ),
    "VOCALIVE_CONVERSATION_WINDOW_START_OPEN": SettingDocumentation(
        description="Start the runtime with the first conversation window already open"
    ),
    "VOCALIVE_CONVERSATION_WINDOW_APPLY_TO_APP_AUDIO": SettingDocumentation(
        description=(
            "Apply conversation-window gating to application-audio STT as well as microphone "
            "speech"
        )
    ),
    "VOCALIVE_CONVERSATION_WINDOW_RESET_POLICY": SettingDocumentation(
        description=(
            "History handling when a closed conversation window reopens: `clear` starts a "
            "fresh session, while `resume_summary` carries forward an LLM-written resume note "
            "with durable goals and constraints"
        )
    ),
    "VOCALIVE_APP_AUDIO_ENABLED": SettingDocumentation(
        description="Enables application-audio capture as an additional live input"
    ),
    "VOCALIVE_APP_AUDIO_MODE": SettingDocumentation(
        description=(
            "`context_only` stores app transcripts in session without immediate assistant "
            "replies; `respond` makes app audio behave like normal live turns"
        )
    ),
    "VOCALIVE_APP_AUDIO_TARGET": SettingDocumentation(
        description=(
            "Required application selector; on macOS it matches application name first then "
            "bundle identifier, and on Windows it matches process name, executable path, or "
            "main window title"
        )
    ),
    "VOCALIVE_APP_AUDIO_SAMPLE_RATE": SettingDocumentation(
        description="Application-audio capture sample rate after helper-side conversion"
    ),
    "VOCALIVE_APP_AUDIO_CHANNELS": SettingDocumentation(
        description="Captured application-audio channel count"
    ),
    "VOCALIVE_APP_AUDIO_BLOCK_MS": SettingDocumentation(
        description="Duration of each buffered application-audio PCM block"
    ),
    "VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD": SettingDocumentation(
        description=(
            "Minimum floor for application-audio speech detection; adaptive VAD treats it as "
            "a fallback absolute threshold"
        )
    ),
    "VOCALIVE_APP_AUDIO_PRE_SPEECH_MS": SettingDocumentation(
        description="Buffered application audio kept before speech onset"
    ),
    "VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS": SettingDocumentation(
        description="Keeps application audio in the speech state briefly after the threshold drops"
    ),
    "VOCALIVE_APP_AUDIO_SILENCE_MS": SettingDocumentation(
        description="Silence required before emitting a buffered application-audio utterance"
    ),
    "VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS": SettingDocumentation(
        description="Minimum buffered application audio before end-of-turn detection may emit"
    ),
    "VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS": SettingDocumentation(
        description="Hard cap for one buffered application-audio utterance"
    ),
    "VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS": SettingDocumentation(
        description="Timeout for application lookup, helper startup, and helper build floor"
    ),
    "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS": SettingDocumentation(
        description=(
            "Minimum delay between accepted application-audio utterances before STT runs "
            "again; increase this to reduce continuous application-audio transcription load"
        )
    ),
    "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS": SettingDocumentation(
        description=(
            "Delay before application-audio STT runs so nearby segments can merge into one "
            "transcription request"
        )
    ),
    "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS": SettingDocumentation(
        description=(
            "Minimum merged application-audio duration required before STT is attempted"
        )
    ),
    "VOCALIVE_APP_AUDIO_ADAPTIVE_VAD": SettingDocumentation(
        description=(
            "Enables adaptive energy-based VAD for application audio; `false` falls back to "
            "fixed thresholding"
        )
    ),
    "VOCALIVE_APP_AUDIO_STT_ENHANCEMENT": SettingDocumentation(
        description="Enables lightweight application-audio speech enhancement before Moonshine STT"
    ),
    "VOCALIVE_STT_PROVIDER": SettingDocumentation(
        description=(
            "STT adapter; accepts `moonshine`, `openai`, and aliases such as "
            "`moonshine voice` or `gpt-4o-mini-transcribe`"
        )
    ),
    "VOCALIVE_MODEL_PROVIDER": SettingDocumentation(
        description="LLM adapter; accepts `gemini` and aliases such as `google gemini`"
    ),
    "VOCALIVE_TTS_PROVIDER": SettingDocumentation(
        description="TTS adapter; accepts `aivis` and aliases such as `aivis speech`"
    ),
    "VOCALIVE_OUTPUT_PROVIDER": SettingDocumentation(description="`memory` or `speaker`"),
    "VOCALIVE_OVERLAY_ENABLED": SettingDocumentation(
        description="Start the local transparent browser overlay with speech-only captions"
    ),
    "VOCALIVE_OVERLAY_HOST": SettingDocumentation(
        description="Host/interface used by the overlay HTTP server"
    ),
    "VOCALIVE_OVERLAY_PORT": SettingDocumentation(
        description="Port used by the overlay HTTP server"
    ),
    "VOCALIVE_OVERLAY_AUTO_OPEN": SettingDocumentation(
        description="Ask the system browser to open the overlay page automatically"
    ),
    "VOCALIVE_OVERLAY_TITLE": SettingDocumentation(
        description="Browser page title for the overlay"
    ),
    "VOCALIVE_OVERLAY_CHARACTER_NAME": SettingDocumentation(
        description="Accessibility label and page text for the overlay character"
    ),
    "VOCALIVE_USER_NAME": SettingDocumentation(
        description=(
            "Optional user name injected before the LLM call so the assistant can answer who "
            "it is speaking with without defaulting to name-based greetings"
        )
    ),
    "VOCALIVE_CONVERSATION_LANGUAGE": SettingDocumentation(
        description="Per-turn language instruction injected before the LLM call; set empty to disable"
    ),
    "VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT": SettingDocumentation(
        description=(
            "Number of recent user/assistant messages kept verbatim in Gemini requests before "
            "older dialogue is compacted"
        )
    ),
    "VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS": SettingDocumentation(
        description=(
            "Maximum age for a message to stay in direct current-turn context; older messages "
            "are summarized as reference-only background. Set 0 to disable the age cutoff."
        )
    ),
    "VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS": SettingDocumentation(
        description=(
            "Character budget for the earlier-conversation summary injected ahead of the "
            "recent raw-message window"
        )
    ),
    "VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT": SettingDocumentation(
        description=(
            "Number of recent application-audio messages kept verbatim in Gemini requests "
            "before older app context is compacted"
        )
    ),
    "VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS": SettingDocumentation(
        description=(
            "Character budget for the earlier application-audio summary injected ahead of the "
            "recent raw app-context window"
        )
    ),
    "VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS": SettingDocumentation(
        description="Minimum normalized application-audio message length kept in the older app-context summary"
    ),
    "VOCALIVE_REPLY_DEBOUNCE_MS": SettingDocumentation(
        description=(
            "Delay before a microphone user utterance is queued for the LLM so nearby "
            "follow-up utterances can merge into one turn"
        )
    ),
    "VOCALIVE_REPLY_POLICY_ENABLED": SettingDocumentation(
        description="Enables conservative microphone reply suppression for low-value live chatter"
    ),
    "VOCALIVE_REPLY_MIN_GAP_MS": SettingDocumentation(
        description=(
            "Minimum time after a completed assistant reply during which short microphone "
            "chatter is more likely to be suppressed"
        )
    ),
    "VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS": SettingDocumentation(
        description="Maximum normalized length treated as a short microphone reaction for suppression heuristics"
    ),
    "VOCALIVE_REPLY_REQUIRE_EXPLICIT_TRIGGER": SettingDocumentation(
        description=(
            "When true, microphone turns only trigger replies when they clearly look like "
            "questions/requests or directly address the assistant; useful for suppressing "
            "think-aloud or read-aloud chatter"
        )
    ),
    "VOCALIVE_PROACTIVE_ENABLED": SettingDocumentation(
        description=(
            "Enables low-priority proactive monologues when the user has been quiet and "
            "new live observations are available"
        )
    ),
    "VOCALIVE_PROACTIVE_MICROPHONE_ENABLED": SettingDocumentation(
        description=(
            "Allows finalized microphone utterances that did not trigger an immediate reply "
            "to become proactive-monologue candidates"
        )
    ),
    "VOCALIVE_PROACTIVE_APPLICATION_AUDIO_ENABLED": SettingDocumentation(
        description=(
            "Allows new `context_only` application-audio transcripts to become proactive-"
            "monologue candidates"
        )
    ),
    "VOCALIVE_PROACTIVE_SCREEN_ENABLED": SettingDocumentation(
        description=(
            "Allows proactive monologues to watch for changed screenshots when screen "
            "capture is enabled and multimodal input is available"
        )
    ),
    "VOCALIVE_PROACTIVE_IDLE_SECONDS": SettingDocumentation(
        description="Minimum quiet time after the latest live user/application activity before a proactive monologue may start"
    ),
    "VOCALIVE_PROACTIVE_COOLDOWN_SECONDS": SettingDocumentation(
        description="Minimum delay between completed proactive monologues"
    ),
    "VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS": SettingDocumentation(
        description="How often proactive mode polls the configured window for changed screenshots while idle"
    ),
    "VOCALIVE_GEMINI_API_KEY": SettingDocumentation(
        description="Gemini API key; `GEMINI_API_KEY` is also accepted"
    ),
    "VOCALIVE_GEMINI_MODEL": SettingDocumentation(
        description="Gemini model name used for `generateContent`"
    ),
    "VOCALIVE_GEMINI_TIMEOUT_SECONDS": SettingDocumentation(description="Gemini HTTP timeout"),
    "VOCALIVE_GEMINI_TEMPERATURE": SettingDocumentation(
        description="Optional Gemini generation temperature"
    ),
    "VOCALIVE_GEMINI_THINKING_BUDGET": SettingDocumentation(
        description="Gemini 2.5 thinking budget; empty unsets it"
    ),
    "VOCALIVE_GEMINI_SYSTEM_INSTRUCTION": SettingDocumentation(
        description="Overrides the default Gemini character prompt; set empty to disable it entirely",
        default_label="Kohaku surreal deadpan persona prompt",
    ),
    "VOCALIVE_SCREEN_CAPTURE_ENABLED": SettingDocumentation(
        description="Enables request-scoped named-window screenshot capture for Gemini turns"
    ),
    "VOCALIVE_SCREEN_WINDOW_NAME": SettingDocumentation(
        description="Required window selector; matches on-screen window title first, then owner name"
    ),
    "VOCALIVE_SCREEN_TRIGGER_PHRASES": SettingDocumentation(
        description="Comma-separated trigger phrases that cause a screenshot to be attached"
    ),
    "VOCALIVE_SCREEN_PASSIVE_ENABLED": SettingDocumentation(
        description=(
            "Allows screen-reference phrases to attach a screenshot opportunistically during "
            "normal conversation"
        )
    ),
    "VOCALIVE_SCREEN_PASSIVE_TRIGGER_PHRASES": SettingDocumentation(
        description=(
            "Comma-separated screen-reference phrases checked only when passive capture is enabled"
        )
    ),
    "VOCALIVE_SCREEN_PASSIVE_COOLDOWN_SECONDS": SettingDocumentation(
        description=(
            "Minimum delay between passive screenshot sends; unchanged passive screenshots are "
            "also skipped"
        )
    ),
    "VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS": SettingDocumentation(
        description="Timeout for window lookup and platform capture helpers"
    ),
    "VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX": SettingDocumentation(
        description=(
            "Resizes captured screenshots so their longest edge stays within this many pixels; "
            "empty disables resizing"
        )
    ),
    "VOCALIVE_MOONSHINE_MODEL": SettingDocumentation(
        description="Moonshine model architecture such as `base` / `tiny`, or a concrete model id such as `base-ja`"
    ),
    "VOCALIVE_OPENAI_API_KEY": SettingDocumentation(
        description="OpenAI API key for STT; `OPENAI_API_KEY` is also accepted"
    ),
    "VOCALIVE_OPENAI_MODEL": SettingDocumentation(
        description="OpenAI transcription model name"
    ),
    "VOCALIVE_OPENAI_BASE_URL": SettingDocumentation(
        description="OpenAI API base URL for audio transcription"
    ),
    "VOCALIVE_OPENAI_TIMEOUT_SECONDS": SettingDocumentation(
        description="OpenAI audio transcription HTTP timeout"
    ),
    "VOCALIVE_AIVIS_BASE_URL": SettingDocumentation(description="AivisSpeech engine base URL"),
    "VOCALIVE_AIVIS_ENGINE_MODE": SettingDocumentation(
        description="AivisSpeech engine startup mode: `external`, `cpu`, or `gpu`"
    ),
    "VOCALIVE_AIVIS_ENGINE_PATH": SettingDocumentation(
        description="Optional path to AivisSpeech Engine `run(.exe)` for managed CPU/GPU startup"
    ),
    "VOCALIVE_AIVIS_CPU_NUM_THREADS": SettingDocumentation(
        description=(
            "Optional managed-startup CPU thread limit passed to AivisSpeech Engine as "
            "`--cpu_num_threads`; lower values can reduce CPU load"
        )
    ),
    "VOCALIVE_AIVIS_STARTUP_TIMEOUT_SECONDS": SettingDocumentation(
        description="How long VocaLive waits for a managed AivisSpeech engine to become ready"
    ),
    "VOCALIVE_AIVIS_SPEAKER_ID": SettingDocumentation(
        description="Explicit AivisSpeech style ID"
    ),
    "VOCALIVE_AIVIS_SPEAKER_NAME": SettingDocumentation(
        description="Speaker name to resolve via `/speakers`"
    ),
    "VOCALIVE_AIVIS_STYLE_NAME": SettingDocumentation(
        description="Style name to resolve via `/speakers`"
    ),
    "VOCALIVE_AIVIS_TIMEOUT_SECONDS": SettingDocumentation(
        description="AivisSpeech API timeout"
    ),
    "VOCALIVE_SPEAKER_COMMAND": SettingDocumentation(
        description=(
            "Override playback command; must include `{path}`. Defaults to `afplay {path}` "
            "on macOS and PowerShell `SoundPlayer` on Windows"
        ),
        default_label="platform default",
    ),
    "VOCALIVE_QUEUE_MAXSIZE": SettingDocumentation(
        description="Maximum queued utterances waiting to run"
    ),
    "VOCALIVE_QUEUE_OVERFLOW": SettingDocumentation(
        description="Overflow strategy: `drop_oldest` or `reject_new`"
    ),
}

if set(_CONTROLLER_SETTING_DOCUMENTATION) != set(_CONTROLLER_SETTING_INDEX):
    raise RuntimeError("Controller setting documentation must cover every controller setting")


def _setting_default_label(definition: SettingDefinition) -> str:
    documentation = _CONTROLLER_SETTING_DOCUMENTATION[definition.env_name]
    if documentation.default_label is not None:
        return documentation.default_label
    if definition.default_raw is None:
        return "unset"
    return definition.default_raw


def controller_setting_rows() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "env_name": definition.env_name,
            "default_label": _setting_default_label(definition),
            "description": _CONTROLLER_SETTING_DOCUMENTATION[definition.env_name].description,
        }
        for definition in CONTROLLER_SETTING_DEFINITIONS
    )


def controller_setting_definitions() -> tuple[SettingDefinition, ...]:
    return CONTROLLER_SETTING_DEFINITIONS


def controller_setting_schema() -> tuple[dict[str, object], ...]:
    schema = []
    for definition in CONTROLLER_SETTING_DEFINITIONS:
        documentation = _CONTROLLER_SETTING_DOCUMENTATION[definition.env_name]
        payload = asdict(definition)
        payload["default_label"] = _setting_default_label(definition)
        payload["description"] = documentation.description
        schema.append(payload)
    return tuple(schema)


def controller_default_values() -> dict[str, str | None]:
    return {
        definition.env_name: definition.default_raw
        for definition in CONTROLLER_SETTING_DEFINITIONS
    }


def controller_secret_env_names() -> tuple[str, ...]:
    return _CONTROLLER_SECRET_ENV_NAMES


def sanitize_persisted_controller_values(
    values: Mapping[str, str | None],
) -> dict[str, str | None]:
    normalized = normalize_controller_values(values)
    for env_name in _CONTROLLER_SECRET_ENV_NAMES:
        normalized[env_name] = _CONTROLLER_SETTING_INDEX[env_name].default_raw
    return normalized


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
    silence_threshold_ms: float = 300.0
    min_utterance_ms: float = 250.0
    max_utterance_ms: float = 15_000.0
    device: str | int | None = None
    prefer_external_device: bool = True
    interrupt_mode: MicrophoneInterruptMode = MicrophoneInterruptMode.ALWAYS


@dataclass
class ConversationWindowSettings:
    enabled: bool = False
    open_duration_seconds: float = 20.0
    closed_duration_seconds: float = 180.0
    start_open: bool = True
    apply_to_application_audio: bool = False
    reset_policy: ConversationWindowResetPolicy = ConversationWindowResetPolicy.CLEAR


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
    transcription_cooldown_seconds: float = 0.0
    transcription_debounce_ms: float = 0.0
    min_transcription_duration_ms: float = 0.0
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
    debounce_ms: float = 200.0
    policy_enabled: bool = True
    min_gap_ms: float = 6000.0
    short_utterance_max_chars: int = 12
    require_explicit_trigger: bool = False


@dataclass
class ProactiveSettings:
    enabled: bool = False
    microphone_enabled: bool = True
    application_audio_enabled: bool = True
    screen_enabled: bool = True
    idle_seconds: float = 20.0
    cooldown_seconds: float = 45.0
    screen_poll_seconds: float = 10.0


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
    passive_enabled: bool = False
    passive_trigger_phrases: tuple[str, ...] = DEFAULT_SCREEN_PASSIVE_TRIGGER_PHRASES
    passive_cooldown_seconds: float = 30.0
    timeout_seconds: float = 5.0
    resize_max_edge_px: int | None = 1280


@dataclass
class ConversationSettings:
    user_name: str | None = None
    language: str | None = "ja"


@dataclass
class ContextSettings:
    recent_message_count: int = 8
    active_message_max_age_seconds: float = 90.0
    conversation_summary_max_chars: int = 1200
    application_recent_message_count: int = 4
    application_summary_max_chars: int = 900
    application_summary_min_message_chars: int = 8


@dataclass
class MoonshineSettings:
    model_name: str = "base"


@dataclass
class OpenAISettings:
    api_key: str | None = None
    model_name: str = "gpt-4o-mini-transcribe"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 30.0


@dataclass
class AivisSpeechSettings:
    base_url: str = "http://127.0.0.1:10101"
    engine_mode: AivisEngineMode = AivisEngineMode.EXTERNAL
    engine_path: str | None = None
    cpu_num_threads: int | None = None
    startup_timeout_seconds: float = 60.0
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
    conversation_window: ConversationWindowSettings = field(
        default_factory=ConversationWindowSettings
    )
    application_audio: ApplicationAudioSettings = field(default_factory=ApplicationAudioSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    reply: ReplySettings = field(default_factory=ReplySettings)
    proactive: ProactiveSettings = field(default_factory=ProactiveSettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    screen_capture: ScreenCaptureSettings = field(default_factory=ScreenCaptureSettings)
    moonshine: MoonshineSettings = field(default_factory=MoonshineSettings)
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    aivis: AivisSpeechSettings = field(default_factory=AivisSpeechSettings)

    def __post_init__(self) -> None:
        self.stt_provider = _normalize_provider_setting("stt", self.stt_provider)
        self.model_provider = _normalize_provider_setting("model", self.model_provider)
        self.tts_provider = _normalize_provider_setting("tts", self.tts_provider)
        if self.application_audio.transcription_cooldown_seconds < 0:
            raise ValueError(
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS must be >= 0"
            )
        if self.conversation_window.open_duration_seconds <= 0:
            raise ValueError(
                "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS must be > 0"
            )
        if self.conversation_window.closed_duration_seconds < 0:
            raise ValueError(
                "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS must be >= 0"
            )
        if self.application_audio.transcription_debounce_ms < 0:
            raise ValueError(
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS must be >= 0"
            )
        if self.application_audio.min_transcription_duration_ms < 0:
            raise ValueError(
                "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS must be >= 0"
            )
        if self.context.active_message_max_age_seconds < 0:
            raise ValueError(
                "VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS must be >= 0"
            )
        if self.proactive.idle_seconds <= 0:
            raise ValueError("VOCALIVE_PROACTIVE_IDLE_SECONDS must be > 0")
        if self.proactive.cooldown_seconds < 0:
            raise ValueError("VOCALIVE_PROACTIVE_COOLDOWN_SECONDS must be >= 0")
        if self.proactive.screen_poll_seconds <= 0:
            raise ValueError("VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS must be > 0")
        if self.aivis.cpu_num_threads is not None and self.aivis.cpu_num_threads < 1:
            raise ValueError(
                "VOCALIVE_AIVIS_CPU_NUM_THREADS must be >= 1 when set"
            )

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str | None]) -> "AppSettings":
        return cls(
            session_id=_read_str_with_default(
                mapping,
                "VOCALIVE_SESSION_ID",
                default=uuid.uuid4().hex,
            ),
            log_level=_read_str_with_default(
                mapping,
                "VOCALIVE_LOG_LEVEL",
                default="INFO",
            ).upper(),
            stt_provider=_read_str_with_default(
                mapping,
                "VOCALIVE_STT_PROVIDER",
                default="mock",
            ),
            model_provider=_read_str_with_default(
                mapping,
                "VOCALIVE_MODEL_PROVIDER",
                default="mock",
            ),
            tts_provider=_read_str_with_default(
                mapping,
                "VOCALIVE_TTS_PROVIDER",
                default="mock",
            ),
            queue=QueueSettings(
                ingress_maxsize=_read_int(mapping, "VOCALIVE_QUEUE_MAXSIZE", default=4),
                overflow_strategy=QueueOverflowStrategy(
                    _read_str_with_default(
                        mapping,
                        "VOCALIVE_QUEUE_OVERFLOW",
                        default=QueueOverflowStrategy.DROP_OLDEST.value,
                    )
                ),
            ),
            conversation=ConversationSettings(
                user_name=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_USER_NAME",
                    default=None,
                ),
                language=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_CONVERSATION_LANGUAGE",
                    default="ja",
                ),
            ),
            context=ContextSettings(
                recent_message_count=_read_int(
                    mapping,
                    "VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT",
                    default=8,
                ),
                active_message_max_age_seconds=_read_float(
                    mapping,
                    "VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS",
                    default=90.0,
                ),
                conversation_summary_max_chars=_read_int(
                    mapping,
                    "VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS",
                    default=1200,
                ),
                application_recent_message_count=_read_int(
                    mapping,
                    "VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT",
                    default=4,
                ),
                application_summary_max_chars=_read_int(
                    mapping,
                    "VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS",
                    default=900,
                ),
                application_summary_min_message_chars=_read_int(
                    mapping,
                    "VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS",
                    default=8,
                ),
            ),
            input=InputSettings(
                provider=InputProvider(
                    _read_str_with_default(
                        mapping,
                        "VOCALIVE_INPUT_PROVIDER",
                        default=InputProvider.STDIN.value,
                    )
                ),
                sample_rate_hz=_read_int(mapping, "VOCALIVE_MIC_SAMPLE_RATE", default=16_000),
                channels=_read_int(mapping, "VOCALIVE_MIC_CHANNELS", default=1),
                block_duration_ms=_read_float(
                    mapping,
                    "VOCALIVE_MIC_BLOCK_MS",
                    default=40.0,
                ),
                speech_threshold=_read_float(
                    mapping,
                    "VOCALIVE_MIC_SPEECH_THRESHOLD",
                    default=0.02,
                ),
                pre_speech_ms=_read_float(
                    mapping,
                    "VOCALIVE_MIC_PRE_SPEECH_MS",
                    default=200.0,
                ),
                speech_hold_ms=_read_float(
                    mapping,
                    "VOCALIVE_MIC_SPEECH_HOLD_MS",
                    default=200.0,
                ),
                silence_threshold_ms=_read_float(
                    mapping,
                    "VOCALIVE_MIC_SILENCE_MS",
                    default=300.0,
                ),
                min_utterance_ms=_read_float(
                    mapping,
                    "VOCALIVE_MIC_MIN_UTTERANCE_MS",
                    default=250.0,
                ),
                max_utterance_ms=_read_float(
                    mapping,
                    "VOCALIVE_MIC_MAX_UTTERANCE_MS",
                    default=15_000.0,
                ),
                device=_read_optional_device(mapping, "VOCALIVE_MIC_DEVICE"),
                prefer_external_device=_read_bool(
                    mapping,
                    "VOCALIVE_MIC_PREFER_EXTERNAL",
                    default=True,
                ),
                interrupt_mode=_read_microphone_interrupt_mode(
                    mapping,
                    "VOCALIVE_MIC_INTERRUPT_MODE",
                    default=MicrophoneInterruptMode.ALWAYS,
                ),
            ),
            conversation_window=ConversationWindowSettings(
                enabled=_read_bool(
                    mapping,
                    "VOCALIVE_CONVERSATION_WINDOW_ENABLED",
                    default=False,
                ),
                open_duration_seconds=_read_float(
                    mapping,
                    "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS",
                    default=20.0,
                ),
                closed_duration_seconds=_read_float(
                    mapping,
                    "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS",
                    default=180.0,
                ),
                start_open=_read_bool(
                    mapping,
                    "VOCALIVE_CONVERSATION_WINDOW_START_OPEN",
                    default=True,
                ),
                apply_to_application_audio=_read_bool(
                    mapping,
                    "VOCALIVE_CONVERSATION_WINDOW_APPLY_TO_APP_AUDIO",
                    default=False,
                ),
                reset_policy=_read_conversation_window_reset_policy(
                    mapping,
                    "VOCALIVE_CONVERSATION_WINDOW_RESET_POLICY",
                    default=ConversationWindowResetPolicy.CLEAR,
                ),
            ),
            application_audio=ApplicationAudioSettings(
                enabled=_read_bool(
                    mapping,
                    "VOCALIVE_APP_AUDIO_ENABLED",
                    default=False,
                ),
                mode=_read_application_audio_mode(
                    mapping,
                    "VOCALIVE_APP_AUDIO_MODE",
                    default=ApplicationAudioMode.CONTEXT_ONLY,
                ),
                target=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_APP_AUDIO_TARGET",
                    default=None,
                ),
                sample_rate_hz=_read_int(
                    mapping,
                    "VOCALIVE_APP_AUDIO_SAMPLE_RATE",
                    default=16_000,
                ),
                channels=_read_int(
                    mapping,
                    "VOCALIVE_APP_AUDIO_CHANNELS",
                    default=1,
                ),
                block_duration_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_BLOCK_MS",
                    default=40.0,
                ),
                speech_threshold=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD",
                    default=0.02,
                ),
                pre_speech_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_PRE_SPEECH_MS",
                    default=200.0,
                ),
                speech_hold_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS",
                    default=320.0,
                ),
                silence_threshold_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_SILENCE_MS",
                    default=650.0,
                ),
                min_utterance_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS",
                    default=250.0,
                ),
                max_utterance_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS",
                    default=15_000.0,
                ),
                timeout_seconds=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS",
                    default=10.0,
                ),
                transcription_cooldown_seconds=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS",
                    default=0.0,
                ),
                transcription_debounce_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS",
                    default=0.0,
                ),
                min_transcription_duration_ms=_read_float(
                    mapping,
                    "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS",
                    default=0.0,
                ),
                adaptive_vad_enabled=_read_bool(
                    mapping,
                    "VOCALIVE_APP_AUDIO_ADAPTIVE_VAD",
                    default=True,
                ),
                stt_enhancement_enabled=_read_bool(
                    mapping,
                    "VOCALIVE_APP_AUDIO_STT_ENHANCEMENT",
                    default=True,
                ),
            ),
            output=OutputSettings(
                provider=OutputProvider(
                    _read_str_with_default(
                        mapping,
                        "VOCALIVE_OUTPUT_PROVIDER",
                        default=OutputProvider.MEMORY.value,
                    )
                ),
                speaker_command=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_SPEAKER_COMMAND",
                    default=None,
                ),
            ),
            overlay=OverlaySettings(
                enabled=_read_bool(mapping, "VOCALIVE_OVERLAY_ENABLED", default=False),
                host=_read_str_with_default(
                    mapping,
                    "VOCALIVE_OVERLAY_HOST",
                    default="127.0.0.1",
                ),
                port=_read_int(mapping, "VOCALIVE_OVERLAY_PORT", default=8765),
                auto_open=_read_bool(mapping, "VOCALIVE_OVERLAY_AUTO_OPEN", default=True),
                title=_read_str_with_default(
                    mapping,
                    "VOCALIVE_OVERLAY_TITLE",
                    default="VocaLive Overlay",
                ),
                character_name=_read_str_with_default(
                    mapping,
                    "VOCALIVE_OVERLAY_CHARACTER_NAME",
                    default="Tora",
                ),
            ),
            reply=ReplySettings(
                debounce_ms=_read_float(
                    mapping,
                    "VOCALIVE_REPLY_DEBOUNCE_MS",
                    default=200.0,
                ),
                policy_enabled=_read_bool(
                    mapping,
                    "VOCALIVE_REPLY_POLICY_ENABLED",
                    default=True,
                ),
                min_gap_ms=_read_float(
                    mapping,
                    "VOCALIVE_REPLY_MIN_GAP_MS",
                    default=6000.0,
                ),
                short_utterance_max_chars=_read_int(
                    mapping,
                    "VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS",
                    default=12,
                ),
                require_explicit_trigger=_read_bool(
                    mapping,
                    "VOCALIVE_REPLY_REQUIRE_EXPLICIT_TRIGGER",
                    default=False,
                ),
            ),
            proactive=ProactiveSettings(
                enabled=_read_bool(
                    mapping,
                    "VOCALIVE_PROACTIVE_ENABLED",
                    default=False,
                ),
                microphone_enabled=_read_bool(
                    mapping,
                    "VOCALIVE_PROACTIVE_MICROPHONE_ENABLED",
                    default=True,
                ),
                application_audio_enabled=_read_bool(
                    mapping,
                    "VOCALIVE_PROACTIVE_APPLICATION_AUDIO_ENABLED",
                    default=True,
                ),
                screen_enabled=_read_bool(
                    mapping,
                    "VOCALIVE_PROACTIVE_SCREEN_ENABLED",
                    default=True,
                ),
                idle_seconds=_read_float(
                    mapping,
                    "VOCALIVE_PROACTIVE_IDLE_SECONDS",
                    default=20.0,
                ),
                cooldown_seconds=_read_float(
                    mapping,
                    "VOCALIVE_PROACTIVE_COOLDOWN_SECONDS",
                    default=45.0,
                ),
                screen_poll_seconds=_read_float(
                    mapping,
                    "VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS",
                    default=10.0,
                ),
            ),
            gemini=GeminiSettings(
                api_key=(
                    _read_optional_str_with_default(
                        mapping,
                        "VOCALIVE_GEMINI_API_KEY",
                        default=None,
                    )
                    or _read_optional_str_with_default(
                        mapping,
                        "GEMINI_API_KEY",
                        default=None,
                    )
                ),
                model_name=_read_str_with_default(
                    mapping,
                    "VOCALIVE_GEMINI_MODEL",
                    default="gemini-2.5-flash",
                ),
                timeout_seconds=_read_float(
                    mapping,
                    "VOCALIVE_GEMINI_TIMEOUT_SECONDS",
                    default=30.0,
                ),
                temperature=_read_optional_float(
                    mapping,
                    "VOCALIVE_GEMINI_TEMPERATURE",
                ),
                thinking_budget=_read_optional_int_with_default(
                    mapping,
                    "VOCALIVE_GEMINI_THINKING_BUDGET",
                    default=0,
                ),
                system_instruction=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_GEMINI_SYSTEM_INSTRUCTION",
                    default=DEFAULT_GEMINI_SYSTEM_INSTRUCTION,
                ),
            ),
            screen_capture=ScreenCaptureSettings(
                enabled=_read_bool(
                    mapping,
                    "VOCALIVE_SCREEN_CAPTURE_ENABLED",
                    default=False,
                ),
                window_name=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_SCREEN_WINDOW_NAME",
                    default=None,
                ),
                trigger_phrases=_read_str_tuple(
                    mapping,
                    "VOCALIVE_SCREEN_TRIGGER_PHRASES",
                    default=DEFAULT_SCREEN_TRIGGER_PHRASES,
                ),
                passive_enabled=_read_bool(
                    mapping,
                    "VOCALIVE_SCREEN_PASSIVE_ENABLED",
                    default=False,
                ),
                passive_trigger_phrases=_read_str_tuple(
                    mapping,
                    "VOCALIVE_SCREEN_PASSIVE_TRIGGER_PHRASES",
                    default=DEFAULT_SCREEN_PASSIVE_TRIGGER_PHRASES,
                ),
                passive_cooldown_seconds=_read_float(
                    mapping,
                    "VOCALIVE_SCREEN_PASSIVE_COOLDOWN_SECONDS",
                    default=30.0,
                ),
                timeout_seconds=_read_float(
                    mapping,
                    "VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS",
                    default=5.0,
                ),
                resize_max_edge_px=_read_optional_int_with_default(
                    mapping,
                    "VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX",
                    default=1280,
                ),
            ),
            moonshine=MoonshineSettings(
                model_name=_read_str_with_default(
                    mapping,
                    "VOCALIVE_MOONSHINE_MODEL",
                    default="base",
                ),
            ),
            openai=OpenAISettings(
                api_key=(
                    _read_optional_str_with_default(
                        mapping,
                        "VOCALIVE_OPENAI_API_KEY",
                        default=None,
                    )
                    or _read_optional_str_with_default(
                        mapping,
                        "OPENAI_API_KEY",
                        default=None,
                    )
                ),
                model_name=_read_str_with_default(
                    mapping,
                    "VOCALIVE_OPENAI_MODEL",
                    default="gpt-4o-mini-transcribe",
                ),
                base_url=_read_str_with_default(
                    mapping,
                    "VOCALIVE_OPENAI_BASE_URL",
                    default="https://api.openai.com/v1",
                ),
                timeout_seconds=_read_float(
                    mapping,
                    "VOCALIVE_OPENAI_TIMEOUT_SECONDS",
                    default=30.0,
                ),
            ),
            aivis=AivisSpeechSettings(
                base_url=_read_str_with_default(
                    mapping,
                    "VOCALIVE_AIVIS_BASE_URL",
                    default="http://127.0.0.1:10101",
                ),
                engine_mode=AivisEngineMode(
                    _read_str_with_default(
                        mapping,
                        "VOCALIVE_AIVIS_ENGINE_MODE",
                        default=AivisEngineMode.EXTERNAL.value,
                    ).strip().lower()
                ),
                engine_path=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_AIVIS_ENGINE_PATH",
                    default=None,
                ),
                cpu_num_threads=_read_optional_int_with_default(
                    mapping,
                    "VOCALIVE_AIVIS_CPU_NUM_THREADS",
                    default=None,
                ),
                startup_timeout_seconds=_read_float(
                    mapping,
                    "VOCALIVE_AIVIS_STARTUP_TIMEOUT_SECONDS",
                    default=60.0,
                ),
                speaker_id=_read_optional_int(mapping, "VOCALIVE_AIVIS_SPEAKER_ID"),
                speaker_name=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_AIVIS_SPEAKER_NAME",
                    default=None,
                ),
                style_name=_read_optional_str_with_default(
                    mapping,
                    "VOCALIVE_AIVIS_STYLE_NAME",
                    default=None,
                ),
                timeout_seconds=_read_float(
                    mapping,
                    "VOCALIVE_AIVIS_TIMEOUT_SECONDS",
                    default=30.0,
                ),
            ),
        )


def _lookup_raw(mapping: Mapping[str, str | None], name: str) -> str | None:
    raw_value = mapping.get(name)
    if raw_value is None:
        return None
    return str(raw_value)


def _read_str_with_default(
    mapping: Mapping[str, str | None],
    name: str,
    default: str,
) -> str:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    return raw_value


def _read_int(mapping: Mapping[str, str | None], name: str, default: int) -> int:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    return int(raw_value)


def _read_float(mapping: Mapping[str, str | None], name: str, default: float) -> float:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    return float(raw_value)


def _read_optional_float(
    mapping: Mapping[str, str | None],
    name: str,
) -> float | None:
    raw_value = _lookup_raw(mapping, name)
    if raw_value in {None, ""}:
        return None
    return float(raw_value)


def _read_optional_int(
    mapping: Mapping[str, str | None],
    name: str,
) -> int | None:
    raw_value = _lookup_raw(mapping, name)
    if raw_value in {None, ""}:
        return None
    return int(raw_value)


def _read_optional_device(
    mapping: Mapping[str, str | None],
    name: str,
) -> str | int | None:
    raw_value = _lookup_raw(mapping, name)
    if raw_value in {None, ""}:
        return None
    normalized_value = raw_value.strip()
    if normalized_value.isdigit():
        return int(normalized_value)
    return normalized_value


def _read_bool(
    mapping: Mapping[str, str | None],
    name: str,
    default: bool,
) -> bool:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    if normalized_value in _TRUTHY_VALUES:
        return True
    if normalized_value in _FALSY_VALUES:
        return False
    raise ValueError(
        f"Environment variable {name} must be one of: "
        f"{', '.join(sorted(_TRUTHY_VALUES | _FALSY_VALUES))}"
    )


def _read_optional_int_with_default(
    mapping: Mapping[str, str | None],
    name: str,
    default: int | None,
) -> int | None:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    if raw_value == "":
        return None
    return int(raw_value)


def _read_optional_str_with_default(
    mapping: Mapping[str, str | None],
    name: str,
    default: str | None,
) -> str | None:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip()
    if not normalized_value:
        return None
    return normalized_value


def _read_str_tuple(
    mapping: Mapping[str, str | None],
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _read_application_audio_mode(
    mapping: Mapping[str, str | None],
    name: str,
    default: ApplicationAudioMode,
) -> ApplicationAudioMode:
    raw_value = _lookup_raw(mapping, name)
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


def _read_microphone_interrupt_mode(
    mapping: Mapping[str, str | None],
    name: str,
    default: MicrophoneInterruptMode,
) -> MicrophoneInterruptMode:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    normalized_value = "_".join(raw_value.strip().lower().replace("-", " ").split())
    aliases = {
        "always": MicrophoneInterruptMode.ALWAYS,
        "explicit": MicrophoneInterruptMode.EXPLICIT,
        "disabled": MicrophoneInterruptMode.DISABLED,
        "never": MicrophoneInterruptMode.DISABLED,
        "off": MicrophoneInterruptMode.DISABLED,
    }
    mode = aliases.get(normalized_value)
    if mode is None:
        supported_values = ", ".join(mode.value for mode in MicrophoneInterruptMode)
        raise ValueError(
            f"Unsupported microphone interrupt mode: {raw_value!r}. "
            f"Supported values: {supported_values}"
        )
    return mode


def _read_conversation_window_reset_policy(
    mapping: Mapping[str, str | None],
    name: str,
    default: ConversationWindowResetPolicy,
) -> ConversationWindowResetPolicy:
    raw_value = _lookup_raw(mapping, name)
    if raw_value is None:
        return default
    normalized_value = "_".join(raw_value.strip().lower().replace("-", " ").split())
    aliases = {
        "clear": ConversationWindowResetPolicy.CLEAR,
        "resume_summary": ConversationWindowResetPolicy.RESUME_SUMMARY,
        "summary": ConversationWindowResetPolicy.RESUME_SUMMARY,
        "resume": ConversationWindowResetPolicy.RESUME_SUMMARY,
    }
    policy = aliases.get(normalized_value)
    if policy is None:
        supported_values = ", ".join(policy.value for policy in ConversationWindowResetPolicy)
        raise ValueError(
            f"Unsupported conversation window reset policy: {raw_value!r}. "
            f"Supported values: {supported_values}"
        )
    return policy


def normalize_controller_values(
    values: Mapping[str, str | None],
    *,
    include_defaults: bool = True,
) -> dict[str, str | None]:
    normalized: dict[str, str | None] = (
        controller_default_values() if include_defaults else {}
    )
    for env_name, raw_value in values.items():
        if env_name not in _CONTROLLER_SETTING_INDEX:
            continue
        normalized[env_name] = None if raw_value is None else str(raw_value)
    return normalized


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
