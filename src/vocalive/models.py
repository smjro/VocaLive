from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

from .util.time import utc_timestamp


MessageRole = Literal["system", "user", "assistant", "application"]
AudioSource = Literal["user", "application_audio"]


@dataclass(frozen=True)
class TurnContext:
    session_id: str
    turn_id: int


@dataclass(frozen=True)
class AudioSegment:
    pcm: bytes
    sample_rate_hz: int
    channels: int = 1
    sample_width_bytes: int = 2
    transcript_hint: str | None = None
    source: AudioSource = "user"
    source_label: str | None = None

    @classmethod
    def from_text(
        cls,
        text: str,
        sample_rate_hz: int = 16_000,
        source: AudioSource = "user",
        source_label: str | None = None,
    ) -> "AudioSegment":
        encoded_text = text.encode("utf-8")
        return cls(
            pcm=encoded_text,
            sample_rate_hz=sample_rate_hz,
            transcript_hint=text,
            source=source,
            source_label=source_label,
        )


@dataclass(frozen=True)
class Transcription:
    text: str
    provider: str
    confidence: float = 1.0
    language: str | None = None


@dataclass(frozen=True)
class ConversationMessage:
    role: MessageRole
    content: str
    created_at: str = field(default_factory=utc_timestamp)


@dataclass(frozen=True)
class ConversationTextPart:
    text: str


@dataclass(frozen=True)
class ConversationInlineDataPart:
    mime_type: str
    data: bytes


ConversationRequestPart = Union[ConversationTextPart, ConversationInlineDataPart]


@dataclass(frozen=True)
class ConversationRequest:
    context: TurnContext
    messages: tuple[ConversationMessage, ...]
    current_user_parts: tuple[ConversationRequestPart, ...] = ()


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    provider: str


@dataclass(frozen=True)
class SynthesizedSpeech:
    text: str
    provider: str
    audio: bytes
    sample_rate_hz: int
    channels: int = 1
    sample_width_bytes: int = 2
    mime_type: str = "audio/L16"
    file_extension: str | None = None
