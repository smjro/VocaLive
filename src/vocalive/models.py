from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .util.time import utc_timestamp


MessageRole = Literal["system", "user", "assistant"]


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

    @classmethod
    def from_text(cls, text: str, sample_rate_hz: int = 16_000) -> "AudioSegment":
        encoded_text = text.encode("utf-8")
        return cls(
            pcm=encoded_text,
            sample_rate_hz=sample_rate_hz,
            transcript_hint=text,
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
class ConversationRequest:
    context: TurnContext
    messages: tuple[ConversationMessage, ...]


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
