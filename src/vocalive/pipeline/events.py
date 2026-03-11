from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vocalive.util.time import utc_timestamp


@dataclass(frozen=True)
class ConversationEvent:
    type: str
    session_id: str
    turn_id: int | None = None
    text: str | None = None
    stage: str | None = None
    reason: str | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    duration_ms: float | None = None
    timestamp: str = field(default_factory=utc_timestamp)


class ConversationEventSink(ABC):
    @abstractmethod
    def emit(self, event: ConversationEvent) -> None:
        """Publish a conversation lifecycle event."""

    def close(self) -> None:
        """Release any resources owned by the sink."""


class NullConversationEventSink(ConversationEventSink):
    def emit(self, event: ConversationEvent) -> None:
        return None
