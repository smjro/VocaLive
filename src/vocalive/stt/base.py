from __future__ import annotations

from abc import ABC, abstractmethod

from vocalive.models import AudioSegment, Transcription, TurnContext
from vocalive.pipeline.interruption import CancellationToken


class SpeechToTextEngine(ABC):
    name: str

    @abstractmethod
    async def transcribe(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> Transcription:
        """Transcribe a single utterance."""
