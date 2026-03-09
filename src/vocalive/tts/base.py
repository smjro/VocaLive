from __future__ import annotations

from abc import ABC, abstractmethod

from vocalive.models import SynthesizedSpeech, TurnContext
from vocalive.pipeline.interruption import CancellationToken


class TextToSpeechEngine(ABC):
    name: str

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> SynthesizedSpeech:
        """Synthesize speech audio from text."""
