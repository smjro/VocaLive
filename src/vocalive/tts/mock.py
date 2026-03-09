from __future__ import annotations

import asyncio

from vocalive.models import SynthesizedSpeech, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.tts.base import TextToSpeechEngine


class MockTextToSpeechEngine(TextToSpeechEngine):
    name = "mock-tts"

    def __init__(self, delay_seconds: float = 0.0, sample_rate_hz: int = 24_000) -> None:
        self.delay_seconds = delay_seconds
        self.sample_rate_hz = sample_rate_hz

    async def synthesize(
        self,
        text: str,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> SynthesizedSpeech:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return SynthesizedSpeech(
            text=text,
            provider=self.name,
            audio=text.encode("utf-8"),
            sample_rate_hz=self.sample_rate_hz,
        )
