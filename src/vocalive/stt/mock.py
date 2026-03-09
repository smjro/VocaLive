from __future__ import annotations

import asyncio

from vocalive.models import AudioSegment, Transcription, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.stt.base import SpeechToTextEngine


class MockSpeechToTextEngine(SpeechToTextEngine):
    name = "mock-stt"

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    async def transcribe(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> Transcription:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        text = segment.transcript_hint
        if not text:
            text = segment.pcm.decode("utf-8", errors="ignore").strip()
        if not text:
            raise ValueError("mock STT needs transcript_hint or decodable PCM bytes")
        return Transcription(text=text, provider=self.name, confidence=1.0, language="ja")
