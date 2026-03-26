from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.output import MemoryAudioOutput
from vocalive.models import AssistantResponse, SynthesizedSpeech, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.pipeline.playback import (
    PlaybackRunner,
    estimate_playback_duration_ms,
    normalize_assistant_response,
    split_response_for_playback,
)
from vocalive.tts.base import TextToSpeechEngine
from vocalive.util.metrics import InMemoryMetricsRecorder


class RecordingTextToSpeechEngine(TextToSpeechEngine):
    name = "recording-tts"

    def __init__(self) -> None:
        self.started_texts: list[str] = []

    async def synthesize(
        self,
        text: str,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> SynthesizedSpeech:
        del context
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.started_texts.append(text)
        return SynthesizedSpeech(
            text=text,
            provider=self.name,
            audio=text.encode("utf-8"),
            sample_rate_hz=24_000,
            duration_ms=250.0,
        )


class PlaybackTests(unittest.IsolatedAsyncioTestCase):
    def test_split_response_for_playback_handles_multiple_sentences(self) -> None:
        self.assertEqual(
            split_response_for_playback("Hello. World!\nNext line"),
            ("Hello.", "World!", "Next line"),
        )

    def test_normalize_assistant_response_flattens_multiline_text(self) -> None:
        response = AssistantResponse(text=" Hello\nworld ", provider="mock")

        normalized = normalize_assistant_response(response)

        self.assertEqual(normalized.text, "Hello world")
        self.assertEqual(normalized.provider, "mock")

    async def test_play_response_emits_chunk_events_and_records_metrics(self) -> None:
        tts_engine = RecordingTextToSpeechEngine()
        output = MemoryAudioOutput()
        metrics = InMemoryMetricsRecorder()
        events = []
        stages: list[str | None] = []
        runner = PlaybackRunner(
            get_tts_engine=lambda: tts_engine,
            get_audio_output=lambda: output,
            metrics=metrics,
            emit_event=events.append,
            set_active_stage=stages.append,
        )
        context = TurnContext(session_id="session", turn_id=7)
        cancellation = CancellationToken()

        await runner.play_response(
            AssistantResponse(text="One. Two.", provider="mock"),
            context,
            cancellation,
        )

        self.assertEqual(tts_engine.started_texts, ["One.", "Two."])
        self.assertEqual(output.completed_texts, ["One.", "Two."])
        self.assertEqual([event.type for event in events], ["assistant_chunk_started", "assistant_chunk_started"])
        self.assertEqual([metric.stage for metric in metrics.events], ["tts", "playback"])
        self.assertEqual(stages, ["tts", "playback", "tts", "playback"])

    def test_estimate_playback_duration_prefers_explicit_duration(self) -> None:
        speech = SynthesizedSpeech(
            text="hello",
            provider="mock",
            audio=b"1234",
            sample_rate_hz=24_000,
            duration_ms=123.0,
        )

        self.assertEqual(estimate_playback_duration_ms(speech), 123.0)


if __name__ == "__main__":
    unittest.main()
