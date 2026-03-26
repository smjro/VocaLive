from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.models import AudioSegment, Transcription, TurnContext
from vocalive.pipeline.interrupt_probe import ExplicitInterruptProbeManager
from vocalive.stt.base import SpeechToTextEngine


class StubSpeechToTextEngine(SpeechToTextEngine):
    name = "stub-stt"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[TurnContext] = []

    async def transcribe(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation=None,
    ) -> Transcription:
        del segment, cancellation
        self.calls.append(context)
        return Transcription(text=self.text, provider=self.name, language="ja")


class ExplicitInterruptProbeManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_segment_adds_transcript_hint_and_detects_interrupt(self) -> None:
        stt_engine = StubSpeechToTextEngine("コハク、どうする？")
        interrupt_reasons: list[str] = []
        manager = ExplicitInterruptProbeManager(
            get_stt_engine=lambda: stt_engine,
            get_active_context=lambda: None,
            get_active_stage=lambda: "playback",
            get_assistant_names=lambda: ("コハク",),
            get_session_id=lambda: "session",
            interrupt_active_turn=lambda reason: _record_interrupt(
                interrupt_reasons,
                reason,
            ),
            logger=logging.getLogger("tests.interrupt_probe"),
        )

        segment = AudioSegment(pcm=b"\0\0", sample_rate_hz=16_000)
        prepared_segment, should_interrupt = await manager.probe_segment(
            segment,
            session_id="session",
            turn_id=5,
        )

        self.assertEqual(stt_engine.calls, [TurnContext(session_id="session", turn_id=5)])
        self.assertEqual(prepared_segment.transcript_hint, "コハク、どうする？")
        self.assertTrue(should_interrupt)
        self.assertEqual(interrupt_reasons, [])

    def test_build_request_returns_none_without_active_context(self) -> None:
        manager = ExplicitInterruptProbeManager(
            get_stt_engine=lambda: StubSpeechToTextEngine("ignored"),
            get_active_context=lambda: None,
            get_active_stage=lambda: None,
            get_assistant_names=lambda: ("コハク",),
            get_session_id=lambda: "session",
            interrupt_active_turn=lambda reason: _record_interrupt([], reason),
            logger=logging.getLogger("tests.interrupt_probe"),
        )

        request = manager.build_request(
            AudioSegment.from_text("hello"),
            reason="utterance_submitted",
        )

        self.assertIsNone(request)


async def _record_interrupt(reasons: list[str], reason: str) -> None:
    reasons.append(reason)


if __name__ == "__main__":
    unittest.main()
