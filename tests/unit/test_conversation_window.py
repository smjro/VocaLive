from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import ConversationWindowSettings
from vocalive.conversation_window import ConversationWindowGate
from vocalive.models import AudioSegment


class _FakeClock:
    def __init__(self, initial_ms: float = 0.0) -> None:
        self.current_ms = initial_ms

    def set_ms(self, value: float) -> None:
        self.current_ms = value

    def __call__(self) -> float:
        return self.current_ms


class ConversationWindowGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_gate_waits_for_user_speech_to_reopen_after_closed_period(self) -> None:
        clock = _FakeClock()
        gate = ConversationWindowGate(
            ConversationWindowSettings(
                enabled=True,
                open_duration_seconds=10.0,
                closed_duration_seconds=20.0,
            ),
            now_ms=clock,
        )

        speech_start_calls: list[str] = []

        async def handler(source: str) -> None:
            speech_start_calls.append(source)

        wrapped_handler = gate.wrap_speech_start_handler(handler)
        assert wrapped_handler is not None

        self.assertTrue(gate.should_forward_segment(AudioSegment.from_text("open")))
        clock.set_ms(10_000.0)
        self.assertFalse(gate.should_forward_segment(AudioSegment.from_text("closed")))
        self.assertTrue(gate.consume_resume_summary_capture_request())
        clock.set_ms(30_000.0)
        self.assertFalse(gate.should_forward_segment(AudioSegment.from_text("still closed")))
        self.assertFalse(gate.consume_history_reset_request())

        await wrapped_handler("user")

        self.assertEqual(speech_start_calls, ["user"])
        self.assertTrue(gate.should_forward_segment(AudioSegment.from_text("open again")))
        self.assertTrue(gate.consume_history_reset_request())

    async def test_closed_window_ignores_speech_start_and_drops_next_user_segment(self) -> None:
        clock = _FakeClock()
        gate = ConversationWindowGate(
            ConversationWindowSettings(
                enabled=True,
                open_duration_seconds=5.0,
                closed_duration_seconds=20.0,
                start_open=False,
            ),
            now_ms=clock,
        )
        speech_start_calls: list[str] = []

        async def handler(source: str) -> None:
            speech_start_calls.append(source)

        wrapped_handler = gate.wrap_speech_start_handler(handler)
        assert wrapped_handler is not None

        await wrapped_handler("user")

        self.assertEqual(speech_start_calls, [])
        self.assertFalse(gate.should_forward_segment(AudioSegment.from_text("drop me")))

        clock.set_ms(20_000.0)
        self.assertFalse(gate.should_forward_segment(AudioSegment.from_text("still waiting")))

        await wrapped_handler("user")

        self.assertEqual(speech_start_calls, ["user"])
        self.assertTrue(gate.should_forward_segment(AudioSegment.from_text("allow me")))
        self.assertTrue(gate.consume_history_reset_request())

    async def test_application_audio_is_ungated_by_default(self) -> None:
        clock = _FakeClock()
        gate = ConversationWindowGate(
            ConversationWindowSettings(
                enabled=True,
                open_duration_seconds=5.0,
                closed_duration_seconds=20.0,
                start_open=False,
            ),
            now_ms=clock,
        )

        self.assertTrue(
            gate.should_forward_segment(
                AudioSegment.from_text(
                    "game audio",
                    source="application_audio",
                    source_label="game",
                )
            )
        )

    async def test_application_audio_can_be_gated_when_enabled(self) -> None:
        clock = _FakeClock()
        gate = ConversationWindowGate(
            ConversationWindowSettings(
                enabled=True,
                open_duration_seconds=5.0,
                closed_duration_seconds=20.0,
                start_open=False,
                apply_to_application_audio=True,
            ),
            now_ms=clock,
        )
        speech_start_calls: list[str] = []

        async def handler(source: str) -> None:
            speech_start_calls.append(source)

        wrapped_handler = gate.wrap_speech_start_handler(handler)
        assert wrapped_handler is not None

        await wrapped_handler("application_audio")

        self.assertEqual(speech_start_calls, [])
        self.assertFalse(
            gate.should_forward_segment(
                AudioSegment.from_text(
                    "game audio",
                    source="application_audio",
                    source_label="game",
                )
            )
        )

    async def test_application_audio_does_not_reopen_window_while_waiting_for_user_speech(
        self,
    ) -> None:
        clock = _FakeClock()
        gate = ConversationWindowGate(
            ConversationWindowSettings(
                enabled=True,
                open_duration_seconds=5.0,
                closed_duration_seconds=20.0,
                start_open=False,
                apply_to_application_audio=True,
            ),
            now_ms=clock,
        )
        speech_start_calls: list[str] = []

        async def handler(source: str) -> None:
            speech_start_calls.append(source)

        wrapped_handler = gate.wrap_speech_start_handler(handler)
        assert wrapped_handler is not None

        clock.set_ms(20_000.0)
        await wrapped_handler("application_audio")

        self.assertEqual(speech_start_calls, [])
        self.assertFalse(
            gate.should_forward_segment(
                AudioSegment.from_text(
                    "still blocked",
                    source="application_audio",
                    source_label="game",
                )
            )
        )

        await wrapped_handler("user")

        self.assertEqual(speech_start_calls, ["user"])
        self.assertTrue(
            gate.should_forward_segment(
                AudioSegment.from_text(
                    "now open",
                    source="application_audio",
                    source_label="game",
                )
            )
        )

    async def test_poll_state_marks_resume_summary_capture_when_window_closes(self) -> None:
        clock = _FakeClock()
        gate = ConversationWindowGate(
            ConversationWindowSettings(
                enabled=True,
                open_duration_seconds=5.0,
                closed_duration_seconds=20.0,
            ),
            now_ms=clock,
        )

        self.assertTrue(gate.poll_state())
        clock.set_ms(5_000.0)

        self.assertFalse(gate.poll_state())
        self.assertTrue(gate.consume_resume_summary_capture_request())

    async def test_summary_describes_window_configuration(self) -> None:
        gate = ConversationWindowGate(
            ConversationWindowSettings(
                enabled=True,
                open_duration_seconds=15.0,
                closed_duration_seconds=120.0,
                apply_to_application_audio=True,
            ),
            now_ms=_FakeClock(),
        )

        summary = gate.summary()

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("15s open", summary)
        self.assertIn("120s closed", summary)
        self.assertIn("reopens on user speech", summary)
        self.assertIn("microphone + app audio", summary)
