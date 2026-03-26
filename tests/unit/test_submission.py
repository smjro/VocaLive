from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.models import AudioSegment
from vocalive.pipeline.submission import DebouncedSegmentBuffer, segment_duration_ms


class DebouncedSegmentBufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_merges_segments_and_flushes_after_delay(self) -> None:
        flushed: list[tuple[AudioSegment, str]] = []
        buffer = DebouncedSegmentBuffer(
            delay_ms=lambda: 20.0,
            on_ready=lambda segment, reason: _record_ready(flushed, segment, reason),
            task_name_prefix="test-live-segment",
        )
        self.addAsyncCleanup(buffer.discard)

        await buffer.submit(
            AudioSegment.from_text("first"),
            ready_reason="debounced_ready",
            flush_replaced_reason="debounced_flushed",
        )
        await asyncio.sleep(0.005)
        await buffer.submit(
            AudioSegment.from_text("second"),
            ready_reason="debounced_ready",
            flush_replaced_reason="debounced_flushed",
        )
        await asyncio.sleep(0.05)

        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0][0].transcript_hint, "first second")
        self.assertEqual(flushed[0][1], "debounced_ready")
        self.assertFalse(buffer.has_pending_segment)

    async def test_flushes_replaced_segment_when_new_segment_is_not_mergeable(self) -> None:
        flushed: list[tuple[AudioSegment, str]] = []
        buffer = DebouncedSegmentBuffer(
            delay_ms=lambda: 50.0,
            on_ready=lambda segment, reason: _record_ready(flushed, segment, reason),
            task_name_prefix="test-application-segment",
        )
        self.addAsyncCleanup(buffer.discard)

        await buffer.submit(
            AudioSegment.from_text("first"),
            ready_reason="debounced_ready",
            flush_replaced_reason="debounced_flushed",
        )
        await buffer.submit(
            AudioSegment.from_text(
                "second",
                source="application_audio",
                source_label="Steam",
            ),
            ready_reason="debounced_ready",
            flush_replaced_reason="debounced_flushed",
        )

        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0][0].transcript_hint, "first")
        self.assertEqual(flushed[0][1], "debounced_flushed")
        self.assertTrue(buffer.has_pending_segment)

    def test_segment_duration_ms_uses_pcm_length(self) -> None:
        segment = AudioSegment(
            pcm=b"\0\0" * 1600,
            sample_rate_hz=16_000,
            channels=1,
            sample_width_bytes=2,
        )

        self.assertEqual(segment_duration_ms(segment), 100.0)


async def _record_ready(
    flushed: list[tuple[AudioSegment, str]],
    segment: AudioSegment,
    reason: str,
) -> bool:
    flushed.append((segment, reason))
    return True


if __name__ == "__main__":
    unittest.main()
