from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from vocalive.models import AudioSegment


class DebouncedSegmentBuffer:
    def __init__(
        self,
        *,
        delay_ms: Callable[[], float],
        on_ready: Callable[[AudioSegment, str], Awaitable[bool]],
        task_name_prefix: str,
    ) -> None:
        self._delay_ms = delay_ms
        self._on_ready = on_ready
        self._task_name_prefix = task_name_prefix
        self._lock = asyncio.Lock()
        self._pending_segment: AudioSegment | None = None
        self._generation = 0
        self._pending_task: asyncio.Task[None] | None = None

    async def submit(
        self,
        segment: AudioSegment,
        *,
        ready_reason: str,
        flush_replaced_reason: str,
    ) -> bool:
        segment_to_flush: AudioSegment | None = None
        async with self._lock:
            pending_segment = self._pending_segment
            if pending_segment is None:
                self._pending_segment = segment
                self._schedule_flush_locked(ready_reason)
                return True
            if segments_can_merge(pending_segment, segment):
                self._pending_segment = merge_segments(pending_segment, segment)
                self._schedule_flush_locked(ready_reason)
                return True
            segment_to_flush = pending_segment
            self._pending_segment = segment
            self._schedule_flush_locked(ready_reason)
        if segment_to_flush is None:
            return True
        return await self._on_ready(segment_to_flush, flush_replaced_reason)

    async def discard(self) -> None:
        async with self._lock:
            self._pending_segment = None
            self._generation += 1
            pending_task = self._pending_task
            self._pending_task = None
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending_task

    def _schedule_flush_locked(self, ready_reason: str) -> None:
        self._generation += 1
        pending_task = self._pending_task
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
        generation = self._generation
        self._pending_task = asyncio.create_task(
            self._flush_after_delay(generation, ready_reason),
            name=f"{self._task_name_prefix}-{generation}",
        )

    async def _flush_after_delay(self, generation: int, ready_reason: str) -> None:
        try:
            await asyncio.sleep(max(0.0, self._delay_ms()) / 1000.0)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if generation != self._generation:
                return
            segment_to_flush = self._pending_segment
            self._pending_segment = None
            self._pending_task = None
        if segment_to_flush is None:
            return
        await self._on_ready(segment_to_flush, ready_reason)

    @property
    def has_pending_segment(self) -> bool:
        return self._pending_segment is not None


def segment_duration_ms(segment: AudioSegment) -> float:
    bytes_per_second = segment.sample_rate_hz * segment.channels * segment.sample_width_bytes
    if bytes_per_second <= 0 or not segment.pcm:
        return 0.0
    return (len(segment.pcm) / bytes_per_second) * 1000.0


def segments_can_merge(first: AudioSegment, second: AudioSegment) -> bool:
    return (
        first.sample_rate_hz == second.sample_rate_hz
        and first.channels == second.channels
        and first.sample_width_bytes == second.sample_width_bytes
        and first.source == second.source
        and first.source_label == second.source_label
    )


def merge_segments(first: AudioSegment, second: AudioSegment) -> AudioSegment:
    if not segments_can_merge(first, second):
        raise ValueError("audio segments are not mergeable")
    return AudioSegment(
        pcm=first.pcm + second.pcm,
        sample_rate_hz=first.sample_rate_hz,
        channels=first.channels,
        sample_width_bytes=first.sample_width_bytes,
        transcript_hint=merge_transcript_hints(first.transcript_hint, second.transcript_hint),
        source=first.source,
        source_label=first.source_label,
    )


def with_transcript_hint(segment: AudioSegment, transcript_hint: str) -> AudioSegment:
    return AudioSegment(
        pcm=segment.pcm,
        sample_rate_hz=segment.sample_rate_hz,
        channels=segment.channels,
        sample_width_bytes=segment.sample_width_bytes,
        transcript_hint=transcript_hint,
        source=segment.source,
        source_label=segment.source_label,
    )


def merge_transcript_hints(first: str | None, second: str | None) -> str | None:
    normalized_parts = [part.strip() for part in (first, second) if part and part.strip()]
    if not normalized_parts:
        return None
    return " ".join(normalized_parts)
