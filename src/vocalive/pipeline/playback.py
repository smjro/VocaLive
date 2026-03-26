from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from vocalive.audio.output import AudioOutput
from vocalive.models import AssistantResponse, SynthesizedSpeech, TurnContext
from vocalive.pipeline.events import ConversationEvent
from vocalive.pipeline.interruption import CancellationToken
from vocalive.tts.base import TextToSpeechEngine
from vocalive.util.metrics import MetricsRecorder
from vocalive.util.time import monotonic_ms


_SENTENCE_END_MARKERS = frozenset(".!?。！？")


@dataclass(frozen=True)
class SynthesizedChunk:
    speech: SynthesizedSpeech
    duration_ms: float


class PlaybackRunner:
    def __init__(
        self,
        *,
        get_tts_engine: Callable[[], TextToSpeechEngine],
        get_audio_output: Callable[[], AudioOutput],
        metrics: MetricsRecorder,
        emit_event: Callable[[ConversationEvent], None],
        set_active_stage: Callable[[str | None], None],
    ) -> None:
        self._get_tts_engine = get_tts_engine
        self._get_audio_output = get_audio_output
        self._metrics = metrics
        self._emit_event = emit_event
        self._set_active_stage = set_active_stage

    async def play_response(
        self,
        response: AssistantResponse,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> None:
        chunks = split_response_for_playback(response.text)
        tts_duration_ms = 0.0
        playback_duration_ms = 0.0
        pending_chunk: asyncio.Task[SynthesizedChunk] | None = None
        try:
            if chunks:
                pending_chunk = asyncio.create_task(
                    self._synthesize_chunk(
                        chunks[0],
                        context=context,
                        cancellation=cancellation,
                    ),
                    name=f"vocalive-tts-{context.turn_id}-0",
                )
            for index, _ in enumerate(chunks):
                cancellation.raise_if_cancelled()
                assert pending_chunk is not None
                self._set_active_stage("tts")
                synthesized_chunk = await pending_chunk
                pending_chunk = None
                tts_duration_ms += synthesized_chunk.duration_ms
                next_index = index + 1
                if next_index < len(chunks):
                    pending_chunk = asyncio.create_task(
                        self._synthesize_chunk(
                            chunks[next_index],
                            context=context,
                            cancellation=cancellation,
                        ),
                        name=f"vocalive-tts-{context.turn_id}-{next_index}",
                    )
                playback_started_ms = monotonic_ms()
                self._set_active_stage("playback")
                self._emit_event(
                    ConversationEvent(
                        type="assistant_chunk_started",
                        session_id=context.session_id,
                        turn_id=context.turn_id,
                        text=synthesized_chunk.speech.text,
                        chunk_index=index,
                        chunk_count=len(chunks),
                        duration_ms=estimate_playback_duration_ms(synthesized_chunk.speech),
                    )
                )
                await self._get_audio_output().play(
                    synthesized_chunk.speech,
                    cancellation=cancellation,
                )
                playback_duration_ms += monotonic_ms() - playback_started_ms
        finally:
            await discard_background_task(pending_chunk)
            self._metrics.record_duration(
                stage="tts",
                duration_ms=tts_duration_ms,
                context=context,
            )
            self._metrics.record_duration(
                stage="playback",
                duration_ms=playback_duration_ms,
                context=context,
            )

    async def _synthesize_chunk(
        self,
        text: str,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> SynthesizedChunk:
        started_ms = monotonic_ms()
        speech = await self._get_tts_engine().synthesize(
            text,
            context,
            cancellation=cancellation,
        )
        return SynthesizedChunk(
            speech=speech,
            duration_ms=monotonic_ms() - started_ms,
        )


def normalize_assistant_response(response: AssistantResponse) -> AssistantResponse:
    normalized_text = normalize_assistant_response_text(response.text)
    if normalized_text == response.text:
        return response
    return AssistantResponse(text=normalized_text, provider=response.provider)


def normalize_assistant_response_text(text: str) -> str:
    stripped_text = text.strip()
    if not stripped_text:
        return stripped_text
    lines = [line.strip() for line in stripped_text.splitlines() if line.strip()]
    if not lines:
        return stripped_text
    return " ".join(lines)


def split_response_for_playback(text: str) -> tuple[str, ...]:
    normalized_text = text.strip()
    if not normalized_text:
        return tuple()
    chunks: list[str] = []
    for paragraph in normalized_text.splitlines():
        normalized_paragraph = paragraph.strip()
        if not normalized_paragraph:
            continue
        current_chunk: list[str] = []
        for index, char in enumerate(normalized_paragraph):
            current_chunk.append(char)
            if char not in _SENTENCE_END_MARKERS:
                continue
            next_char = normalized_paragraph[index + 1] if index + 1 < len(normalized_paragraph) else None
            if next_char is not None and next_char in _SENTENCE_END_MARKERS:
                continue
            completed_chunk = "".join(current_chunk).strip()
            if completed_chunk:
                chunks.append(completed_chunk)
            current_chunk.clear()
        trailing_chunk = "".join(current_chunk).strip()
        if trailing_chunk:
            chunks.append(trailing_chunk)
    return tuple(chunks) if chunks else (normalized_text,)


def estimate_playback_duration_ms(speech: SynthesizedSpeech) -> float | None:
    if speech.duration_ms is not None and speech.duration_ms > 0:
        return speech.duration_ms
    bytes_per_second = speech.sample_rate_hz * speech.channels * speech.sample_width_bytes
    if bytes_per_second <= 0 or not speech.audio:
        return None
    return (len(speech.audio) / bytes_per_second) * 1000.0


async def discard_background_task(task: asyncio.Task[SynthesizedChunk] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
