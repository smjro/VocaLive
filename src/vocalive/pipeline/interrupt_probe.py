from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from vocalive.models import AudioSegment, TurnContext
from vocalive.pipeline.reply_policy import looks_like_explicit_assistant_address
from vocalive.stt.base import SpeechToTextEngine
from vocalive.util.logging import log_event


@dataclass(frozen=True)
class ExplicitInterruptProbeRequest:
    segment: AudioSegment
    session_id: str
    turn_id: int
    reason: str


class ExplicitInterruptProbeManager:
    def __init__(
        self,
        *,
        get_stt_engine: Callable[[], SpeechToTextEngine],
        get_active_context: Callable[[], TurnContext | None],
        get_active_stage: Callable[[], str | None],
        get_assistant_names: Callable[[], tuple[str, ...]],
        get_session_id: Callable[[], str],
        interrupt_active_turn: Callable[[str], Awaitable[None]],
        logger: logging.Logger,
    ) -> None:
        self._get_stt_engine = get_stt_engine
        self._get_active_context = get_active_context
        self._get_active_stage = get_active_stage
        self._get_assistant_names = get_assistant_names
        self._get_session_id = get_session_id
        self._interrupt_active_turn = interrupt_active_turn
        self._logger = logger
        self._task: asyncio.Task[None] | None = None
        self._pending_request: ExplicitInterruptProbeRequest | None = None
        self._probed_transcript_hints: OrderedDict[int, str] = OrderedDict()

    def apply_cached_transcript_hint(self, segment: AudioSegment) -> AudioSegment:
        transcript_hint = self._probed_transcript_hints.pop(id(segment), None)
        if transcript_hint is None or segment.transcript_hint:
            return segment
        return _with_transcript_hint(segment, transcript_hint)

    async def probe_segment(
        self,
        segment: AudioSegment,
        *,
        session_id: str,
        turn_id: int,
    ) -> tuple[AudioSegment, bool]:
        transcription_text = (segment.transcript_hint or "").strip()
        if not transcription_text:
            probe_context = TurnContext(session_id=session_id, turn_id=turn_id)
            try:
                transcription = await self._get_stt_engine().transcribe(segment, probe_context)
            except Exception as exc:
                log_event(
                    self._logger,
                    "interrupt_probe_failed",
                    session_id=session_id,
                    stage=self._get_active_stage(),
                    audio_source=segment.source,
                    error=str(exc),
                )
                return segment, False
            transcription_text = transcription.text.strip()
            if not transcription_text:
                return segment, False
            segment = _with_transcript_hint(segment, transcription_text)

        should_interrupt = looks_like_explicit_assistant_address(
            transcription_text,
            assistant_names=self._get_assistant_names(),
        )
        return segment, should_interrupt

    def build_request(
        self,
        segment: AudioSegment,
        *,
        reason: str,
    ) -> ExplicitInterruptProbeRequest | None:
        active_context = self._get_active_context()
        if active_context is None:
            return None
        return ExplicitInterruptProbeRequest(
            segment=segment,
            session_id=active_context.session_id,
            turn_id=active_context.turn_id,
            reason=reason,
        )

    def schedule(self, request: ExplicitInterruptProbeRequest) -> None:
        task = self._task
        if task is not None and not task.done():
            self._pending_request = request
            return
        self._start(request)

    async def discard(self) -> None:
        self._pending_request = None
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _start(self, request: ExplicitInterruptProbeRequest) -> None:
        task = asyncio.create_task(
            self._run(request),
            name=f"vocalive-explicit-interrupt-probe-{request.turn_id}",
        )
        self._task = task
        task.add_done_callback(self._finalize)

    async def _run(self, request: ExplicitInterruptProbeRequest) -> None:
        probe_context = TurnContext(
            session_id=request.session_id,
            turn_id=request.turn_id + 1,
        )
        try:
            transcription = await self._get_stt_engine().transcribe(
                request.segment,
                probe_context,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                self._logger,
                "interrupt_probe_failed",
                session_id=request.session_id,
                stage=self._get_active_stage(),
                audio_source=request.segment.source,
                error=str(exc),
            )
            return
        transcription_text = transcription.text.strip()
        if not transcription_text:
            return
        self._remember_probed_transcript_hint(request.segment, transcription_text)
        if not looks_like_explicit_assistant_address(
            transcription_text,
            assistant_names=self._get_assistant_names(),
        ):
            return
        active_context = self._get_active_context()
        if active_context is None:
            return
        if (
            active_context.session_id != request.session_id
            or active_context.turn_id != request.turn_id
            or self._get_active_stage() not in {"tts", "playback"}
        ):
            return
        await self._interrupt_active_turn(request.reason)

    def _finalize(self, task: asyncio.Task[None]) -> None:
        if self._task is task:
            self._task = None
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log_event(
                self._logger,
                "interrupt_probe_failed",
                session_id=self._get_session_id(),
                stage=self._get_active_stage(),
                error=str(exc),
            )
        pending_request = self._pending_request
        self._pending_request = None
        if pending_request is not None:
            self._start(pending_request)

    def _remember_probed_transcript_hint(
        self,
        segment: AudioSegment,
        transcript_hint: str,
    ) -> None:
        self._probed_transcript_hints.pop(id(segment), None)
        self._probed_transcript_hints[id(segment)] = transcript_hint
        while len(self._probed_transcript_hints) > 32:
            self._probed_transcript_hints.popitem(last=False)


def _with_transcript_hint(segment: AudioSegment, transcript_hint: str) -> AudioSegment:
    return AudioSegment(
        pcm=segment.pcm,
        sample_rate_hz=segment.sample_rate_hz,
        channels=segment.channels,
        sample_width_bytes=segment.sample_width_bytes,
        transcript_hint=transcript_hint,
        source=segment.source,
        source_label=segment.source_label,
    )
