from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from vocalive.audio.output import AudioOutput
from vocalive.config.settings import (
    AppSettings,
    ApplicationAudioMode,
    InputProvider,
    MicrophoneInterruptMode,
)
from vocalive.llm.base import LanguageModel
from vocalive.models import (
    AssistantResponse,
    AudioSegment,
    AudioSource,
    ConversationInlineDataPart,
    ConversationMessage,
    ConversationRequest,
    ConversationRequestPart,
    ConversationTextPart,
    SynthesizedSpeech,
    TurnContext,
)
from vocalive.pipeline.context import build_compacted_messages
from vocalive.pipeline.events import ConversationEvent, ConversationEventSink, NullConversationEventSink
from vocalive.pipeline.interruption import (
    CancellationToken,
    InterruptionController,
    TurnCancelledError,
)
from vocalive.pipeline.queues import BoundedAsyncQueue
from vocalive.pipeline.reply_policy import (
    ReplyDecision,
    decide_reply,
    looks_like_explicit_assistant_address,
)
from vocalive.pipeline.session import ConversationSession
from vocalive.screen.base import ScreenCaptureEngine
from vocalive.stt.base import SpeechToTextEngine
from vocalive.tts.base import TextToSpeechEngine
from vocalive.util.logging import get_logger, log_event
from vocalive.util.metrics import InMemoryMetricsRecorder, MetricsRecorder, timed_stage
from vocalive.util.time import monotonic_ms


_SENTENCE_END_MARKERS = frozenset(".!?。！？")


@dataclass(frozen=True)
class _SynthesizedChunk:
    speech: SynthesizedSpeech
    duration_ms: float


class ConversationOrchestrator:
    def __init__(
        self,
        settings: AppSettings,
        stt_engine: SpeechToTextEngine,
        language_model: LanguageModel,
        tts_engine: TextToSpeechEngine,
        audio_output: AudioOutput,
        event_sink: ConversationEventSink | None = None,
        screen_capture_engine: ScreenCaptureEngine | None = None,
        logger: logging.Logger | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.stt_engine = stt_engine
        self.language_model = language_model
        self.tts_engine = tts_engine
        self.audio_output = audio_output
        self.event_sink = event_sink or NullConversationEventSink()
        self.screen_capture_engine = screen_capture_engine
        self.logger = logger or get_logger("vocalive.orchestrator")
        self.metrics = metrics or InMemoryMetricsRecorder()
        self.session = ConversationSession(session_id=settings.session_id)
        self._queue = BoundedAsyncQueue[AudioSegment](
            maxsize=settings.queue.ingress_maxsize,
            overflow_strategy=settings.queue.overflow_strategy,
        )
        self._application_context_queue = BoundedAsyncQueue[AudioSegment](
            maxsize=settings.queue.ingress_maxsize,
            overflow_strategy=settings.queue.overflow_strategy,
        )
        self._interruptions = InterruptionController()
        self._idle_event = asyncio.Event()
        self._work_available = asyncio.Event()
        self._pending_submission_lock = asyncio.Lock()
        self._queue_submission_lock = asyncio.Lock()
        self._pending_live_segment: AudioSegment | None = None
        self._pending_live_segment_generation = 0
        self._pending_live_segment_task: asyncio.Task[None] | None = None
        self._idle_event.set()
        self._worker_task: asyncio.Task[None] | None = None
        self._turn_counter = 0
        self._active_context: TurnContext | None = None
        self._active_stage: str | None = None
        self._last_assistant_response_ms: float | None = None
        self._last_application_audio_submission_ms: float | None = None
        self._last_screen_observation_ms: float | None = None
        self._last_screen_capture_fingerprint: str | None = None

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._run(), name="vocalive-orchestrator")

    async def stop(self) -> None:
        await self._discard_pending_live_segment()
        await self._interrupt_active_turn(reason="shutdown", force_stop_audio=True)
        worker_task = self._worker_task
        self._worker_task = None
        if worker_task is None:
            return
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    async def submit_utterance(self, segment: AudioSegment) -> bool:
        application_audio_submission_ms = self._begin_application_audio_submission(segment)
        if segment.source == "application_audio" and application_audio_submission_ms is None:
            return True
        if self._should_capture_application_audio_as_context(segment):
            accepted = await self.submit_application_context(segment)
        elif self._should_debounce_live_segment(segment):
            accepted = await self._submit_debounced_live_segment(segment)
        else:
            accepted = await self._queue_turn_segment(segment, reason="utterance_submitted")
        if segment.source == "application_audio" and accepted and application_audio_submission_ms is not None:
            self._last_application_audio_submission_ms = application_audio_submission_ms
        return accepted

    async def _queue_turn_segment(self, segment: AudioSegment, *, reason: str) -> bool:
        self._idle_event.clear()
        async with self._queue_submission_lock:
            queued_segment, should_interrupt = await self._prepare_segment_for_queue(segment)
            if should_interrupt:
                await self._interrupt_active_turn(reason=reason)
            accepted = await self._queue.put(queued_segment)
        if not accepted:
            self._log_queue_overflow(queue_name="conversation", queue_size=self._queue.qsize())
            self._set_idle_if_drained()
            return False
        self._work_available.set()
        return True

    async def _submit_debounced_live_segment(self, segment: AudioSegment) -> bool:
        self._idle_event.clear()
        segment_to_flush: AudioSegment | None = None
        async with self._pending_submission_lock:
            pending_segment = self._pending_live_segment
            if pending_segment is None:
                self._pending_live_segment = segment
                self._schedule_pending_live_segment_flush_locked()
                return True
            if _segments_can_merge(pending_segment, segment):
                self._pending_live_segment = _merge_segments(pending_segment, segment)
                self._schedule_pending_live_segment_flush_locked()
                return True
            segment_to_flush = pending_segment
            self._pending_live_segment = segment
            self._schedule_pending_live_segment_flush_locked()
        if segment_to_flush is None:
            return True
        return await self._queue_turn_segment(
            segment_to_flush,
            reason="debounced_utterance_flushed",
        )

    def _schedule_pending_live_segment_flush_locked(self) -> None:
        self._pending_live_segment_generation += 1
        pending_task = self._pending_live_segment_task
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
        generation = self._pending_live_segment_generation
        self._pending_live_segment_task = asyncio.create_task(
            self._flush_pending_live_segment_after_delay(generation),
            name=f"vocalive-pending-live-segment-{generation}",
        )

    async def _flush_pending_live_segment_after_delay(self, generation: int) -> None:
        try:
            await asyncio.sleep(max(0.0, self.settings.reply.debounce_ms) / 1000.0)
        except asyncio.CancelledError:
            return
        segment_to_flush: AudioSegment | None = None
        async with self._pending_submission_lock:
            if generation != self._pending_live_segment_generation:
                return
            segment_to_flush = self._pending_live_segment
            self._pending_live_segment = None
            self._pending_live_segment_task = None
        if segment_to_flush is None:
            self._set_idle_if_drained()
            return
        await self._queue_turn_segment(
            segment_to_flush,
            reason="debounced_utterance_ready",
        )

    async def _discard_pending_live_segment(self) -> None:
        async with self._pending_submission_lock:
            self._pending_live_segment = None
            self._pending_live_segment_generation += 1
            pending_task = self._pending_live_segment_task
            self._pending_live_segment_task = None
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending_task
        self._set_idle_if_drained()

    async def submit_application_context(self, segment: AudioSegment) -> bool:
        if segment.source != "application_audio":
            raise ValueError("application context submission requires application_audio segments")
        self._idle_event.clear()
        accepted = await self._application_context_queue.put(segment)
        if not accepted:
            self._log_queue_overflow(
                queue_name="application_context",
                queue_size=self._application_context_queue.qsize(),
            )
            self._set_idle_if_drained()
            return False
        self._work_available.set()
        return True

    async def handle_user_speech_start(self, source: AudioSource = "user") -> None:
        if self._active_stage not in {"tts", "playback"}:
            return
        if source == "application_audio":
            await self._interrupt_active_turn(reason="speech_started")
            return
        if self.settings.input.interrupt_mode is not MicrophoneInterruptMode.ALWAYS:
            return
        await self._interrupt_active_turn(reason="speech_started")

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def _run(self) -> None:
        while True:
            segment, task_done = await self._await_next_segment()
            self._turn_counter += 1
            context = TurnContext(
                session_id=self.session.session_id,
                turn_id=self._turn_counter,
            )
            cancellation = self._interruptions.begin_turn()
            self._active_context = context
            self._active_stage = None
            try:
                await self._process_turn(segment=segment, context=context, cancellation=cancellation)
            except TurnCancelledError:
                self._emit_event(
                    ConversationEvent(
                        type="turn_cancelled",
                        session_id=context.session_id,
                        turn_id=context.turn_id,
                    )
                )
                log_event(
                    self.logger,
                    "turn_cancelled",
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                )
            except Exception as exc:
                log_event(
                    self.logger,
                    "turn_failed",
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    error=str(exc),
                )
            finally:
                if self._active_context == context:
                    self._active_context = None
                    self._active_stage = None
                self._interruptions.clear_if_current(cancellation)
                task_done()
                if self._set_idle_if_drained():
                    self._emit_event(ConversationEvent(type="session_idle", session_id=context.session_id))

    async def _process_turn(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> None:
        turn_started_ms = monotonic_ms()
        self._active_stage = "stt"
        with timed_stage(self.metrics, "stt", context):
            transcription = await self.stt_engine.transcribe(segment, context, cancellation=cancellation)
        log_event(
            self.logger,
            "transcription_ready",
            session_id=context.session_id,
            turn_id=context.turn_id,
            text=transcription.text,
            stt_provider=transcription.provider,
            audio_source=segment.source,
            audio_source_label=segment.source_label,
        )
        self._emit_event(
            ConversationEvent(
                type="transcription_ready",
                session_id=context.session_id,
                turn_id=context.turn_id,
                text=transcription.text,
            )
        )
        session_message_text = _build_session_message_text(segment, transcription.text)
        if segment.source == "application_audio":
            self.session.append_application_message(session_message_text)
            if self._should_capture_application_audio_as_context(segment):
                self._active_stage = None
                self.metrics.record_duration(
                    stage="turn_total",
                    duration_ms=monotonic_ms() - turn_started_ms,
                    context=context,
                )
                return
            current_user_parts = tuple()
        else:
            self.session.append_user_message(session_message_text)
            reply_decision = self._decide_user_reply(
                transcription_text=transcription.text,
                segment=segment,
            )
            if not reply_decision.should_reply:
                log_event(
                    self.logger,
                    "response_suppressed",
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    reason=reply_decision.reason,
                    text=transcription.text,
                    audio_source=segment.source,
                )
                self._active_stage = None
                self.metrics.record_duration(
                    stage="turn_total",
                    duration_ms=monotonic_ms() - turn_started_ms,
                    context=context,
                )
                return
            current_user_parts = await self._maybe_capture_current_user_parts(
                user_text=transcription.text,
                context=context,
                cancellation=cancellation,
            )

        request = ConversationRequest(
            context=context,
            messages=self._build_request_messages(
                conversation_language=transcription.language or self.settings.conversation.language
            ),
            current_user_parts=current_user_parts,
        )
        self._active_stage = "llm"
        with timed_stage(self.metrics, "llm", context):
            response = await self.language_model.generate(request, cancellation=cancellation)
        log_event(
            self.logger,
            "response_ready",
            session_id=context.session_id,
            turn_id=context.turn_id,
            text=response.text,
            llm_provider=response.provider,
        )
        self._emit_event(
            ConversationEvent(
                type="response_ready",
                session_id=context.session_id,
                turn_id=context.turn_id,
                text=response.text,
            )
        )

        self._active_stage = "tts"
        await self._play_response(response=response, context=context, cancellation=cancellation)

        self.session.append_assistant_message(response.text)
        self._last_assistant_response_ms = monotonic_ms()
        self._emit_event(
            ConversationEvent(
                type="assistant_message_committed",
                session_id=context.session_id,
                turn_id=context.turn_id,
                text=response.text,
            )
        )
        self._active_stage = None
        self.metrics.record_duration(
            stage="turn_total",
            duration_ms=monotonic_ms() - turn_started_ms,
            context=context,
        )

    async def _await_next_segment(self) -> tuple[AudioSegment, Callable[[], None]]:
        while True:
            segment = self._queue.get_nowait()
            if segment is not None:
                return segment, self._queue.task_done
            segment = self._application_context_queue.get_nowait()
            if segment is not None:
                return segment, self._application_context_queue.task_done
            self._work_available.clear()
            if not self._queue.empty() or not self._application_context_queue.empty():
                self._work_available.set()
                continue
            await self._work_available.wait()

    def _should_capture_application_audio_as_context(self, segment: AudioSegment) -> bool:
        return (
            segment.source == "application_audio"
            and self.settings.application_audio.mode is ApplicationAudioMode.CONTEXT_ONLY
        )

    def _begin_application_audio_submission(self, segment: AudioSegment) -> float | None:
        if segment.source != "application_audio":
            return None
        cooldown_seconds = self.settings.application_audio.transcription_cooldown_seconds
        if cooldown_seconds <= 0.0:
            return monotonic_ms()
        now_ms = monotonic_ms()
        last_submission_ms = self._last_application_audio_submission_ms
        if (
            last_submission_ms is not None
            and (now_ms - last_submission_ms) < (cooldown_seconds * 1000.0)
        ):
            log_event(
                self.logger,
                "application_audio_skipped",
                session_id=self.session.session_id,
                reason="cooldown",
                cooldown_seconds=cooldown_seconds,
                source_label=segment.source_label,
            )
            return None
        return now_ms

    def _log_queue_overflow(self, queue_name: str, queue_size: int) -> None:
        log_event(
            self.logger,
            "queue_overflow",
            session_id=self.session.session_id,
            queue_name=queue_name,
            queue_size=queue_size,
            strategy=self.settings.queue.overflow_strategy.value,
        )

    def _set_idle_if_drained(self) -> bool:
        if (
            self._pending_live_segment is None
            and self._queue.empty()
            and self._application_context_queue.empty()
            and not self._interruptions.has_active_turn
        ):
            if not self._idle_event.is_set():
                self._idle_event.set()
                return True
        return False

    def _should_debounce_live_segment(self, segment: AudioSegment) -> bool:
        return (
            self.settings.input.provider is InputProvider.MICROPHONE
            and segment.source == "user"
            and self.settings.reply.debounce_ms > 0.0
        )

    async def _play_response(
        self,
        response: AssistantResponse,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> None:
        chunks = _split_response_for_playback(response.text)
        tts_duration_ms = 0.0
        playback_duration_ms = 0.0
        pending_chunk: asyncio.Task[_SynthesizedChunk] | None = None
        try:
            if chunks:
                pending_chunk = asyncio.create_task(
                    self._synthesize_chunk(chunks[0], context=context, cancellation=cancellation),
                    name=f"vocalive-tts-{context.turn_id}-0",
                )
            for index, _ in enumerate(chunks):
                cancellation.raise_if_cancelled()
                assert pending_chunk is not None
                self._active_stage = "tts"
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
                self._active_stage = "playback"
                self._emit_event(
                    ConversationEvent(
                        type="assistant_chunk_started",
                        session_id=context.session_id,
                        turn_id=context.turn_id,
                        text=synthesized_chunk.speech.text,
                        chunk_index=index,
                        chunk_count=len(chunks),
                        duration_ms=_estimate_playback_duration_ms(synthesized_chunk.speech),
                    )
                )
                await self.audio_output.play(synthesized_chunk.speech, cancellation=cancellation)
                playback_duration_ms += monotonic_ms() - playback_started_ms
        finally:
            await _discard_background_task(pending_chunk)
            self.metrics.record_duration(
                stage="tts",
                duration_ms=tts_duration_ms,
                context=context,
            )
            self.metrics.record_duration(
                stage="playback",
                duration_ms=playback_duration_ms,
                context=context,
            )

    async def _interrupt_active_turn(self, reason: str, force_stop_audio: bool = False) -> None:
        interrupted_now = self._interruptions.interrupt_active_turn()
        if interrupted_now or force_stop_audio:
            await self.audio_output.stop()
        active_context = self._active_context
        if not interrupted_now or active_context is None:
            return
        log_event(
            self.logger,
            "turn_interrupted",
            session_id=active_context.session_id,
            turn_id=active_context.turn_id,
            stage=self._active_stage,
            reason=reason,
        )
        self._emit_event(
            ConversationEvent(
                type="turn_interrupted",
                session_id=active_context.session_id,
                turn_id=active_context.turn_id,
                stage=self._active_stage,
                reason=reason,
            )
        )

    async def _synthesize_chunk(
        self,
        text: str,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> _SynthesizedChunk:
        started_ms = monotonic_ms()
        speech = await self.tts_engine.synthesize(text, context, cancellation=cancellation)
        return _SynthesizedChunk(
            speech=speech,
            duration_ms=monotonic_ms() - started_ms,
        )

    def _build_request_messages(self, conversation_language: str | None) -> tuple[ConversationMessage, ...]:
        messages = list(
            build_compacted_messages(
                self.session.snapshot(),
                recent_message_count=self.settings.context.recent_message_count,
                conversation_summary_max_chars=(
                    self.settings.context.conversation_summary_max_chars
                ),
                application_recent_message_count=(
                    self.settings.context.application_recent_message_count
                ),
                application_summary_max_chars=(
                    self.settings.context.application_summary_max_chars
                ),
                application_summary_min_message_chars=(
                    self.settings.context.application_summary_min_message_chars
                ),
            )
        )
        identity_instruction = _build_participant_identity_instruction(self.settings)
        if identity_instruction is not None:
            messages.insert(0, ConversationMessage(role="system", content=identity_instruction))
        language_instruction = _build_conversation_language_instruction(conversation_language)
        if language_instruction is not None:
            insertion_index = 1 if identity_instruction is not None else 0
            messages.insert(
                insertion_index,
                ConversationMessage(role="system", content=language_instruction),
            )
        return tuple(messages)

    def _emit_event(self, event: ConversationEvent) -> None:
        try:
            self.event_sink.emit(event)
        except Exception as exc:
            log_event(
                self.logger,
                "event_sink_failed",
                session_id=event.session_id,
                turn_id=event.turn_id,
                event_type=event.type,
                error=str(exc),
            )

    def _decide_user_reply(
        self,
        transcription_text: str,
        segment: AudioSegment,
    ) -> ReplyDecision:
        if (
            self.settings.input.provider is not InputProvider.MICROPHONE
            or segment.source != "user"
        ):
            return ReplyDecision(
                should_reply=True,
                reason="policy_not_applicable",
            )
        return decide_reply(
            transcription_text,
            settings=self.settings.reply,
            last_assistant_response_ms=self._last_assistant_response_ms,
            now_ms=monotonic_ms(),
        )

    async def _maybe_capture_current_user_parts(
        self,
        user_text: str,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> tuple[ConversationRequestPart, ...]:
        settings = self.settings.screen_capture
        if not settings.enabled or self.screen_capture_engine is None:
            return tuple()
        if not self.language_model.supports_multimodal_input:
            return tuple()
        capture_mode = _classify_screen_capture_request(
            user_text,
            trigger_phrases=settings.trigger_phrases,
            passive_enabled=settings.passive_enabled,
            passive_trigger_phrases=settings.passive_trigger_phrases,
        )
        if capture_mode is None:
            return tuple()

        capture_timestamp_ms: float | None = None
        if capture_mode == "passive":
            capture_timestamp_ms = monotonic_ms()
            if _passive_screen_capture_is_on_cooldown(
                now_ms=capture_timestamp_ms,
                last_observation_ms=self._last_screen_observation_ms,
                cooldown_seconds=settings.passive_cooldown_seconds,
            ):
                log_event(
                    self.logger,
                    "screen_capture_skipped",
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    screen_capture_provider=self.screen_capture_engine.name,
                    screen_window_name=settings.window_name,
                    reason="passive_cooldown",
                    trigger_mode=capture_mode,
                )
                return tuple()

        self._active_stage = "screen_capture"
        try:
            with timed_stage(self.metrics, "screen_capture", context):
                screenshot = await self.screen_capture_engine.capture(
                    context=context,
                    cancellation=cancellation,
                )
        except TurnCancelledError:
            raise
        except Exception as exc:
            log_event(
                self.logger,
                "screen_capture_failed",
                session_id=context.session_id,
                turn_id=context.turn_id,
                screen_capture_provider=self.screen_capture_engine.name,
                screen_window_name=settings.window_name,
                error=str(exc),
            )
            return tuple()

        if capture_timestamp_ms is None:
            capture_timestamp_ms = monotonic_ms()
        screenshot_fingerprint = _screen_capture_fingerprint(screenshot)
        self._last_screen_observation_ms = capture_timestamp_ms
        if (
            capture_mode == "passive"
            and screenshot_fingerprint == self._last_screen_capture_fingerprint
        ):
            log_event(
                self.logger,
                "screen_capture_skipped",
                session_id=context.session_id,
                turn_id=context.turn_id,
                screen_capture_provider=self.screen_capture_engine.name,
                screen_window_name=settings.window_name,
                reason="passive_unchanged",
                trigger_mode=capture_mode,
                mime_type=screenshot.mime_type,
                byte_length=len(screenshot.data),
            )
            return tuple()

        self._last_screen_capture_fingerprint = screenshot_fingerprint
        log_event(
            self.logger,
            "screen_capture_ready",
            session_id=context.session_id,
            turn_id=context.turn_id,
            screen_capture_provider=self.screen_capture_engine.name,
            screen_window_name=settings.window_name,
            trigger_mode=capture_mode,
            mime_type=screenshot.mime_type,
            byte_length=len(screenshot.data),
        )
        return _build_screen_capture_parts(
            screenshot,
            settings.window_name,
            capture_mode=capture_mode,
        )

    async def _prepare_segment_for_queue(
        self,
        segment: AudioSegment,
    ) -> tuple[AudioSegment, bool]:
        if not self._is_microphone_user_segment(segment):
            return segment, True

        interrupt_mode = self.settings.input.interrupt_mode
        if interrupt_mode is MicrophoneInterruptMode.ALWAYS:
            return segment, True
        if interrupt_mode is MicrophoneInterruptMode.DISABLED:
            return segment, False
        if self._active_stage not in {"tts", "playback"}:
            return segment, False

        explicit_segment = await self._probe_explicit_microphone_segment(segment)
        if explicit_segment is None:
            return segment, False
        return explicit_segment, True

    async def _probe_explicit_microphone_segment(
        self,
        segment: AudioSegment,
    ) -> AudioSegment | None:
        transcription_text = (segment.transcript_hint or "").strip()
        if not transcription_text:
            probe_context = TurnContext(
                session_id=self.session.session_id,
                turn_id=self._turn_counter + 1,
            )
            try:
                transcription = await self.stt_engine.transcribe(segment, probe_context)
            except Exception as exc:
                log_event(
                    self.logger,
                    "microphone_interrupt_probe_failed",
                    session_id=self.session.session_id,
                    stage=self._active_stage,
                    error=str(exc),
                )
                return None
            transcription_text = transcription.text.strip()
            if not transcription_text:
                return None
            segment = _with_transcript_hint(segment, transcription_text)

        if not looks_like_explicit_assistant_address(
            transcription_text,
            assistant_names=_assistant_names_for_interrupt(self.settings),
        ):
            return None
        return segment

    def _is_microphone_user_segment(self, segment: AudioSegment) -> bool:
        return (
            self.settings.input.provider is InputProvider.MICROPHONE
            and segment.source == "user"
        )


def _build_participant_identity_instruction(settings: AppSettings) -> str:
    user_name = (settings.conversation.user_name or "").strip()
    if user_name:
        return (
            "Your name is コハク. "
            f"The current user's name is {user_name}. "
            "If the user asks what their name is or who you are speaking with, answer with that name. "
            "Do not begin replies by addressing the user by name unless the user asks for that "
            "or the name is genuinely needed for clarity."
        )
    return (
        "Your name is コハク. "
        "You are speaking directly with the current user. "
        "Do not begin replies by addressing the user by name unless the user asks for that "
        "or the name is genuinely needed for clarity."
    )


def _build_conversation_language_instruction(language: str | None) -> str | None:
    normalized_language = _normalize_language(language)
    if normalized_language is None:
        return None
    language_name = _language_name(normalized_language)
    return (
        f"The conversation language is {language_name}. "
        f"Reply in {language_name} unless the user explicitly asks to switch languages."
    )


def _build_session_message_text(segment: AudioSegment, transcription_text: str) -> str:
    normalized_text = transcription_text.strip()
    if segment.source != "application_audio":
        return normalized_text
    source_label = (segment.source_label or "unknown application").strip() or "unknown application"
    return f"Application audio ({source_label}): {normalized_text}"


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized_language = language.strip()
    if not normalized_language:
        return None
    return normalized_language


def _language_name(language: str) -> str:
    normalized_language = language.lower()
    language_names = {
        "ja": "Japanese",
        "ja-jp": "Japanese",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
    }
    return language_names.get(normalized_language, language)


def _build_screen_capture_parts(
    screenshot: ConversationInlineDataPart,
    window_name: str | None,
    capture_mode: Literal["explicit", "passive"],
) -> tuple[ConversationRequestPart, ...]:
    if capture_mode == "passive" and window_name:
        context_text = (
            f"Configured target window: {window_name}. "
            "The attached image is a screenshot of that window for this turn because the user "
            "appears to be referring to the current screen."
        )
    elif capture_mode == "passive":
        context_text = (
            "The attached image is a screenshot of the requested window for this turn because "
            "the user appears to be referring to the current screen."
        )
    elif window_name:
        context_text = (
            f"Configured target window: {window_name}. "
            "The attached image is a screenshot of that window for this turn."
        )
    else:
        context_text = "The attached image is a screenshot of the requested window for this turn."
    return (
        ConversationTextPart(text=context_text),
        screenshot,
    )


def _classify_screen_capture_request(
    user_text: str,
    trigger_phrases: tuple[str, ...],
    passive_enabled: bool,
    passive_trigger_phrases: tuple[str, ...],
) -> Literal["explicit", "passive"] | None:
    if _matches_screen_trigger(user_text, trigger_phrases):
        return "explicit"
    if passive_enabled and _matches_screen_trigger(user_text, passive_trigger_phrases):
        return "passive"
    return None


def _matches_screen_trigger(user_text: str, trigger_phrases: tuple[str, ...]) -> bool:
    normalized_user_text = _normalize_screen_trigger_text(user_text)
    if not normalized_user_text:
        return False
    for trigger_phrase in trigger_phrases:
        normalized_trigger = _normalize_screen_trigger_text(trigger_phrase)
        if normalized_trigger and normalized_trigger in normalized_user_text:
            return True
    return False


def _normalize_screen_trigger_text(value: str) -> str:
    return "".join(value.lower().split())


def _passive_screen_capture_is_on_cooldown(
    now_ms: float,
    last_observation_ms: float | None,
    cooldown_seconds: float,
) -> bool:
    if last_observation_ms is None or cooldown_seconds <= 0:
        return False
    return (now_ms - last_observation_ms) < (cooldown_seconds * 1000.0)


def _screen_capture_fingerprint(screenshot: ConversationInlineDataPart) -> str:
    digest = hashlib.sha256()
    digest.update(screenshot.mime_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(screenshot.data)
    return digest.hexdigest()


def _assistant_names_for_interrupt(settings: AppSettings) -> tuple[str, ...]:
    normalized_names: list[str] = []
    for candidate in (settings.overlay.character_name, "コハク"):
        if candidate is None:
            continue
        normalized = candidate.strip()
        if normalized and normalized not in normalized_names:
            normalized_names.append(normalized)
    return tuple(normalized_names)


def _estimate_playback_duration_ms(speech: SynthesizedSpeech) -> float | None:
    if speech.duration_ms is not None and speech.duration_ms > 0:
        return speech.duration_ms
    bytes_per_second = speech.sample_rate_hz * speech.channels * speech.sample_width_bytes
    if bytes_per_second <= 0 or not speech.audio:
        return None
    return (len(speech.audio) / bytes_per_second) * 1000.0


def _split_response_for_playback(text: str) -> tuple[str, ...]:
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


async def _discard_background_task(task: asyncio.Task[_SynthesizedChunk] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _segments_can_merge(first: AudioSegment, second: AudioSegment) -> bool:
    return (
        first.sample_rate_hz == second.sample_rate_hz
        and first.channels == second.channels
        and first.sample_width_bytes == second.sample_width_bytes
        and first.source == second.source
        and first.source_label == second.source_label
    )


def _merge_segments(first: AudioSegment, second: AudioSegment) -> AudioSegment:
    if not _segments_can_merge(first, second):
        raise ValueError("audio segments are not mergeable")
    return AudioSegment(
        pcm=first.pcm + second.pcm,
        sample_rate_hz=first.sample_rate_hz,
        channels=first.channels,
        sample_width_bytes=first.sample_width_bytes,
        transcript_hint=_merge_transcript_hints(first.transcript_hint, second.transcript_hint),
        source=first.source,
        source_label=first.source_label,
    )


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


def _merge_transcript_hints(first: str | None, second: str | None) -> str | None:
    normalized_parts = [part.strip() for part in (first, second) if part and part.strip()]
    if not normalized_parts:
        return None
    return " ".join(normalized_parts)
