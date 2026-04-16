from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from vocalive.audio.output import AudioOutput
from vocalive.config.settings import (
    AppSettings,
    ApplicationAudioMode,
    ConversationWindowResetPolicy,
    InputProvider,
    MicrophoneInterruptMode,
    QueueOverflowStrategy,
)
from vocalive.llm.base import LanguageModel
from vocalive.models import (
    AudioSegment,
    AudioSource,
    ConversationMessage,
    ConversationRequestPart,
    TurnContext,
)
from vocalive.pipeline.events import ConversationEvent, ConversationEventSink, NullConversationEventSink
from vocalive.pipeline.interruption import (
    CancellationToken,
    InterruptionController,
    TurnCancelledError,
)
from vocalive.pipeline.interrupt_probe import (
    ExplicitInterruptProbeManager,
    ExplicitInterruptProbeRequest,
)
from vocalive.pipeline.playback import PlaybackRunner
from vocalive.pipeline.proactive import ProactiveCoordinator, ProactiveTurnRequest
from vocalive.pipeline.queues import BoundedAsyncQueue
from vocalive.pipeline.reply_policy import (
    ReplyDecision,
    decide_reply,
    looks_like_explicit_assistant_address,
    looks_like_explicit_request,
)
from vocalive.pipeline.resume_summary import (
    ConversationResumeSummarizer,
    build_resume_system_message,
)
from vocalive.pipeline.request_building import build_session_message_text
from vocalive.pipeline.screen_capture_turn import CurrentTurnScreenCaptureCoordinator
from vocalive.pipeline.session import ConversationSession
from vocalive.pipeline.submission import (
    DebouncedSegmentBuffer,
    segment_duration_ms,
)
from vocalive.pipeline.turn_execution import TurnExecutor
from vocalive.screen.base import ScreenCaptureEngine
from vocalive.stt.base import SpeechToTextEngine
from vocalive.tts.base import TextToSpeechEngine
from vocalive.util.logging import get_logger, log_event
from vocalive.util.metrics import InMemoryMetricsRecorder, MetricsRecorder, timed_stage
from vocalive.util.time import monotonic_ms


@dataclass
class _PendingAudibleAssistantContext:
    turn_id: int
    text: str


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
        resume_summarizer: ConversationResumeSummarizer | None = None,
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
        self.resume_summarizer = resume_summarizer
        self.logger = logger or get_logger("vocalive.orchestrator")
        self.metrics = metrics or InMemoryMetricsRecorder()
        self.session = ConversationSession(session_id=settings.session_id)
        self._queue = BoundedAsyncQueue[AudioSegment](
            maxsize=settings.queue.ingress_maxsize,
            overflow_strategy=settings.queue.overflow_strategy,
        )
        self._application_context_queue = BoundedAsyncQueue[tuple[AudioSegment, int]](
            maxsize=settings.queue.ingress_maxsize,
            overflow_strategy=settings.queue.overflow_strategy,
        )
        self._proactive_queue = BoundedAsyncQueue[ProactiveTurnRequest](
            maxsize=1,
            overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
        )
        self._interruptions = InterruptionController()
        self._idle_event = asyncio.Event()
        self._work_available = asyncio.Event()
        self._queue_submission_lock = asyncio.Lock()
        self._resume_summary_lock = asyncio.Lock()
        self._prepared_resume_summary_text: str | None = None
        self._prepared_resume_summary_revision: int | None = None
        self._session_generation = 0
        self._application_context_inflight = 0
        self._pending_audible_assistant_context: _PendingAudibleAssistantContext | None = None
        self._idle_event.set()
        self._worker_task: asyncio.Task[None] | None = None
        self._application_context_worker_task: asyncio.Task[None] | None = None
        self._proactive_monitor_task: asyncio.Task[None] | None = None
        self._turn_counter = 0
        self._active_context: TurnContext | None = None
        self._active_stage: str | None = None
        self._last_assistant_response_ms: float | None = None
        self._last_application_audio_submission_ms: float | None = None
        self._playback_runner = PlaybackRunner(
            get_tts_engine=lambda: self.tts_engine,
            get_audio_output=lambda: self.audio_output,
            metrics=self.metrics,
            emit_event=self._emit_event,
            set_active_stage=self._set_active_stage,
        )
        self._live_segment_debouncer = DebouncedSegmentBuffer(
            delay_ms=lambda: self.settings.reply.debounce_ms,
            on_ready=lambda segment, reason: self._queue_turn_segment(segment, reason=reason),
            task_name_prefix="vocalive-pending-live-segment",
        )
        self._application_segment_debouncer = DebouncedSegmentBuffer(
            delay_ms=lambda: self.settings.application_audio.transcription_debounce_ms,
            on_ready=(
                lambda segment, reason: self._submit_application_audio_segment_now(
                    segment,
                    reason=reason,
                )
            ),
            task_name_prefix="vocalive-pending-application-segment",
        )
        self._interrupt_probe_manager = ExplicitInterruptProbeManager(
            get_stt_engine=lambda: self.stt_engine,
            get_active_context=lambda: self._active_context,
            get_active_stage=lambda: self._active_stage,
            get_assistant_names=lambda: _assistant_names_for_interrupt(self.settings),
            get_session_id=lambda: self.session.session_id,
            interrupt_active_turn=self._interrupt_active_turn,
            logger=self.logger,
        )
        self._screen_capture = CurrentTurnScreenCaptureCoordinator(
            settings=self.settings.screen_capture,
            get_screen_capture_engine=lambda: self.screen_capture_engine,
            get_language_model=lambda: self.language_model,
            now_ms=lambda: monotonic_ms(),
            set_active_stage=self._set_active_stage,
            logger=self.logger,
            metrics=self.metrics,
        )
        self._proactive = ProactiveCoordinator(
            settings=self.settings,
            proactive_queue=self._proactive_queue,
            idle_event=self._idle_event,
            work_available=self._work_available,
            get_language_model=lambda: self.language_model,
            get_screen_capture_engine=lambda: self.screen_capture_engine,
            now_ms=lambda: monotonic_ms(),
            get_session_id=lambda: self.session.session_id,
            build_poll_context=lambda: TurnContext(
                session_id=self.session.session_id,
                turn_id=self._turn_counter + 1,
            ),
            is_assistant_idle_for_proactive_work=self._is_assistant_idle_for_proactive_work,
            is_microphone_user_segment=self._is_microphone_user_segment,
            set_idle_if_drained=self._set_idle_if_drained,
            logger=self.logger,
        )
        self._turn_executor = TurnExecutor(
            settings=self.settings,
            get_stt_engine=lambda: self.stt_engine,
            get_language_model=lambda: self.language_model,
            get_session=lambda: self.session,
            playback_runner=self._playback_runner,
            screen_capture=self._screen_capture,
            proactive=self._proactive,
            prepare_segment=self._interrupt_probe_manager.apply_cached_transcript_hint,
            mark_live_user_activity=self._mark_live_user_activity,
            decide_user_reply=self._decide_user_reply,
            consume_recent_audible_assistant_context=(
                self._consume_recent_audible_assistant_context
            ),
            should_capture_application_audio_as_context=(
                self._should_capture_application_audio_as_context
            ),
            set_last_assistant_response_ms=self._set_last_assistant_response_ms,
            set_active_stage=self._set_active_stage,
            emit_event=self._emit_event,
            logger=self.logger,
            metrics=self.metrics,
        )

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._proactive.initialize_user_activity()
        self._worker_task = asyncio.create_task(self._run(), name="vocalive-orchestrator")
        if self._application_context_worker_task is None:
            self._application_context_worker_task = asyncio.create_task(
                self._run_application_context_worker(),
                name="vocalive-application-context",
            )
        if self.settings.proactive.enabled and self._proactive_monitor_task is None:
            self._proactive_monitor_task = asyncio.create_task(
                self._run_proactive_monitor(),
                name="vocalive-proactive-monitor",
            )

    async def stop(self) -> None:
        proactive_monitor_task = self._proactive_monitor_task
        self._proactive_monitor_task = None
        if proactive_monitor_task is not None and not proactive_monitor_task.done():
            proactive_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await proactive_monitor_task
        await self._discard_pending_live_segment()
        await self._discard_pending_application_segment()
        await self._interrupt_probe_manager.discard()
        self._discard_pending_proactive_turns()
        await self._interrupt_active_turn(reason="shutdown", force_stop_audio=True)
        application_context_worker_task = self._application_context_worker_task
        self._application_context_worker_task = None
        if (
            application_context_worker_task is not None
            and not application_context_worker_task.done()
        ):
            application_context_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await application_context_worker_task
        worker_task = self._worker_task
        self._worker_task = None
        if worker_task is None:
            return
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    async def reset_session_history(
        self,
        *,
        reason: str = "session_reset",
        carry_forward_messages: tuple[ConversationMessage, ...] = (),
    ) -> None:
        await self._discard_pending_live_segment()
        await self._discard_pending_application_segment()
        await self._interrupt_probe_manager.discard()
        await self._interrupt_active_turn(reason=reason, force_stop_audio=True)
        drained_conversation_count = _drain_queue(self._queue)
        drained_application_count = _drain_queue(self._application_context_queue)
        drained_proactive_count = self._discard_pending_proactive_turns()
        self._session_generation += 1
        self.session = ConversationSession(session_id=self.session.session_id)
        for message in carry_forward_messages:
            if message.role != "system":
                continue
            self.session.append_system_message(message.content)
        self._proactive.reset_session_state()
        self._last_assistant_response_ms = None
        self._last_application_audio_submission_ms = None
        self._screen_capture.reset()
        self._clear_prepared_resume_summary()
        self._clear_pending_audible_assistant_context()
        self._set_idle_if_drained()
        log_event(
            self.logger,
            "session_history_reset",
            session_id=self.session.session_id,
            reason=reason,
            drained_conversation_count=drained_conversation_count,
            drained_application_count=drained_application_count,
            drained_proactive_count=drained_proactive_count,
            carry_forward_message_count=len(carry_forward_messages),
        )

    async def prepare_conversation_window_resume_summary(self) -> None:
        if (
            self.settings.conversation_window.reset_policy
            is not ConversationWindowResetPolicy.RESUME_SUMMARY
        ):
            return
        if self.resume_summarizer is None:
            return
        async with self._resume_summary_lock:
            if self.session.revision == self._prepared_resume_summary_revision:
                return
            await self.wait_for_idle()
            snapshot = self.session.snapshot()
            revision = self.session.revision
            if revision == self._prepared_resume_summary_revision:
                return
            if not _has_resume_summary_source_messages(snapshot):
                self._prepared_resume_summary_text = None
                self._prepared_resume_summary_revision = revision
                return
            try:
                summary_text = await self.resume_summarizer.summarize(
                    session_id=self.session.session_id,
                    messages=snapshot,
                    closed_duration_seconds=(
                        self.settings.conversation_window.closed_duration_seconds
                    ),
                )
            except Exception as exc:
                log_event(
                    self.logger,
                    "conversation_window_resume_summary_failed",
                    session_id=self.session.session_id,
                    revision=revision,
                    error=str(exc),
                )
                return
            if revision != self.session.revision:
                log_event(
                    self.logger,
                    "conversation_window_resume_summary_discarded",
                    session_id=self.session.session_id,
                    summarized_revision=revision,
                    current_revision=self.session.revision,
                )
                return
            self._prepared_resume_summary_text = summary_text
            self._prepared_resume_summary_revision = revision
            log_event(
                self.logger,
                "conversation_window_resume_summary_ready",
                session_id=self.session.session_id,
                revision=revision,
                char_count=len(summary_text or ""),
                has_summary=summary_text is not None,
            )

    async def handle_conversation_window_reopened(
        self,
        *,
        reason: str = "conversation_window_reopened",
    ) -> None:
        if (
            self.settings.conversation_window.reset_policy
            is not ConversationWindowResetPolicy.RESUME_SUMMARY
        ):
            await self.reset_session_history(reason=reason)
            return
        await self.prepare_conversation_window_resume_summary()
        carry_forward_messages = self._consume_prepared_resume_messages()
        await self.reset_session_history(
            reason=reason,
            carry_forward_messages=carry_forward_messages,
        )

    async def submit_utterance(self, segment: AudioSegment) -> bool:
        if segment.source == "application_audio":
            return await self._submit_application_audio_segment(segment)
        if self._should_debounce_live_segment(segment):
            accepted = await self._submit_debounced_live_segment(segment)
        else:
            accepted = await self._queue_turn_segment(segment, reason="utterance_submitted")
        return accepted

    async def _submit_application_audio_segment(self, segment: AudioSegment) -> bool:
        if segment.source != "application_audio":
            raise ValueError("application-audio submission requires application_audio segments")
        if self._should_debounce_application_segment(segment):
            return await self._submit_debounced_application_segment(segment)
        return await self._submit_application_audio_segment_now(
            segment,
            reason="utterance_submitted",
        )

    async def _submit_application_audio_segment_now(
        self,
        segment: AudioSegment,
        *,
        reason: str,
    ) -> bool:
        if self._should_skip_application_audio_segment(segment):
            return True
        application_audio_submission_ms = self._begin_application_audio_submission(segment)
        if application_audio_submission_ms is None:
            return True
        if self._should_capture_application_audio_as_context(segment):
            accepted = await self.submit_application_context(segment)
        else:
            accepted = await self._queue_turn_segment(segment, reason=reason)
        if accepted:
            self._last_application_audio_submission_ms = application_audio_submission_ms
        return accepted

    def _consume_prepared_resume_messages(self) -> tuple[ConversationMessage, ...]:
        current_revision = self.session.revision
        summary_text = None
        if self._prepared_resume_summary_revision == current_revision:
            summary_text = self._prepared_resume_summary_text
        self._clear_prepared_resume_summary()
        if not summary_text:
            return tuple()
        return (
            ConversationMessage(
                role="system",
                content=build_resume_system_message(
                    summary_text,
                    closed_duration_seconds=(
                        self.settings.conversation_window.closed_duration_seconds
                    ),
                ),
            ),
        )

    def _clear_prepared_resume_summary(self) -> None:
        self._prepared_resume_summary_text = None
        self._prepared_resume_summary_revision = None

    async def _queue_turn_segment(self, segment: AudioSegment, *, reason: str) -> bool:
        self._idle_event.clear()
        async with self._queue_submission_lock:
            (
                queued_segment,
                should_interrupt,
                interrupt_probe_request,
            ) = await self._prepare_segment_for_queue(segment, reason=reason)
            if should_interrupt:
                await self._interrupt_active_turn(reason=reason)
            accepted = await self._queue.put(queued_segment)
        if not accepted:
            self._log_queue_overflow(queue_name="conversation", queue_size=self._queue.qsize())
            self._set_idle_if_drained()
            return False
        if interrupt_probe_request is not None:
            self._interrupt_probe_manager.schedule(interrupt_probe_request)
        self._work_available.set()
        return True

    async def _submit_debounced_live_segment(self, segment: AudioSegment) -> bool:
        self._idle_event.clear()
        return await self._live_segment_debouncer.submit(
            segment,
            ready_reason="debounced_utterance_ready",
            flush_replaced_reason="debounced_utterance_flushed",
        )

    async def _discard_pending_live_segment(self) -> None:
        await self._live_segment_debouncer.discard()
        self._set_idle_if_drained()

    async def _submit_debounced_application_segment(self, segment: AudioSegment) -> bool:
        self._idle_event.clear()
        return await self._application_segment_debouncer.submit(
            segment,
            ready_reason="application_audio_debounced_ready",
            flush_replaced_reason="application_audio_debounced_flushed",
        )

    async def _discard_pending_application_segment(self) -> None:
        await self._application_segment_debouncer.discard()
        self._set_idle_if_drained()

    async def submit_application_context(self, segment: AudioSegment) -> bool:
        if segment.source != "application_audio":
            raise ValueError("application context submission requires application_audio segments")
        self._idle_event.clear()
        accepted = await self._application_context_queue.put(
            (segment, self._session_generation)
        )
        if not accepted:
            self._log_queue_overflow(
                queue_name="application_context",
                queue_size=self._application_context_queue.qsize(),
            )
            self._set_idle_if_drained()
            return False
        return True

    async def handle_user_speech_start(self, source: AudioSource = "user") -> None:
        self._mark_live_user_activity()
        if self._active_stage not in {"tts", "playback"}:
            return
        if source != "user":
            return
        if self.settings.input.interrupt_mode is not MicrophoneInterruptMode.ALWAYS:
            return
        await self._interrupt_active_turn(reason="speech_started")

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def _run(self) -> None:
        while True:
            work_item, task_done = await self._await_next_work_item()
            context = self._allocate_turn_context()
            cancellation = self._interruptions.begin_turn()
            self._active_context = context
            self._active_stage = None
            try:
                if isinstance(work_item, ProactiveTurnRequest):
                    self._proactive.mark_request_started(work_item)
                    await self._process_proactive_turn(
                        request=work_item,
                        context=context,
                        cancellation=cancellation,
                    )
                else:
                    await self._process_turn(
                        segment=work_item,
                        context=context,
                        cancellation=cancellation,
                    )
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
                if isinstance(work_item, ProactiveTurnRequest):
                    self._proactive.mark_request_finished(work_item)
                if self._active_context == context:
                    self._active_context = None
                    self._active_stage = None
                self._interruptions.clear_if_current(cancellation)
                task_done()
                if self._set_idle_if_drained():
                    self._emit_event(ConversationEvent(type="session_idle", session_id=context.session_id))

    async def _run_application_context_worker(self) -> None:
        while True:
            queued_item = await self._application_context_queue.get()
            segment, generation = queued_item
            context = self._allocate_turn_context()
            self._application_context_inflight += 1
            try:
                await self._process_application_context_segment(
                    segment=segment,
                    context=context,
                    generation=generation,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    self.logger,
                    "turn_failed",
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    error=str(exc),
                    audio_source=segment.source,
                    context_only=True,
                )
            finally:
                self._application_context_inflight = max(
                    0,
                    self._application_context_inflight - 1,
                )
                self._application_context_queue.task_done()
                if self._set_idle_if_drained():
                    self._emit_event(
                        ConversationEvent(
                            type="session_idle",
                            session_id=context.session_id,
                        )
                    )

    async def _process_turn(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> None:
        await self._turn_executor.execute_turn(
            segment=segment,
            context=context,
            cancellation=cancellation,
        )

    async def _process_proactive_turn(
        self,
        request: ProactiveTurnRequest,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> None:
        await self._turn_executor.execute_proactive_turn(
            request=request,
            context=context,
            cancellation=cancellation,
        )

    async def _process_application_context_segment(
        self,
        *,
        segment: AudioSegment,
        context: TurnContext,
        generation: int,
    ) -> None:
        if segment.source != "application_audio":
            raise ValueError("application context worker requires application_audio segments")
        turn_started_ms = monotonic_ms()
        with timed_stage(self.metrics, "stt", context):
            transcription = await self.stt_engine.transcribe(segment, context)
        log_event(
            self.logger,
            "transcription_ready",
            session_id=context.session_id,
            turn_id=context.turn_id,
            text=transcription.text,
            stt_provider=transcription.provider,
            audio_source=segment.source,
            audio_source_label=segment.source_label,
            context_only=True,
        )
        self._emit_event(
            ConversationEvent(
                type="transcription_ready",
                session_id=context.session_id,
                turn_id=context.turn_id,
                text=transcription.text,
            )
        )
        self._mark_live_user_activity()
        if generation != self._session_generation:
            self.metrics.record_duration(
                stage="turn_total",
                duration_ms=monotonic_ms() - turn_started_ms,
                context=context,
            )
            return
        self.session.append_application_message(
            build_session_message_text(segment, transcription.text)
        )
        self._proactive.record_application_audio_observation(segment)
        self.metrics.record_duration(
            stage="turn_total",
            duration_ms=monotonic_ms() - turn_started_ms,
            context=context,
        )

    async def _await_next_work_item(
        self,
    ) -> tuple[AudioSegment | ProactiveTurnRequest, Callable[[], None]]:
        while True:
            segment = self._queue.get_nowait()
            if segment is not None:
                return segment, self._queue.task_done
            proactive_request = self._proactive_queue.get_nowait()
            if proactive_request is not None:
                self._proactive.mark_request_dequeued(proactive_request)
                if not self._should_run_proactive_request(proactive_request):
                    self._proactive_queue.task_done()
                    self._set_idle_if_drained()
                    continue
                return proactive_request, self._proactive_queue.task_done
            self._work_available.clear()
            if (
                not self._queue.empty()
                or not self._proactive_queue.empty()
            ):
                self._work_available.set()
                continue
            await self._work_available.wait()

    def _should_capture_application_audio_as_context(self, segment: AudioSegment) -> bool:
        return (
            segment.source == "application_audio"
            and self.settings.application_audio.mode is ApplicationAudioMode.CONTEXT_ONLY
        )

    def _should_debounce_application_segment(self, segment: AudioSegment) -> bool:
        return (
            segment.source == "application_audio"
            and self.settings.application_audio.transcription_debounce_ms > 0.0
        )

    def _should_skip_application_audio_segment(self, segment: AudioSegment) -> bool:
        if segment.source != "application_audio" or segment.transcript_hint:
            return False
        min_duration_ms = self.settings.application_audio.min_transcription_duration_ms
        if min_duration_ms <= 0.0:
            return False
        duration_ms = segment_duration_ms(segment)
        if duration_ms >= min_duration_ms:
            return False
        log_event(
            self.logger,
            "application_audio_skipped",
            session_id=self.session.session_id,
            reason="too_short",
            source_label=segment.source_label,
            duration_ms=duration_ms,
            min_transcription_duration_ms=min_duration_ms,
        )
        return True

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
            not self._live_segment_debouncer.has_pending_segment
            and not self._application_segment_debouncer.has_pending_segment
            and self._queue.empty()
            and self._application_context_queue.empty()
            and self._application_context_inflight == 0
            and self._proactive_queue.empty()
            and not self._interruptions.has_active_turn
        ):
            if not self._idle_event.is_set():
                self._idle_event.set()
                return True
        return False

    def _mark_live_user_activity(self) -> None:
        self._proactive.record_live_user_activity()

    def _record_proactive_microphone_observation(self, segment: AudioSegment) -> None:
        self._proactive.record_microphone_observation(segment)

    def _record_proactive_application_audio_observation(self, segment: AudioSegment) -> None:
        self._proactive.record_application_audio_observation(segment)

    def _record_proactive_observation(self) -> None:
        self._proactive.record_observation()

    def _consume_proactive_observation(self, observation_version: int) -> None:
        self._proactive.consume_observation(observation_version)

    def _clear_proactive_observations(self) -> None:
        self._proactive.clear_observations()

    def _discard_pending_proactive_turns(self) -> int:
        return self._proactive.discard_pending_turns()

    def _is_current_proactive_observation(self, observation_version: int) -> bool:
        return self._proactive.is_current_observation(observation_version)

    def _is_assistant_idle_for_proactive_work(self) -> bool:
        return (
            self._active_context is None
            and self._active_stage is None
            and not self._live_segment_debouncer.has_pending_segment
            and not self._application_segment_debouncer.has_pending_segment
            and self._queue.empty()
            and self._application_context_queue.empty()
            and self._application_context_inflight == 0
            and self._proactive_queue.empty()
            and not self._interruptions.has_active_turn
        )

    def _can_start_proactive_turn(self, *, now_ms: float) -> bool:
        return self._proactive.can_start_turn(now_ms=now_ms)

    def _should_run_proactive_request(self, request: ProactiveTurnRequest) -> bool:
        return self._proactive.should_run_request(request)

    async def _run_proactive_monitor(self) -> None:
        await self._proactive.run_monitor()

    async def _maybe_enqueue_proactive_turn(self) -> None:
        await self._proactive.maybe_enqueue_turn()

    async def _poll_proactive_screen_if_due(self) -> None:
        await self._proactive.poll_screen_if_due()

    def _proactive_screen_capture_supported(self) -> bool:
        return self._proactive.screen_capture_supported()

    @property
    def _last_user_activity_ms(self) -> float | None:
        return self._proactive.last_user_activity_ms

    @_last_user_activity_ms.setter
    def _last_user_activity_ms(self, value: float | None) -> None:
        self._proactive.last_user_activity_ms = value

    @property
    def _last_proactive_response_ms(self) -> float | None:
        return self._proactive.last_proactive_response_ms

    @_last_proactive_response_ms.setter
    def _last_proactive_response_ms(self, value: float | None) -> None:
        self._proactive.last_proactive_response_ms = value

    def _should_debounce_live_segment(self, segment: AudioSegment) -> bool:
        if not (
            self.settings.input.provider is InputProvider.MICROPHONE
            and segment.source == "user"
            and self.settings.reply.debounce_ms > 0.0
        ):
            return False
        transcript_hint = (segment.transcript_hint or "").strip()
        if transcript_hint and (
            looks_like_explicit_assistant_address(
                transcript_hint,
                assistant_names=_assistant_names_for_interrupt(self.settings),
            )
            or looks_like_explicit_request(transcript_hint)
        ):
            return False
        return True

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

    def _build_proactive_current_user_parts(self) -> tuple[ConversationRequestPart, ...]:
        return self._proactive.build_current_user_parts()

    def _set_active_stage(self, stage: str | None) -> None:
        self._active_stage = stage

    def _set_last_assistant_response_ms(self, value: float) -> None:
        self._last_assistant_response_ms = value

    def _emit_event(self, event: ConversationEvent) -> None:
        if event.type == "assistant_chunk_started" and event.turn_id is not None:
            self._record_audible_assistant_chunk(
                turn_id=event.turn_id,
                text=event.text,
            )
        elif event.type == "assistant_message_committed" and event.turn_id is not None:
            self._clear_pending_audible_assistant_context(turn_id=event.turn_id)
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

    def _allocate_turn_context(self) -> TurnContext:
        self._turn_counter += 1
        return TurnContext(
            session_id=self.session.session_id,
            turn_id=self._turn_counter,
        )

    def _record_audible_assistant_chunk(self, *, turn_id: int, text: str | None) -> None:
        normalized_text = " ".join((text or "").split())
        if not normalized_text:
            return
        pending = self._pending_audible_assistant_context
        if pending is None or pending.turn_id != turn_id:
            self._pending_audible_assistant_context = _PendingAudibleAssistantContext(
                turn_id=turn_id,
                text=normalized_text,
            )
            return
        if normalized_text == pending.text:
            return
        if pending.text.endswith(normalized_text):
            return
        pending.text = f"{pending.text} {normalized_text}".strip()

    def _clear_pending_audible_assistant_context(
        self,
        *,
        turn_id: int | None = None,
    ) -> None:
        pending = self._pending_audible_assistant_context
        if pending is None:
            return
        if turn_id is not None and pending.turn_id != turn_id:
            return
        self._pending_audible_assistant_context = None

    def _consume_recent_audible_assistant_context(self) -> str | None:
        pending = self._pending_audible_assistant_context
        self._pending_audible_assistant_context = None
        if pending is None:
            return None
        return pending.text

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
            assistant_names=_assistant_names_for_interrupt(self.settings),
        )

    async def _maybe_capture_current_user_parts(
        self,
        user_text: str,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> tuple[ConversationRequestPart, ...]:
        return await self._screen_capture.maybe_capture_current_user_parts(
            user_text=user_text,
            context=context,
            cancellation=cancellation,
        )

    async def _prepare_segment_for_queue(
        self,
        segment: AudioSegment,
        *,
        reason: str,
    ) -> tuple[AudioSegment, bool, ExplicitInterruptProbeRequest | None]:
        if segment.source == "application_audio":
            return await self._prepare_application_audio_segment_for_queue(segment, reason=reason)
        if not self._is_microphone_user_segment(segment):
            return segment, True, None

        interrupt_mode = self.settings.input.interrupt_mode
        if interrupt_mode is MicrophoneInterruptMode.ALWAYS:
            return segment, True, None
        if interrupt_mode is MicrophoneInterruptMode.DISABLED:
            return segment, False, None
        if self._active_stage is None:
            return segment, False, None
        if segment.transcript_hint:
            prepared_segment, should_interrupt = await self._interrupt_probe_manager.probe_segment(
                segment,
                session_id=self.session.session_id,
                turn_id=self._turn_counter + 1,
                interrupt_on_explicit_request=True,
            )
            return prepared_segment, should_interrupt, None
        return segment, False, self._interrupt_probe_manager.build_request(
            segment,
            reason=reason,
            interrupt_on_explicit_request=True,
        )

    async def _prepare_application_audio_segment_for_queue(
        self,
        segment: AudioSegment,
        *,
        reason: str,
    ) -> tuple[AudioSegment, bool, ExplicitInterruptProbeRequest | None]:
        if self._active_stage not in {"tts", "playback"}:
            return segment, False, None
        if segment.transcript_hint:
            prepared_segment, should_interrupt = await self._interrupt_probe_manager.probe_segment(
                segment,
                session_id=self.session.session_id,
                turn_id=self._turn_counter + 1,
            )
            return prepared_segment, should_interrupt, None
        return segment, False, self._interrupt_probe_manager.build_request(
            segment,
            reason=reason,
        )

    def _is_microphone_user_segment(self, segment: AudioSegment) -> bool:
        return (
            self.settings.input.provider is InputProvider.MICROPHONE
            and segment.source == "user"
        )


def _assistant_names_for_interrupt(settings: AppSettings) -> tuple[str, ...]:
    normalized_names: list[str] = []
    for candidate in (settings.overlay.character_name, "コハク", "こはく", "琥珀"):
        if candidate is None:
            continue
        normalized = candidate.strip()
        if normalized and normalized not in normalized_names:
            normalized_names.append(normalized)
    return tuple(normalized_names)


def _drain_queue(queue: BoundedAsyncQueue[object]) -> int:
    drained_count = 0
    while True:
        item = queue.get_nowait()
        if item is None:
            return drained_count
        queue.task_done()
        drained_count += 1


def _has_resume_summary_source_messages(
    messages: tuple[ConversationMessage, ...],
) -> bool:
    return any(message.role in {"user", "assistant", "application"} for message in messages)
