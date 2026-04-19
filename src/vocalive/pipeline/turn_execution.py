from __future__ import annotations

import logging
from collections.abc import Callable

from vocalive.config.settings import AppSettings
from vocalive.llm.base import LanguageModel
from vocalive.models import AudioSegment, ConversationRequest, TurnContext
from vocalive.pipeline.events import ConversationEvent
from vocalive.pipeline.interruption import CancellationToken
from vocalive.pipeline.playback import PlaybackRunner, normalize_assistant_response
from vocalive.pipeline.proactive import ProactiveCoordinator, ProactiveTurnRequest
from vocalive.pipeline.reply_policy import ReplyDecision
from vocalive.pipeline.reply_policy import looks_like_explicit_request
from vocalive.pipeline.request_building import (
    build_recent_audible_assistant_instruction,
    build_proactive_request_messages,
    build_request_messages,
    build_session_message_text,
    inject_recent_audible_assistant_message,
)
from vocalive.pipeline.screen_capture_turn import CurrentTurnScreenCaptureCoordinator
from vocalive.pipeline.session import ConversationSession
from vocalive.stt.base import SpeechToTextEngine
from vocalive.util.logging import log_event
from vocalive.util.metrics import MetricsRecorder, timed_stage
from vocalive.util.time import monotonic_ms


class TurnExecutor:
    def __init__(
        self,
        *,
        settings: AppSettings,
        get_stt_engine: Callable[[], SpeechToTextEngine],
        get_language_model: Callable[[], LanguageModel],
        get_session: Callable[[], ConversationSession],
        playback_runner: PlaybackRunner,
        screen_capture: CurrentTurnScreenCaptureCoordinator,
        proactive: ProactiveCoordinator,
        prepare_segment: Callable[[AudioSegment], AudioSegment],
        mark_live_user_activity: Callable[[], None],
        decide_user_reply: Callable[[str, AudioSegment], ReplyDecision],
        consume_recent_audible_assistant_context: Callable[[], str | None],
        should_capture_application_audio_as_context: Callable[[AudioSegment], bool],
        set_last_assistant_response_ms: Callable[[float], None],
        set_active_stage: Callable[[str | None], None],
        emit_event: Callable[[ConversationEvent], None],
        logger: logging.Logger,
        metrics: MetricsRecorder,
    ) -> None:
        self._settings = settings
        self._get_stt_engine = get_stt_engine
        self._get_language_model = get_language_model
        self._get_session = get_session
        self._playback_runner = playback_runner
        self._screen_capture = screen_capture
        self._proactive = proactive
        self._prepare_segment = prepare_segment
        self._mark_live_user_activity = mark_live_user_activity
        self._decide_user_reply = decide_user_reply
        self._consume_recent_audible_assistant_context = (
            consume_recent_audible_assistant_context
        )
        self._should_capture_application_audio_as_context = (
            should_capture_application_audio_as_context
        )
        self._set_last_assistant_response_ms = set_last_assistant_response_ms
        self._set_active_stage = set_active_stage
        self._emit_event = emit_event
        self._logger = logger
        self._metrics = metrics

    async def execute_turn(
        self,
        *,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> None:
        segment = self._prepare_segment(segment)
        pending_screen_capture = self._screen_capture.start_pending_capture(
            segment=segment,
            context=context,
            cancellation=cancellation,
        )
        turn_started_ms = monotonic_ms()
        try:
            self._set_active_stage("stt")
            with timed_stage(self._metrics, "stt", context):
                transcription = await self._get_stt_engine().transcribe(
                    segment,
                    context,
                    cancellation=cancellation,
                )
            log_event(
                self._logger,
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
            self._mark_live_user_activity()
            session_message_text = build_session_message_text(segment, transcription.text)
            session = self._get_session()
            if segment.source == "application_audio":
                session.append_application_message(session_message_text)
                if self._should_capture_application_audio_as_context(segment):
                    self._proactive.record_application_audio_observation(segment)
                    self._set_active_stage(None)
                    self._metrics.record_duration(
                        stage="turn_total",
                        duration_ms=monotonic_ms() - turn_started_ms,
                        context=context,
                    )
                    return
                current_user_parts = tuple()
            else:
                session.append_user_message(session_message_text)
                if looks_like_explicit_request(transcription.text):
                    self._proactive.clear_observations()
                reply_decision = self._decide_user_reply(
                    transcription.text,
                    segment,
                )
                if not reply_decision.should_reply:
                    log_event(
                        self._logger,
                        "response_suppressed",
                        session_id=context.session_id,
                        turn_id=context.turn_id,
                        reason=reply_decision.reason,
                        text=transcription.text,
                        audio_source=segment.source,
                    )
                    self._proactive.record_microphone_observation(segment)
                    self._set_active_stage(None)
                    self._metrics.record_duration(
                        stage="turn_total",
                        duration_ms=monotonic_ms() - turn_started_ms,
                        context=context,
                    )
                    return
                current_user_parts = await self._screen_capture.maybe_capture_current_user_parts(
                    user_text=transcription.text,
                    context=context,
                    cancellation=cancellation,
                    pending_capture=pending_screen_capture,
                )
                pending_screen_capture = None
            transient_system_messages = tuple()
            request_messages = None
            if segment.source != "application_audio":
                recent_audible_assistant_text = (
                    self._consume_recent_audible_assistant_context()
                )
                if recent_audible_assistant_text:
                    transient_system_messages = (
                        build_recent_audible_assistant_instruction(),
                    )
                    request_messages = inject_recent_audible_assistant_message(
                        build_request_messages(
                            self._get_session().snapshot(),
                            settings=self._settings,
                            conversation_language=(
                                transcription.language or self._settings.conversation.language
                            ),
                            transient_system_messages=transient_system_messages,
                        ),
                        recent_audible_assistant_text,
                    )
            if request_messages is None:
                request_messages = build_request_messages(
                    self._get_session().snapshot(),
                    settings=self._settings,
                    conversation_language=(
                        transcription.language or self._settings.conversation.language
                    ),
                    transient_system_messages=transient_system_messages,
                )

            request = ConversationRequest(
                context=context,
                messages=request_messages,
                current_user_parts=current_user_parts,
            )
            self._set_active_stage("llm")
            with timed_stage(self._metrics, "llm", context):
                response = await self._get_language_model().generate(
                    request,
                    cancellation=cancellation,
                )
            response = normalize_assistant_response(response)
            log_event(
                self._logger,
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
            self._set_active_stage("tts")
            await self._playback_runner.play_response(
                response=response,
                context=context,
                cancellation=cancellation,
            )

            committed_at_ms = monotonic_ms()
            self._get_session().append_assistant_message(response.text)
            self._set_last_assistant_response_ms(committed_at_ms)
            self._emit_event(
                ConversationEvent(
                    type="assistant_message_committed",
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    text=response.text,
                )
            )
            self._set_active_stage(None)
            self._metrics.record_duration(
                stage="turn_total",
                duration_ms=committed_at_ms - turn_started_ms,
                context=context,
            )
        finally:
            await self._screen_capture.discard_pending_capture(pending_screen_capture)

    async def execute_proactive_turn(
        self,
        *,
        request: ProactiveTurnRequest,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> None:
        if not self._proactive.is_current_observation(request.observation_version):
            return
        turn_started_ms = monotonic_ms()
        proactive_request = ConversationRequest(
            context=context,
            messages=build_proactive_request_messages(
                self._get_session().snapshot(),
                settings=self._settings,
            ),
            current_user_parts=self._proactive.build_current_user_parts(),
        )
        self._set_active_stage("llm")
        with timed_stage(self._metrics, "llm", context):
            response = await self._get_language_model().generate(
                proactive_request,
                cancellation=cancellation,
            )
        response = normalize_assistant_response(response)
        log_event(
            self._logger,
            "response_ready",
            session_id=context.session_id,
            turn_id=context.turn_id,
            text=response.text,
            llm_provider=response.provider,
            proactive=True,
        )
        self._emit_event(
            ConversationEvent(
                type="response_ready",
                session_id=context.session_id,
                turn_id=context.turn_id,
                text=response.text,
            )
        )
        self._set_active_stage("tts")
        await self._playback_runner.play_response(
            response=response,
            context=context,
            cancellation=cancellation,
        )
        committed_at_ms = monotonic_ms()
        self._get_session().append_assistant_message(response.text)
        self._set_last_assistant_response_ms(committed_at_ms)
        self._proactive.record_proactive_response(committed_at_ms)
        self._proactive.consume_observation(request.observation_version)
        self._emit_event(
            ConversationEvent(
                type="assistant_message_committed",
                session_id=context.session_id,
                turn_id=context.turn_id,
                text=response.text,
            )
        )
        self._set_active_stage(None)
        self._metrics.record_duration(
            stage="turn_total",
            duration_ms=committed_at_ms - turn_started_ms,
            context=context,
        )
