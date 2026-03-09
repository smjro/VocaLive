from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from vocalive.audio.output import AudioOutput
from vocalive.config.settings import AppSettings
from vocalive.llm.base import LanguageModel
from vocalive.models import (
    AssistantResponse,
    AudioSegment,
    ConversationMessage,
    ConversationRequest,
    SynthesizedSpeech,
    TurnContext,
)
from vocalive.pipeline.interruption import (
    CancellationToken,
    InterruptionController,
    TurnCancelledError,
)
from vocalive.pipeline.queues import BoundedAsyncQueue
from vocalive.pipeline.session import ConversationSession
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
        logger: logging.Logger | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.stt_engine = stt_engine
        self.language_model = language_model
        self.tts_engine = tts_engine
        self.audio_output = audio_output
        self.logger = logger or get_logger("vocalive.orchestrator")
        self.metrics = metrics or InMemoryMetricsRecorder()
        self.session = ConversationSession(session_id=settings.session_id)
        self._queue = BoundedAsyncQueue[AudioSegment](
            maxsize=settings.queue.ingress_maxsize,
            overflow_strategy=settings.queue.overflow_strategy,
        )
        self._interruptions = InterruptionController()
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._worker_task: asyncio.Task[None] | None = None
        self._turn_counter = 0
        self._active_context: TurnContext | None = None
        self._active_stage: str | None = None

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._run(), name="vocalive-orchestrator")

    async def stop(self) -> None:
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
        self._idle_event.clear()
        await self._interrupt_active_turn(reason="utterance_submitted")
        accepted = await self._queue.put(segment)
        if not accepted:
            log_event(
                self.logger,
                "queue_overflow",
                session_id=self.session.session_id,
                queue_size=self._queue.qsize(),
                strategy=self.settings.queue.overflow_strategy.value,
            )
            if not self._interruptions.has_active_turn and self._queue.empty():
                self._idle_event.set()
            return False
        return True

    async def handle_user_speech_start(self) -> None:
        await self._interrupt_active_turn(reason="speech_started")

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def _run(self) -> None:
        while True:
            segment = await self._queue.get()
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
                self._queue.task_done()
                if self._queue.empty() and not self._interruptions.has_active_turn:
                    self._idle_event.set()

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
        )
        self.session.append_user_message(transcription.text)

        request = ConversationRequest(
            context=context,
            messages=self._build_request_messages(
                conversation_language=transcription.language or self.settings.conversation.language
            ),
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

        self._active_stage = "tts"
        await self._play_response(response=response, context=context, cancellation=cancellation)

        self.session.append_assistant_message(response.text)
        self._active_stage = None
        self.metrics.record_duration(
            stage="turn_total",
            duration_ms=monotonic_ms() - turn_started_ms,
            context=context,
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
        messages = list(self.session.snapshot())
        language_instruction = _build_conversation_language_instruction(conversation_language)
        if language_instruction is not None:
            messages.insert(0, ConversationMessage(role="system", content=language_instruction))
        return tuple(messages)


def _build_conversation_language_instruction(language: str | None) -> str | None:
    normalized_language = _normalize_language(language)
    if normalized_language is None:
        return None
    language_name = _language_name(normalized_language)
    return (
        f"The conversation language is {language_name}. "
        f"Reply in {language_name} unless the user explicitly asks to switch languages."
    )


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
