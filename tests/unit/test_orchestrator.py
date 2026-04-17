from __future__ import annotations

import asyncio
import io
import logging
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.output import MemoryAudioOutput
from vocalive.config.settings import (
    AppSettings,
    ApplicationAudioMode,
    ApplicationAudioSettings,
    ConversationSettings,
    ConversationWindowResetPolicy,
    ConversationWindowSettings,
    ContextSettings,
    InputProvider,
    InputSettings,
    MicrophoneInterruptMode,
    ProactiveSettings,
    QueueSettings,
    QueueOverflowStrategy,
    ReplySettings,
    ScreenCaptureSettings,
)
from vocalive.llm.base import LanguageModel
from vocalive.llm.echo import EchoLanguageModel
from vocalive.models import (
    AssistantResponse,
    AudioSegment,
    ConversationInlineDataPart,
    ConversationRequest,
    ConversationTextPart,
    SynthesizedSpeech,
    Transcription,
    TurnContext,
)
from vocalive.pipeline.events import ConversationEvent, ConversationEventSink
from vocalive.pipeline.orchestrator import ConversationOrchestrator
from vocalive.pipeline.interruption import CancellationToken, TurnCancelledError
from vocalive.screen.base import ScreenCaptureEngine
from vocalive.stt.mock import MockSpeechToTextEngine
from vocalive.tts.base import TextToSpeechEngine
from vocalive.tts.mock import MockTextToSpeechEngine
from vocalive.util.metrics import InMemoryMetricsRecorder


class CapturingLanguageModel(LanguageModel):
    def __init__(self) -> None:
        self.requests: list[ConversationRequest] = []

    async def generate(
        self,
        request: ConversationRequest,
        cancellation: CancellationToken | None = None,
    ) -> AssistantResponse:
        self.requests.append(request)
        return AssistantResponse(text="captured", provider="capture")


class MultimodalCapturingLanguageModel(CapturingLanguageModel):
    supports_multimodal_input = True


class SequencedCapturingLanguageModel(CapturingLanguageModel):
    def __init__(self, responses: tuple[str, ...]) -> None:
        super().__init__()
        self.responses = responses

    async def generate(
        self,
        request: ConversationRequest,
        cancellation: CancellationToken | None = None,
    ) -> AssistantResponse:
        del cancellation
        self.requests.append(request)
        response_index = min(len(self.requests), len(self.responses)) - 1
        return AssistantResponse(
            text=self.responses[response_index],
            provider="capture",
        )


class StaticLanguageModel(LanguageModel):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    async def generate(
        self,
        request: ConversationRequest,
        cancellation: CancellationToken | None = None,
    ) -> AssistantResponse:
        return AssistantResponse(text=self.response_text, provider="static")


class StubResumeSummarizer:
    def __init__(self, summary_text: str | None) -> None:
        self.summary_text = summary_text
        self.calls: list[tuple[str, float, tuple[str, ...]]] = []

    async def summarize(
        self,
        *,
        session_id: str,
        messages,
        closed_duration_seconds: float,
    ) -> str | None:
        transcript = tuple(message.content for message in messages)
        self.calls.append((session_id, closed_duration_seconds, transcript))
        return self.summary_text


class RecordingTextToSpeechEngine(TextToSpeechEngine):
    name = "recording-tts"

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.started_texts: list[str] = []
        self.completed_texts: list[str] = []

    async def synthesize(
        self,
        text: str,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> SynthesizedSpeech:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.started_texts.append(text)
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.completed_texts.append(text)
        return SynthesizedSpeech(
            text=text,
            provider=self.name,
            audio=text.encode("utf-8"),
            sample_rate_hz=24_000,
            duration_ms=max(320.0, len(text) * 70.0),
        )


class RecordingEventSink(ConversationEventSink):
    def __init__(self) -> None:
        self.events: list[ConversationEvent] = []

    def emit(self, event: ConversationEvent) -> None:
        self.events.append(event)


class StubScreenCaptureEngine(ScreenCaptureEngine):
    name = "stub-screen"

    def __init__(self, image_data: bytes = b"png-bytes") -> None:
        self.image_data = image_data
        self.calls: list[int] = []

    async def capture(
        self,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> ConversationInlineDataPart:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        self.calls.append(context.turn_id)
        return ConversationInlineDataPart(mime_type="image/png", data=self.image_data)


class CancelledScreenCaptureEngine(ScreenCaptureEngine):
    name = "cancelled-screen"

    async def capture(
        self,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> ConversationInlineDataPart:
        del context, cancellation
        raise TurnCancelledError("turn cancelled")


class FailingScreenCaptureEngine(ScreenCaptureEngine):
    name = "failing-screen"

    def __init__(self, error_message: str) -> None:
        self.error_message = error_message
        self.calls = 0

    async def capture(
        self,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> ConversationInlineDataPart:
        del context, cancellation
        self.calls += 1
        raise RuntimeError(self.error_message)


class CountingSpeechToTextEngine(MockSpeechToTextEngine):
    name = "counting-stt"

    def __init__(
        self,
        *,
        text_resolver: Callable[[AudioSegment], str] | None = None,
    ) -> None:
        super().__init__(delay_seconds=0.0)
        self.text_resolver = text_resolver
        self.transcribe_call_count = 0
        self.backend_call_count = 0

    async def transcribe(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> Transcription:
        del context
        self.transcribe_call_count += 1
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        text = segment.transcript_hint
        if not text:
            self.backend_call_count += 1
            if self.text_resolver is not None:
                text = self.text_resolver(segment)
            else:
                text = segment.pcm.decode("utf-8", errors="ignore").strip()
        if not text:
            raise ValueError("counting STT needs transcript_hint or a text resolver")
        return Transcription(text=text, provider=self.name, confidence=1.0, language="ja")


class BlockingSpeechToTextEngine(CountingSpeechToTextEngine):
    def __init__(
        self,
        release_event: asyncio.Event,
        *,
        text_resolver: Callable[[AudioSegment], str] | None = None,
    ) -> None:
        super().__init__(text_resolver=text_resolver)
        self.release_event = release_event
        self.blocking_transcribe_started = asyncio.Event()

    async def transcribe(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> Transcription:
        if not segment.transcript_hint:
            self.blocking_transcribe_started.set()
            await self.release_event.wait()
        return await super().transcribe(segment, context, cancellation=cancellation)


def _pcm_silence(duration_ms: float, *, sample_rate_hz: int = 16_000) -> bytes:
    frame_count = max(1, int(sample_rate_hz * duration_ms / 1000.0))
    return b"\0\0" * frame_count


def _raw_audio_segment(
    duration_ms: float,
    *,
    source: str = "user",
    source_label: str | None = None,
) -> AudioSegment:
    return AudioSegment(
        pcm=_pcm_silence(duration_ms),
        sample_rate_hz=16_000,
        channels=1,
        sample_width_bytes=2,
        source=source,
        source_label=source_label,
    )


class ConversationOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.metrics = InMemoryMetricsRecorder()
        self.output = MemoryAudioOutput()
        self.orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=self.output,
            metrics=self.metrics,
        )
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()

    async def _wait_for(
        self,
        predicate: Callable[[], bool],
        timeout_seconds: float = 0.5,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                self.fail("Timed out while waiting for asynchronous condition")
            await asyncio.sleep(0.005)

    async def test_single_turn_flows_through_pipeline(self) -> None:
        accepted = await self.orchestrator.submit_utterance(AudioSegment.from_text("hello"))
        self.assertTrue(accepted)

        await self.orchestrator.wait_for_idle()

        self.assertEqual(self.output.completed_texts, ["Assistant: hello"])
        self.assertEqual(
            [message.role for message in self.orchestrator.session.snapshot()],
            ["user", "assistant"],
        )
        self.assertEqual(
            {event.stage for event in self.metrics.events},
            {"stt", "llm", "tts", "playback", "turn_total"},
        )

    async def test_reset_session_history_clears_previous_context_for_future_turns(self) -> None:
        language_model = CapturingLanguageModel()
        self.orchestrator.language_model = language_model
        self.orchestrator.settings.conversation.language = None

        accepted = await self.orchestrator.submit_utterance(AudioSegment.from_text("first"))
        self.assertTrue(accepted)
        await self.orchestrator.wait_for_idle()

        await self.orchestrator.reset_session_history(reason="test_reset")

        accepted = await self.orchestrator.submit_utterance(AudioSegment.from_text("second"))
        self.assertTrue(accepted)
        await self.orchestrator.wait_for_idle()

        latest_request = language_model.requests[-1]
        self.assertEqual(
            [message.content for message in latest_request.messages if message.role == "user"],
            ["second"],
        )
        self.assertEqual(
            [message.content for message in self.orchestrator.session.snapshot()],
            ["second", "captured"],
        )

    async def test_conversation_window_reopen_injects_llm_resume_note(self) -> None:
        language_model = CapturingLanguageModel()
        resume_summarizer = StubResumeSummarizer(
            "Carry forward:\n"
            "- The user is rushing to finish a four-hour mission.\n"
            "Assistant approach:\n"
            "- Keep guidance concise and progress-first.\n"
            "Freshness cautions:\n"
            "- Exact current screen state may have changed."
        )
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                conversation=ConversationSettings(language=None),
                conversation_window=ConversationWindowSettings(
                    reset_policy=ConversationWindowResetPolicy.RESUME_SUMMARY,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            resume_summarizer=resume_summarizer,  # type: ignore[arg-type]
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        accepted = await orchestrator.submit_utterance(
            AudioSegment.from_text("4時間以内にクリアしたいから急いでる")
        )
        self.assertTrue(accepted)
        await orchestrator.wait_for_idle()

        await orchestrator.prepare_conversation_window_resume_summary()
        await orchestrator.handle_conversation_window_reopened()

        accepted = await orchestrator.submit_utterance(AudioSegment.from_text("次どうする"))
        self.assertTrue(accepted)
        await orchestrator.wait_for_idle()

        latest_request = language_model.requests[-1]
        system_messages = [
            message.content
            for message in latest_request.messages
            if message.role == "system"
        ]
        joined_system_messages = "\n".join(system_messages)
        self.assertIn("Conversation window resume note:", joined_system_messages)
        self.assertIn("four-hour mission", joined_system_messages)
        self.assertIn("Keep guidance concise", joined_system_messages)
        self.assertEqual(
            [message.content for message in latest_request.messages if message.role == "user"],
            ["次どうする"],
        )
        self.assertEqual(len(resume_summarizer.calls), 1)

    async def test_new_turn_interrupts_existing_playback(self) -> None:
        self.orchestrator.audio_output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        self.orchestrator.tts_engine = MockTextToSpeechEngine(delay_seconds=0.0)

        await self.orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await asyncio.sleep(0.03)
        await self.orchestrator.submit_utterance(AudioSegment.from_text("second"))
        await self.orchestrator.wait_for_idle()

        self.assertEqual(
            [message.content for message in self.orchestrator.session.snapshot()],
            ["first", "second", "Assistant: second"],
        )
        output = self.orchestrator.audio_output
        self.assertIn("Assistant: first", output.started_texts)
        self.assertIn("Assistant: second", output.completed_texts)

    async def test_interrupted_audible_assistant_context_becomes_transient_assistant_history(
        self,
    ) -> None:
        output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        language_model = SequencedCapturingLanguageModel(
            ("First sentence. Second sentence.", "Follow-up reply.")
        )
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=output,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("first"))
            self.assertTrue(accepted)
            await self._wait_for(lambda: output.started_texts == ["First sentence."])

            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text("その話でいいよ")
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 2)
        request_messages = language_model.requests[1].messages
        recent_instruction_messages = [
            message.content
            for message in request_messages
            if message.role == "system"
            and "already heard before an interruption" in message.content
        ]
        self.assertEqual(len(recent_instruction_messages), 1)
        assistant_messages = [
            message.content for message in request_messages if message.role == "assistant"
        ]
        self.assertEqual(assistant_messages, ["First sentence."])
        latest_user_index = max(
            index for index, message in enumerate(request_messages) if message.role == "user"
        )
        self.assertEqual(request_messages[latest_user_index - 1].role, "assistant")
        self.assertEqual(request_messages[latest_user_index - 1].content, "First sentence.")
        self.assertEqual(
            [message.content for message in orchestrator.session.snapshot()],
            ["first", "その話でいいよ", "Follow-up reply."],
        )

    async def test_user_speech_start_interrupts_existing_turn_before_submit(self) -> None:
        output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        self.orchestrator.audio_output = output
        self.orchestrator.tts_engine = MockTextToSpeechEngine(delay_seconds=0.0)

        await self.orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        await self.orchestrator.handle_user_speech_start()
        await self._wait_for(lambda: output.interrupted_texts == ["Assistant: first"])

        await self.orchestrator.submit_utterance(AudioSegment.from_text("second"))
        await self.orchestrator.wait_for_idle()

        self.assertEqual(output.stop_calls, 1)
        self.assertNotIn("Assistant: first", output.completed_texts)
        self.assertIn("Assistant: second", output.completed_texts)
        self.assertEqual(
            [message.content for message in self.orchestrator.session.snapshot()],
            ["first", "second", "Assistant: second"],
        )

    async def test_user_speech_start_does_not_interrupt_in_explicit_microphone_mode(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.EXPLICIT,
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        await orchestrator.handle_user_speech_start(source="user")
        await asyncio.sleep(0.03)

        self.assertEqual(output.stop_calls, 0)
        self.assertEqual(output.interrupted_texts, [])

    async def test_application_audio_speech_start_does_not_interrupt_active_playback(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.ALWAYS,
                ),
                application_audio=ApplicationAudioSettings(
                    enabled=True,
                    mode=ApplicationAudioMode.RESPOND,
                    target="YouTube",
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        await orchestrator.handle_user_speech_start(source="application_audio")
        await asyncio.sleep(0.03)

        self.assertEqual(output.stop_calls, 0)
        self.assertEqual(output.interrupted_texts, [])

    async def test_user_speech_start_resets_proactive_idle_timer(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                proactive=ProactiveSettings(
                    enabled=True,
                    screen_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        orchestrator._record_proactive_observation()
        orchestrator._last_user_activity_ms = 0.0

        self.assertTrue(orchestrator._can_start_proactive_turn(now_ms=1000.0))

        await orchestrator.handle_user_speech_start(source="user")

        self.assertIsNotNone(orchestrator._last_user_activity_ms)
        assert orchestrator._last_user_activity_ms is not None
        self.assertGreater(orchestrator._last_user_activity_ms, 0.0)
        self.assertFalse(
            orchestrator._can_start_proactive_turn(
                now_ms=orchestrator._last_user_activity_ms + 1.0
            )
        )

    async def test_application_audio_speech_start_resets_proactive_idle_timer(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                application_audio=ApplicationAudioSettings(
                    enabled=True,
                    mode=ApplicationAudioMode.CONTEXT_ONLY,
                    target="Steam",
                ),
                proactive=ProactiveSettings(
                    enabled=True,
                    microphone_enabled=False,
                    screen_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        orchestrator._record_proactive_observation()
        orchestrator._last_user_activity_ms = 0.0

        self.assertTrue(orchestrator._can_start_proactive_turn(now_ms=1000.0))

        await orchestrator.handle_user_speech_start(source="application_audio")

        self.assertIsNotNone(orchestrator._last_user_activity_ms)
        assert orchestrator._last_user_activity_ms is not None
        self.assertGreater(orchestrator._last_user_activity_ms, 0.0)
        self.assertFalse(
            orchestrator._can_start_proactive_turn(
                now_ms=orchestrator._last_user_activity_ms + 1.0
            )
        )

    async def test_user_speech_start_does_not_cancel_screen_capture_stage(self) -> None:
        token = self.orchestrator._interruptions.begin_turn()
        self.orchestrator._active_context = TurnContext(
            session_id="test-session",
            turn_id=99,
        )
        self.orchestrator._active_stage = "screen_capture"

        await self.orchestrator.handle_user_speech_start()

        self.assertFalse(token.is_cancelled())
        self.assertEqual(self.output.stop_calls, 0)

    async def test_non_explicit_microphone_turn_does_not_interrupt_active_playback_in_explicit_mode(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.EXPLICIT,
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        accepted = await orchestrator.submit_utterance(AudioSegment.from_text("そのまま行く"))
        self.assertTrue(accepted)
        await orchestrator.wait_for_idle()

        self.assertEqual(output.stop_calls, 0)
        self.assertIn("Assistant: first", output.completed_texts)
        self.assertEqual(
            [message.content for message in orchestrator.session.snapshot()],
            ["first", "Assistant: first", "そのまま行く"],
        )

    async def test_explicit_microphone_turn_interrupts_active_playback_in_explicit_mode(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.EXPLICIT,
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        accepted = await orchestrator.submit_utterance(AudioSegment.from_text("どうする？"))
        self.assertTrue(accepted)
        await orchestrator.wait_for_idle()

        self.assertEqual(output.stop_calls, 1)
        self.assertNotIn("Assistant: first", output.completed_texts)
        self.assertIn("Assistant: どうする？", output.completed_texts)

    async def test_non_explicit_application_audio_turn_does_not_interrupt_active_playback(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.ALWAYS,
                ),
                application_audio=ApplicationAudioSettings(
                    enabled=True,
                    mode=ApplicationAudioMode.RESPOND,
                    target="YouTube",
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        accepted = await orchestrator.submit_utterance(
            AudioSegment.from_text(
                "そのまま行く",
                source="application_audio",
                source_label="YouTube",
            )
        )
        self.assertTrue(accepted)
        await orchestrator.wait_for_idle()

        self.assertEqual(output.stop_calls, 0)
        self.assertIn("Assistant: first", output.completed_texts)
        self.assertEqual(output.interrupted_texts, [])

    async def test_explicit_application_audio_turn_interrupts_active_playback(self) -> None:
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.ALWAYS,
                ),
                application_audio=ApplicationAudioSettings(
                    enabled=True,
                    mode=ApplicationAudioMode.RESPOND,
                    target="YouTube",
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        accepted = await orchestrator.submit_utterance(
            AudioSegment.from_text(
                "コハク、どうする？",
                source="application_audio",
                source_label="YouTube",
            )
        )
        self.assertTrue(accepted)
        await orchestrator.wait_for_idle()

        self.assertEqual(output.stop_calls, 1)
        self.assertEqual(output.interrupted_texts, ["Assistant: first"])
        self.assertNotIn("Assistant: first", output.completed_texts)

    async def test_explicit_interrupt_probe_reuses_transcription_for_turn_processing(self) -> None:
        stt_engine = CountingSpeechToTextEngine(
            text_resolver=lambda segment: "どうする？" if segment.source == "user" else "ignored"
        )
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.EXPLICIT,
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=stt_engine,
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        accepted = await orchestrator.submit_utterance(_raw_audio_segment(800.0))
        self.assertTrue(accepted)
        await orchestrator.wait_for_idle()

        self.assertEqual(stt_engine.backend_call_count, 1)
        self.assertEqual(stt_engine.transcribe_call_count, 3)
        self.assertIn("Assistant: どうする？", output.completed_texts)

    async def test_raw_explicit_interrupt_probe_does_not_block_submit(self) -> None:
        release_event = asyncio.Event()
        explicit_interrupt_text = "assistant help"
        stt_engine = BlockingSpeechToTextEngine(
            release_event,
            text_resolver=(
                lambda segment: explicit_interrupt_text
                if segment.source == "user"
                else "ignored"
            ),
        )
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.EXPLICIT,
                ),
                reply=ReplySettings(debounce_ms=0.0),
            ),
            stt_engine=stt_engine,
            language_model=EchoLanguageModel(delay_seconds=0.0),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        self.addAsyncCleanup(orchestrator.stop)

        output = orchestrator.audio_output
        assert isinstance(output, MemoryAudioOutput)

        await orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        accepted = await asyncio.wait_for(
            orchestrator.submit_utterance(_raw_audio_segment(800.0)),
            timeout=0.05,
        )
        self.assertTrue(accepted)
        await self._wait_for(lambda: stt_engine.blocking_transcribe_started.is_set())

        self.assertEqual(output.stop_calls, 0)

        release_event.set()
        await orchestrator.wait_for_idle()

        self.assertGreaterEqual(stt_engine.backend_call_count, 1)
        self.assertIn("Assistant: assistant help", output.completed_texts)

    async def test_conversation_language_instruction_is_included_for_llm(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("hello"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            [(message.role, message.content) for message in language_model.requests[0].messages[:3]],
            [
                (
                    "system",
                    "Your name is コハク. "
                    "You are speaking directly with the current user. "
                    "Do not begin replies by addressing the user by name unless the user asks "
                    "for that or the name is genuinely needed for clarity.",
                ),
                (
                    "system",
                    "The conversation language is Japanese. "
                    "Reply in Japanese unless the user explicitly asks to switch languages.",
                ),
                ("user", "hello"),
            ],
        )

    async def test_configured_user_name_is_included_in_identity_instruction(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                conversation=ConversationSettings(user_name="ましま", language="ja"),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("hello"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(
            language_model.requests[0].messages[0].content,
            "Your name is コハク. "
            "The current user's name is ましま. "
            "If the user asks what their name is or who you are speaking with, answer with that name. "
            "Do not begin replies by addressing the user by name unless the user asks for that "
            "or the name is genuinely needed for clarity.",
        )

    async def test_long_conversation_is_compacted_into_summary_plus_recent_window(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                context=ContextSettings(
                    recent_message_count=1,
                    conversation_summary_max_chars=220,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("alpha"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()

            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("bravo"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 2)
        request_messages = language_model.requests[1].messages
        self.assertEqual(request_messages[0].role, "system")
        self.assertIn("Your name is コハク.", request_messages[0].content)
        self.assertEqual(request_messages[1].role, "system")
        self.assertIn("The conversation language is Japanese.", request_messages[1].content)
        self.assertEqual(request_messages[2].role, "system")
        self.assertIn("Earlier conversation summary:", request_messages[2].content)
        self.assertIn("alpha", request_messages[2].content)
        self.assertIn("captured", request_messages[2].content)
        self.assertEqual(
            [(message.role, message.content) for message in request_messages[3:]],
            [("user", "bravo")],
        )

    async def test_microphone_turn_waits_for_reply_debounce_before_llm(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(debounce_ms=40.0),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("hello"))
            self.assertTrue(accepted)
            await asyncio.sleep(0.01)
            self.assertEqual(language_model.requests, [])
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)

    async def test_explicit_microphone_transcript_hint_skips_reply_debounce(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(debounce_ms=80.0),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text("なんで？")
            )
            self.assertTrue(accepted)
            await self._wait_for(lambda: len(language_model.requests) == 1, timeout_seconds=0.05)
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)

    async def test_microphone_turns_within_reply_debounce_are_merged(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(debounce_ms=40.0),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("first"))
            self.assertTrue(accepted)
            await asyncio.sleep(0.01)
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("second"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "first second"),
                ("assistant", "captured"),
            ],
        )

    async def test_short_microphone_reaction_is_suppressed_after_recent_reply(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    min_gap_ms=5000.0,
                    short_utterance_max_chars=12,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("こんにちは"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()

            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("やばい"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "こんにちは"),
                ("assistant", "captured"),
                ("user", "やばい"),
            ],
        )

    async def test_explicit_microphone_question_bypasses_reply_suppression(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    min_gap_ms=5000.0,
                    short_utterance_max_chars=12,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("こんにちは"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()

            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("なんで？"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 2)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "こんにちは"),
                ("assistant", "captured"),
                ("user", "なんで？"),
                ("assistant", "captured"),
            ],
        )

    async def test_require_explicit_trigger_suppresses_read_aloud_monologue(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    require_explicit_trigger=True,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text("なぜ世界は滅びたのかという古い文書らしい")
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(language_model.requests, [])
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [("user", "なぜ世界は滅びたのかという古い文書らしい")],
        )

    async def test_require_explicit_trigger_allows_direct_assistant_address(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    require_explicit_trigger=True,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text("コハク さっきの続きなんだけど")
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "コハク さっきの続きなんだけど"),
                ("assistant", "captured"),
            ],
        )

    async def test_require_explicit_trigger_allows_assistant_name_variants(self) -> None:
        for utterance in ("こはく さっきの続きなんだけど", "琥珀 さっきの続きなんだけど"):
            with self.subTest(utterance=utterance):
                language_model = CapturingLanguageModel()
                orchestrator = ConversationOrchestrator(
                    settings=AppSettings(
                        session_id="test-session",
                        input=InputSettings(provider=InputProvider.MICROPHONE),
                        reply=ReplySettings(
                            debounce_ms=0.0,
                            require_explicit_trigger=True,
                        ),
                        queue=QueueSettings(
                            ingress_maxsize=4,
                            overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                        ),
                    ),
                    stt_engine=MockSpeechToTextEngine(),
                    language_model=language_model,
                    tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
                    audio_output=MemoryAudioOutput(),
                    metrics=InMemoryMetricsRecorder(),
                )
                await orchestrator.start()
                try:
                    accepted = await orchestrator.submit_utterance(AudioSegment.from_text(utterance))
                    self.assertTrue(accepted)
                    await orchestrator.wait_for_idle()
                finally:
                    await orchestrator.stop()

                self.assertEqual(len(language_model.requests), 1)
                self.assertEqual(
                    [(message.role, message.content) for message in orchestrator.session.snapshot()],
                    [
                        ("user", utterance),
                        ("assistant", "captured"),
                    ],
                )

    async def test_suppressed_microphone_turn_can_trigger_proactive_reply(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    require_explicit_trigger=True,
                ),
                proactive=ProactiveSettings(
                    enabled=True,
                    screen_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text("なぜ世界は滅びたのかという古い文書らしい")
            )
            self.assertTrue(accepted)
            await self._wait_for(lambda: len(language_model.requests) == 1, timeout_seconds=1.0)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertIn(
            "The user has not made a direct request right now.",
            language_model.requests[0].messages[0].content,
        )
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "なぜ世界は滅びたのかという古い文書らしい"),
                ("assistant", "captured"),
            ],
        )

    async def test_explicit_microphone_reply_does_not_schedule_proactive_follow_up(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    require_explicit_trigger=True,
                ),
                proactive=ProactiveSettings(
                    enabled=True,
                    screen_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("なんで？"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
            await asyncio.sleep(0.2)
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "なんで？"),
                ("assistant", "captured"),
            ],
        )

    async def test_explicit_microphone_reply_discards_stale_proactive_topic(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(provider=InputProvider.MICROPHONE),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    require_explicit_trigger=True,
                ),
                proactive=ProactiveSettings(
                    enabled=True,
                    screen_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            orchestrator._record_proactive_observation()
            orchestrator._last_user_activity_ms = 0.0

            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("どうする？"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
            await asyncio.sleep(0.2)
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "どうする？"),
                ("assistant", "captured"),
            ],
        )

    async def test_context_only_application_audio_can_trigger_proactive_reply(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                application_audio=ApplicationAudioSettings(
                    mode=ApplicationAudioMode.CONTEXT_ONLY,
                ),
                proactive=ProactiveSettings(
                    enabled=True,
                    microphone_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_enabled=False,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "ボスが来た",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await self._wait_for(lambda: len(language_model.requests) == 1, timeout_seconds=1.0)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("application", "Application audio (Steam): ボスが来た"),
                ("assistant", "captured"),
            ],
        )

    async def test_application_audio_respond_mode_does_not_schedule_extra_proactive_reply(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                application_audio=ApplicationAudioSettings(
                    mode=ApplicationAudioMode.RESPOND,
                ),
                proactive=ProactiveSettings(
                    enabled=True,
                    microphone_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_enabled=False,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "ボスが来た",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
            await asyncio.sleep(0.2)
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("application", "Application audio (Steam): ボスが来た"),
                ("assistant", "captured"),
            ],
        )

    async def test_proactive_screen_capture_uses_changed_image_without_repeating_same_frame(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                proactive=ProactiveSettings(
                    enabled=True,
                    microphone_enabled=False,
                    application_audio_enabled=False,
                    screen_enabled=True,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=0.05,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    window_name="YouTube",
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            await self._wait_for(lambda: len(language_model.requests) == 1, timeout_seconds=1.0)
            await orchestrator.wait_for_idle()
            await asyncio.sleep(0.2)
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        self.assertEqual(
            language_model.requests[0].current_user_parts,
            (
                ConversationTextPart(
                    text=(
                        "Configured target window: YouTube. "
                        "The attached image is the latest changed screenshot observed while the "
                        "user was quiet."
                    )
                ),
                ConversationInlineDataPart(mime_type="image/png", data=b"png-bytes"),
            ),
        )

    async def test_proactive_screen_capture_failure_is_rate_limited(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = FailingScreenCaptureEngine(
            "macOS window lookup helper build timed out"
        )
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                proactive=ProactiveSettings(
                    enabled=True,
                    microphone_enabled=False,
                    application_audio_enabled=False,
                    screen_enabled=True,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=0.05,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    window_name="Google Chrome",
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            await asyncio.sleep(0.25)
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, 1)
        self.assertEqual(language_model.requests, [])

    async def test_proactive_reply_waits_for_active_playback_to_finish(self) -> None:
        language_model = SequencedCapturingLanguageModel(
            ("A fairly long first sentence.", "proactive captured")
        )
        output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.DISABLED,
                ),
                reply=ReplySettings(
                    debounce_ms=0.0,
                    require_explicit_trigger=True,
                ),
                proactive=ProactiveSettings(
                    enabled=True,
                    screen_enabled=False,
                    idle_seconds=0.05,
                    cooldown_seconds=0.0,
                    screen_poll_seconds=1.0,
                ),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=RecordingTextToSpeechEngine(),
            audio_output=output,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("なんで？"))
            self.assertTrue(accepted)
            await self._wait_for(
                lambda: output.started_texts == ["A fairly long first sentence."],
                timeout_seconds=1.0,
            )
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text("古い本をそのまま読んでるだけ")
            )
            self.assertTrue(accepted)
            await self._wait_for(lambda: len(language_model.requests) == 2, timeout_seconds=1.5)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(output.interrupted_texts, [])
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("user", "なんで？"),
                ("assistant", "A fairly long first sentence."),
                ("user", "古い本をそのまま読んでるだけ"),
                ("assistant", "proactive captured"),
            ],
        )

    async def test_trigger_phrase_adds_screen_capture_parts_to_current_turn(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    window_name="YouTube",
                    trigger_phrases=("画面見て",),
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("この画面見て"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, [1])
        self.assertEqual(
            language_model.requests[0].current_user_parts,
            (
                ConversationTextPart(
                    text=(
                        "Configured target window: YouTube. "
                        "The attached image is a screenshot of that window for this turn."
                    )
                ),
                ConversationInlineDataPart(mime_type="image/png", data=b"png-bytes"),
            ),
        )

    async def test_non_trigger_phrase_skips_screen_capture(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    trigger_phrases=("画面見て",),
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("こんにちは"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, [])
        self.assertEqual(language_model.requests[0].current_user_parts, ())

    async def test_always_attach_adds_screen_capture_parts_without_trigger_phrase(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    window_name="YouTube",
                    always_attach=True,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("縺薙ｓ縺ｫ縺｡縺ｯ"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, [1])
        self.assertEqual(
            language_model.requests[0].current_user_parts,
            (
                ConversationTextPart(
                    text=(
                        "Configured target window: YouTube. "
                        "The attached image is a screenshot of that window for this turn because "
                        "always-attach screen capture is enabled."
                    )
                ),
                ConversationInlineDataPart(mime_type="image/png", data=b"png-bytes"),
            ),
        )

    async def test_passive_trigger_phrase_adds_screen_capture_parts(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    window_name="YouTube",
                    passive_enabled=True,
                    passive_trigger_phrases=("この画面",),
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            with patch("vocalive.pipeline.orchestrator.monotonic_ms", return_value=1_000.0):
                accepted = await orchestrator.submit_utterance(AudioSegment.from_text("この画面どう見える？"))
                self.assertTrue(accepted)
                await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, [1])
        self.assertEqual(
            language_model.requests[0].current_user_parts,
            (
                ConversationTextPart(
                    text=(
                        "Configured target window: YouTube. "
                        "The attached image is a screenshot of that window for this turn because "
                        "the user appears to be referring to the current screen."
                    )
                ),
                ConversationInlineDataPart(mime_type="image/png", data=b"png-bytes"),
            ),
        )

    async def test_passive_trigger_phrase_respects_cooldown(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    passive_enabled=True,
                    passive_trigger_phrases=("この画面",),
                    passive_cooldown_seconds=30.0,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            with patch("vocalive.pipeline.orchestrator.monotonic_ms", return_value=1_000.0):
                accepted = await orchestrator.submit_utterance(AudioSegment.from_text("この画面どう？"))
                self.assertTrue(accepted)
                await orchestrator.wait_for_idle()

            with patch("vocalive.pipeline.orchestrator.monotonic_ms", return_value=5_000.0):
                accepted = await orchestrator.submit_utterance(AudioSegment.from_text("この画面どう？"))
                self.assertTrue(accepted)
                await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, [1])
        self.assertEqual(language_model.requests[0].current_user_parts[1].data, b"png-bytes")
        self.assertEqual(language_model.requests[1].current_user_parts, ())

    async def test_passive_trigger_phrase_skips_unchanged_screenshot(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    passive_enabled=True,
                    passive_trigger_phrases=("この画面",),
                    passive_cooldown_seconds=1.0,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            with patch("vocalive.pipeline.orchestrator.monotonic_ms", return_value=1_000.0):
                accepted = await orchestrator.submit_utterance(AudioSegment.from_text("この画面どう？"))
                self.assertTrue(accepted)
                await orchestrator.wait_for_idle()

            with patch("vocalive.pipeline.orchestrator.monotonic_ms", return_value=3_500.0):
                accepted = await orchestrator.submit_utterance(AudioSegment.from_text("この画面どう？"))
                self.assertTrue(accepted)
                await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, [1, 2])
        self.assertEqual(language_model.requests[0].current_user_parts[1].data, b"png-bytes")
        self.assertEqual(language_model.requests[1].current_user_parts, ())

    async def test_application_audio_is_committed_as_context_without_immediate_response(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "ボスが来た",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("application", "Application audio (Steam): ボスが来た"),
            ],
        )
        self.assertEqual(language_model.requests, [])

    async def test_application_audio_transcription_cooldown_skips_frequent_context_segments(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                application_audio=ApplicationAudioSettings(
                    transcription_cooldown_seconds=2.0,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            with patch("vocalive.pipeline.orchestrator.monotonic_ms", return_value=1_000.0):
                accepted = await orchestrator.submit_utterance(
                    AudioSegment.from_text(
                        "first clip",
                        source="application_audio",
                        source_label="Steam",
                    )
                )
                self.assertTrue(accepted)
                await orchestrator.wait_for_idle()

            with patch("vocalive.pipeline.orchestrator.monotonic_ms", return_value=2_000.0):
                accepted = await orchestrator.submit_utterance(
                    AudioSegment.from_text(
                        "second clip",
                        source="application_audio",
                        source_label="Steam",
                    )
                )
                self.assertTrue(accepted)
                await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("application", "Application audio (Steam): first clip"),
            ],
        )
        self.assertEqual(language_model.requests, [])

    async def test_short_application_audio_segment_is_skipped_before_stt(self) -> None:
        stt_engine = CountingSpeechToTextEngine(text_resolver=lambda segment: "ignored")
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                application_audio=ApplicationAudioSettings(
                    min_transcription_duration_ms=500.0,
                ),
            ),
            stt_engine=stt_engine,
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                _raw_audio_segment(
                    220.0,
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(stt_engine.backend_call_count, 0)
        self.assertEqual(orchestrator.session.snapshot(), ())
        self.assertEqual(language_model.requests, [])

    async def test_application_audio_segments_are_merged_before_transcription_when_debounced(self) -> None:
        stt_engine = CountingSpeechToTextEngine(
            text_resolver=lambda segment: f"merged-{len(segment.pcm)}"
        )
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                application_audio=ApplicationAudioSettings(
                    transcription_debounce_ms=40.0,
                    min_transcription_duration_ms=700.0,
                ),
            ),
            stt_engine=stt_engine,
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                _raw_audio_segment(
                    320.0,
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await asyncio.sleep(0.01)
            accepted = await orchestrator.submit_utterance(
                _raw_audio_segment(
                    420.0,
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(stt_engine.backend_call_count, 1)
        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [("application", f"Application audio (Steam): merged-{len(_pcm_silence(320.0)) + len(_pcm_silence(420.0))}")],
        )
        self.assertEqual(language_model.requests, [])

    async def test_application_audio_can_still_trigger_response_in_respond_mode(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                application_audio=ApplicationAudioSettings(
                    mode=ApplicationAudioMode.RESPOND,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "ボスが来た",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(
            [(message.role, message.content) for message in orchestrator.session.snapshot()],
            [
                ("application", "Application audio (Steam): ボスが来た"),
                ("assistant", "captured"),
            ],
        )
        self.assertEqual(
            [(message.role, message.content) for message in language_model.requests[0].messages[:3]],
            [
                (
                    "system",
                    "Your name is コハク. "
                    "You are speaking directly with the current user. "
                    "Do not begin replies by addressing the user by name unless the user asks "
                    "for that or the name is genuinely needed for clarity.",
                ),
                (
                    "system",
                    "The conversation language is Japanese. "
                    "Reply in Japanese unless the user explicitly asks to switch languages.",
                ),
                ("application", "Application audio (Steam): ボスが来た"),
            ],
        )

    async def test_context_only_application_audio_does_not_trigger_screen_capture(self) -> None:
        language_model = MultimodalCapturingLanguageModel()
        screen_capture_engine = StubScreenCaptureEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    trigger_phrases=("画面見て",),
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=screen_capture_engine,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "この画面見て",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(screen_capture_engine.calls, [])
        self.assertEqual(language_model.requests, [])

    async def test_older_application_audio_is_compacted_into_separate_summary(self) -> None:
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                context=ContextSettings(
                    recent_message_count=8,
                    conversation_summary_max_chars=220,
                    application_recent_message_count=1,
                    application_summary_max_chars=220,
                    application_summary_min_message_chars=4,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "boss incoming",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()

            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "door opened",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()

            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("どうする？"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 1)
        request_messages = language_model.requests[0].messages
        self.assertEqual(request_messages[1].role, "system")
        self.assertIn("The conversation language is Japanese.", request_messages[1].content)
        self.assertEqual(request_messages[2].role, "system")
        self.assertIn("Earlier application audio summary:", request_messages[2].content)
        self.assertIn("boss incoming", request_messages[2].content)
        self.assertEqual(request_messages[3].role, "system")
        self.assertIn("reply_target", request_messages[3].content)
        self.assertEqual(request_messages[4].role, "system")
        self.assertIn("recent_context:", request_messages[4].content)
        self.assertIn("Application audio (Steam): door opened", request_messages[4].content)
        self.assertEqual(request_messages[5].role, "user")
        self.assertTrue(request_messages[5].content.startswith("reply_target: "))
        return
        self.assertEqual(
            [(message.role, message.content) for message in request_messages[3:]],
            [
                ("application", "Application audio (Steam): door opened"),
                ("user", "どうする？"),
            ],
        )

    async def test_context_only_application_audio_does_not_interrupt_active_playback(self) -> None:
        output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        self.orchestrator.audio_output = output
        self.orchestrator.tts_engine = MockTextToSpeechEngine(delay_seconds=0.0)

        await self.orchestrator.submit_utterance(AudioSegment.from_text("first"))
        await self._wait_for(lambda: output.started_texts == ["Assistant: first"])

        accepted = await self.orchestrator.submit_utterance(
            AudioSegment.from_text(
                "cutscene line",
                source="application_audio",
                source_label="Steam",
            )
        )
        self.assertTrue(accepted)
        await self.orchestrator.wait_for_idle()

        self.assertEqual(output.stop_calls, 0)
        self.assertIn("Assistant: first", output.completed_texts)
        self.assertEqual(
            [(message.role, message.content) for message in self.orchestrator.session.snapshot()],
            [
                ("user", "first"),
                ("application", "Application audio (Steam): cutscene line"),
                ("assistant", "Assistant: first"),
            ],
        )

    async def test_context_only_application_audio_commits_before_next_queued_user_turn(self) -> None:
        output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        language_model = CapturingLanguageModel()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    interrupt_mode=MicrophoneInterruptMode.DISABLED,
                ),
                reply=ReplySettings(debounce_ms=0.0, policy_enabled=False),
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=language_model,
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=output,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("first"))
            self.assertTrue(accepted)
            await self._wait_for(lambda: output.started_texts == ["captured"])

            accepted = await orchestrator.submit_utterance(
                AudioSegment.from_text(
                    "boss incoming",
                    source="application_audio",
                    source_label="Steam",
                )
            )
            self.assertTrue(accepted)
            await self._wait_for(
                lambda: (
                    [message.content for message in orchestrator.session.snapshot()]
                    == ["first", "Application audio (Steam): boss incoming"]
                )
            )
            self.assertEqual(output.completed_texts, [])

            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("what now"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(len(language_model.requests), 2)
        self.assertEqual(
            [(message.role, message.content) for message in language_model.requests[1].messages[2:]],
            [
                ("user", "first"),
                ("application", "Application audio (Steam): boss incoming"),
                ("assistant", "captured"),
                ("user", "what now"),
            ],
        )

    async def test_screen_capture_cancellation_propagates_without_failure_log(self) -> None:
        stream = io.StringIO()
        logger = logging.getLogger("tests.screen_capture_cancelled")
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
                screen_capture=ScreenCaptureSettings(
                    enabled=True,
                    window_name="Steam",
                    trigger_phrases=("画面見て",),
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=MultimodalCapturingLanguageModel(),
            tts_engine=MockTextToSpeechEngine(delay_seconds=0.0),
            audio_output=MemoryAudioOutput(),
            screen_capture_engine=CancelledScreenCaptureEngine(),
            logger=logger,
            metrics=InMemoryMetricsRecorder(),
        )
        cancellation = CancellationToken()
        context = TurnContext(session_id="test-session", turn_id=1)
        try:
            with self.assertRaisesRegex(TurnCancelledError, "turn cancelled"):
                await orchestrator._maybe_capture_current_user_parts(
                    user_text="この画面見て",
                    context=context,
                    cancellation=cancellation,
                )
        finally:
            logger.removeHandler(handler)

        self.assertEqual(stream.getvalue(), "")

    async def test_multi_sentence_response_is_played_sentence_by_sentence(self) -> None:
        response_text = "最初の文です。次の文です！最後の文です"
        output = MemoryAudioOutput()
        tts_engine = RecordingTextToSpeechEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=StaticLanguageModel(response_text),
            tts_engine=tts_engine,
            audio_output=output,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("hello"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(
            tts_engine.started_texts,
            ["最初の文です。", "次の文です！", "最後の文です"],
        )
        self.assertEqual(
            output.completed_texts,
            ["最初の文です。", "次の文です！", "最後の文です"],
        )
        self.assertEqual(
            [message.content for message in orchestrator.session.snapshot()],
            ["hello", response_text],
        )

    async def test_multiline_response_is_flattened_before_playback_and_commit(self) -> None:
        response_text = "最初の文です。\n次の文です！\n\n最後の文です"
        event_sink = RecordingEventSink()
        output = MemoryAudioOutput()
        tts_engine = RecordingTextToSpeechEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=StaticLanguageModel(response_text),
            tts_engine=tts_engine,
            audio_output=output,
            event_sink=event_sink,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("hello"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        flattened_text = "最初の文です。 次の文です！ 最後の文です"
        self.assertEqual(
            tts_engine.started_texts,
            ["最初の文です。", "次の文です！", "最後の文です"],
        )
        self.assertEqual(
            output.completed_texts,
            ["最初の文です。", "次の文です！", "最後の文です"],
        )
        self.assertEqual(
            [message.content for message in orchestrator.session.snapshot()],
            ["hello", flattened_text],
        )
        self.assertEqual(
            [event.text for event in event_sink.events if event.type == "response_ready"],
            [flattened_text],
        )
        self.assertEqual(
            [
                event.text
                for event in event_sink.events
                if event.type == "assistant_message_committed"
            ],
            [flattened_text],
        )

    async def test_next_sentence_tts_is_prefetched_during_current_playback(self) -> None:
        output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        tts_engine = RecordingTextToSpeechEngine()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=StaticLanguageModel("First sentence. Second sentence."),
            tts_engine=tts_engine,
            audio_output=output,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("hello"))
            self.assertTrue(accepted)
            await self._wait_for(lambda: output.started_texts == ["First sentence."])
            await self._wait_for(lambda: len(tts_engine.started_texts) >= 2)
            self.assertEqual(
                tts_engine.started_texts[:2],
                ["First sentence.", "Second sentence."],
            )
            self.assertNotIn("First sentence.", output.completed_texts)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

    async def test_ui_events_follow_turn_lifecycle_and_chunk_playback(self) -> None:
        event_sink = RecordingEventSink()
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=StaticLanguageModel("First sentence. Second sentence."),
            tts_engine=RecordingTextToSpeechEngine(),
            audio_output=MemoryAudioOutput(),
            event_sink=event_sink,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            accepted = await orchestrator.submit_utterance(AudioSegment.from_text("hello"))
            self.assertTrue(accepted)
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        self.assertEqual(
            [event.type for event in event_sink.events],
            [
                "transcription_ready",
                "response_ready",
                "assistant_chunk_started",
                "assistant_chunk_started",
                "assistant_message_committed",
                "session_idle",
            ],
        )
        self.assertEqual(
            [event.text for event in event_sink.events if event.type == "assistant_chunk_started"],
            ["First sentence.", "Second sentence."],
        )
        self.assertTrue(
            all(
                (event.duration_ms or 0) > 0
                for event in event_sink.events
                if event.type == "assistant_chunk_started"
            )
        )

    async def test_interrupted_turn_emits_interruption_and_cancellation_events(self) -> None:
        event_sink = RecordingEventSink()
        output = MemoryAudioOutput(chunk_delay_seconds=0.02, chunk_size_bytes=1)
        orchestrator = ConversationOrchestrator(
            settings=AppSettings(
                session_id="test-session",
                queue=QueueSettings(
                    ingress_maxsize=4,
                    overflow_strategy=QueueOverflowStrategy.DROP_OLDEST,
                ),
            ),
            stt_engine=MockSpeechToTextEngine(),
            language_model=StaticLanguageModel("A fairly long first sentence."),
            tts_engine=RecordingTextToSpeechEngine(),
            audio_output=output,
            event_sink=event_sink,
            metrics=InMemoryMetricsRecorder(),
        )
        await orchestrator.start()
        try:
            await orchestrator.submit_utterance(AudioSegment.from_text("first"))
            await self._wait_for(lambda: output.started_texts == ["A fairly long first sentence."])
            await orchestrator.submit_utterance(AudioSegment.from_text("second"))
            await orchestrator.wait_for_idle()
        finally:
            await orchestrator.stop()

        event_types = [event.type for event in event_sink.events]
        self.assertIn("turn_interrupted", event_types)
        self.assertIn("turn_cancelled", event_types)
