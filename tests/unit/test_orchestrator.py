from __future__ import annotations

import asyncio
import io
import logging
import sys
import unittest
from collections.abc import Callable
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.output import MemoryAudioOutput
from vocalive.config.settings import (
    AppSettings,
    ApplicationAudioMode,
    ApplicationAudioSettings,
    ContextSettings,
    QueueSettings,
    QueueOverflowStrategy,
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
    TurnContext,
)
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


class StaticLanguageModel(LanguageModel):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    async def generate(
        self,
        request: ConversationRequest,
        cancellation: CancellationToken | None = None,
    ) -> AssistantResponse:
        return AssistantResponse(text=self.response_text, provider="static")


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
        )


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
            [(message.role, message.content) for message in language_model.requests[0].messages[:2]],
            [
                (
                    "system",
                    "The conversation language is Japanese. "
                    "Reply in Japanese unless the user explicitly asks to switch languages.",
                ),
                ("user", "hello"),
            ],
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
        self.assertIn("The conversation language is Japanese.", request_messages[0].content)
        self.assertEqual(request_messages[1].role, "system")
        self.assertIn("Earlier conversation summary:", request_messages[1].content)
        self.assertIn("alpha", request_messages[1].content)
        self.assertIn("captured", request_messages[1].content)
        self.assertEqual(
            [(message.role, message.content) for message in request_messages[2:]],
            [("user", "bravo")],
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
            [(message.role, message.content) for message in language_model.requests[0].messages[:2]],
            [
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
                ("assistant", "Assistant: first"),
                ("application", "Application audio (Steam): cutscene line"),
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
