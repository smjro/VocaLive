from __future__ import annotations

import asyncio
import sys
import unittest
from collections.abc import Callable
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.output import MemoryAudioOutput
from vocalive.config.settings import AppSettings, QueueSettings, QueueOverflowStrategy
from vocalive.llm.base import LanguageModel
from vocalive.llm.echo import EchoLanguageModel
from vocalive.models import (
    AssistantResponse,
    AudioSegment,
    ConversationRequest,
    SynthesizedSpeech,
    TurnContext,
)
from vocalive.pipeline.events import ConversationEvent, ConversationEventSink
from vocalive.pipeline.orchestrator import ConversationOrchestrator
from vocalive.pipeline.interruption import CancellationToken
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
            duration_ms=max(320.0, len(text) * 70.0),
        )


class RecordingEventSink(ConversationEventSink):
    def __init__(self) -> None:
        self.events: list[ConversationEvent] = []

    def emit(self, event: ConversationEvent) -> None:
        self.events.append(event)


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
