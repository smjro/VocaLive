from __future__ import annotations

import asyncio
import contextlib
import sys

from vocalive.audio.input import AudioInput, CombinedAudioInput, MicrophoneAudioInput
from vocalive.audio.output import MemoryAudioOutput, SpeakerAudioOutput, parse_playback_command
from vocalive.conversation_window import ConversationWindowGate
from vocalive.config.settings import (
    AppSettings,
    AivisEngineMode,
    ApplicationAudioMode,
    ConversationWindowResetPolicy,
    InputProvider,
    OutputProvider,
)
from vocalive.llm.echo import EchoLanguageModel
from vocalive.llm.gemini import GeminiLanguageModel
from vocalive.models import AudioSegment
from vocalive.pipeline.events import ConversationEventSink
from vocalive.pipeline.orchestrator import ConversationOrchestrator
from vocalive.pipeline.resume_summary import ConversationResumeSummarizer
from vocalive.stt.mock import MockSpeechToTextEngine
from vocalive.stt.moonshine import MoonshineSpeechToTextEngine
from vocalive.stt.openai import OpenAITranscriptionSpeechToTextEngine
from vocalive.tts.aivis import AivisSpeechTextToSpeechEngine
from vocalive.tts.aivis_manager import ManagedAivisSpeechEngine
from vocalive.tts.mock import MockTextToSpeechEngine
from vocalive.ui.overlay import OverlayServer
from vocalive.util.logging import get_logger, log_event


logger = get_logger(__name__)


async def run_headless(settings: AppSettings) -> int:
    overlay = build_overlay(settings)
    orchestrator = build_orchestrator(settings, event_sink=overlay)
    audio_input = build_audio_input(settings)
    managed_aivis_engine = build_managed_aivis_engine(settings)

    try:
        if managed_aivis_engine is not None:
            await managed_aivis_engine.start()
        if overlay is not None:
            await overlay.start()
            print(f"VocaLive overlay: {overlay.url}")
        await orchestrator.start()
        if audio_input is None:
            return await run_stdin_shell(orchestrator)
        return await run_microphone_loop(orchestrator, audio_input, settings=settings)
    finally:
        if audio_input is not None:
            await audio_input.close()
        await orchestrator.stop()
        if overlay is not None:
            await overlay.stop()
        if managed_aivis_engine is not None:
            await managed_aivis_engine.close()


def build_orchestrator(
    settings: AppSettings,
    event_sink: ConversationEventSink | None = None,
) -> ConversationOrchestrator:
    if _uses_live_audio_input(settings) and settings.stt_provider == "mock":
        raise ValueError(
            "microphone/application audio input requires a real STT adapter; "
            "set VOCALIVE_STT_PROVIDER=moonshine or openai"
        )
    if settings.stt_provider == "moonshine":
        stt_engine = MoonshineSpeechToTextEngine(
            model_name=settings.moonshine.model_name,
            default_language=settings.conversation.language,
            application_audio_enhancement_enabled=(
                settings.application_audio.stt_enhancement_enabled
            ),
        )
    elif settings.stt_provider == "openai":
        stt_engine = OpenAITranscriptionSpeechToTextEngine(
            api_key=settings.openai.api_key,
            model_name=settings.openai.model_name,
            base_url=settings.openai.base_url,
            timeout_seconds=settings.openai.timeout_seconds,
            default_language=settings.conversation.language,
        )
    else:
        stt_engine = MockSpeechToTextEngine()
    if settings.model_provider == "gemini":
        language_model = GeminiLanguageModel(
            api_key=settings.gemini.api_key,
            model_name=settings.gemini.model_name,
            timeout_seconds=settings.gemini.timeout_seconds,
            temperature=settings.gemini.temperature,
            thinking_budget=settings.gemini.thinking_budget,
            system_instruction=settings.gemini.system_instruction,
        )
    else:
        language_model = EchoLanguageModel()
    resume_summarizer = None
    if settings.conversation_window.reset_policy is ConversationWindowResetPolicy.RESUME_SUMMARY:
        if settings.model_provider != "gemini":
            raise ValueError(
                "conversation-window resume summaries currently require "
                "VOCALIVE_MODEL_PROVIDER=gemini"
            )
        resume_summarizer = ConversationResumeSummarizer(
            GeminiLanguageModel(
                api_key=settings.gemini.api_key,
                model_name=settings.gemini.model_name,
                timeout_seconds=settings.gemini.timeout_seconds,
                temperature=0.0,
                thinking_budget=None,
                system_instruction=None,
            )
        )
    if settings.tts_provider == "aivis":
        tts_engine = AivisSpeechTextToSpeechEngine(
            base_url=settings.aivis.base_url,
            speaker_id=settings.aivis.speaker_id,
            speaker_name=settings.aivis.speaker_name,
            style_name=settings.aivis.style_name,
            timeout_seconds=settings.aivis.timeout_seconds,
        )
    else:
        tts_engine = MockTextToSpeechEngine()
    if settings.output.provider == OutputProvider.SPEAKER:
        if settings.tts_provider != "aivis":
            raise ValueError(
                "speaker output currently requires VOCALIVE_TTS_PROVIDER=aivis"
            )
        audio_output = SpeakerAudioOutput(
            playback_command=parse_playback_command(settings.output.speaker_command)
        )
    else:
        audio_output = MemoryAudioOutput()
    screen_capture_engine = None
    if settings.screen_capture.enabled:
        if settings.model_provider != "gemini":
            raise ValueError(
                "screen capture input currently requires VOCALIVE_MODEL_PROVIDER=gemini"
            )
        if not settings.screen_capture.window_name:
            raise ValueError(
                "screen capture input currently requires VOCALIVE_SCREEN_WINDOW_NAME"
            )
        screen_capture_engine_class = screen_capture_engine_class_for_platform(sys.platform)
        if screen_capture_engine_class is None:
            raise ValueError(
                "screen capture input currently supports macOS and Windows only"
            )
        screen_capture_engine = screen_capture_engine_class(
            window_name=settings.screen_capture.window_name,
            timeout_seconds=settings.screen_capture.timeout_seconds,
            resize_max_edge_px=settings.screen_capture.resize_max_edge_px,
        )
    return ConversationOrchestrator(
        settings=settings,
        stt_engine=stt_engine,
        language_model=language_model,
        tts_engine=tts_engine,
        audio_output=audio_output,
        event_sink=event_sink,
        screen_capture_engine=screen_capture_engine,
        resume_summarizer=resume_summarizer,
    )


def build_audio_input(settings: AppSettings) -> AudioInput | None:
    live_inputs: list[AudioInput] = []
    if settings.input.provider == InputProvider.MICROPHONE:
        live_inputs.append(
            MicrophoneAudioInput(
                sample_rate_hz=settings.input.sample_rate_hz,
                channels=settings.input.channels,
                block_duration_ms=settings.input.block_duration_ms,
                speech_threshold=settings.input.speech_threshold,
                pre_speech_ms=settings.input.pre_speech_ms,
                speech_hold_ms=settings.input.speech_hold_ms,
                silence_threshold_ms=settings.input.silence_threshold_ms,
                min_utterance_ms=settings.input.min_utterance_ms,
                max_utterance_ms=settings.input.max_utterance_ms,
                device=settings.input.device,
                prefer_external_device=settings.input.prefer_external_device,
            )
        )
    if settings.application_audio.enabled:
        if not settings.application_audio.target:
            raise ValueError(
                "application audio input currently requires VOCALIVE_APP_AUDIO_TARGET"
            )
        application_audio_input_class = application_audio_input_class_for_platform(sys.platform)
        if application_audio_input_class is None:
            raise ValueError(
                "application audio input currently supports macOS and Windows only"
            )
        if sys.platform == "win32":
            log_event(
                logger,
                "windows_application_audio_loopback_enabled",
                output_provider=settings.output.provider,
            )
        live_inputs.append(
            application_audio_input_class(
                target=settings.application_audio.target,
                sample_rate_hz=settings.application_audio.sample_rate_hz,
                channels=settings.application_audio.channels,
                block_duration_ms=settings.application_audio.block_duration_ms,
                speech_threshold=settings.application_audio.speech_threshold,
                pre_speech_ms=settings.application_audio.pre_speech_ms,
                speech_hold_ms=settings.application_audio.speech_hold_ms,
                silence_threshold_ms=settings.application_audio.silence_threshold_ms,
                min_utterance_ms=settings.application_audio.min_utterance_ms,
                max_utterance_ms=settings.application_audio.max_utterance_ms,
                timeout_seconds=settings.application_audio.timeout_seconds,
                adaptive_vad_enabled=settings.application_audio.adaptive_vad_enabled,
                speech_start_events_enabled=(
                    settings.application_audio.mode is ApplicationAudioMode.RESPOND
                    or (
                        settings.conversation_window.enabled
                        and settings.conversation_window.apply_to_application_audio
                    )
                ),
            )
        )
    if not live_inputs:
        return None
    if len(live_inputs) == 1:
        return live_inputs[0]
    return CombinedAudioInput(live_inputs)


def build_overlay(settings: AppSettings) -> OverlayServer | None:
    if not settings.overlay.enabled:
        return None
    return OverlayServer(settings.overlay)


def build_managed_aivis_engine(settings: AppSettings) -> ManagedAivisSpeechEngine | None:
    if settings.tts_provider != "aivis":
        return None
    if settings.aivis.engine_mode is AivisEngineMode.EXTERNAL:
        return None
    return ManagedAivisSpeechEngine(settings.aivis)


async def run_stdin_shell(orchestrator: ConversationOrchestrator) -> int:
    print("VocaLive shell. Type text to simulate microphone input. Use /quit to exit.")
    while True:
        user_text = await asyncio.to_thread(input, "you> ")
        normalized_text = user_text.strip()
        if not normalized_text:
            continue
        if normalized_text in {"/quit", "quit", "exit"}:
            return 0
        accepted = await orchestrator.submit_utterance(AudioSegment.from_text(normalized_text))
        if not accepted:
            print("assistant> queue full, utterance dropped")
            continue
        await orchestrator.wait_for_idle()
        message = orchestrator.session.last_assistant_message()
        if message is not None:
            print(f"assistant> {message.content}")


async def run_microphone_loop(
    orchestrator: ConversationOrchestrator,
    audio_input: AudioInput,
    *,
    settings: AppSettings | None = None,
) -> int:
    resolved_settings = settings or AppSettings()
    conversation_window = configure_live_audio_input(
        audio_input,
        orchestrator,
        settings=resolved_settings,
    )
    monitor_task = _maybe_start_conversation_window_monitor(
        orchestrator,
        conversation_window=conversation_window,
    )
    selected_input = await audio_input.start()
    display_label = _format_live_input_label(selected_input, conversation_window)
    if selected_input:
        print(f"VocaLive live audio mode. Using {display_label}. Ctrl-C to exit.")
    elif display_label:
        print(f"VocaLive live audio mode. {display_label}. Ctrl-C to exit.")
    else:
        print("VocaLive live audio mode. Ctrl-C to exit.")
    try:
        return await forward_live_audio_segments(
            audio_input,
            orchestrator,
            conversation_window=conversation_window,
            print_queue_overflow=True,
        )
    finally:
        if monitor_task is not None:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task


def uses_live_audio_input(settings: AppSettings) -> bool:
    return (
        settings.input.provider == InputProvider.MICROPHONE
        or settings.application_audio.enabled
    )


def application_audio_input_class_for_platform(
    platform: str,
) -> type[AudioInput] | None:
    if platform == "darwin":
        from vocalive.audio.macos_application import MacOSApplicationAudioInput

        return MacOSApplicationAudioInput
    if platform == "win32":
        from vocalive.audio.windows_application import WindowsApplicationAudioInput

        return WindowsApplicationAudioInput
    return None


def screen_capture_engine_class_for_platform(platform: str):
    if platform == "darwin":
        from vocalive.screen.macos import MacOSWindowScreenCapture

        return MacOSWindowScreenCapture
    if platform == "win32":
        from vocalive.screen.windows import WindowsWindowScreenCapture

        return WindowsWindowScreenCapture
    return None


def _uses_live_audio_input(settings: AppSettings) -> bool:
    return uses_live_audio_input(settings)


def configure_live_audio_input(
    audio_input: AudioInput,
    orchestrator: ConversationOrchestrator,
    *,
    settings: AppSettings,
) -> ConversationWindowGate:
    conversation_window = ConversationWindowGate(
        settings.conversation_window,
        session_id=settings.session_id,
    )
    audio_input.set_speech_start_handler(
        conversation_window.wrap_speech_start_handler(orchestrator.handle_user_speech_start)
    )
    return conversation_window


async def forward_live_audio_segments(
    audio_input: AudioInput,
    orchestrator: ConversationOrchestrator,
    *,
    conversation_window: ConversationWindowGate,
    print_queue_overflow: bool = False,
) -> int:
    while True:
        segment = await audio_input.read()
        if segment is None:
            return 0
        if not conversation_window.should_forward_segment(segment):
            continue
        if conversation_window.consume_history_reset_request():
            await orchestrator.handle_conversation_window_reopened(
                reason="conversation_window_reopened"
            )
        accepted = await orchestrator.submit_utterance(segment)
        if accepted or not print_queue_overflow:
            continue
        print("assistant> queue full, utterance dropped")


def _maybe_start_conversation_window_monitor(
    orchestrator: ConversationOrchestrator,
    *,
    conversation_window: ConversationWindowGate,
) -> asyncio.Task[None] | None:
    if not conversation_window.enabled:
        return None
    return asyncio.create_task(
        _monitor_conversation_window(orchestrator, conversation_window=conversation_window),
        name="vocalive-conversation-window-monitor",
    )


async def _monitor_conversation_window(
    orchestrator: ConversationOrchestrator,
    *,
    conversation_window: ConversationWindowGate,
) -> None:
    while True:
        await asyncio.sleep(1.0)
        conversation_window.poll_state()
        if not conversation_window.consume_resume_summary_capture_request():
            continue
        await orchestrator.prepare_conversation_window_resume_summary()


def _format_live_input_label(
    selected_input: str | None,
    conversation_window: ConversationWindowGate,
) -> str:
    summary = conversation_window.summary()
    if not selected_input:
        return summary or ""
    if summary is None:
        return selected_input
    return f"{selected_input} ({summary})"
