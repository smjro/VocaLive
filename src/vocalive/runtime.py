from __future__ import annotations

import asyncio
import sys

from vocalive.audio.input import AudioInput, CombinedAudioInput, MicrophoneAudioInput
from vocalive.audio.output import MemoryAudioOutput, SpeakerAudioOutput, parse_playback_command
from vocalive.config.settings import (
    AppSettings,
    ApplicationAudioMode,
    InputProvider,
    OutputProvider,
)
from vocalive.llm.echo import EchoLanguageModel
from vocalive.llm.gemini import GeminiLanguageModel
from vocalive.models import AudioSegment
from vocalive.pipeline.events import ConversationEventSink
from vocalive.pipeline.orchestrator import ConversationOrchestrator
from vocalive.stt.mock import MockSpeechToTextEngine
from vocalive.stt.moonshine import MoonshineSpeechToTextEngine
from vocalive.tts.aivis import AivisSpeechTextToSpeechEngine
from vocalive.tts.mock import MockTextToSpeechEngine
from vocalive.tts.voicevox import VoicevoxTextToSpeechEngine
from vocalive.ui.overlay import OverlayServer
from vocalive.util.logging import get_logger, log_event


logger = get_logger(__name__)


async def run_headless(settings: AppSettings) -> int:
    overlay = build_overlay(settings)
    orchestrator = build_orchestrator(settings, event_sink=overlay)
    audio_input = build_audio_input(settings)

    try:
        if overlay is not None:
            await overlay.start()
            print(f"VocaLive overlay: {overlay.url}")
        await orchestrator.start()
        if audio_input is None:
            return await run_stdin_shell(orchestrator)
        return await run_microphone_loop(orchestrator, audio_input)
    finally:
        if audio_input is not None:
            await audio_input.close()
        await orchestrator.stop()
        if overlay is not None:
            await overlay.stop()


def build_orchestrator(
    settings: AppSettings,
    event_sink: ConversationEventSink | None = None,
) -> ConversationOrchestrator:
    if _uses_live_audio_input(settings) and settings.stt_provider == "mock":
        raise ValueError(
            "microphone/application audio input requires a real STT adapter; "
            "set VOCALIVE_STT_PROVIDER=moonshine"
        )
    if settings.stt_provider == "moonshine":
        stt_engine = MoonshineSpeechToTextEngine(
            model_name=settings.moonshine.model_name,
            default_language=settings.conversation.language,
            application_audio_enhancement_enabled=(
                settings.application_audio.stt_enhancement_enabled
            ),
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
    if settings.tts_provider == "aivis":
        tts_engine = AivisSpeechTextToSpeechEngine(
            base_url=settings.aivis.base_url,
            speaker_id=settings.aivis.speaker_id,
            speaker_name=settings.aivis.speaker_name,
            style_name=settings.aivis.style_name,
            timeout_seconds=settings.aivis.timeout_seconds,
        )
    elif settings.tts_provider == "voicevox":
        tts_engine = VoicevoxTextToSpeechEngine(
            base_url=settings.voicevox.base_url,
            speaker_id=settings.voicevox.speaker_id,
            speaker_name=settings.voicevox.speaker_name,
            style_name=settings.voicevox.style_name,
            timeout_seconds=settings.voicevox.timeout_seconds,
        )
    else:
        tts_engine = MockTextToSpeechEngine()
    if settings.output.provider == OutputProvider.SPEAKER:
        if settings.tts_provider not in {"aivis", "voicevox"}:
            raise ValueError(
                "speaker output currently requires VOCALIVE_TTS_PROVIDER=aivis or voicevox"
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
) -> int:
    audio_input.set_speech_start_handler(orchestrator.handle_user_speech_start)
    selected_input = await audio_input.start()
    if selected_input:
        print(f"VocaLive live audio mode. Using {selected_input}. Ctrl-C to exit.")
    else:
        print("VocaLive live audio mode. Ctrl-C to exit.")
    while True:
        segment = await audio_input.read()
        if segment is None:
            return 0
        accepted = await orchestrator.submit_utterance(segment)
        if not accepted:
            print("assistant> queue full, utterance dropped")
            continue


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
