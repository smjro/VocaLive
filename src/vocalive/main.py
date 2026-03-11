from __future__ import annotations

import asyncio
import sys

from vocalive.audio.input import AudioInput, MicrophoneAudioInput
from vocalive.audio.output import MemoryAudioOutput, SpeakerAudioOutput, parse_playback_command
from vocalive.config.settings import AppSettings, InputProvider, OutputProvider
from vocalive.llm.echo import EchoLanguageModel
from vocalive.llm.gemini import GeminiLanguageModel
from vocalive.models import AudioSegment
from vocalive.pipeline.orchestrator import ConversationOrchestrator
from vocalive.screen.macos import MacOSWindowScreenCapture
from vocalive.stt.mock import MockSpeechToTextEngine
from vocalive.stt.moonshine import MoonshineSpeechToTextEngine
from vocalive.tts.mock import MockTextToSpeechEngine
from vocalive.tts.aivis import AivisSpeechTextToSpeechEngine
from vocalive.util.logging import configure_logging


async def run_cli() -> int:
    settings = AppSettings.from_env()
    configure_logging(settings.log_level)
    orchestrator = build_orchestrator(settings)
    audio_input = build_audio_input(settings)
    await orchestrator.start()

    try:
        if audio_input is None:
            return await _run_stdin_shell(orchestrator)
        return await _run_microphone_loop(orchestrator, audio_input)
    finally:
        if audio_input is not None:
            await audio_input.close()
        await orchestrator.stop()


def build_orchestrator(settings: AppSettings) -> ConversationOrchestrator:
    if settings.input.provider == InputProvider.MICROPHONE and settings.stt_provider == "mock":
        raise ValueError(
            "microphone input requires a real STT adapter; set VOCALIVE_STT_PROVIDER=moonshine"
        )
    if settings.stt_provider == "moonshine":
        stt_engine = MoonshineSpeechToTextEngine(
            model_name=settings.moonshine.model_name,
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
        if sys.platform != "darwin":
            raise ValueError(
                "screen capture input currently supports macOS only"
            )
        screen_capture_engine = MacOSWindowScreenCapture(
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
        screen_capture_engine=screen_capture_engine,
    )


def build_audio_input(settings: AppSettings) -> AudioInput | None:
    if settings.input.provider == InputProvider.STDIN:
        return None
    return MicrophoneAudioInput(
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


async def _run_stdin_shell(orchestrator: ConversationOrchestrator) -> int:
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


async def _run_microphone_loop(
    orchestrator: ConversationOrchestrator,
    audio_input: AudioInput,
) -> int:
    if isinstance(audio_input, MicrophoneAudioInput):
        audio_input.set_speech_start_handler(orchestrator.handle_user_speech_start)
        selected_device = await audio_input.start()
        print(f"VocaLive microphone mode. Using {selected_device}. Ctrl-C to exit.")
    else:
        print("VocaLive microphone mode. Speak into the selected input device. Ctrl-C to exit.")
    while True:
        segment = await audio_input.read()
        if segment is None:
            return 0
        accepted = await orchestrator.submit_utterance(segment)
        if not accepted:
            print("assistant> queue full, utterance dropped")
            continue


def main() -> int:
    try:
        return asyncio.run(run_cli())
    except KeyboardInterrupt:
        return 130
