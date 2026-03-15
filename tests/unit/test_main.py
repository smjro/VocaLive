from __future__ import annotations

import asyncio
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.input import MicrophoneAudioInput
from vocalive.audio.output import MemoryAudioOutput
from vocalive.config.settings import (
    AppSettings,
    ApplicationAudioMode,
    ApplicationAudioSettings,
    InputProvider,
    InputSettings,
    OverlaySettings,
    OutputProvider,
    OutputSettings,
    ScreenCaptureSettings,
)
from vocalive.llm.gemini import GeminiLanguageModel
from vocalive.main import (
    _run_microphone_loop,
    build_audio_input,
    build_orchestrator,
    build_overlay,
    load_headless_settings,
    main,
)
from vocalive.models import AudioSegment
from vocalive.screen.macos import MacOSWindowScreenCapture
from vocalive.screen.windows import WindowsWindowScreenCapture
from vocalive.stt.moonshine import MoonshineSpeechToTextEngine
from vocalive.tts.aivis import AivisSpeechTextToSpeechEngine
from vocalive.tts.voicevox import VoicevoxTextToSpeechEngine
from vocalive.ui.overlay import OverlayServer


class BuildOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def tearDown(self) -> None:
        asyncio.set_event_loop(None)
        self._loop.close()

    def test_build_orchestrator_uses_real_provider_adapters(self) -> None:
        orchestrator = build_orchestrator(
            AppSettings(
                stt_provider="Moonshine Voice",
                model_provider="Google Gemini",
                tts_provider="Aivis Speech",
            )
        )

        self.assertIsInstance(orchestrator.stt_engine, MoonshineSpeechToTextEngine)
        assert isinstance(orchestrator.stt_engine, MoonshineSpeechToTextEngine)
        self.assertTrue(orchestrator.stt_engine.application_audio_enhancement_enabled)
        self.assertIsInstance(orchestrator.language_model, GeminiLanguageModel)
        self.assertEqual(orchestrator.language_model.thinking_budget, 0)
        self.assertIsInstance(orchestrator.tts_engine, AivisSpeechTextToSpeechEngine)
        self.assertIsInstance(orchestrator.audio_output, MemoryAudioOutput)

    def test_build_orchestrator_supports_voicevox_tts_with_speaker_output(self) -> None:
        orchestrator = build_orchestrator(
            AppSettings(
                tts_provider="VOICEVOX",
                output=OutputSettings(
                    provider=OutputProvider.SPEAKER,
                    speaker_command="playback {path}",
                ),
            )
        )

        self.assertIsInstance(orchestrator.tts_engine, VoicevoxTextToSpeechEngine)
        self.assertEqual(
            orchestrator.audio_output.playback_command,
            ("playback", "{path}"),
        )

    def test_build_audio_input_propagates_device_preferences(self) -> None:
        audio_input = build_audio_input(
            AppSettings(
                input=InputSettings(
                    provider=InputProvider.MICROPHONE,
                    device="external",
                    prefer_external_device=False,
                    pre_speech_ms=180.0,
                    speech_hold_ms=350.0,
                ),
            )
        )

        self.assertIsInstance(audio_input, MicrophoneAudioInput)
        assert isinstance(audio_input, MicrophoneAudioInput)
        self.assertEqual(audio_input.device, "external")
        self.assertFalse(audio_input.prefer_external_device)
        self.assertEqual(audio_input._accumulator.pre_speech_ms, 180.0)
        self.assertEqual(audio_input._accumulator.speech_hold_ms, 350.0)

    def test_build_overlay_returns_server_when_enabled(self) -> None:
        overlay = build_overlay(
            AppSettings(
                overlay=OverlaySettings(
                    enabled=True,
                    auto_open=False,
                )
            )
        )

        self.assertIsInstance(overlay, OverlayServer)

    def test_build_audio_input_combines_microphone_and_application_audio(self) -> None:
        captured_kwargs: list[dict[str, object]] = []

        class _FakeApplicationAudioInput:
            def __init__(self, **kwargs) -> None:
                captured_kwargs.append(kwargs)
                self.kwargs = kwargs

            async def start(self) -> str:
                return "application audio"

            def set_speech_start_handler(self, handler) -> None:
                del handler

            async def read(self) -> AudioSegment | None:
                return None

            async def close(self) -> None:
                return None

        with patch("vocalive.runtime.sys.platform", "darwin"), patch(
            "vocalive.runtime.application_audio_input_class_for_platform",
            return_value=_FakeApplicationAudioInput,
        ):
            audio_input = build_audio_input(
                AppSettings(
                    input=InputSettings(provider=InputProvider.MICROPHONE),
                    application_audio=ApplicationAudioSettings(
                        enabled=True,
                        target="Steam",
                        timeout_seconds=12.0,
                        adaptive_vad_enabled=False,
                    ),
                )
            )

        self.assertIsNotNone(audio_input)
        assert audio_input is not None
        self.assertEqual(type(audio_input).__name__, "CombinedAudioInput")
        self.assertEqual(len(captured_kwargs), 1)
        self.assertFalse(captured_kwargs[0]["adaptive_vad_enabled"])
        self.assertFalse(captured_kwargs[0]["speech_start_events_enabled"])

    def test_build_audio_input_enables_application_audio_speech_events_in_respond_mode(self) -> None:
        captured_kwargs: list[dict[str, object]] = []

        class _FakeApplicationAudioInput:
            def __init__(self, **kwargs) -> None:
                captured_kwargs.append(kwargs)
                self.kwargs = kwargs

            async def start(self) -> str:
                return "application audio"

            def set_speech_start_handler(self, handler) -> None:
                del handler

            async def read(self) -> AudioSegment | None:
                return None

            async def close(self) -> None:
                return None

        with patch("vocalive.runtime.sys.platform", "darwin"), patch(
            "vocalive.runtime.application_audio_input_class_for_platform",
            return_value=_FakeApplicationAudioInput,
        ):
            build_audio_input(
                AppSettings(
                    application_audio=ApplicationAudioSettings(
                        enabled=True,
                        mode=ApplicationAudioMode.RESPOND,
                        target="Steam",
                    ),
                )
            )

        self.assertEqual(len(captured_kwargs), 1)
        self.assertTrue(captured_kwargs[0]["speech_start_events_enabled"])

    def test_build_audio_input_supports_windows_application_audio(self) -> None:
        with patch("vocalive.runtime.sys.platform", "win32"):
            audio_input = build_audio_input(
                AppSettings(
                    application_audio=ApplicationAudioSettings(
                        enabled=True,
                        target="chrome",
                    ),
                )
            )

        self.assertIsNotNone(audio_input)
        assert audio_input is not None
        self.assertEqual(type(audio_input).__name__, "WindowsApplicationAudioInput")

    def test_build_orchestrator_can_disable_application_audio_enhancement(self) -> None:
        orchestrator = build_orchestrator(
            AppSettings(
                stt_provider="moonshine",
                application_audio=ApplicationAudioSettings(
                    stt_enhancement_enabled=False,
                ),
            )
        )

        self.assertIsInstance(orchestrator.stt_engine, MoonshineSpeechToTextEngine)
        assert isinstance(orchestrator.stt_engine, MoonshineSpeechToTextEngine)
        self.assertFalse(orchestrator.stt_engine.application_audio_enhancement_enabled)

    def test_build_orchestrator_rejects_application_audio_with_mock_stt(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "microphone/application audio input requires a real STT adapter",
        ):
            build_orchestrator(
                AppSettings(
                    application_audio=ApplicationAudioSettings(
                        enabled=True,
                        target="Steam",
                    ),
                )
            )

    def test_build_orchestrator_enables_screen_capture_for_gemini_on_macos(self) -> None:
        with patch("vocalive.runtime.sys.platform", "darwin"):
            orchestrator = build_orchestrator(
                AppSettings(
                    model_provider="gemini",
                    screen_capture=ScreenCaptureSettings(
                        enabled=True,
                        window_name="YouTube",
                        timeout_seconds=7.0,
                        resize_max_edge_px=960,
                    ),
                )
            )

        self.assertIsInstance(orchestrator.language_model, GeminiLanguageModel)
        self.assertIsInstance(orchestrator.screen_capture_engine, MacOSWindowScreenCapture)
        assert isinstance(orchestrator.screen_capture_engine, MacOSWindowScreenCapture)
        self.assertEqual(orchestrator.screen_capture_engine.window_name, "YouTube")
        self.assertEqual(orchestrator.screen_capture_engine.timeout_seconds, 7.0)
        self.assertEqual(orchestrator.screen_capture_engine.resize_max_edge_px, 960)

    def test_build_orchestrator_enables_screen_capture_for_gemini_on_windows(self) -> None:
        with patch("vocalive.runtime.sys.platform", "win32"):
            orchestrator = build_orchestrator(
                AppSettings(
                    model_provider="gemini",
                    screen_capture=ScreenCaptureSettings(
                        enabled=True,
                        window_name="Chrome",
                        timeout_seconds=6.0,
                        resize_max_edge_px=1024,
                    ),
                )
            )

        self.assertIsInstance(orchestrator.language_model, GeminiLanguageModel)
        self.assertIsInstance(orchestrator.screen_capture_engine, WindowsWindowScreenCapture)
        assert isinstance(orchestrator.screen_capture_engine, WindowsWindowScreenCapture)
        self.assertEqual(orchestrator.screen_capture_engine.window_name, "Chrome")
        self.assertEqual(orchestrator.screen_capture_engine.timeout_seconds, 6.0)
        self.assertEqual(orchestrator.screen_capture_engine.resize_max_edge_px, 1024)

    def test_build_orchestrator_rejects_screen_capture_without_gemini(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "screen capture input currently requires VOCALIVE_MODEL_PROVIDER=gemini",
        ):
            build_orchestrator(
                AppSettings(
                    screen_capture=ScreenCaptureSettings(enabled=True),
                )
            )

    def test_build_orchestrator_rejects_screen_capture_without_window_name(self) -> None:
        with patch("vocalive.runtime.sys.platform", "darwin"):
            with self.assertRaisesRegex(
                ValueError,
                "screen capture input currently requires VOCALIVE_SCREEN_WINDOW_NAME",
            ):
                build_orchestrator(
                    AppSettings(
                        model_provider="gemini",
                        screen_capture=ScreenCaptureSettings(enabled=True),
                    )
                )

    def test_build_orchestrator_rejects_screen_capture_on_unsupported_platform(self) -> None:
        with patch("vocalive.runtime.sys.platform", "linux"):
            with self.assertRaisesRegex(
                ValueError,
                "screen capture input currently supports macOS and Windows only",
            ):
                build_orchestrator(
                    AppSettings(
                        model_provider="gemini",
                        screen_capture=ScreenCaptureSettings(
                            enabled=True,
                            window_name="YouTube",
                        ),
                    )
                )


class _ScriptedMicrophoneInput(MicrophoneAudioInput):
    def __init__(self, segments: list[AudioSegment | None]) -> None:
        self._segments = list(segments)
        self.speech_start_handler = None

    def set_speech_start_handler(self, handler):  # type: ignore[override]
        self.speech_start_handler = handler

    async def start(self) -> str:
        return "test microphone"

    async def read(self) -> AudioSegment | None:
        await asyncio.sleep(0)
        return self._segments.pop(0) if self._segments else None


class _RecordingOrchestrator:
    def __init__(self) -> None:
        self.submitted: list[str | None] = []
        self.wait_for_idle_calls = 0

    async def submit_utterance(self, segment: AudioSegment) -> bool:
        self.submitted.append(segment.transcript_hint)
        return True

    async def handle_user_speech_start(self, source="user") -> None:
        del source
        return None

    async def wait_for_idle(self) -> None:
        self.wait_for_idle_calls += 1


class MicrophoneLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_microphone_loop_keeps_reading_without_waiting_for_idle(self) -> None:
        audio_input = _ScriptedMicrophoneInput(
            [
                AudioSegment.from_text("first"),
                AudioSegment.from_text("second"),
                None,
            ]
        )
        orchestrator = _RecordingOrchestrator()

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = await _run_microphone_loop(orchestrator, audio_input)

        self.assertEqual(exit_code, 0)
        self.assertEqual(orchestrator.submitted, ["first", "second"])
        self.assertEqual(orchestrator.wait_for_idle_calls, 0)
        self.assertIsNotNone(audio_input.speech_start_handler)


class MainEntrypointTests(unittest.TestCase):
    def test_load_headless_settings_merges_saved_values_with_env_overrides(self) -> None:
        class _Store:
            def load_values(self) -> dict[str, str | None]:
                return {
                    "VOCALIVE_INPUT_PROVIDER": "microphone",
                    "VOCALIVE_MODEL_PROVIDER": "mock",
                    "VOCALIVE_GEMINI_API_KEY": None,
                }

        settings = load_headless_settings(
            store=_Store(),  # type: ignore[arg-type]
            environ={
                "VOCALIVE_MODEL_PROVIDER": "gemini",
                "GEMINI_API_KEY": "secret",
            },
        )

        self.assertEqual(settings.input.provider, InputProvider.MICROPHONE)
        self.assertEqual(settings.model_provider, "gemini")
        self.assertEqual(settings.gemini.api_key, "secret")

    def test_main_defaults_to_controller_mode(self) -> None:
        with patch("vocalive.main.run_controller", new=AsyncMock(return_value=0)) as controller_run, patch(
            "vocalive.main.configure_logging"
        ):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        controller_run.assert_awaited_once()

    def test_main_run_mode_uses_headless_runner(self) -> None:
        settings = AppSettings(
            input=InputSettings(provider=InputProvider.MICROPHONE),
        )
        with patch("vocalive.main.load_headless_settings", return_value=settings) as load_settings, patch(
            "vocalive.main.run_headless",
            new=AsyncMock(return_value=0),
        ) as headless_run, patch("vocalive.main.configure_logging"):
            exit_code = main(["run"])

        self.assertEqual(exit_code, 0)
        load_settings.assert_called_once_with()
        headless_run.assert_awaited_once_with(settings)
