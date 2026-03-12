from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import (
    AppSettings,
    ApplicationAudioMode,
    DEFAULT_GEMINI_SYSTEM_INSTRUCTION,
    DEFAULT_SCREEN_TRIGGER_PHRASES,
)


class AppSettingsTests(unittest.TestCase):
    def test_from_env_normalizes_real_provider_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_STT_PROVIDER": "Moonshine Voice",
                "VOCALIVE_MODEL_PROVIDER": "Google Gemini",
                "VOCALIVE_TTS_PROVIDER": "Aivis Speech",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.stt_provider, "moonshine")
        self.assertEqual(settings.model_provider, "gemini")
        self.assertEqual(settings.tts_provider, "aivis")

    def test_from_env_rejects_unknown_provider_names(self) -> None:
        with patch.dict(os.environ, {"VOCALIVE_TTS_PROVIDER": "unsupported"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported tts provider"):
                AppSettings.from_env()

    def test_from_env_defaults_gemini_thinking_budget_to_zero(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.gemini.thinking_budget, 0)
        self.assertEqual(
            settings.gemini.system_instruction,
            DEFAULT_GEMINI_SYSTEM_INSTRUCTION,
        )
        self.assertEqual(settings.conversation.language, "ja")

    def test_from_env_allows_overriding_gemini_thinking_budget(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_GEMINI_THINKING_BUDGET": "32"},
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.gemini.thinking_budget, 32)

    def test_from_env_parses_microphone_device_and_external_preference(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_MIC_DEVICE": "7",
                "VOCALIVE_MIC_PREFER_EXTERNAL": "false",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.input.device, 7)
        self.assertFalse(settings.input.prefer_external_device)

    def test_from_env_preserves_named_microphone_device(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_MIC_DEVICE": "AirPods Pro Hands-Free",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.input.device, "AirPods Pro Hands-Free")

    def test_from_env_reads_microphone_turn_detection_tuning(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_MIC_PRE_SPEECH_MS": "180",
                "VOCALIVE_MIC_SPEECH_HOLD_MS": "320",
                "VOCALIVE_MIC_SILENCE_MS": "750",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.input.pre_speech_ms, 180.0)
        self.assertEqual(settings.input.speech_hold_ms, 320.0)
        self.assertEqual(settings.input.silence_threshold_ms, 750.0)

    def test_from_env_allows_disabling_default_conversation_language(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_CONVERSATION_LANGUAGE": "",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertIsNone(settings.conversation.language)

    def test_from_env_allows_disabling_default_gemini_system_instruction(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_GEMINI_SYSTEM_INSTRUCTION": "",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertIsNone(settings.gemini.system_instruction)

    def test_from_env_reads_screen_capture_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_SCREEN_CAPTURE_ENABLED": "true",
                "VOCALIVE_SCREEN_WINDOW_NAME": "YouTube",
                "VOCALIVE_SCREEN_TRIGGER_PHRASES": "画面見て, screen please",
                "VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS": "7.5",
                "VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX": "960",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertTrue(settings.screen_capture.enabled)
        self.assertEqual(settings.screen_capture.window_name, "YouTube")
        self.assertEqual(
            settings.screen_capture.trigger_phrases,
            ("画面見て", "screen please"),
        )
        self.assertEqual(settings.screen_capture.timeout_seconds, 7.5)
        self.assertEqual(settings.screen_capture.resize_max_edge_px, 960)

    def test_from_env_reads_application_audio_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_APP_AUDIO_ENABLED": "true",
                "VOCALIVE_APP_AUDIO_MODE": "respond",
                "VOCALIVE_APP_AUDIO_TARGET": "Steam",
                "VOCALIVE_APP_AUDIO_SAMPLE_RATE": "22050",
                "VOCALIVE_APP_AUDIO_CHANNELS": "2",
                "VOCALIVE_APP_AUDIO_BLOCK_MS": "60",
                "VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD": "0.015",
                "VOCALIVE_APP_AUDIO_PRE_SPEECH_MS": "80",
                "VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS": "240",
                "VOCALIVE_APP_AUDIO_SILENCE_MS": "650",
                "VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS": "320",
                "VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS": "9000",
                "VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS": "12.5",
                "VOCALIVE_APP_AUDIO_ADAPTIVE_VAD": "false",
                "VOCALIVE_APP_AUDIO_STT_ENHANCEMENT": "false",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertTrue(settings.application_audio.enabled)
        self.assertEqual(settings.application_audio.mode, ApplicationAudioMode.RESPOND)
        self.assertEqual(settings.application_audio.target, "Steam")
        self.assertEqual(settings.application_audio.sample_rate_hz, 22_050)
        self.assertEqual(settings.application_audio.channels, 2)
        self.assertEqual(settings.application_audio.block_duration_ms, 60.0)
        self.assertEqual(settings.application_audio.speech_threshold, 0.015)
        self.assertEqual(settings.application_audio.pre_speech_ms, 80.0)
        self.assertEqual(settings.application_audio.speech_hold_ms, 240.0)
        self.assertEqual(settings.application_audio.silence_threshold_ms, 650.0)
        self.assertEqual(settings.application_audio.min_utterance_ms, 320.0)
        self.assertEqual(settings.application_audio.max_utterance_ms, 9000.0)
        self.assertEqual(settings.application_audio.timeout_seconds, 12.5)
        self.assertFalse(settings.application_audio.adaptive_vad_enabled)
        self.assertFalse(settings.application_audio.stt_enhancement_enabled)

    def test_from_env_reads_context_compaction_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT": "5",
                "VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS": "640",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.context.recent_message_count, 5)
        self.assertEqual(settings.context.conversation_summary_max_chars, 640)

    def test_from_env_defaults_screen_capture_triggers(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(
            settings.screen_capture.trigger_phrases,
            DEFAULT_SCREEN_TRIGGER_PHRASES,
        )
        self.assertEqual(settings.context.recent_message_count, 8)
        self.assertEqual(settings.context.conversation_summary_max_chars, 1200)
        self.assertFalse(settings.application_audio.enabled)
        self.assertEqual(
            settings.application_audio.mode,
            ApplicationAudioMode.CONTEXT_ONLY,
        )
        self.assertIsNone(settings.application_audio.target)
        self.assertEqual(settings.application_audio.pre_speech_ms, 200.0)
        self.assertEqual(settings.application_audio.speech_hold_ms, 320.0)
        self.assertEqual(settings.application_audio.silence_threshold_ms, 650.0)
        self.assertTrue(settings.application_audio.adaptive_vad_enabled)
        self.assertTrue(settings.application_audio.stt_enhancement_enabled)

    def test_from_env_rejects_unknown_application_audio_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_APP_AUDIO_MODE": "unsupported"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported application audio mode"):
                AppSettings.from_env()
