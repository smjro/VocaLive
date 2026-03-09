from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import AppSettings


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
