from __future__ import annotations

import os
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import (
    AppSettings,
    AivisEngineMode,
    ApplicationAudioMode,
    ConversationWindowResetPolicy,
    DEFAULT_GEMINI_SYSTEM_INSTRUCTION,
    DEFAULT_SCREEN_PASSIVE_TRIGGER_PHRASES,
    DEFAULT_SCREEN_TRIGGER_PHRASES,
    MicrophoneInterruptMode,
    controller_setting_definitions,
    controller_setting_rows,
    controller_setting_schema,
    normalize_controller_values,
)


def _extract_markdown_table_rows(
    path: Path,
    *,
    header: str,
    end_marker: str,
) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    start = text.index(header)
    end = text.index(end_marker, start)
    lines = text[start:end].splitlines()[2:]

    rows: dict[str, dict[str, str]] = {}
    for line in lines:
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 3:
            continue
        env_name, default_label, description = parts
        rows[env_name.strip("`")] = {
            "default_label": default_label.strip("`"),
            "description": description,
        }
    return rows


class AppSettingsTests(unittest.TestCase):
    def test_from_mapping_matches_from_env(self) -> None:
        payload = {
            "VOCALIVE_SESSION_ID": "session-fixed",
            "VOCALIVE_STT_PROVIDER": "Moonshine Voice",
            "VOCALIVE_MODEL_PROVIDER": "Google Gemini",
            "VOCALIVE_TTS_PROVIDER": "Aivis Speech",
            "VOCALIVE_INPUT_PROVIDER": "microphone",
            "VOCALIVE_APP_AUDIO_ENABLED": "true",
            "VOCALIVE_APP_AUDIO_TARGET": "Steam",
            "VOCALIVE_OVERLAY_ENABLED": "true",
            "VOCALIVE_GEMINI_API_KEY": "secret",
            "VOCALIVE_GEMINI_SYSTEM_INSTRUCTION": "",
        }

        with patch.dict(os.environ, payload, clear=True):
            from_env = AppSettings.from_env()
            from_mapping = AppSettings.from_mapping(dict(os.environ))

        self.assertEqual(asdict(from_env), asdict(from_mapping))

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

    def test_from_env_normalizes_openai_stt_alias_and_api_key_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_STT_PROVIDER": "gpt-4o-mini-transcribe",
                "OPENAI_API_KEY": "secret",
                "VOCALIVE_OPENAI_BASE_URL": "https://example.test/v1",
                "VOCALIVE_OPENAI_TIMEOUT_SECONDS": "12.5",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.stt_provider, "openai")
        self.assertEqual(settings.openai.api_key, "secret")
        self.assertEqual(settings.openai.model_name, "gpt-4o-mini-transcribe")
        self.assertEqual(settings.openai.base_url, "https://example.test/v1")
        self.assertEqual(settings.openai.timeout_seconds, 12.5)

    def test_from_env_rejects_unknown_provider_names(self) -> None:
        with patch.dict(os.environ, {"VOCALIVE_TTS_PROVIDER": "unsupported"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported tts provider"):
                AppSettings.from_env()

    def test_from_env_reads_aivis_engine_startup_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_AIVIS_ENGINE_MODE": "gpu",
                "VOCALIVE_AIVIS_ENGINE_PATH": r"C:\AivisSpeech\run.exe",
                "VOCALIVE_AIVIS_CPU_NUM_THREADS": "4",
                "VOCALIVE_AIVIS_STARTUP_TIMEOUT_SECONDS": "75.0",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.aivis.engine_mode, AivisEngineMode.GPU)
        self.assertEqual(settings.aivis.engine_path, r"C:\AivisSpeech\run.exe")
        self.assertEqual(settings.aivis.cpu_num_threads, 4)
        self.assertEqual(settings.aivis.startup_timeout_seconds, 75.0)

    def test_from_env_rejects_non_positive_aivis_cpu_thread_limit(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_AIVIS_CPU_NUM_THREADS": "0",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "VOCALIVE_AIVIS_CPU_NUM_THREADS"):
                AppSettings.from_env()

    def test_from_env_reads_application_audio_transcription_cooldown(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS": "2.5",
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS": "1200",
                "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS": "800",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.application_audio.transcription_cooldown_seconds, 2.5)
        self.assertEqual(settings.application_audio.transcription_debounce_ms, 1200.0)
        self.assertEqual(settings.application_audio.min_transcription_duration_ms, 800.0)

    def test_from_env_rejects_negative_application_audio_transcription_cooldown(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS": "-1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS",
            ):
                AppSettings.from_env()

    def test_from_env_rejects_negative_application_audio_transcription_debounce(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS": "-1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS",
            ):
                AppSettings.from_env()

    def test_from_env_rejects_negative_application_audio_min_transcription_duration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS": "-1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS",
            ):
                AppSettings.from_env()

    def test_from_env_defaults_gemini_thinking_budget_to_zero(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.gemini.thinking_budget, 0)
        self.assertEqual(
            settings.gemini.system_instruction,
            DEFAULT_GEMINI_SYSTEM_INSTRUCTION,
        )
        assert settings.gemini.system_instruction is not None
        self.assertIn("your name is コハク", settings.gemini.system_instruction.lower())
        self.assertIn("Do not use emoji", settings.gemini.system_instruction)
        self.assertIn("Stay on one main topic per reply", settings.gemini.system_instruction)
        self.assertIn("Do not start replies by addressing the user by name", settings.gemini.system_instruction)
        self.assertNotIn("ましま", settings.gemini.system_instruction)
        self.assertEqual(settings.conversation.language, "ja")

    def test_from_env_allows_overriding_gemini_thinking_budget(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_GEMINI_THINKING_BUDGET": "32"},
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.gemini.thinking_budget, 32)

    def test_from_env_reads_conversation_window_reset_policy(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_CONVERSATION_WINDOW_RESET_POLICY": "resume_summary"},
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(
            settings.conversation_window.reset_policy,
            ConversationWindowResetPolicy.RESUME_SUMMARY,
        )

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
                "VOCALIVE_MIC_INTERRUPT_MODE": "explicit",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.input.pre_speech_ms, 180.0)
        self.assertEqual(settings.input.speech_hold_ms, 320.0)
        self.assertEqual(settings.input.silence_threshold_ms, 750.0)
        self.assertEqual(settings.input.interrupt_mode, MicrophoneInterruptMode.EXPLICIT)

    def test_from_env_reads_conversation_window_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_CONVERSATION_WINDOW_ENABLED": "true",
                "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS": "30",
                "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS": "240",
                "VOCALIVE_CONVERSATION_WINDOW_START_OPEN": "false",
                "VOCALIVE_CONVERSATION_WINDOW_APPLY_TO_APP_AUDIO": "true",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertTrue(settings.conversation_window.enabled)
        self.assertEqual(settings.conversation_window.open_duration_seconds, 30.0)
        self.assertEqual(settings.conversation_window.closed_duration_seconds, 240.0)
        self.assertFalse(settings.conversation_window.start_open)
        self.assertTrue(settings.conversation_window.apply_to_application_audio)

    def test_from_env_rejects_non_positive_conversation_window_open_seconds(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS": "0",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS",
            ):
                AppSettings.from_env()

    def test_from_env_rejects_negative_conversation_window_closed_seconds(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS": "-1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS",
            ):
                AppSettings.from_env()

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

    def test_from_env_reads_optional_user_name(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_USER_NAME": "ましま",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.conversation.user_name, "ましま")

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

    def test_from_env_reads_overlay_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_OVERLAY_ENABLED": "true",
                "VOCALIVE_OVERLAY_HOST": "0.0.0.0",
                "VOCALIVE_OVERLAY_PORT": "9876",
                "VOCALIVE_OVERLAY_AUTO_OPEN": "false",
                "VOCALIVE_OVERLAY_TITLE": "Overlay Test",
                "VOCALIVE_OVERLAY_CHARACTER_NAME": "Stripe",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertTrue(settings.overlay.enabled)
        self.assertEqual(settings.overlay.host, "0.0.0.0")
        self.assertEqual(settings.overlay.port, 9876)
        self.assertFalse(settings.overlay.auto_open)
        self.assertEqual(settings.overlay.title, "Overlay Test")
        self.assertEqual(settings.overlay.character_name, "Stripe")

    def test_from_env_reads_screen_capture_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_SCREEN_CAPTURE_ENABLED": "true",
                "VOCALIVE_SCREEN_WINDOW_NAME": "YouTube",
                "VOCALIVE_SCREEN_TRIGGER_PHRASES": "画面見て, screen please",
                "VOCALIVE_SCREEN_PASSIVE_ENABLED": "true",
                "VOCALIVE_SCREEN_PASSIVE_TRIGGER_PHRASES": "この画面, 見えてる",
                "VOCALIVE_SCREEN_PASSIVE_COOLDOWN_SECONDS": "22.5",
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
        self.assertTrue(settings.screen_capture.passive_enabled)
        self.assertEqual(
            settings.screen_capture.passive_trigger_phrases,
            ("この画面", "見えてる"),
        )
        self.assertEqual(settings.screen_capture.passive_cooldown_seconds, 22.5)
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
                "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS": "900",
                "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS": "700",
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
        self.assertEqual(settings.application_audio.transcription_debounce_ms, 900.0)
        self.assertEqual(settings.application_audio.min_transcription_duration_ms, 700.0)
        self.assertFalse(settings.application_audio.adaptive_vad_enabled)
        self.assertFalse(settings.application_audio.stt_enhancement_enabled)

    def test_from_env_reads_context_compaction_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT": "5",
                "VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS": "75",
                "VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS": "640",
                "VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT": "2",
                "VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS": "420",
                "VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS": "5",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.context.recent_message_count, 5)
        self.assertEqual(settings.context.active_message_max_age_seconds, 75.0)
        self.assertEqual(settings.context.conversation_summary_max_chars, 640)
        self.assertEqual(settings.context.application_recent_message_count, 2)
        self.assertEqual(settings.context.application_summary_max_chars, 420)
        self.assertEqual(settings.context.application_summary_min_message_chars, 5)

    def test_from_env_reads_reply_debounce_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_REPLY_DEBOUNCE_MS": "250",
                "VOCALIVE_REPLY_POLICY_ENABLED": "false",
                "VOCALIVE_REPLY_MIN_GAP_MS": "4200",
                "VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS": "9",
                "VOCALIVE_REPLY_REQUIRE_EXPLICIT_TRIGGER": "true",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertEqual(settings.reply.debounce_ms, 250.0)
        self.assertFalse(settings.reply.policy_enabled)
        self.assertEqual(settings.reply.min_gap_ms, 4200.0)
        self.assertEqual(settings.reply.short_utterance_max_chars, 9)
        self.assertTrue(settings.reply.require_explicit_trigger)

    def test_from_env_reads_proactive_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VOCALIVE_PROACTIVE_ENABLED": "true",
                "VOCALIVE_PROACTIVE_MICROPHONE_ENABLED": "false",
                "VOCALIVE_PROACTIVE_APPLICATION_AUDIO_ENABLED": "false",
                "VOCALIVE_PROACTIVE_SCREEN_ENABLED": "false",
                "VOCALIVE_PROACTIVE_IDLE_SECONDS": "12.5",
                "VOCALIVE_PROACTIVE_COOLDOWN_SECONDS": "33.0",
                "VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS": "4.0",
            },
            clear=True,
        ):
            settings = AppSettings.from_env()

        self.assertTrue(settings.proactive.enabled)
        self.assertFalse(settings.proactive.microphone_enabled)
        self.assertFalse(settings.proactive.application_audio_enabled)
        self.assertFalse(settings.proactive.screen_enabled)
        self.assertEqual(settings.proactive.idle_seconds, 12.5)
        self.assertEqual(settings.proactive.cooldown_seconds, 33.0)
        self.assertEqual(settings.proactive.screen_poll_seconds, 4.0)

    def test_from_env_rejects_non_positive_proactive_idle_seconds(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_PROACTIVE_IDLE_SECONDS": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "VOCALIVE_PROACTIVE_IDLE_SECONDS"):
                AppSettings.from_env()

    def test_from_env_rejects_negative_proactive_cooldown_seconds(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_PROACTIVE_COOLDOWN_SECONDS": "-1"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "VOCALIVE_PROACTIVE_COOLDOWN_SECONDS"):
                AppSettings.from_env()

    def test_from_env_rejects_non_positive_proactive_screen_poll_seconds(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS"):
                AppSettings.from_env()

    def test_from_env_defaults_screen_capture_triggers(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(
            settings.screen_capture.trigger_phrases,
            DEFAULT_SCREEN_TRIGGER_PHRASES,
        )
        self.assertFalse(settings.screen_capture.passive_enabled)
        self.assertEqual(
            settings.screen_capture.passive_trigger_phrases,
            DEFAULT_SCREEN_PASSIVE_TRIGGER_PHRASES,
        )
        self.assertEqual(settings.screen_capture.passive_cooldown_seconds, 30.0)
        self.assertEqual(settings.context.recent_message_count, 8)
        self.assertEqual(settings.context.active_message_max_age_seconds, 90.0)
        self.assertEqual(settings.context.conversation_summary_max_chars, 1200)
        self.assertEqual(settings.context.application_recent_message_count, 4)
        self.assertEqual(settings.context.application_summary_max_chars, 900)
        self.assertEqual(settings.context.application_summary_min_message_chars, 8)
        self.assertEqual(settings.reply.debounce_ms, 200.0)
        self.assertTrue(settings.reply.policy_enabled)
        self.assertEqual(settings.reply.min_gap_ms, 6000.0)
        self.assertEqual(settings.reply.short_utterance_max_chars, 12)
        self.assertFalse(settings.reply.require_explicit_trigger)
        self.assertFalse(settings.proactive.enabled)
        self.assertTrue(settings.proactive.microphone_enabled)
        self.assertTrue(settings.proactive.application_audio_enabled)
        self.assertTrue(settings.proactive.screen_enabled)
        self.assertEqual(settings.proactive.idle_seconds, 20.0)
        self.assertEqual(settings.proactive.cooldown_seconds, 45.0)
        self.assertEqual(settings.proactive.screen_poll_seconds, 10.0)
        self.assertEqual(settings.input.interrupt_mode, MicrophoneInterruptMode.ALWAYS)
        self.assertEqual(
            settings.conversation_window.reset_policy,
            ConversationWindowResetPolicy.CLEAR,
        )
        self.assertFalse(settings.application_audio.enabled)
        self.assertEqual(
            settings.application_audio.mode,
            ApplicationAudioMode.CONTEXT_ONLY,
        )
        self.assertIsNone(settings.application_audio.target)
        self.assertEqual(settings.application_audio.pre_speech_ms, 200.0)
        self.assertEqual(settings.application_audio.speech_hold_ms, 320.0)
        self.assertEqual(settings.application_audio.silence_threshold_ms, 650.0)
        self.assertEqual(settings.application_audio.transcription_debounce_ms, 0.0)
        self.assertEqual(settings.application_audio.min_transcription_duration_ms, 0.0)
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

    def test_from_env_rejects_negative_active_message_max_age(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS": "-1"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS must be >= 0",
            ):
                AppSettings.from_env()

    def test_from_env_rejects_unknown_microphone_interrupt_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"VOCALIVE_MIC_INTERRUPT_MODE": "unsupported"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported microphone interrupt mode"):
                AppSettings.from_env()

    def test_controller_setting_definitions_cover_expected_env_names(self) -> None:
        expected_names = {
            "VOCALIVE_SESSION_ID",
            "VOCALIVE_LOG_LEVEL",
            "VOCALIVE_STT_PROVIDER",
            "VOCALIVE_MODEL_PROVIDER",
            "VOCALIVE_TTS_PROVIDER",
            "VOCALIVE_QUEUE_MAXSIZE",
            "VOCALIVE_QUEUE_OVERFLOW",
            "VOCALIVE_CONVERSATION_LANGUAGE",
            "VOCALIVE_USER_NAME",
            "VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT",
            "VOCALIVE_CONTEXT_ACTIVE_MESSAGE_MAX_AGE_SECONDS",
            "VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS",
            "VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT",
            "VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS",
            "VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS",
            "VOCALIVE_INPUT_PROVIDER",
            "VOCALIVE_MIC_SAMPLE_RATE",
            "VOCALIVE_MIC_CHANNELS",
            "VOCALIVE_MIC_BLOCK_MS",
            "VOCALIVE_MIC_SPEECH_THRESHOLD",
            "VOCALIVE_MIC_PRE_SPEECH_MS",
            "VOCALIVE_MIC_SPEECH_HOLD_MS",
            "VOCALIVE_MIC_SILENCE_MS",
            "VOCALIVE_MIC_MIN_UTTERANCE_MS",
            "VOCALIVE_MIC_MAX_UTTERANCE_MS",
            "VOCALIVE_MIC_DEVICE",
            "VOCALIVE_MIC_PREFER_EXTERNAL",
            "VOCALIVE_MIC_INTERRUPT_MODE",
            "VOCALIVE_CONVERSATION_WINDOW_ENABLED",
            "VOCALIVE_CONVERSATION_WINDOW_OPEN_SECONDS",
            "VOCALIVE_CONVERSATION_WINDOW_CLOSED_SECONDS",
            "VOCALIVE_CONVERSATION_WINDOW_START_OPEN",
            "VOCALIVE_CONVERSATION_WINDOW_APPLY_TO_APP_AUDIO",
            "VOCALIVE_CONVERSATION_WINDOW_RESET_POLICY",
            "VOCALIVE_APP_AUDIO_ENABLED",
            "VOCALIVE_APP_AUDIO_MODE",
            "VOCALIVE_APP_AUDIO_TARGET",
            "VOCALIVE_APP_AUDIO_SAMPLE_RATE",
            "VOCALIVE_APP_AUDIO_CHANNELS",
            "VOCALIVE_APP_AUDIO_BLOCK_MS",
            "VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD",
            "VOCALIVE_APP_AUDIO_PRE_SPEECH_MS",
            "VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS",
            "VOCALIVE_APP_AUDIO_SILENCE_MS",
            "VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS",
            "VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS",
            "VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS",
            "VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS",
            "VOCALIVE_APP_AUDIO_TRANSCRIPTION_DEBOUNCE_MS",
            "VOCALIVE_APP_AUDIO_MIN_TRANSCRIPTION_MS",
            "VOCALIVE_APP_AUDIO_ADAPTIVE_VAD",
            "VOCALIVE_APP_AUDIO_STT_ENHANCEMENT",
            "VOCALIVE_OUTPUT_PROVIDER",
            "VOCALIVE_SPEAKER_COMMAND",
            "VOCALIVE_OVERLAY_ENABLED",
            "VOCALIVE_OVERLAY_HOST",
            "VOCALIVE_OVERLAY_PORT",
            "VOCALIVE_OVERLAY_AUTO_OPEN",
            "VOCALIVE_OVERLAY_TITLE",
            "VOCALIVE_OVERLAY_CHARACTER_NAME",
            "VOCALIVE_REPLY_DEBOUNCE_MS",
            "VOCALIVE_REPLY_POLICY_ENABLED",
            "VOCALIVE_REPLY_MIN_GAP_MS",
            "VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS",
            "VOCALIVE_REPLY_REQUIRE_EXPLICIT_TRIGGER",
            "VOCALIVE_PROACTIVE_ENABLED",
            "VOCALIVE_PROACTIVE_MICROPHONE_ENABLED",
            "VOCALIVE_PROACTIVE_APPLICATION_AUDIO_ENABLED",
            "VOCALIVE_PROACTIVE_SCREEN_ENABLED",
            "VOCALIVE_PROACTIVE_IDLE_SECONDS",
            "VOCALIVE_PROACTIVE_COOLDOWN_SECONDS",
            "VOCALIVE_PROACTIVE_SCREEN_POLL_SECONDS",
            "VOCALIVE_GEMINI_API_KEY",
            "VOCALIVE_GEMINI_MODEL",
            "VOCALIVE_GEMINI_TIMEOUT_SECONDS",
            "VOCALIVE_GEMINI_TEMPERATURE",
            "VOCALIVE_GEMINI_THINKING_BUDGET",
            "VOCALIVE_GEMINI_SYSTEM_INSTRUCTION",
            "VOCALIVE_SCREEN_CAPTURE_ENABLED",
            "VOCALIVE_SCREEN_WINDOW_NAME",
            "VOCALIVE_SCREEN_TRIGGER_PHRASES",
            "VOCALIVE_SCREEN_PASSIVE_ENABLED",
            "VOCALIVE_SCREEN_PASSIVE_TRIGGER_PHRASES",
            "VOCALIVE_SCREEN_PASSIVE_COOLDOWN_SECONDS",
            "VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS",
            "VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX",
            "VOCALIVE_MOONSHINE_MODEL",
            "VOCALIVE_OPENAI_API_KEY",
            "VOCALIVE_OPENAI_MODEL",
            "VOCALIVE_OPENAI_BASE_URL",
            "VOCALIVE_OPENAI_TIMEOUT_SECONDS",
            "VOCALIVE_AIVIS_BASE_URL",
            "VOCALIVE_AIVIS_ENGINE_MODE",
            "VOCALIVE_AIVIS_ENGINE_PATH",
            "VOCALIVE_AIVIS_CPU_NUM_THREADS",
            "VOCALIVE_AIVIS_STARTUP_TIMEOUT_SECONDS",
            "VOCALIVE_AIVIS_SPEAKER_ID",
            "VOCALIVE_AIVIS_SPEAKER_NAME",
            "VOCALIVE_AIVIS_STYLE_NAME",
            "VOCALIVE_AIVIS_TIMEOUT_SECONDS",
        }

        actual_names = {
            definition.env_name for definition in controller_setting_definitions()
        }
        self.assertEqual(actual_names, expected_names)

    def test_controller_setting_schema_includes_documentation(self) -> None:
        schema = controller_setting_schema()

        self.assertTrue(schema)
        for field in schema:
            self.assertIn("default_label", field)
            self.assertIn("description", field)
            self.assertIsInstance(field["default_label"], str)
            self.assertIsInstance(field["description"], str)
            self.assertTrue(field["description"])

    def test_readme_configuration_table_matches_controller_setting_rows(self) -> None:
        readme_path = Path(__file__).resolve().parents[2] / "README.md"
        actual_rows = _extract_markdown_table_rows(
            readme_path,
            header="| Variable | Default | Purpose |",
            end_marker="\n\nCurrent provider support:",
        )

        expected_rows = {
            row["env_name"]: {
                "default_label": row["default_label"],
                "description": row["description"],
            }
            for row in controller_setting_rows()
        }

        self.assertEqual(actual_rows, expected_rows)

    def test_development_configuration_table_matches_controller_setting_rows(self) -> None:
        development_path = Path(__file__).resolve().parents[2] / "docs" / "development.md"
        actual_rows = _extract_markdown_table_rows(
            development_path,
            header="| Variable | Default | Notes |",
            end_marker="\n\nIf you add a new setting",
        )

        expected_rows = {
            row["env_name"]: {
                "default_label": row["default_label"],
                "description": row["description"],
            }
            for row in controller_setting_rows()
        }

        self.assertEqual(actual_rows, expected_rows)

    def test_normalize_controller_values_filters_unknown_keys_and_applies_defaults(self) -> None:
        normalized = normalize_controller_values(
            {
                "VOCALIVE_LOG_LEVEL": "DEBUG",
                "VOCALIVE_OVERLAY_ENABLED": "true",
                "UNRELATED_VALUE": "ignored",
            }
        )

        self.assertEqual(normalized["VOCALIVE_LOG_LEVEL"], "DEBUG")
        self.assertEqual(normalized["VOCALIVE_OVERLAY_ENABLED"], "true")
        self.assertEqual(normalized["VOCALIVE_INPUT_PROVIDER"], "stdin")
        self.assertNotIn("UNRELATED_VALUE", normalized)
