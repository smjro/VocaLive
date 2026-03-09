from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.input import MicrophoneAudioInput
from vocalive.audio.output import MemoryAudioOutput
from vocalive.config.settings import AppSettings, InputProvider, InputSettings
from vocalive.llm.gemini import GeminiLanguageModel
from vocalive.main import build_audio_input, build_orchestrator
from vocalive.stt.moonshine import MoonshineSpeechToTextEngine
from vocalive.tts.aivis import AivisSpeechTextToSpeechEngine


class BuildOrchestratorTests(unittest.TestCase):
    def test_build_orchestrator_uses_real_provider_adapters(self) -> None:
        orchestrator = build_orchestrator(
            AppSettings(
                stt_provider="Moonshine Voice",
                model_provider="Google Gemini",
                tts_provider="Aivis Speech",
            )
        )

        self.assertIsInstance(orchestrator.stt_engine, MoonshineSpeechToTextEngine)
        self.assertIsInstance(orchestrator.language_model, GeminiLanguageModel)
        self.assertEqual(orchestrator.language_model.thinking_budget, 0)
        self.assertIsInstance(orchestrator.tts_engine, AivisSpeechTextToSpeechEngine)
        self.assertIsInstance(orchestrator.audio_output, MemoryAudioOutput)

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
