from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.models import AudioSegment, TurnContext
from vocalive.stt.moonshine import MoonshineSpeechToTextEngine, _resolve_moonshine_model


class MoonshineSpeechToTextEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_uses_transcript_hint_for_stdin_simulation(self) -> None:
        engine = MoonshineSpeechToTextEngine(default_language="ja")
        context = TurnContext(session_id="session", turn_id=1)

        with patch(
            "vocalive.stt.moonshine._import_moonshine",
            side_effect=AssertionError("Moonshine backend should not be imported"),
        ):
            transcription = await engine.transcribe(
                AudioSegment.from_text("typed input"),
                context,
            )

        self.assertEqual(transcription.text, "typed input")
        self.assertEqual(transcription.provider, "moonshine")
        self.assertEqual(transcription.language, "ja")

    async def test_transcribe_uses_moonshine_voice_transcriber(self) -> None:
        created_transcribers: list[FakeTranscriber] = []

        class FakeTranscriber:
            def __init__(self, model_path: str, model_arch: str) -> None:
                self.model_path = model_path
                self.model_arch = model_arch
                self.calls: list[tuple[list[float], int]] = []
                created_transcribers.append(self)

            def transcribe_without_streaming(
                self,
                audio_data: list[float],
                sample_rate: int = 16_000,
                flags: int = 0,
            ) -> object:
                self.calls.append((audio_data, sample_rate))
                return SimpleNamespace(
                    lines=[
                        SimpleNamespace(text="hello"),
                        SimpleNamespace(text="world"),
                    ]
                )

        fake_moonshine = SimpleNamespace(
            string_to_model_arch=lambda value: f"arch:{value}",
            get_model_for_language=lambda language, model_arch: (
                f"/models/{language}",
                model_arch,
            ),
            Transcriber=FakeTranscriber,
        )
        engine = MoonshineSpeechToTextEngine(model_name="base", default_language="ja")
        context = TurnContext(session_id="session", turn_id=1)
        segment = AudioSegment(
            pcm=b"\x00\x00\xff\x7f\x00\x80",
            sample_rate_hz=16_000,
        )

        with patch("vocalive.stt.moonshine._import_moonshine", return_value=fake_moonshine):
            transcription = await engine.transcribe(segment, context)

        self.assertEqual(transcription.text, "hello world")
        self.assertEqual(transcription.provider, "moonshine")
        self.assertEqual(transcription.language, "ja")
        self.assertEqual(len(created_transcribers), 1)
        self.assertEqual(created_transcribers[0].model_path, "/models/ja")
        self.assertEqual(created_transcribers[0].model_arch, "arch:base")
        self.assertEqual(created_transcribers[0].calls[0][1], 16_000)
        self.assertAlmostEqual(created_transcribers[0].calls[0][0][0], 0.0)
        self.assertAlmostEqual(created_transcribers[0].calls[0][0][1], 32767.0 / 32768.0)
        self.assertAlmostEqual(created_transcribers[0].calls[0][0][2], -1.0)

    def test_resolve_moonshine_model_supports_legacy_alias(self) -> None:
        resolved_model = _resolve_moonshine_model("moonshine/base", "ja-JP")

        self.assertEqual(resolved_model.architecture_name, "base")
        self.assertEqual(resolved_model.language, "ja")
