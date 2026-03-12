from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.models import AudioSegment, TurnContext
from vocalive.stt.moonshine import (
    MoonshineSpeechToTextEngine,
    _APPLICATION_AUDIO_PADDING_LEAD_MS,
    _APPLICATION_AUDIO_PADDING_TAIL_MS,
    _enhance_application_audio,
    _pcm16le_to_float_mono,
    _resolve_moonshine_model,
)


def _mixed_sine_pcm(
    frame_count: int,
    components: tuple[tuple[int, float], ...],
    sample_rate_hz: int = 16_000,
) -> bytes:
    samples = bytearray()
    for index in range(frame_count):
        sample = 0.0
        for amplitude, frequency_hz in components:
            sample += amplitude * math.sin((2.0 * math.pi * frequency_hz * index) / sample_rate_hz)
        clamped_sample = max(-32768, min(32767, int(round(sample))))
        samples.extend(clamped_sample.to_bytes(2, byteorder="little", signed=True))
    return bytes(samples)


def _rms(audio_data: list[float]) -> float:
    if not audio_data:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in audio_data) / len(audio_data))


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

    async def test_transcribe_enhances_application_audio_before_moonshine(self) -> None:
        created_transcribers: list[FakeTranscriber] = []

        class FakeTranscriber:
            def __init__(self, model_path: str, model_arch: str) -> None:
                del model_path, model_arch
                self.calls: list[tuple[list[float], int]] = []
                created_transcribers.append(self)

            def transcribe_without_streaming(
                self,
                audio_data: list[float],
                sample_rate: int = 16_000,
                flags: int = 0,
            ) -> object:
                del flags
                self.calls.append((audio_data, sample_rate))
                return "enhanced"

        fake_moonshine = SimpleNamespace(
            string_to_model_arch=lambda value: value,
            get_model_for_language=lambda language, model_arch: (language, model_arch),
            Transcriber=FakeTranscriber,
        )
        engine = MoonshineSpeechToTextEngine(
            model_name="base",
            default_language="ja",
            application_audio_enhancement_enabled=True,
        )
        context = TurnContext(session_id="session", turn_id=1)
        segment = AudioSegment(
            pcm=_mixed_sine_pcm(
                frame_count=640,
                components=((5_000, 90.0), (7_000, 420.0)),
            ),
            sample_rate_hz=16_000,
            source="application_audio",
        )

        with patch("vocalive.stt.moonshine._import_moonshine", return_value=fake_moonshine):
            transcription = await engine.transcribe(segment, context)

        self.assertEqual(transcription.text, "enhanced")
        self.assertEqual(len(created_transcribers), 1)
        raw_audio = _pcm16le_to_float_mono(segment)
        enhanced_audio = created_transcribers[0].calls[0][0]
        self.assertNotEqual(enhanced_audio, raw_audio)
        self.assertGreater(max(abs(sample) for sample in enhanced_audio), 0.1)

    async def test_transcribe_does_not_enhance_non_application_audio(self) -> None:
        created_transcribers: list[FakeTranscriber] = []

        class FakeTranscriber:
            def __init__(self, model_path: str, model_arch: str) -> None:
                del model_path, model_arch
                self.calls: list[tuple[list[float], int]] = []
                created_transcribers.append(self)

            def transcribe_without_streaming(
                self,
                audio_data: list[float],
                sample_rate: int = 16_000,
                flags: int = 0,
            ) -> object:
                del flags
                self.calls.append((audio_data, sample_rate))
                return "plain"

        fake_moonshine = SimpleNamespace(
            string_to_model_arch=lambda value: value,
            get_model_for_language=lambda language, model_arch: (language, model_arch),
            Transcriber=FakeTranscriber,
        )
        engine = MoonshineSpeechToTextEngine(
            model_name="base",
            default_language="ja",
            application_audio_enhancement_enabled=True,
        )
        context = TurnContext(session_id="session", turn_id=1)
        segment = AudioSegment(
            pcm=b"\x00\x00\xff\x7f\x00\x80",
            sample_rate_hz=16_000,
            source="user",
        )

        with patch("vocalive.stt.moonshine._import_moonshine", return_value=fake_moonshine):
            transcription = await engine.transcribe(segment, context)

        self.assertEqual(transcription.text, "plain")
        self.assertEqual(len(created_transcribers), 1)
        self.assertEqual(created_transcribers[0].calls[0][0], _pcm16le_to_float_mono(segment))

    async def test_transcribe_retries_application_audio_with_raw_fallback_when_enhanced_pass_is_empty(self) -> None:
        created_transcribers: list[FakeTranscriber] = []

        class FakeTranscriber:
            def __init__(self, model_path: str, model_arch: str) -> None:
                del model_path, model_arch
                self.calls: list[tuple[list[float], int]] = []
                created_transcribers.append(self)

            def transcribe_without_streaming(
                self,
                audio_data: list[float],
                sample_rate: int = 16_000,
                flags: int = 0,
            ) -> object:
                del flags
                self.calls.append((audio_data, sample_rate))
                return "" if len(self.calls) == 1 else "fallback"

        fake_moonshine = SimpleNamespace(
            string_to_model_arch=lambda value: value,
            get_model_for_language=lambda language, model_arch: (language, model_arch),
            Transcriber=FakeTranscriber,
        )
        engine = MoonshineSpeechToTextEngine(
            model_name="base",
            default_language="ja",
            application_audio_enhancement_enabled=True,
        )
        context = TurnContext(session_id="session", turn_id=1)
        segment = AudioSegment(
            pcm=_mixed_sine_pcm(
                frame_count=640,
                components=((2_500, 85.0), (1_300, 170.0), (800, 260.0)),
            ),
            sample_rate_hz=16_000,
            source="application_audio",
        )

        with patch("vocalive.stt.moonshine._import_moonshine", return_value=fake_moonshine):
            transcription = await engine.transcribe(segment, context)

        self.assertEqual(transcription.text, "fallback")
        self.assertEqual(len(created_transcribers), 1)
        self.assertEqual(len(created_transcribers[0].calls), 2)

    def test_application_audio_enhancement_preserves_low_frequency_dialogue(self) -> None:
        segment = AudioSegment(
            pcm=_mixed_sine_pcm(
                frame_count=6_400,
                components=((7_000, 95.0), (1_200, 210.0)),
            ),
            sample_rate_hz=16_000,
            source="application_audio",
        )

        raw_audio = _pcm16le_to_float_mono(segment)
        enhanced_audio = _enhance_application_audio(raw_audio, sample_rate=segment.sample_rate_hz)
        lead_padding_samples = int(
            (_APPLICATION_AUDIO_PADDING_LEAD_MS / 1000.0) * segment.sample_rate_hz
        )
        tail_padding_samples = int(
            (_APPLICATION_AUDIO_PADDING_TAIL_MS / 1000.0) * segment.sample_rate_hz
        )
        trimmed_audio = enhanced_audio[lead_padding_samples:-tail_padding_samples]

        self.assertEqual(len(trimmed_audio), len(raw_audio))
        self.assertTrue(all(sample == 0.0 for sample in enhanced_audio[:lead_padding_samples]))
        self.assertTrue(all(sample == 0.0 for sample in enhanced_audio[-tail_padding_samples:]))
        self.assertGreater(_rms(trimmed_audio), _rms(raw_audio) * 0.4)

    def test_resolve_moonshine_model_supports_legacy_alias(self) -> None:
        resolved_model = _resolve_moonshine_model("moonshine/base", "ja-JP")

        self.assertEqual(resolved_model.architecture_name, "base")
        self.assertEqual(resolved_model.language, "ja")
