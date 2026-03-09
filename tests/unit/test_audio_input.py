from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.devices import resolve_input_device
from vocalive.audio.input import UtteranceAccumulator
from vocalive.audio.vad import FixedSilenceTurnDetector


def _pcm_chunk(frame_count: int, amplitude: int) -> bytes:
    frame = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    return frame * frame_count


class _FakeSounddevice:
    def __init__(self, devices: list[dict[str, object]], default_input_index: int | None) -> None:
        self._devices = devices
        self.default = type("DefaultSettings", (), {"device": [default_input_index, None]})()

    def query_devices(self) -> list[dict[str, object]]:
        return list(self._devices)


class UtteranceAccumulatorTests(unittest.TestCase):
    def test_includes_preroll_audio_before_speech_crosses_threshold(self) -> None:
        accumulator = UtteranceAccumulator(
            sample_rate_hz=1_000,
            speech_threshold=0.01,
            pre_speech_ms=120.0,
            speech_hold_ms=0.0,
            min_utterance_ms=100.0,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=100.0),
        )

        weak_prefix = _pcm_chunk(frame_count=100, amplitude=200)
        voiced_chunk = _pcm_chunk(frame_count=100, amplitude=2_000)
        trailing_silence = _pcm_chunk(frame_count=100, amplitude=0)

        self.assertIsNone(accumulator.add_chunk(weak_prefix))
        self.assertIsNone(accumulator.add_chunk(voiced_chunk))
        segment = accumulator.add_chunk(trailing_silence)

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertTrue(segment.pcm.startswith(weak_prefix + voiced_chunk))

    def test_limits_preroll_to_the_most_recent_configured_audio(self) -> None:
        accumulator = UtteranceAccumulator(
            sample_rate_hz=1_000,
            speech_threshold=0.01,
            pre_speech_ms=100.0,
            speech_hold_ms=0.0,
            min_utterance_ms=100.0,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=100.0),
        )

        older_prefix = _pcm_chunk(frame_count=100, amplitude=200)
        latest_prefix = _pcm_chunk(frame_count=100, amplitude=300)
        voiced_chunk = _pcm_chunk(frame_count=100, amplitude=2_000)
        trailing_silence = _pcm_chunk(frame_count=100, amplitude=0)

        self.assertIsNone(accumulator.add_chunk(older_prefix))
        self.assertIsNone(accumulator.add_chunk(latest_prefix))
        self.assertIsNone(accumulator.add_chunk(voiced_chunk))
        segment = accumulator.add_chunk(trailing_silence)

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertFalse(segment.pcm.startswith(older_prefix))
        self.assertTrue(segment.pcm.startswith(latest_prefix + voiced_chunk))

    def test_calls_speech_start_callback_once_per_utterance(self) -> None:
        started: list[str] = []
        accumulator = UtteranceAccumulator(
            sample_rate_hz=1_000,
            speech_threshold=0.01,
            speech_hold_ms=0.0,
            min_utterance_ms=100.0,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=100.0),
            on_speech_start=lambda: started.append("speech"),
        )

        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=100, amplitude=2_000)))
        self.assertEqual(started, ["speech"])
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=100, amplitude=2_000)))
        self.assertEqual(started, ["speech"])
        segment = accumulator.add_chunk(_pcm_chunk(frame_count=100, amplitude=0))

        self.assertIsNotNone(segment)
        self.assertEqual(started, ["speech"])

    def test_emits_after_detected_silence(self) -> None:
        accumulator = UtteranceAccumulator(
            sample_rate_hz=16_000,
            speech_threshold=0.01,
            speech_hold_ms=0.0,
            min_utterance_ms=100.0,
        )

        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=2000)))
        segment = accumulator.add_chunk(_pcm_chunk(frame_count=8000, amplitude=0))

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertEqual(segment.sample_rate_hz, 16_000)
        self.assertGreater(len(segment.pcm), 0)

    def test_does_not_emit_during_short_mid_utterance_pause(self) -> None:
        accumulator = UtteranceAccumulator(
            sample_rate_hz=16_000,
            speech_threshold=0.01,
            speech_hold_ms=200.0,
            min_utterance_ms=100.0,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=300.0),
        )

        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=2000)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=2000)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=2000)))

    def test_emits_after_pause_exceeds_hold_and_silence_threshold(self) -> None:
        accumulator = UtteranceAccumulator(
            sample_rate_hz=16_000,
            speech_threshold=0.01,
            speech_hold_ms=200.0,
            min_utterance_ms=100.0,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=300.0),
        )

        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=2000)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=2000)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0)))

        segment = accumulator.add_chunk(_pcm_chunk(frame_count=1600, amplitude=0))

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertEqual(segment.sample_rate_hz, 16_000)
        self.assertGreater(len(segment.pcm), 0)


class InputDeviceResolutionTests(unittest.TestCase):
    def test_prefers_external_headset_when_default_is_builtin(self) -> None:
        sounddevice = _FakeSounddevice(
            devices=[
                {"name": "MacBook Pro Microphone", "max_input_channels": 1},
                {"name": "AirPods Pro Hands-Free", "max_input_channels": 1},
            ],
            default_input_index=0,
        )

        match = resolve_input_device(sounddevice, requested_device=None, prefer_external=True)

        self.assertEqual(match.index, 1)
        self.assertEqual(match.selection, "external")
        self.assertIn("AirPods Pro Hands-Free", match.label)

    def test_keeps_system_default_when_it_is_already_external(self) -> None:
        sounddevice = _FakeSounddevice(
            devices=[
                {"name": "MacBook Pro Microphone", "max_input_channels": 1},
                {"name": "USB Audio Interface", "max_input_channels": 1},
            ],
            default_input_index=1,
        )

        match = resolve_input_device(sounddevice, requested_device=None, prefer_external=True)

        self.assertEqual(match.index, 1)
        self.assertEqual(match.selection, "default")

    def test_honors_explicit_named_device_selection(self) -> None:
        sounddevice = _FakeSounddevice(
            devices=[
                {"name": "MacBook Pro Microphone", "max_input_channels": 1},
                {"name": "USB Audio Interface", "max_input_channels": 1},
            ],
            default_input_index=0,
        )

        match = resolve_input_device(sounddevice, requested_device="USB Audio", prefer_external=True)

        self.assertEqual(match.index, 1)
        self.assertEqual(match.selection, "explicit")

    def test_explicit_external_device_raises_when_no_external_input_exists(self) -> None:
        sounddevice = _FakeSounddevice(
            devices=[
                {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            ],
            default_input_index=0,
        )

        with self.assertRaisesRegex(ValueError, "No external input device was found"):
            resolve_input_device(sounddevice, requested_device="external", prefer_external=True)
