from __future__ import annotations

import asyncio
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.devices import resolve_input_device
from vocalive.audio.input import CombinedAudioInput, MicrophoneAudioInput, UtteranceAccumulator
from vocalive.audio.speech_detection import AdaptiveEnergySpeechDetector
from vocalive.models import AudioSegment
from vocalive.audio.vad import FixedSilenceTurnDetector


def _pcm_chunk(frame_count: int, amplitude: int) -> bytes:
    frame = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    return frame * frame_count


def _mixed_sine_chunk(
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
            on_speech_start=lambda source: started.append(source),
        )

        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=100, amplitude=2_000)))
        self.assertEqual(started, ["user"])
        self.assertIsNone(accumulator.add_chunk(_pcm_chunk(frame_count=100, amplitude=2_000)))
        self.assertEqual(started, ["user"])
        segment = accumulator.add_chunk(_pcm_chunk(frame_count=100, amplitude=0))

        self.assertIsNotNone(segment)
        self.assertEqual(started, ["user"])

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

    def test_adaptive_speech_detector_ignores_steady_background_until_dialogue_arrives(self) -> None:
        accumulator = UtteranceAccumulator(
            sample_rate_hz=16_000,
            speech_threshold=0.02,
            pre_speech_ms=120.0,
            speech_hold_ms=0.0,
            min_utterance_ms=40.0,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=40.0),
            speech_detector=AdaptiveEnergySpeechDetector(speech_threshold=0.02),
        )

        background_chunk = _mixed_sine_chunk(
            frame_count=640,
            components=((4_800, 90.0),),
        )
        dialogue_chunk = _mixed_sine_chunk(
            frame_count=640,
            components=((4_800, 90.0), (7_200, 420.0)),
        )
        trailing_silence = _pcm_chunk(frame_count=640, amplitude=0)

        for _ in range(8):
            self.assertIsNone(accumulator.add_chunk(background_chunk))
        self.assertIsNone(accumulator.add_chunk(dialogue_chunk))

        segment = accumulator.add_chunk(trailing_silence)

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertGreater(len(segment.pcm), len(dialogue_chunk))

    def test_adaptive_speech_detector_detects_low_male_voice_over_low_background(self) -> None:
        detector = AdaptiveEnergySpeechDetector(speech_threshold=0.02)
        background_chunk = _mixed_sine_chunk(
            frame_count=640,
            components=((2_600, 60.0), (1_600, 90.0)),
        )
        low_voice_chunk = _mixed_sine_chunk(
            frame_count=640,
            components=((2_200, 85.0), (1_200, 170.0), (700, 260.0)),
        )

        for _ in range(8):
            self.assertFalse(detector.is_speech(background_chunk, sample_width_bytes=2))

        self.assertTrue(detector.is_speech(low_voice_chunk, sample_width_bytes=2))


class InputDeviceResolutionTests(unittest.TestCase):
    def test_keeps_default_when_only_external_candidate_is_hands_free(self) -> None:
        sounddevice = _FakeSounddevice(
            devices=[
                {"name": "MacBook Pro Microphone", "max_input_channels": 1},
                {"name": "AirPods Pro Hands-Free", "max_input_channels": 1},
            ],
            default_input_index=0,
        )

        match = resolve_input_device(sounddevice, requested_device=None, prefer_external=True)

        self.assertEqual(match.index, 0)
        self.assertEqual(match.selection, "default")
        self.assertIn("MacBook Pro Microphone", match.label)

    def test_prefers_usb_input_over_hands_free_when_default_is_builtin(self) -> None:
        sounddevice = _FakeSounddevice(
            devices=[
                {"name": "MacBook Pro Microphone", "max_input_channels": 1},
                {"name": "AirPods Pro Hands-Free", "max_input_channels": 1},
                {"name": "USB Audio Interface", "max_input_channels": 1},
            ],
            default_input_index=0,
        )

        match = resolve_input_device(sounddevice, requested_device=None, prefer_external=True)

        self.assertEqual(match.index, 2)
        self.assertEqual(match.selection, "external")
        self.assertIn("USB Audio Interface", match.label)

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

    def test_explicit_external_can_still_select_hands_free_input(self) -> None:
        sounddevice = _FakeSounddevice(
            devices=[
                {"name": "MacBook Pro Microphone", "max_input_channels": 1},
                {"name": "AirPods Pro Hands-Free", "max_input_channels": 1},
            ],
            default_input_index=0,
        )

        match = resolve_input_device(sounddevice, requested_device="external", prefer_external=True)

        self.assertEqual(match.index, 1)
        self.assertEqual(match.selection, "external")
        self.assertIn("AirPods Pro Hands-Free", match.label)


class _FakeRawInputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.callback = kwargs.get("callback")
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class _FakeStreamingSounddevice(_FakeSounddevice):
    def __init__(self, devices: list[dict[str, object]], default_input_index: int | None) -> None:
        super().__init__(devices=devices, default_input_index=default_input_index)
        self.streams: list[_FakeRawInputStream] = []

    def RawInputStream(self, **kwargs) -> _FakeRawInputStream:
        stream = _FakeRawInputStream(**kwargs)
        self.streams.append(stream)
        return stream


class MicrophoneAudioInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_microphone_chunks_via_callback_stream(self) -> None:
        sounddevice = _FakeStreamingSounddevice(
            devices=[
                {"name": "Built-in Microphone", "max_input_channels": 1},
            ],
            default_input_index=0,
        )
        audio_input = MicrophoneAudioInput(
            sample_rate_hz=1_000,
            block_duration_ms=100.0,
            speech_threshold=0.01,
            speech_hold_ms=0.0,
            silence_threshold_ms=100.0,
            min_utterance_ms=100.0,
        )

        with patch("vocalive.audio.input._import_sounddevice", return_value=sounddevice):
            selected_label = await audio_input.start()
            self.assertEqual(selected_label, "Built-in Microphone (id=0)")
            self.assertEqual(len(sounddevice.streams), 1)
            stream = sounddevice.streams[0]
            self.assertTrue(stream.started)
            self.assertIsNotNone(stream.callback)
            read_task = asyncio.create_task(audio_input.read())

            assert stream.callback is not None
            stream.callback(_pcm_chunk(frame_count=100, amplitude=2_000), 100, None, None)
            stream.callback(_pcm_chunk(frame_count=100, amplitude=0), 100, None, None)

            segment = await asyncio.wait_for(read_task, timeout=1.0)

        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertEqual(segment.sample_rate_hz, 1_000)
        self.assertGreater(len(segment.pcm), 0)

        await audio_input.close()
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)


class _ScriptedAudioInput:
    def __init__(self, segments: list[AudioSegment | None]) -> None:
        self._segments = list(segments)
        self.closed = False

    async def start(self) -> str:
        return "scripted input"

    def set_speech_start_handler(self, handler) -> None:
        del handler

    async def read(self) -> AudioSegment | None:
        await asyncio.sleep(0)
        return self._segments.pop(0) if self._segments else None

    async def close(self) -> None:
        self.closed = True


class CombinedAudioInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_combines_segments_from_multiple_inputs(self) -> None:
        audio_input = CombinedAudioInput(
            (
                _ScriptedAudioInput([AudioSegment.from_text("mic"), None]),
                _ScriptedAudioInput(
                    [
                        AudioSegment.from_text(
                            "npc",
                            source="application_audio",
                            source_label="Steam",
                        ),
                        None,
                    ]
                ),
            )
        )

        await audio_input.start()
        first_segment = await audio_input.read()
        second_segment = await audio_input.read()
        terminal_segment = await audio_input.read()

        assert first_segment is not None
        assert second_segment is not None
        self.assertEqual(
            {(first_segment.transcript_hint, first_segment.source), (second_segment.transcript_hint, second_segment.source)},
            {("mic", "user"), ("npc", "application_audio")},
        )
        self.assertEqual(terminal_segment, None)
