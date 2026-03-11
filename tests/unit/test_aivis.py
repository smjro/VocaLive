from __future__ import annotations

import io
import sys
import unittest
import wave
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.tts.aivis import _read_wave_duration_ms, _read_wave_metadata, _select_style


class AivisSpeechTests(unittest.TestCase):
    def test_select_style_prefers_requested_speaker_and_style(self) -> None:
        speakers = [
            {
                "name": "Speaker A",
                "styles": [
                    {"id": 10, "name": "normal"},
                    {"id": 11, "name": "happy"},
                ],
            },
            {
                "name": "Speaker B",
                "styles": [{"id": 20, "name": "normal"}],
            },
        ]

        style_id = _select_style(
            speakers=speakers,
            speaker_name="Speaker A",
            style_name="happy",
        )

        self.assertEqual(style_id, 11)

    def test_read_wave_metadata_returns_basic_pcm_values(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(b"\x00\x00" * 64)

        self.assertEqual(_read_wave_metadata(buffer.getvalue()), (24_000, 1, 2))

    def test_read_wave_duration_ms_returns_length(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(b"\x00\x00" * 240)

        self.assertEqual(_read_wave_duration_ms(buffer.getvalue()), 10.0)
