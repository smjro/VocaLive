from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.models import AudioSegment, TurnContext
from vocalive.stt.openai import OpenAITranscriptionSpeechToTextEngine


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class OpenAITranscriptionSpeechToTextEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_uses_transcript_hint_for_stdin_simulation(self) -> None:
        engine = OpenAITranscriptionSpeechToTextEngine(
            api_key=None,
            default_language="ja-JP",
        )
        context = TurnContext(session_id="session", turn_id=1)

        with patch(
            "vocalive.stt.openai.urllib.request.urlopen",
            side_effect=AssertionError("OpenAI STT request should not run"),
        ):
            transcription = await engine.transcribe(
                AudioSegment.from_text("typed input"),
                context,
            )

        self.assertEqual(transcription.text, "typed input")
        self.assertEqual(transcription.provider, "openai")
        self.assertEqual(transcription.language, "ja")

    async def test_transcribe_posts_wav_audio_to_openai(self) -> None:
        captured_request = {}
        context = TurnContext(session_id="session", turn_id=1)
        engine = OpenAITranscriptionSpeechToTextEngine(
            api_key="secret",
            model_name="gpt-4o-mini-transcribe",
            timeout_seconds=12.5,
            default_language="ja-JP",
        )
        segment = AudioSegment(
            pcm=b"\x01\x00\x02\x00",
            sample_rate_hz=16_000,
            channels=1,
            sample_width_bytes=2,
        )

        def _fake_urlopen(request, timeout):
            captured_request["request"] = request
            captured_request["timeout"] = timeout
            return _FakeHTTPResponse({"text": "hello world", "language": "ja"})

        with patch(
            "vocalive.stt.openai.urllib.request.urlopen",
            side_effect=_fake_urlopen,
        ):
            transcription = await engine.transcribe(segment, context)

        self.assertEqual(transcription.text, "hello world")
        self.assertEqual(transcription.provider, "openai")
        self.assertEqual(transcription.language, "ja")
        self.assertEqual(captured_request["timeout"], 12.5)

        request = captured_request["request"]
        header_items = {
            key.lower(): value for key, value in request.header_items()
        }
        self.assertEqual(header_items["authorization"], "Bearer secret")
        self.assertIn("multipart/form-data; boundary=", header_items["content-type"])

        body = request.data
        assert isinstance(body, bytes)
        self.assertIn(b'name="model"', body)
        self.assertIn(b"gpt-4o-mini-transcribe", body)
        self.assertIn(b'name="language"', body)
        self.assertIn(b"\r\nja\r\n", body)
        self.assertIn(b'name="response_format"', body)
        self.assertIn(b"\r\njson\r\n", body)
        self.assertIn(b'filename="audio.wav"', body)
        self.assertIn(b"RIFF", body)
        self.assertIn(b"WAVE", body)

    async def test_transcribe_requires_api_key_for_real_audio(self) -> None:
        engine = OpenAITranscriptionSpeechToTextEngine(api_key=None)
        context = TurnContext(session_id="session", turn_id=1)
        segment = AudioSegment(
            pcm=b"\x00\x00",
            sample_rate_hz=16_000,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "VOCALIVE_OPENAI_API_KEY or OPENAI_API_KEY",
        ):
            await engine.transcribe(segment, context)
