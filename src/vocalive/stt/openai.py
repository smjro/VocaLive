from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
import uuid
import wave
from io import BytesIO
from typing import Any

from vocalive.models import AudioSegment, Transcription, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.stt.base import SpeechToTextEngine


class OpenAITranscriptionSpeechToTextEngine(SpeechToTextEngine):
    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        model_name: str = "gpt-4o-mini-transcribe",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        default_language: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_language = _normalize_transcription_language(default_language)

    async def transcribe(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> Transcription:
        del context
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if segment.transcript_hint:
            return Transcription(
                text=segment.transcript_hint,
                provider=self.name,
                confidence=1.0,
                language=self.default_language,
            )
        if not self.api_key:
            raise RuntimeError(
                "OpenAI STT requires VOCALIVE_OPENAI_API_KEY or OPENAI_API_KEY."
            )
        response_body = await asyncio.to_thread(self._transcribe_sync, segment)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        text, language = _extract_transcription(response_body)
        return Transcription(
            text=text,
            provider=self.name,
            confidence=1.0,
            language=language or self.default_language,
        )

    def _transcribe_sync(self, segment: AudioSegment) -> dict[str, Any]:
        request = _build_transcription_request(
            api_key=self.api_key or "",
            base_url=self.base_url,
            model_name=self.model_name,
            segment=segment,
            language=self.default_language,
        )
        return _open_json(request=request, timeout_seconds=self.timeout_seconds)


def _build_transcription_request(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    segment: AudioSegment,
    language: str | None,
) -> urllib.request.Request:
    body, boundary = _build_multipart_form_data(
        fields={
            "model": model_name,
            "language": language,
            "response_format": "json",
        },
        file_field_name="file",
        filename="audio.wav",
        file_content_type="audio/wav",
        file_bytes=_segment_to_wav_bytes(segment),
    )
    return urllib.request.Request(
        url=f"{base_url}/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )


def _build_multipart_form_data(
    *,
    fields: dict[str, str | None],
    file_field_name: str,
    filename: str,
    file_content_type: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    boundary = f"----VocaLiveBoundary{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    body = bytearray()

    for name, value in fields.items():
        if value is None:
            continue
        body.extend(b"--" + boundary_bytes + b"\r\n")
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    body.extend(b"--" + boundary_bytes + b"\r\n")
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field_name}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8")
    )
    body.extend(f"Content-Type: {file_content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(b"--" + boundary_bytes + b"--\r\n")
    return bytes(body), boundary


def _segment_to_wav_bytes(segment: AudioSegment) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(segment.channels)
        wav_file.setsampwidth(segment.sample_width_bytes)
        wav_file.setframerate(segment.sample_rate_hz)
        wav_file.writeframes(segment.pcm)
    return buffer.getvalue()


def _normalize_transcription_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized_language = language.strip()
    if not normalized_language:
        return None
    primary_subtag = normalized_language.replace("_", "-").split("-", 1)[0].strip()
    return primary_subtag.lower() or None


def _extract_transcription(response_body: Any) -> tuple[str, str | None]:
    if not isinstance(response_body, dict):
        raise RuntimeError(f"OpenAI STT returned an invalid response: {response_body!r}")
    raw_text = response_body.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise RuntimeError(f"OpenAI STT response did not include text: {response_body}")
    raw_language = response_body.get("language")
    language = raw_language.strip() if isinstance(raw_language, str) else None
    return raw_text.strip(), language or None


def _open_json(request: urllib.request.Request, timeout_seconds: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI STT request failed with status {exc.code}: {raw_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI STT request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"OpenAI STT returned an invalid response: {payload!r}")
    return payload
