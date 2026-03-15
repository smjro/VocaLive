from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import wave
from io import BytesIO
from typing import Any

from vocalive.models import SynthesizedSpeech, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.tts.base import TextToSpeechEngine


class HttpStyleTextToSpeechEngine(TextToSpeechEngine):
    name = "style-api-tts"
    provider_label = "TTS"

    def __init__(
        self,
        *,
        base_url: str,
        speaker_id: int | None = None,
        speaker_name: str | None = None,
        style_name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.style_name = style_name
        self.timeout_seconds = timeout_seconds
        self._resolved_speaker_id: int | None = speaker_id

    async def synthesize(
        self,
        text: str,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> SynthesizedSpeech:
        del context
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        speaker_id = await asyncio.to_thread(self._resolve_speaker_id)
        audio_query = await asyncio.to_thread(self._request_audio_query, text, speaker_id)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        audio_bytes = await asyncio.to_thread(self._request_synthesis, audio_query, speaker_id)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        sample_rate_hz, channels, sample_width_bytes = _read_wave_metadata(audio_bytes)
        duration_ms = _read_wave_duration_ms(audio_bytes)
        return SynthesizedSpeech(
            text=text,
            provider=self.name,
            audio=audio_bytes,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
            duration_ms=duration_ms,
            mime_type="audio/wav",
            file_extension=".wav",
        )

    def _resolve_speaker_id(self) -> int:
        if self._resolved_speaker_id is not None:
            return self._resolved_speaker_id
        speakers = self._get_json("/speakers")
        if not isinstance(speakers, list) or not speakers:
            raise RuntimeError(f"{self.provider_label} returned no speakers from /speakers")
        matching_style = _select_style(
            speakers=speakers,
            speaker_name=self.speaker_name,
            style_name=self.style_name,
            provider_label=self.provider_label,
        )
        self._resolved_speaker_id = matching_style
        return matching_style

    def _request_audio_query(self, text: str, speaker_id: int) -> dict[str, Any]:
        query = urllib.parse.urlencode({"text": text, "speaker": speaker_id})
        response = self._post_json(f"/audio_query?{query}", payload=None)
        if not isinstance(response, dict):
            raise RuntimeError(
                f"{self.provider_label} returned an invalid audio query: {response!r}"
            )
        return response

    def _request_synthesis(self, audio_query: dict[str, Any], speaker_id: int) -> bytes:
        query = urllib.parse.urlencode({"speaker": speaker_id})
        return self._post_bytes(f"/synthesis?{query}", payload=audio_query)

    def _get_json(self, path: str) -> Any:
        request = urllib.request.Request(url=f"{self.base_url}{path}", method="GET")
        return self._open_json(request=request)

    def _post_json(self, path: str, payload: dict[str, Any] | None) -> Any:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_json(request=request)

    def _post_bytes(self, path: str, payload: dict[str, Any]) -> bytes:
        request = urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_bytes(request=request)

    def _open_json(self, request: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.provider_label} API request failed with status {exc.code}: {raw_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.provider_label} API request failed: {exc.reason}") from exc

    def _open_bytes(self, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.provider_label} API request failed with status {exc.code}: {raw_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.provider_label} API request failed: {exc.reason}") from exc


def _select_style(
    speakers: list[Any],
    speaker_name: str | None,
    style_name: str | None,
    provider_label: str = "TTS",
) -> int:
    speaker_name_normalized = speaker_name.strip() if speaker_name else None
    style_name_normalized = style_name.strip() if style_name else None
    first_style_id: int | None = None
    speaker_found = False

    for speaker in speakers:
        if not isinstance(speaker, dict):
            continue
        current_speaker_name = speaker.get("name")
        styles = speaker.get("styles")
        if not isinstance(styles, list):
            continue
        candidate_style_ids = []
        for style in styles:
            if not isinstance(style, dict):
                continue
            style_id = style.get("id", style.get("style_id"))
            if not isinstance(style_id, int):
                continue
            if first_style_id is None:
                first_style_id = style_id
            candidate_style_ids.append((style_id, style.get("name")))
        if speaker_name_normalized and current_speaker_name != speaker_name_normalized:
            continue
        if speaker_name_normalized:
            speaker_found = True
        for style_id, current_style_name in candidate_style_ids:
            if style_name_normalized is None or current_style_name == style_name_normalized:
                return style_id

    if speaker_name_normalized and not speaker_found:
        raise RuntimeError(f"{provider_label} speaker not found: {speaker_name_normalized}")
    if style_name_normalized:
        raise RuntimeError(
            f"{provider_label} style not found. "
            f"speaker={speaker_name_normalized!r} style={style_name_normalized!r}"
        )
    if first_style_id is not None:
        return first_style_id
    raise RuntimeError(
        f"Could not resolve a {provider_label} style ID from /speakers. "
        "Set an explicit speaker ID or provide matching speaker/style names."
    )


def _read_wave_metadata(audio_bytes: bytes) -> tuple[int, int, int]:
    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        return (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
        )


def _read_wave_duration_ms(audio_bytes: bytes) -> float:
    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise RuntimeError("TTS engine returned invalid WAV metadata: frame rate must be > 0")
        return (wav_file.getnframes() / frame_rate) * 1000.0
