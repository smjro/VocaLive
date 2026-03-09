from __future__ import annotations

import array
import asyncio
import importlib
import sys
import threading
from dataclasses import dataclass
from typing import Any

from vocalive.models import AudioSegment, Transcription, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.stt.base import SpeechToTextEngine


@dataclass(frozen=True)
class _ResolvedMoonshineModel:
    architecture_name: str
    language: str


@dataclass
class _MoonshineBackend:
    transcriber: Any


_MODEL_ARCHITECTURES = {
    "tiny",
    "base",
    "tiny-streaming",
    "base-streaming",
    "small-streaming",
    "medium-streaming",
}


class MoonshineSpeechToTextEngine(SpeechToTextEngine):
    name = "moonshine"

    def __init__(
        self,
        model_name: str = "base",
        default_language: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.resolved_model = _resolve_moonshine_model(model_name, default_language)
        self._backend: _MoonshineBackend | None = None
        self._backend_lock = threading.Lock()

    async def transcribe(
        self,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> Transcription:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if segment.transcript_hint:
            normalized_text = segment.transcript_hint.strip()
            if normalized_text:
                return Transcription(
                    text=normalized_text,
                    provider=self.name,
                    confidence=1.0,
                    language=self.resolved_model.language,
                )
        if segment.sample_width_bytes != 2:
            raise ValueError("Moonshine STT currently expects 16-bit PCM input")
        text = await asyncio.to_thread(self._transcribe_sync, segment)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Moonshine STT returned an empty transcription")
        return Transcription(
            text=normalized_text,
            provider=self.name,
            confidence=0.0,
            language=self.resolved_model.language,
        )

    def _transcribe_sync(self, segment: AudioSegment) -> str:
        backend = self._get_backend()
        audio_data = _pcm16le_to_float_mono(segment)
        result = backend.transcriber.transcribe_without_streaming(
            audio_data,
            sample_rate=segment.sample_rate_hz,
        )
        return _normalize_moonshine_output(result)

    def _get_backend(self) -> _MoonshineBackend:
        backend = self._backend
        if backend is not None:
            return backend
        with self._backend_lock:
            backend = self._backend
            if backend is not None:
                return backend
            moonshine = _import_moonshine()
            model_arch = moonshine.string_to_model_arch(self.resolved_model.architecture_name)
            model_path, resolved_model_arch = moonshine.get_model_for_language(
                self.resolved_model.language,
                model_arch,
            )
            backend = _MoonshineBackend(
                transcriber=moonshine.Transcriber(
                    model_path=model_path,
                    model_arch=resolved_model_arch,
                )
            )
            self._backend = backend
            return backend


def _import_moonshine() -> Any:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "Moonshine STT requires Python 3.10+ and the optional `moonshine-voice` "
            "package. Upgrade Python, install it, and set VOCALIVE_STT_PROVIDER=moonshine."
        )
    try:
        return importlib.import_module("moonshine_voice")
    except ImportError as exc:
        raise RuntimeError(
            "Moonshine STT requires the optional `moonshine-voice` package. "
            "Install it and set VOCALIVE_STT_PROVIDER=moonshine."
        ) from exc


def _pcm16le_to_float_mono(segment: AudioSegment) -> list[float]:
    samples = array.array("h")
    samples.frombytes(segment.pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    if segment.channels <= 0:
        raise ValueError("Moonshine STT expects a positive channel count")
    if segment.channels == 1:
        return [sample / 32768.0 for sample in samples]
    if len(samples) % segment.channels != 0:
        raise ValueError("PCM frame count does not align with the configured channel count")
    mono_samples: list[float] = []
    for index in range(0, len(samples), segment.channels):
        frame = samples[index : index + segment.channels]
        mono_samples.append((sum(frame) / segment.channels) / 32768.0)
    return mono_samples


def _normalize_moonshine_output(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        parts = [str(item).strip() for item in result if str(item).strip()]
        return " ".join(parts)
    if hasattr(result, "text"):
        return str(result.text)
    lines = getattr(result, "lines", None)
    if lines is not None:
        parts = [str(getattr(line, "text", "")).strip() for line in lines]
        return " ".join(part for part in parts if part)
    raise TypeError(f"Unsupported Moonshine transcription result: {type(result)!r}")


def _resolve_moonshine_model(
    model_name: str,
    default_language: str | None,
) -> _ResolvedMoonshineModel:
    normalized_model_name = (model_name or "").strip().lower()
    if not normalized_model_name:
        normalized_model_name = "base"
    if normalized_model_name.startswith("moonshine/"):
        normalized_model_name = normalized_model_name.split("/", 1)[1]
    if normalized_model_name in _MODEL_ARCHITECTURES:
        return _ResolvedMoonshineModel(
            architecture_name=normalized_model_name,
            language=_normalize_model_language(default_language) or "en",
        )
    if "-" in normalized_model_name:
        architecture_name, language = normalized_model_name.rsplit("-", 1)
        if architecture_name in _MODEL_ARCHITECTURES and language:
            return _ResolvedMoonshineModel(
                architecture_name=architecture_name,
                language=_normalize_model_language(language) or language,
            )
    raise ValueError(
        f"Unsupported Moonshine model {model_name!r}. "
        "Use a model architecture such as 'base' or a Moonshine model id such as 'base-ja'."
    )


def _normalize_model_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized_language = language.strip().replace("_", "-").lower()
    if not normalized_language:
        return None
    primary_subtag = normalized_language.split("-", 1)[0]
    return primary_subtag or normalized_language
