from __future__ import annotations

import array
import asyncio
import importlib
import math
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

_APPLICATION_AUDIO_PADDING_LEAD_MS = 80.0
_APPLICATION_AUDIO_PADDING_TAIL_MS = 120.0


class MoonshineSpeechToTextEngine(SpeechToTextEngine):
    name = "moonshine"

    def __init__(
        self,
        model_name: str = "base",
        default_language: str | None = None,
        application_audio_enhancement_enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.resolved_model = _resolve_moonshine_model(model_name, default_language)
        self.application_audio_enhancement_enabled = application_audio_enhancement_enabled
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
        if segment.source == "application_audio" and self.application_audio_enhancement_enabled:
            enhanced_audio = _enhance_application_audio(
                audio_data,
                sample_rate=segment.sample_rate_hz,
            )
            text = self._transcribe_audio_data(
                backend,
                enhanced_audio,
                sample_rate=segment.sample_rate_hz,
            )
            if text.strip():
                return text
            fallback_audio = _pad_with_silence(
                audio_data,
                sample_rate=segment.sample_rate_hz,
                lead_ms=_APPLICATION_AUDIO_PADDING_LEAD_MS,
                tail_ms=_APPLICATION_AUDIO_PADDING_TAIL_MS,
            )
            return self._transcribe_audio_data(
                backend,
                fallback_audio,
                sample_rate=segment.sample_rate_hz,
            )
        return self._transcribe_audio_data(
            backend,
            audio_data,
            sample_rate=segment.sample_rate_hz,
        )

    def _transcribe_audio_data(
        self,
        backend: _MoonshineBackend,
        audio_data: list[float],
        *,
        sample_rate: int,
    ) -> str:
        result = backend.transcriber.transcribe_without_streaming(
            audio_data,
            sample_rate=sample_rate,
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


def _enhance_application_audio(
    audio_data: list[float],
    sample_rate: int,
) -> list[float]:
    if not audio_data:
        return []
    voiced_audio = _high_pass_filter(audio_data, sample_rate=sample_rate, cutoff_hz=70.0)
    presence_audio = _apply_pre_emphasis(voiced_audio, coefficient=0.82)
    enhanced_audio = _blend_audio(audio_data, presence_audio, primary_weight=0.78, secondary_weight=0.22)
    enhanced_audio = _apply_soft_noise_gate(
        enhanced_audio,
        noise_percentile=0.12,
        threshold_scale=1.15,
        minimum_threshold=0.0012,
        quiet_gain=0.35,
    )
    enhanced_audio = _normalize_peak(enhanced_audio, target_peak=0.8, max_gain=3.0)
    return _pad_with_silence(
        enhanced_audio,
        sample_rate=sample_rate,
        lead_ms=_APPLICATION_AUDIO_PADDING_LEAD_MS,
        tail_ms=_APPLICATION_AUDIO_PADDING_TAIL_MS,
    )


def _high_pass_filter(
    audio_data: list[float],
    sample_rate: int,
    cutoff_hz: float,
) -> list[float]:
    if sample_rate <= 0 or len(audio_data) < 2:
        return list(audio_data)
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    filtered_audio = [0.0]
    previous_input = audio_data[0]
    previous_output = 0.0
    for sample in audio_data[1:]:
        output = alpha * (previous_output + sample - previous_input)
        filtered_audio.append(output)
        previous_input = sample
        previous_output = output
    return filtered_audio


def _apply_pre_emphasis(audio_data: list[float], coefficient: float) -> list[float]:
    if len(audio_data) < 2:
        return list(audio_data)
    emphasized_audio = [audio_data[0]]
    previous_sample = audio_data[0]
    for sample in audio_data[1:]:
        emphasized_audio.append(sample - (coefficient * previous_sample))
        previous_sample = sample
    return emphasized_audio


def _blend_audio(
    primary_audio: list[float],
    secondary_audio: list[float],
    *,
    primary_weight: float,
    secondary_weight: float,
) -> list[float]:
    if not primary_audio:
        return []
    if len(primary_audio) != len(secondary_audio):
        raise ValueError("audio streams must be aligned before blending")
    blended_audio: list[float] = []
    for primary_sample, secondary_sample in zip(primary_audio, secondary_audio):
        blended_audio.append(
            max(
                -1.0,
                min(
                    1.0,
                    (primary_sample * primary_weight) + (secondary_sample * secondary_weight),
                ),
            )
        )
    return blended_audio


def _apply_soft_noise_gate(
    audio_data: list[float],
    *,
    noise_percentile: float = 0.12,
    threshold_scale: float = 1.15,
    minimum_threshold: float = 0.0012,
    quiet_gain: float = 0.35,
) -> list[float]:
    if not audio_data:
        return []
    sample_step = max(1, len(audio_data) // 2048)
    magnitudes = sorted(abs(audio_data[index]) for index in range(0, len(audio_data), sample_step))
    if not magnitudes:
        return list(audio_data)
    percentile = min(max(noise_percentile, 0.0), 1.0)
    noise_floor = magnitudes[int((len(magnitudes) - 1) * percentile)]
    threshold = max(minimum_threshold, noise_floor * threshold_scale)
    if threshold >= 0.95:
        return [0.0 for _ in audio_data]
    normalized_quiet_gain = min(max(quiet_gain, 0.0), 1.0)
    gate_span = max(threshold * 2.5, 0.01)
    gated_audio: list[float] = []
    for sample in audio_data:
        magnitude = abs(sample)
        if magnitude <= threshold:
            adjusted = magnitude * normalized_quiet_gain
        else:
            blend = min(1.0, (magnitude - threshold) / gate_span)
            retained_gain = normalized_quiet_gain + ((1.0 - normalized_quiet_gain) * blend)
            adjusted = min(1.0, magnitude * retained_gain)
        gated_audio.append(math.copysign(adjusted, sample))
    return gated_audio


def _normalize_peak(
    audio_data: list[float],
    target_peak: float,
    max_gain: float,
) -> list[float]:
    if not audio_data:
        return []
    peak = max(abs(sample) for sample in audio_data)
    if peak <= 1e-6:
        return list(audio_data)
    gain = min(max_gain, target_peak / peak)
    return [max(-1.0, min(1.0, sample * gain)) for sample in audio_data]


def _pad_with_silence(
    audio_data: list[float],
    *,
    sample_rate: int,
    lead_ms: float,
    tail_ms: float,
) -> list[float]:
    if not audio_data or sample_rate <= 0:
        return list(audio_data)
    lead_padding = [0.0] * max(0, int(sample_rate * lead_ms / 1000.0))
    tail_padding = [0.0] * max(0, int(sample_rate * tail_ms / 1000.0))
    return lead_padding + list(audio_data) + tail_padding


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
