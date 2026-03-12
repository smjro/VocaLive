from __future__ import annotations

import array
import math
import sys
from abc import ABC, abstractmethod


_PCM16_MAX_ABS = 32768.0


class SpeechDetector(ABC):
    @abstractmethod
    def is_speech(self, chunk: bytes, *, sample_width_bytes: int) -> bool:
        """Return True when the current chunk should be treated as speech."""


class FixedThresholdSpeechDetector(SpeechDetector):
    def __init__(self, speech_threshold: float = 0.02) -> None:
        self.speech_threshold = max(0.0, speech_threshold)

    def is_speech(self, chunk: bytes, *, sample_width_bytes: int) -> bool:
        return _normalized_rms(chunk, sample_width_bytes) >= self.speech_threshold


class AdaptiveEnergySpeechDetector(SpeechDetector):
    """Treat steady background audio as noise and react to voice-like changes."""

    def __init__(self, speech_threshold: float = 0.02) -> None:
        self.speech_threshold = max(0.0, speech_threshold)
        self.minimum_absolute_energy = max(0.0015, self.speech_threshold * 0.35)
        self.minimum_noise_floor = max(0.001, self.speech_threshold * 0.3)
        self.minimum_raw_delta = max(0.0015, self.speech_threshold * 0.12)
        self.minimum_texture = max(0.018, self.speech_threshold * 0.9)
        self._enhanced_noise_floor = self.minimum_noise_floor
        self._raw_noise_floor = self.minimum_absolute_energy
        self._texture_floor = self.minimum_texture
        self._previous_sample = 0.0
        self._speech_active = False
        self._initialized = False

    def is_speech(self, chunk: bytes, *, sample_width_bytes: int) -> bool:
        if not chunk:
            self._speech_active = False
            return False
        raw_energy = _normalized_rms(chunk, sample_width_bytes)
        if sample_width_bytes == 2:
            enhanced_energy = self._preemphasized_rms(chunk)
            texture = self._waveform_texture(chunk, raw_energy)
        else:
            enhanced_energy = raw_energy
            texture = self.minimum_texture
        if not self._initialized:
            bootstrap_energy = enhanced_energy if enhanced_energy > 0.0 else raw_energy
            self._enhanced_noise_floor = max(
                self.minimum_noise_floor,
                bootstrap_energy * 0.75,
            )
            self._raw_noise_floor = max(
                self.minimum_absolute_energy,
                raw_energy * 0.85,
            )
            self._texture_floor = max(
                self.minimum_texture,
                texture * 0.85,
            )
            self._initialized = True
        enhanced_ratio_threshold = 1.2 if self._speech_active else 1.45
        raw_ratio_threshold = 1.16 if self._speech_active else 1.3
        texture_ratio_threshold = 1.18 if self._speech_active else 1.32
        enhanced_threshold = max(
            self.minimum_absolute_energy,
            self._enhanced_noise_floor * enhanced_ratio_threshold,
        )
        raw_threshold = max(
            self.minimum_absolute_energy * 1.8,
            self._raw_noise_floor * raw_ratio_threshold,
        )
        texture_threshold = max(
            self.minimum_texture,
            self._texture_floor * texture_ratio_threshold,
        )
        raw_delta_threshold = self.minimum_raw_delta * (1.0 if self._speech_active else 1.4)
        enhanced_detected = (
            enhanced_energy >= enhanced_threshold
            and raw_energy >= self.minimum_absolute_energy
        )
        raw_detected = (
            raw_energy >= raw_threshold
            and (raw_energy - self._raw_noise_floor) >= raw_delta_threshold
            and enhanced_energy >= (self.minimum_noise_floor * 0.8)
        )
        texture_detected = (
            texture >= texture_threshold
            and raw_energy >= max(self.minimum_absolute_energy * 2.0, self._raw_noise_floor * 0.78)
        )
        is_speech = enhanced_detected or raw_detected or texture_detected
        self._enhanced_noise_floor = self._update_noise_floor(
            self._enhanced_noise_floor,
            enhanced_energy,
            is_speech,
            minimum_floor=self.minimum_noise_floor,
        )
        self._raw_noise_floor = self._update_noise_floor(
            self._raw_noise_floor,
            raw_energy,
            is_speech,
            minimum_floor=self.minimum_absolute_energy,
        )
        self._texture_floor = self._update_noise_floor(
            self._texture_floor,
            texture,
            is_speech,
            minimum_floor=self.minimum_texture,
        )
        self._speech_active = is_speech
        return is_speech

    def _preemphasized_rms(self, chunk: bytes) -> float:
        samples = _read_pcm16_samples(chunk)
        if not samples:
            return 0.0
        total = 0.0
        previous_sample = self._previous_sample
        for sample in samples:
            current_sample = float(sample)
            emphasized = current_sample - (0.96 * previous_sample)
            total += emphasized * emphasized
            previous_sample = current_sample
        self._previous_sample = previous_sample
        return math.sqrt(total / len(samples)) / _PCM16_MAX_ABS

    def _waveform_texture(self, chunk: bytes, raw_energy: float) -> float:
        samples = _read_pcm16_samples(chunk)
        if len(samples) < 2:
            return self.minimum_texture
        total_delta = 0.0
        previous_sample = float(samples[0])
        for sample in samples[1:]:
            current_sample = float(sample)
            total_delta += abs(current_sample - previous_sample)
            previous_sample = current_sample
        average_delta = (total_delta / (len(samples) - 1)) / _PCM16_MAX_ABS
        return average_delta / max(raw_energy, 1e-6)

    def _update_noise_floor(
        self,
        current_floor: float,
        energy: float,
        is_speech: bool,
        *,
        minimum_floor: float,
    ) -> float:
        if is_speech:
            alpha = 0.995 if energy >= current_floor else 0.9
        else:
            alpha = 0.92 if energy >= current_floor else 0.75
        updated_floor = (alpha * current_floor) + ((1.0 - alpha) * energy)
        return max(minimum_floor, updated_floor)


def _normalized_rms(chunk: bytes, sample_width_bytes: int) -> float:
    if not chunk:
        return 0.0
    if sample_width_bytes <= 0:
        raise ValueError("sample_width_bytes must be greater than zero")
    usable_length = len(chunk) - (len(chunk) % sample_width_bytes)
    if usable_length == 0:
        return 0.0
    max_amplitude = float(1 << (8 * sample_width_bytes - 1))
    if sample_width_bytes == 2:
        samples = _read_pcm16_samples(chunk[:usable_length])
    else:
        samples = [
            int.from_bytes(
                chunk[offset : offset + sample_width_bytes],
                byteorder="little",
                signed=True,
            )
            for offset in range(0, usable_length, sample_width_bytes)
        ]
    if not samples:
        return 0.0
    mean_square = sum(float(sample) * float(sample) for sample in samples) / len(samples)
    return math.sqrt(mean_square) / max_amplitude


def _read_pcm16_samples(chunk: bytes) -> array.array:
    usable_length = len(chunk) - (len(chunk) % 2)
    samples = array.array("h")
    samples.frombytes(chunk[:usable_length])
    if sys.byteorder != "little":
        samples.byteswap()
    return samples
