from __future__ import annotations

import asyncio
import audioop
import importlib
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from vocalive.audio.devices import InputDeviceMatch, resolve_input_device
from vocalive.audio.vad import FixedSilenceTurnDetector, TurnDetector
from vocalive.models import AudioSegment
from vocalive.util.logging import get_logger, log_event


logger = get_logger(__name__)


class AudioInput(ABC):
    @abstractmethod
    async def read(self) -> AudioSegment | None:
        """Return the next utterance-sized audio segment."""

    async def close(self) -> None:
        """Release input resources."""


class QueueAudioInput(AudioInput):
    def __init__(self, maxsize: int = 4) -> None:
        self._queue: asyncio.Queue[AudioSegment | None] = asyncio.Queue(maxsize=maxsize)

    async def push(self, segment: AudioSegment) -> None:
        await self._queue.put(segment)

    async def close(self) -> None:
        await self._queue.put(None)

    async def read(self) -> AudioSegment | None:
        return await self._queue.get()


class UtteranceAccumulator:
    def __init__(
        self,
        sample_rate_hz: int,
        channels: int = 1,
        sample_width_bytes: int = 2,
        speech_threshold: float = 0.02,
        pre_speech_ms: float = 200.0,
        speech_hold_ms: float = 200.0,
        min_utterance_ms: float = 250.0,
        max_utterance_ms: float = 15_000.0,
        turn_detector: TurnDetector | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.speech_threshold = speech_threshold
        self.pre_speech_ms = max(0.0, pre_speech_ms)
        self.speech_hold_ms = max(0.0, speech_hold_ms)
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.turn_detector = turn_detector or FixedSilenceTurnDetector()
        self._buffer = bytearray()
        self._buffered_ms = 0.0
        self._preroll_chunks: deque[tuple[bytes, float]] = deque()
        self._preroll_buffered_ms = 0.0
        self._silence_ms = 0.0
        self._speech_hold_remaining_ms = 0.0
        self._started = False

    def add_chunk(self, chunk: bytes) -> AudioSegment | None:
        if not chunk:
            return None
        duration_ms = (
            len(chunk)
            / (self.sample_rate_hz * self.channels * self.sample_width_bytes)
            * 1000.0
        )
        max_amplitude = float(1 << (8 * self.sample_width_bytes - 1))
        rms = audioop.rms(chunk, self.sample_width_bytes)
        measured_speech = (rms / max_amplitude) >= self.speech_threshold

        if not self._started:
            if measured_speech:
                self._start_buffer_from_preroll()
            else:
                self._buffer_preroll_chunk(chunk, duration_ms)
                return None
        if measured_speech:
            self._speech_hold_remaining_ms = self.speech_hold_ms

        has_speech = measured_speech
        if not measured_speech and self._speech_hold_remaining_ms > 0.0:
            has_speech = True
            self._speech_hold_remaining_ms = max(
                0.0,
                self._speech_hold_remaining_ms - duration_ms,
            )
        elif measured_speech:
            self._silence_ms = 0.0

        self._buffer.extend(chunk)
        self._buffered_ms += duration_ms
        if not has_speech:
            self._silence_ms += duration_ms

        should_emit = self._buffered_ms >= self.max_utterance_ms or (
            self._buffered_ms >= self.min_utterance_ms
            and self.turn_detector.is_end_of_turn(self._buffered_ms, self._silence_ms)
        )
        if should_emit:
            return self._drain_segment()
        return None

    def flush(self) -> AudioSegment | None:
        if not self._buffer:
            return None
        return self._drain_segment()

    def _drain_segment(self) -> AudioSegment:
        segment = AudioSegment(
            pcm=bytes(self._buffer),
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            sample_width_bytes=self.sample_width_bytes,
        )
        self._buffer.clear()
        self._buffered_ms = 0.0
        self._clear_preroll()
        self._silence_ms = 0.0
        self._speech_hold_remaining_ms = 0.0
        self._started = False
        return segment

    def _buffer_preroll_chunk(self, chunk: bytes, duration_ms: float) -> None:
        if self.pre_speech_ms <= 0.0:
            self._clear_preroll()
            return
        self._preroll_chunks.append((chunk, duration_ms))
        self._preroll_buffered_ms += duration_ms
        while len(self._preroll_chunks) > 1 and self._preroll_buffered_ms > self.pre_speech_ms:
            _, removed_duration_ms = self._preroll_chunks.popleft()
            self._preroll_buffered_ms -= removed_duration_ms

    def _start_buffer_from_preroll(self) -> None:
        self._started = True
        for buffered_chunk, buffered_duration_ms in self._preroll_chunks:
            self._buffer.extend(buffered_chunk)
            self._buffered_ms += buffered_duration_ms
        self._clear_preroll()

    def _clear_preroll(self) -> None:
        self._preroll_chunks.clear()
        self._preroll_buffered_ms = 0.0


class MicrophoneAudioInput(AudioInput):
    def __init__(
        self,
        sample_rate_hz: int = 16_000,
        channels: int = 1,
        sample_width_bytes: int = 2,
        block_duration_ms: float = 40.0,
        speech_threshold: float = 0.02,
        pre_speech_ms: float = 200.0,
        speech_hold_ms: float = 200.0,
        silence_threshold_ms: float = 500.0,
        min_utterance_ms: float = 250.0,
        max_utterance_ms: float = 15_000.0,
        device: str | int | None = None,
        prefer_external_device: bool = True,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.block_duration_ms = block_duration_ms
        self.frames_per_block = max(1, int(sample_rate_hz * block_duration_ms / 1000.0))
        self.device = device
        self.prefer_external_device = prefer_external_device
        self._accumulator = UtteranceAccumulator(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
            speech_threshold=speech_threshold,
            pre_speech_ms=pre_speech_ms,
            speech_hold_ms=speech_hold_ms,
            min_utterance_ms=min_utterance_ms,
            max_utterance_ms=max_utterance_ms,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=silence_threshold_ms),
        )
        self._stream: Any | None = None
        self._selected_device: InputDeviceMatch | None = None
        self._closed = False

    async def start(self) -> str:
        await asyncio.to_thread(self._ensure_stream)
        return self.selected_device_label

    async def read(self) -> AudioSegment | None:
        if self._closed:
            return None
        stream = self._ensure_stream()
        while not self._closed:
            chunk = await asyncio.to_thread(self._read_chunk, stream)
            segment = self._accumulator.add_chunk(chunk)
            if segment is not None:
                return segment
        return self._accumulator.flush()

    async def close(self) -> None:
        self._closed = True
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        await asyncio.to_thread(stream.stop)
        await asyncio.to_thread(stream.close)
        if self._selected_device is not None:
            log_event(
                logger,
                "microphone_stream_closed",
                device=self._selected_device.label,
            )

    @property
    def selected_device_label(self) -> str:
        if self._selected_device is None:
            if self.device is None:
                return "system default input"
            if isinstance(self.device, int):
                return f"input device id {self.device}"
            return f"input device {self.device!r}"
        return self._selected_device.label

    def _ensure_stream(self) -> Any:
        if self._stream is not None:
            return self._stream
        sounddevice = _import_sounddevice()
        selected_device = resolve_input_device(
            sounddevice,
            requested_device=self.device,
            prefer_external=self.prefer_external_device,
        )
        stream = sounddevice.RawInputStream(
            samplerate=self.sample_rate_hz,
            blocksize=self.frames_per_block,
            device=selected_device.index,
            channels=self.channels,
            dtype="int16",
        )
        stream.start()
        self._selected_device = selected_device
        log_event(
            logger,
            "microphone_stream_started",
            device=selected_device.label,
            requested_device=self.device,
            selection=selected_device.selection,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
        )
        self._stream = stream
        return stream

    def _read_chunk(self, stream: Any) -> bytes:
        chunk, overflowed = stream.read(self.frames_per_block)
        if overflowed:
            # Preserve the chunk even when the backend reports an overrun.
            return bytes(chunk)
        return bytes(chunk)


def _import_sounddevice() -> Any:
    try:
        return importlib.import_module("sounddevice")
    except ImportError as exc:
        raise RuntimeError(
            "Microphone input requires the optional `sounddevice` package. "
            "Install it and set VOCALIVE_INPUT_PROVIDER=microphone."
        ) from exc
