from __future__ import annotations

import asyncio
import inspect
import importlib
import queue
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from vocalive.audio.devices import InputDeviceMatch, resolve_input_device
from vocalive.audio.speech_detection import FixedThresholdSpeechDetector, SpeechDetector
from vocalive.audio.vad import FixedSilenceTurnDetector, TurnDetector
from vocalive.models import AudioSegment
from vocalive.util.logging import get_logger, log_event


logger = get_logger(__name__)


class AudioInput(ABC):
    async def start(self) -> str | None:
        """Start any long-lived input resources and return a human-readable label."""
        return None

    def set_speech_start_handler(
        self,
        handler: Callable[[], Awaitable[None] | None] | None,
    ) -> None:
        del handler

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


class CombinedAudioInput(AudioInput):
    def __init__(
        self,
        inputs: tuple[AudioInput, ...] | list[AudioInput],
        maxsize: int = 8,
    ) -> None:
        self.inputs = tuple(inputs)
        if not self.inputs:
            raise ValueError("CombinedAudioInput requires at least one input")
        self._queue: asyncio.Queue[AudioSegment | None] = asyncio.Queue(maxsize=maxsize)
        self._forward_tasks: set[asyncio.Task[None]] = set()
        self._active_forwarders = 0
        self._closed = False
        self._error: Exception | None = None

    async def start(self) -> str | None:
        labels: list[str] = []
        for audio_input in self.inputs:
            label = await audio_input.start()
            if label:
                labels.append(label)
        self._ensure_forward_tasks()
        if not labels:
            return None
        return ", ".join(labels)

    def set_speech_start_handler(
        self,
        handler: Callable[[], Awaitable[None] | None] | None,
    ) -> None:
        for audio_input in self.inputs:
            audio_input.set_speech_start_handler(handler)

    async def read(self) -> AudioSegment | None:
        self._ensure_forward_tasks()
        segment = await self._queue.get()
        if segment is None and self._error is not None:
            raise self._error
        return segment

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(
            *(audio_input.close() for audio_input in self.inputs),
            return_exceptions=True,
        )
        if self._forward_tasks:
            await asyncio.gather(*tuple(self._forward_tasks), return_exceptions=True)
        if self._queue.empty():
            await self._queue.put(None)

    def _ensure_forward_tasks(self) -> None:
        if self._forward_tasks:
            return
        self._active_forwarders = len(self.inputs)
        for index, audio_input in enumerate(self.inputs):
            task = asyncio.create_task(
                self._forward_segments(audio_input),
                name=f"vocalive-audio-input-{index}",
            )
            self._forward_tasks.add(task)
            task.add_done_callback(self._forward_tasks.discard)

    async def _forward_segments(self, audio_input: AudioInput) -> None:
        try:
            while not self._closed:
                segment = await audio_input.read()
                if segment is None:
                    return
                await self._queue.put(segment)
        except Exception as exc:
            if self._error is None:
                self._error = exc
            self._closed = True
            await asyncio.gather(
                *(input_candidate.close() for input_candidate in self.inputs),
                return_exceptions=True,
            )
        finally:
            self._active_forwarders -= 1
            if self._active_forwarders == 0:
                await self._queue.put(None)


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
        segment_source: str = "user",
        segment_source_label: str | None = None,
        turn_detector: TurnDetector | None = None,
        speech_detector: SpeechDetector | None = None,
        on_speech_start: Callable[[], None] | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.speech_threshold = speech_threshold
        self.pre_speech_ms = max(0.0, pre_speech_ms)
        self.speech_hold_ms = max(0.0, speech_hold_ms)
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.segment_source = segment_source
        self.segment_source_label = segment_source_label
        self.turn_detector = turn_detector or FixedSilenceTurnDetector()
        self.speech_detector = speech_detector or FixedThresholdSpeechDetector(
            speech_threshold=speech_threshold
        )
        self.on_speech_start = on_speech_start
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
        measured_speech = self.speech_detector.is_speech(
            chunk,
            sample_width_bytes=self.sample_width_bytes,
        )

        if not self._started:
            if measured_speech:
                self._start_buffer_from_preroll()
                if self.on_speech_start is not None:
                    self.on_speech_start()
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
            source=self.segment_source,
            source_label=self.segment_source_label,
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
        self._on_speech_start: Callable[[], Awaitable[None] | None] | None = None
        self._background_tasks: set[asyncio.Future[Any]] = set()
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
            on_speech_start=self._emit_speech_start,
        )
        self._stream: Any | None = None
        self._selected_device: InputDeviceMatch | None = None
        self._chunk_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=32)
        self._dropped_chunk_count = 0
        self._closed = False

    async def start(self) -> str:
        await asyncio.to_thread(self._ensure_stream)
        return self.selected_device_label

    async def read(self) -> AudioSegment | None:
        if self._closed:
            return None
        self._ensure_stream()
        while True:
            chunk = await asyncio.to_thread(self._read_chunk)
            if chunk is None:
                return self._accumulator.flush()
            segment = self._accumulator.add_chunk(chunk)
            if segment is not None:
                return segment

    async def close(self) -> None:
        self._closed = True
        self._signal_reader_shutdown()
        stream = self._stream
        self._stream = None
        if stream is None:
            await self._wait_for_background_tasks()
            return
        await asyncio.to_thread(stream.stop)
        await asyncio.to_thread(stream.close)
        await self._wait_for_background_tasks()
        if self._selected_device is not None:
            log_event(
                logger,
                "microphone_stream_closed",
                device=self._selected_device.label,
            )

    def set_speech_start_handler(
        self,
        handler: Callable[[], Awaitable[None] | None] | None,
    ) -> None:
        self._on_speech_start = handler

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
            callback=self._handle_stream_chunk,
        )
        self._selected_device = selected_device
        stream.start()
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

    def _handle_stream_chunk(
        self,
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del frames, time_info, status
        if self._closed:
            return
        self._push_chunk(bytes(indata))

    def _read_chunk(self) -> bytes | None:
        return self._chunk_queue.get()

    def _push_chunk(self, chunk: bytes | None) -> None:
        while True:
            try:
                self._chunk_queue.put_nowait(chunk)
                return
            except queue.Full:
                try:
                    self._chunk_queue.get_nowait()
                except queue.Empty:
                    return
                if chunk is not None:
                    self._dropped_chunk_count += 1
                    if self._dropped_chunk_count == 1:
                        log_event(
                            logger,
                            "microphone_chunk_queue_overflow",
                            device=self.selected_device_label,
                        )

    def _signal_reader_shutdown(self) -> None:
        self._push_chunk(None)

    async def _wait_for_background_tasks(self) -> None:
        if not self._background_tasks:
            return
        await asyncio.gather(*tuple(self._background_tasks), return_exceptions=True)

    def _emit_speech_start(self) -> None:
        handler = self._on_speech_start
        if handler is None:
            return
        result = handler()
        if not inspect.isawaitable(result):
            return
        task = asyncio.ensure_future(result)
        self._background_tasks.add(task)
        task.add_done_callback(self._finalize_background_task)

    def _finalize_background_task(self, task: asyncio.Future[Any]) -> None:
        self._background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log_event(
                logger,
                "microphone_speech_start_handler_failed",
                error=str(exc),
            )


def _import_sounddevice() -> Any:
    try:
        return importlib.import_module("sounddevice")
    except ImportError as exc:
        raise RuntimeError(
            "Microphone input requires the optional `sounddevice` package. "
            "Install it and set VOCALIVE_INPUT_PROVIDER=microphone."
        ) from exc
