from __future__ import annotations

import asyncio
import shlex
import shutil
import tempfile
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path

from vocalive.models import SynthesizedSpeech
from vocalive.pipeline.interruption import CancellationToken


class AudioOutput(ABC):
    @abstractmethod
    async def play(
        self,
        speech: SynthesizedSpeech,
        cancellation: CancellationToken | None = None,
    ) -> None:
        """Play synthesized audio."""

    async def stop(self) -> None:
        """Stop playback if the backend supports it."""


class MemoryAudioOutput(AudioOutput):
    def __init__(self, chunk_delay_seconds: float = 0.0, chunk_size_bytes: int = 64) -> None:
        self.chunk_delay_seconds = chunk_delay_seconds
        self.chunk_size_bytes = max(1, chunk_size_bytes)
        self.started_texts: list[str] = []
        self.completed_texts: list[str] = []
        self.interrupted_texts: list[str] = []
        self.stop_calls = 0

    async def play(
        self,
        speech: SynthesizedSpeech,
        cancellation: CancellationToken | None = None,
    ) -> None:
        self.started_texts.append(speech.text)
        chunk_count = max(1, ceil(len(speech.audio) / self.chunk_size_bytes))
        for _ in range(chunk_count):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if self.chunk_delay_seconds > 0:
                await asyncio.sleep(self.chunk_delay_seconds)
        self.completed_texts.append(speech.text)

    async def stop(self) -> None:
        self.stop_calls += 1


class SpeakerAudioOutput(AudioOutput):
    def __init__(
        self,
        playback_command: tuple[str, ...] | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.playback_command = playback_command or tuple(_default_playback_command())
        self.poll_interval_seconds = poll_interval_seconds
        self._current_process: asyncio.subprocess.Process | None = None

    async def play(
        self,
        speech: SynthesizedSpeech,
        cancellation: CancellationToken | None = None,
    ) -> None:
        if speech.file_extension is None:
            raise ValueError(
                "speaker output requires synthesized speech with a file extension "
                "such as `.wav` or `.aiff`"
            )
        with tempfile.NamedTemporaryFile(suffix=speech.file_extension, delete=False) as handle:
            temp_path = handle.name
            handle.write(speech.audio)
        command = [part.format(path=temp_path) for part in self.playback_command]
        process = await asyncio.create_subprocess_exec(*command)
        self._current_process = process
        try:
            while True:
                if cancellation is not None and cancellation.is_cancelled():
                    await self._terminate_process(process)
                    cancellation.raise_if_cancelled()
                try:
                    await asyncio.wait_for(process.wait(), timeout=self.poll_interval_seconds)
                    break
                except asyncio.TimeoutError:
                    continue
            if process.returncode != 0:
                if cancellation is not None and cancellation.is_cancelled():
                    cancellation.raise_if_cancelled()
                raise RuntimeError(
                    f"speaker playback command failed with exit code {process.returncode}"
                )
        finally:
            if self._current_process is process:
                self._current_process = None
            Path(temp_path).unlink(missing_ok=True)

    async def stop(self) -> None:
        process = self._current_process
        if process is None:
            return
        await self._terminate_process(process)
        if self._current_process is process:
            self._current_process = None

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.kill()
        try:
            await process.wait()
        except ProcessLookupError:
            pass


def _default_playback_command() -> list[str]:
    afplay = shutil.which("afplay")
    if afplay is None:
        raise RuntimeError(
            "speaker output requires a playback command. "
            "On macOS `afplay` is used by default; otherwise set VOCALIVE_SPEAKER_COMMAND."
        )
    return [afplay, "{path}"]


def parse_playback_command(command: str | None) -> tuple[str, ...] | None:
    if not command:
        return None
    parsed = tuple(shlex.split(command))
    if "{path}" not in parsed:
        raise ValueError("VOCALIVE_SPEAKER_COMMAND must include a `{path}` placeholder")
    return parsed
