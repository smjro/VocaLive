from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

from vocalive.config.settings import AivisEngineMode, AivisSpeechSettings
from vocalive.util.logging import get_logger, log_event


logger = get_logger(__name__)

_AIVIS_ENGINE_STOP_TIMEOUT_SECONDS = 5.0


class ManagedAivisSpeechEngine:
    def __init__(self, settings: AivisSpeechSettings) -> None:
        self.base_url = settings.base_url.rstrip("/")
        self.engine_mode = settings.engine_mode
        self.engine_path = settings.engine_path
        self.cpu_num_threads = settings.cpu_num_threads
        self.startup_timeout_seconds = settings.startup_timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=8)

    async def start(self) -> None:
        if self.engine_mode is AivisEngineMode.EXTERNAL:
            return
        if _aivis_api_is_reachable(self.base_url, timeout_seconds=1.0):
            raise RuntimeError(
                f"AivisSpeech is already responding at {self.base_url}. "
                "Managed Aivis startup cannot take over an existing server. "
                "Stop the running engine or set VOCALIVE_AIVIS_ENGINE_MODE=external."
            )
        executable_path = self._resolve_engine_path()
        host, port = _parse_managed_base_url(self.base_url)
        command = [str(executable_path), "--host", host, "--port", str(port)]
        if self.engine_mode is AivisEngineMode.GPU:
            command.append("--use_gpu")
        if self.cpu_num_threads is not None:
            command.extend(["--cpu_num_threads", str(self.cpu_num_threads)])
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"AivisSpeech engine executable was not found: {executable_path}"
            ) from exc
        self._process = process
        self._stderr_tail.clear()
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(process),
            name="vocalive-aivis-engine-stderr",
        )
        log_event(
            logger,
            "aivis_engine_started",
            base_url=self.base_url,
            executable_path=str(executable_path),
            mode=self.engine_mode,
            cpu_num_threads=self.cpu_num_threads,
        )
        try:
            await self._wait_until_ready(process)
        except Exception as exc:
            log_event(
                logger,
                "aivis_engine_start_failed",
                base_url=self.base_url,
                mode=self.engine_mode,
                cpu_num_threads=self.cpu_num_threads,
                error=str(exc),
            )
            await self.close()
            raise
        log_event(
            logger,
            "aivis_engine_ready",
            base_url=self.base_url,
            mode=self.engine_mode,
        )

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=_AIVIS_ENGINE_STOP_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
        if process is not None:
            log_event(
                logger,
                "aivis_engine_stopped",
                base_url=self.base_url,
                mode=self.engine_mode,
                returncode=process.returncode,
            )

    async def _wait_until_ready(self, process: asyncio.subprocess.Process) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(self.startup_timeout_seconds, 0.1)
        while True:
            if _aivis_api_is_reachable(self.base_url, timeout_seconds=1.0):
                return
            if process.returncode is not None:
                detail = _format_stderr_tail(self._stderr_tail)
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    "AivisSpeech engine exited before becoming ready"
                    f"{suffix}"
                )
            if loop.time() >= deadline:
                detail = _format_stderr_tail(self._stderr_tail)
                suffix = f" Last stderr: {detail}" if detail else ""
                raise RuntimeError(
                    "AivisSpeech engine did not become ready within "
                    f"{self.startup_timeout_seconds:.1f}s at {self.base_url}.{suffix}"
                )
            await asyncio.sleep(0.25)

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        stderr = process.stderr
        if stderr is None:
            return
        while True:
            line = await stderr.readline()
            if not line:
                return
            normalized_line = line.decode("utf-8", errors="replace").strip()
            if normalized_line:
                self._stderr_tail.append(normalized_line)

    def _resolve_engine_path(self) -> Path:
        if self.engine_path:
            return _normalize_engine_path(Path(self.engine_path).expanduser())
        default_path = find_default_aivis_engine_path()
        if default_path is not None:
            return default_path
        raise RuntimeError(
            "Could not locate AivisSpeech Engine. "
            "Set VOCALIVE_AIVIS_ENGINE_PATH to the local `run(.exe)` path "
            "or set VOCALIVE_AIVIS_ENGINE_MODE=external."
        )


def find_default_aivis_engine_path() -> Path | None:
    for candidate in _default_aivis_engine_paths():
        normalized_candidate = _normalize_engine_path(candidate)
        if normalized_candidate.is_file():
            return normalized_candidate
    return None


def _default_aivis_engine_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    if sys.platform == "win32":
        for root in (
            os.environ.get("LOCALAPPDATA"),
            os.environ.get("ProgramFiles"),
        ):
            if not root:
                continue
            candidates.extend(
                (
                    Path(root) / "Programs" / "AivisSpeech" / "AivisSpeech-Engine" / "run.exe",
                    Path(root) / "AivisSpeech" / "AivisSpeech-Engine" / "run.exe",
                    Path(root) / "AivisSpeech-Engine" / "run.exe",
                )
            )
    if sys.platform == "darwin":
        candidates.extend(
            (
                Path("/Applications/AivisSpeech.app/Contents/Resources/AivisSpeech-Engine/run"),
                Path.home()
                / "Applications"
                / "AivisSpeech.app"
                / "Contents"
                / "Resources"
                / "AivisSpeech-Engine"
                / "run",
            )
        )
    return tuple(candidates)


def _normalize_engine_path(path: Path) -> Path:
    if path.is_dir():
        return path / ("run.exe" if sys.platform == "win32" else "run")
    return path


def _parse_managed_base_url(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http":
        raise RuntimeError(
            "Managed Aivis startup requires an http base URL, for example "
            "`http://127.0.0.1:10101`."
        )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RuntimeError(
            "Managed Aivis startup requires a bare base URL without a path, query, or fragment."
        )
    host = parsed.hostname
    if not host:
        raise RuntimeError("Managed Aivis startup requires a base URL host.")
    return host, parsed.port or 10101


def _aivis_api_is_reachable(base_url: str, timeout_seconds: float) -> bool:
    request = urllib.request.Request(f"{base_url}/speakers", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds):
            return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def _format_stderr_tail(lines: deque[str]) -> str:
    return "; ".join(lines)
