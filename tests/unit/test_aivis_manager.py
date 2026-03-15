from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import AivisEngineMode, AivisSpeechSettings
from vocalive.tts.aivis_manager import ManagedAivisSpeechEngine, find_default_aivis_engine_path


class _FakeStream:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = iter(lines or [b""])

    async def readline(self) -> bytes:
        return next(self._lines, b"")


class _FakeProcess:
    def __init__(self, *, stderr_lines: list[bytes] | None = None) -> None:
        self.returncode: int | None = None
        self.stderr = _FakeStream(stderr_lines)
        self.terminate_called = False
        self.kill_called = False

    def terminate(self) -> None:
        self.terminate_called = True
        self.returncode = 0

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9

    async def wait(self) -> int | None:
        return self.returncode


class ManagedAivisSpeechEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_uses_gpu_flag_and_waits_for_readiness(self) -> None:
        engine = ManagedAivisSpeechEngine(
            AivisSpeechSettings(
                base_url="http://127.0.0.1:10101",
                engine_mode=AivisEngineMode.GPU,
                engine_path="C:/AivisSpeech/run.exe",
                cpu_num_threads=4,
                startup_timeout_seconds=1.0,
            )
        )
        process = _FakeProcess()

        with (
            patch(
                "vocalive.tts.aivis_manager.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ) as create_subprocess_exec,
            patch(
                "vocalive.tts.aivis_manager._aivis_api_is_reachable",
                side_effect=[False, True],
            ),
        ):
            await engine.start()
            await engine.close()

        create_subprocess_exec.assert_awaited_once_with(
            str(Path("C:/AivisSpeech/run.exe")),
            "--host",
            "127.0.0.1",
            "--port",
            "10101",
            "--use_gpu",
            "--cpu_num_threads",
            "4",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self.assertTrue(process.terminate_called)
        self.assertFalse(process.kill_called)

    async def test_start_rejects_existing_server_in_managed_mode(self) -> None:
        engine = ManagedAivisSpeechEngine(
            AivisSpeechSettings(
                base_url="http://127.0.0.1:10101",
                engine_mode=AivisEngineMode.CPU,
                engine_path="C:/AivisSpeech/run.exe",
            )
        )

        with patch(
            "vocalive.tts.aivis_manager._aivis_api_is_reachable",
            return_value=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "already responding"):
                await engine.start()


class AivisEnginePathResolutionTests(unittest.TestCase):
    def test_find_default_aivis_engine_path_prefers_known_windows_install_location(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            local_app_data = Path(tempdir)
            expected_path = (
                local_app_data
                / "Programs"
                / "AivisSpeech"
                / "AivisSpeech-Engine"
                / "run.exe"
            )
            expected_path.parent.mkdir(parents=True)
            expected_path.write_bytes(b"")

            with patch("vocalive.tts.aivis_manager.sys.platform", "win32"), patch.dict(
                os.environ,
                {"LOCALAPPDATA": str(local_app_data), "ProgramFiles": ""},
                clear=False,
            ):
                actual_path = find_default_aivis_engine_path()

        self.assertEqual(actual_path, expected_path)
