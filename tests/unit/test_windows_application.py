from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.windows_application import (
    WindowsApplicationAudioInput,
    _APPLICATION_AUDIO_HELPER_SOURCE,
    _WindowsApplicationInfo,
    _select_application,
)


class WindowsApplicationSelectionTests(unittest.TestCase):
    def test_helper_source_uses_process_scoped_loopback(self) -> None:
        self.assertIn(r'VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"', _APPLICATION_AUDIO_HELPER_SOURCE)
        self.assertIn(
            "PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE",
            _APPLICATION_AUDIO_HELPER_SOURCE,
        )
        self.assertNotIn("GetDefaultAudioEndpoint", _APPLICATION_AUDIO_HELPER_SOURCE)

    def test_select_application_prefers_process_name_match(self) -> None:
        applications = (
            _WindowsApplicationInfo(
                process_id=1,
                application_name="chrome",
                bundle_identifier=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                window_title="YouTube - Google Chrome",
            ),
            _WindowsApplicationInfo(
                process_id=2,
                application_name="steam",
                bundle_identifier=r"C:\Program Files (x86)\Steam\steam.exe",
                window_title="Steam",
            ),
        )

        selected = _select_application(applications, "steam")

        self.assertEqual(selected, applications[1])

    def test_select_application_matches_executable_path(self) -> None:
        applications = (
            _WindowsApplicationInfo(
                process_id=11,
                application_name="chrome",
                bundle_identifier=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                window_title="YouTube - Google Chrome",
            ),
        )

        selected = _select_application(
            applications,
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        )

        self.assertEqual(selected, applications[0])

    def test_select_application_falls_back_to_window_title_match(self) -> None:
        applications = (
            _WindowsApplicationInfo(
                process_id=21,
                application_name="chrome",
                bundle_identifier=None,
                window_title="YouTube - Google Chrome",
            ),
        )

        selected = _select_application(applications, "YouTube")

        self.assertEqual(selected, applications[0])


class _StubProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


class WindowsApplicationAudioInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_target_application_uses_helper_output(self) -> None:
        audio_input = WindowsApplicationAudioInput(target="steam", timeout_seconds=4.0)
        process = _StubProcess()

        with (
            patch(
                "vocalive.audio.windows_application.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ) as create_subprocess_exec,
            patch(
                "vocalive.audio.windows_application.communicate_with_cancellation",
                AsyncMock(
                    return_value=(
                        (
                            b'[{"processId": 42, "applicationName": "steam", '
                            b'"bundleIdentifier": "C:\\\\Steam\\\\steam.exe", '
                            b'"windowTitle": "Steam"}]'
                        ),
                        b"",
                    )
                ),
            ) as communicate,
        ):
            application = await audio_input._resolve_target_application(
                Path("C:/tmp/windows-application-audio.exe")
            )

        self.assertEqual(application.process_id, 42)
        self.assertEqual(application.application_name, "steam")
        create_subprocess_exec.assert_awaited_once_with(
            str(Path("C:/tmp/windows-application-audio.exe")),
            "--list-applications",
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
        )
        communicate.assert_awaited_once_with(
            process=process,
            cancellation=None,
            timeout_seconds=4.0,
        )

    async def test_ensure_process_uses_selected_process_id(self) -> None:
        audio_input = WindowsApplicationAudioInput(target="steam", timeout_seconds=4.0)
        selected_application = _WindowsApplicationInfo(
            process_id=42,
            application_name="steam",
            bundle_identifier=r"C:\Steam\steam.exe",
            window_title="Steam",
        )
        process = _StubProcess()

        with (
            patch.object(
                audio_input,
                "_ensure_helper",
                AsyncMock(return_value=Path("C:/tmp/windows-application-audio.exe")),
            ),
            patch.object(
                audio_input,
                "_resolve_target_application",
                AsyncMock(return_value=selected_application),
            ),
            patch.object(
                audio_input,
                "_drain_stderr",
                AsyncMock(return_value=None),
            ),
            patch("vocalive.audio.windows_application.log_event") as log_event,
            patch(
                "vocalive.audio.windows_application.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ) as create_subprocess_exec,
        ):
            resolved_process = await audio_input._ensure_process()

        self.assertIs(resolved_process, process)
        self.assertEqual(audio_input._selected_application, selected_application)
        self.assertEqual(audio_input._accumulator.segment_source_label, "steam")
        create_subprocess_exec.assert_awaited_once_with(
            str(Path("C:/tmp/windows-application-audio.exe")),
            "--capture-audio",
            "--process-id",
            "42",
            "--sample-rate-hz",
            "16000",
            "--channels",
            "1",
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
        )
        log_event.assert_called_once_with(
            unittest.mock.ANY,
            "application_audio_stream_started",
            application_name="steam",
            executable_path=r"C:\Steam\steam.exe",
            process_id=42,
            window_title="Steam",
            sample_rate_hz=16000,
            channels=1,
            speech_threshold=0.02,
            adaptive_vad=True,
            backend="windows_process_loopback",
        )
