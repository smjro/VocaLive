from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, call, patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.screen.windows import WindowsWindowScreenCapture, _WindowsWindowInfo, _select_window


class WindowsWindowSelectionTests(unittest.TestCase):
    def test_select_window_prefers_window_title_match(self) -> None:
        windows = (
            _WindowsWindowInfo(
                window_id=11,
                owner_name="chrome",
                window_name="docs.google.com",
                layer=0,
            ),
            _WindowsWindowInfo(
                window_id=12,
                owner_name="chrome",
                window_name="YouTube - Google Chrome",
                layer=0,
            ),
        )

        selected = _select_window(windows, "YouTube")

        self.assertEqual(selected, windows[1])

    def test_select_window_falls_back_to_owner_name_match(self) -> None:
        windows = (
            _WindowsWindowInfo(
                window_id=21,
                owner_name="steam",
                window_name=None,
                layer=0,
            ),
        )

        selected = _select_window(windows, "steam")

        self.assertEqual(selected, windows[0])

    def test_select_window_ignores_nonzero_window_layer(self) -> None:
        windows = (
            _WindowsWindowInfo(
                window_id=31,
                owner_name="chrome",
                window_name="YouTube",
                layer=3,
            ),
            _WindowsWindowInfo(
                window_id=32,
                owner_name="chrome",
                window_name="YouTube",
                layer=0,
            ),
        )

        selected = _select_window(windows, "youtube")

        self.assertEqual(selected, windows[1])


class _StubProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


class WindowsWindowHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_window_id_uses_configured_timeout(self) -> None:
        engine = WindowsWindowScreenCapture(window_name="Steam", timeout_seconds=3.5)
        process = _StubProcess()

        with (
            patch.object(
                engine,
                "_ensure_window_capture_helper",
                AsyncMock(return_value=Path("C:/tmp/vocalive-window-capture.exe")),
            ),
            patch(
                "vocalive.screen.windows.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ) as create_subprocess_exec,
            patch(
                "vocalive.screen.windows.communicate_with_cancellation",
                AsyncMock(
                    return_value=(
                        b'[{"windowID": 42, "ownerName": "steam", "windowName": "Steam", "layer": 0}]',
                        b"",
                    )
                ),
            ) as communicate,
        ):
            window_id = await engine._resolve_window_id()

        self.assertEqual(window_id, 42)
        create_subprocess_exec.assert_awaited_once_with(
            str(Path("C:/tmp/vocalive-window-capture.exe")),
            "--list-windows",
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
        )
        communicate.assert_awaited_once_with(
            process=process,
            cancellation=None,
            timeout_seconds=3.5,
        )

    async def test_capture_window_passes_resize_arg_and_reads_output(self) -> None:
        engine = WindowsWindowScreenCapture(
            window_name="Steam",
            timeout_seconds=3.5,
            resize_max_edge_px=1024,
        )
        process = _StubProcess()

        async def create_subprocess_exec(*args, **kwargs):
            del kwargs
            output_index = args.index("--output-path") + 1
            Path(args[output_index]).write_bytes(b"png-data")
            return process

        with (
            patch.object(
                engine,
                "_ensure_window_capture_helper",
                AsyncMock(return_value=Path("C:/tmp/vocalive-window-capture.exe")),
            ),
            patch(
                "vocalive.screen.windows.asyncio.create_subprocess_exec",
                AsyncMock(side_effect=create_subprocess_exec),
            ) as create_subprocess_exec_mock,
            patch(
                "vocalive.screen.windows.communicate_with_cancellation",
                AsyncMock(return_value=(b"", b"")),
            ) as communicate,
        ):
            screenshot = await engine._capture_window(window_id=42)

        self.assertEqual(screenshot.mime_type, "image/png")
        self.assertEqual(screenshot.data, b"png-data")
        create_subprocess_exec_mock.assert_awaited_once()
        self.assertIn(call(
            str(Path("C:/tmp/vocalive-window-capture.exe")),
            "--capture-window",
            "--window-id",
            "42",
            "--output-path",
            unittest.mock.ANY,
            "--max-edge-px",
            "1024",
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
        ), create_subprocess_exec_mock.await_args_list)
        communicate.assert_awaited_once_with(
            process=process,
            cancellation=None,
            timeout_seconds=3.5,
        )

    async def test_window_capture_helper_build_is_cached(self) -> None:
        engine = WindowsWindowScreenCapture(window_name="Steam", timeout_seconds=3.5)

        with tempfile.TemporaryDirectory() as directory:
            helper_dir = Path(directory)
            source_path = helper_dir / "windows-window-capture.cs"
            binary_path = helper_dir / "windows-window-capture.exe"

            async def ensure_helper(**kwargs):
                del kwargs
                binary_path.write_bytes(b"binary")
                return binary_path

            with (
                patch("vocalive.screen.windows._WINDOW_CAPTURE_HELPER_DIR", helper_dir),
                patch("vocalive.screen.windows._WINDOW_CAPTURE_HELPER_SOURCE_PATH", source_path),
                patch("vocalive.screen.windows._WINDOW_CAPTURE_HELPER_BINARY_PATH", binary_path),
                patch(
                    "vocalive.screen.windows.ensure_csharp_helper",
                    AsyncMock(side_effect=ensure_helper),
                ) as ensure_csharp_helper_mock,
            ):
                first_path = await engine._ensure_window_capture_helper()
                second_path = await engine._ensure_window_capture_helper()

        self.assertEqual(first_path, binary_path)
        self.assertEqual(second_path, binary_path)
        ensure_csharp_helper_mock.assert_awaited_once()
