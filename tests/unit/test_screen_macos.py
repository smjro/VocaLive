from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, call, patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.screen.macos import MacOSWindowScreenCapture, _MacOSWindowInfo, _select_window


class MacOSWindowSelectionTests(unittest.TestCase):
    def test_select_window_prefers_window_title_match(self) -> None:
        windows = (
            _MacOSWindowInfo(
                window_id=11,
                owner_name="Google Chrome",
                window_name="docs.google.com",
                layer=0,
            ),
            _MacOSWindowInfo(
                window_id=12,
                owner_name="Google Chrome",
                window_name="YouTube - Video",
                layer=0,
            ),
        )

        selected = _select_window(windows, "YouTube")

        self.assertEqual(selected, windows[1])

    def test_select_window_falls_back_to_owner_name_match(self) -> None:
        windows = (
            _MacOSWindowInfo(
                window_id=21,
                owner_name="Google Chrome",
                window_name=None,
                layer=0,
            ),
        )

        selected = _select_window(windows, "chrome")

        self.assertEqual(selected, windows[0])

    def test_select_window_ignores_nonzero_window_layer(self) -> None:
        windows = (
            _MacOSWindowInfo(
                window_id=31,
                owner_name="Google Chrome",
                window_name="YouTube",
                layer=3,
            ),
            _MacOSWindowInfo(
                window_id=32,
                owner_name="Google Chrome",
                window_name="YouTube",
                layer=0,
            ),
        )

        selected = _select_window(windows, "youtube")

        self.assertEqual(selected, windows[1])


class _StubProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


class MacOSWindowHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_window_id_uses_configured_timeout(self) -> None:
        engine = MacOSWindowScreenCapture(window_name="Steam", timeout_seconds=3.5)
        process = _StubProcess()

        with (
            patch.object(
                engine,
                "_ensure_window_query_helper",
                AsyncMock(return_value=Path("/tmp/vocalive-window-query")),
            ),
            patch(
                "vocalive.screen.macos.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ) as create_subprocess_exec,
            patch(
                "vocalive.screen.macos._communicate_with_cancellation",
                AsyncMock(
                    return_value=(
                        b'[{"windowID": 42, "ownerName": "Steam", "windowName": "Steam", "layer": 0}]',
                        b"",
                    )
                ),
            ) as communicate,
        ):
            window_id = await engine._resolve_window_id()

        self.assertEqual(window_id, 42)
        create_subprocess_exec.assert_awaited_once_with(
            "/tmp/vocalive-window-query",
            stdout=unittest.mock.ANY,
            stderr=unittest.mock.ANY,
        )
        communicate.assert_awaited_once_with(
            process=process,
            cancellation=None,
            timeout_seconds=3.5,
        )

    async def test_window_query_helper_build_is_cached(self) -> None:
        engine = MacOSWindowScreenCapture(window_name="Steam", timeout_seconds=3.5)
        process = _StubProcess()

        with tempfile.TemporaryDirectory() as directory:
            helper_dir = Path(directory)
            source_path = helper_dir / "window-query.m"
            binary_path = helper_dir / "window-query"

            async def create_subprocess_exec(*args, **kwargs):
                del kwargs
                Path(args[-1]).write_bytes(b"compiled-binary")
                return process

            with (
                patch("vocalive.screen.macos._WINDOW_QUERY_HELPER_DIR", helper_dir),
                patch("vocalive.screen.macos._WINDOW_QUERY_HELPER_SOURCE_PATH", source_path),
                patch("vocalive.screen.macos._WINDOW_QUERY_HELPER_BINARY_PATH", binary_path),
                patch(
                    "vocalive.screen.macos.asyncio.create_subprocess_exec",
                    AsyncMock(side_effect=create_subprocess_exec),
                ) as compile_process,
                patch(
                    "vocalive.screen.macos._communicate_with_cancellation",
                    AsyncMock(return_value=(b"", b"")),
                ),
            ):
                first_path = await engine._ensure_window_query_helper()
                second_path = await engine._ensure_window_query_helper()
                self.assertEqual(first_path, binary_path)
                self.assertEqual(second_path, binary_path)
                self.assertTrue(binary_path.exists())
                compile_process.assert_awaited_once()

    async def test_capture_window_resizes_image_before_returning(self) -> None:
        engine = MacOSWindowScreenCapture(
            window_name="Steam",
            timeout_seconds=3.5,
            resize_max_edge_px=1024,
        )
        screencapture_process = _StubProcess()
        resize_process = _StubProcess()

        async def create_subprocess_exec(*args, **kwargs):
            del kwargs
            command = args[0]
            if command == "screencapture":
                Path(args[-1]).write_bytes(b"original-png")
                return screencapture_process
            if command == "sips":
                output_path = Path(args[args.index("--out") + 1])
                output_path.write_bytes(b"resized-png")
                return resize_process
            raise AssertionError(f"unexpected command: {command}")

        with (
            patch(
                "vocalive.screen.macos.asyncio.create_subprocess_exec",
                AsyncMock(side_effect=create_subprocess_exec),
            ) as create_subprocess_exec,
            patch(
                "vocalive.screen.macos._communicate_with_cancellation",
                AsyncMock(return_value=(b"", b"")),
            ) as communicate,
        ):
            screenshot = await engine._capture_window(window_id=42)

        self.assertEqual(screenshot.mime_type, "image/png")
        self.assertEqual(screenshot.data, b"resized-png")
        self.assertEqual(create_subprocess_exec.await_count, 2)
        communicate.assert_has_awaits(
            [
                call(
                    process=screencapture_process,
                    cancellation=None,
                    timeout_seconds=3.5,
                ),
                call(
                    process=resize_process,
                    cancellation=None,
                    timeout_seconds=3.5,
                ),
            ]
        )
