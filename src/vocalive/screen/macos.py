from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vocalive.models import ConversationInlineDataPart, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.screen.base import ScreenCaptureEngine


_WINDOW_QUERY_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS = 10.0
_WINDOW_QUERY_HELPER_DIR = Path(tempfile.gettempdir()) / "vocalive-screen-capture"
_WINDOW_QUERY_HELPER_SOURCE = """
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#include <stdio.h>

int main(void) {
    @autoreleasepool {
        CFArrayRef info = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID
        );
        if (info == NULL) {
            fprintf(stderr, "CGWindowListCopyWindowInfo returned null\\n");
            return 1;
        }

        NSArray *windows = CFBridgingRelease(info);
        NSMutableArray *result = [NSMutableArray arrayWithCapacity:windows.count];
        for (NSDictionary *item in windows) {
            NSNumber *windowID = item[(id)kCGWindowNumber];
            if (windowID == nil) {
                continue;
            }
            [result addObject:@{
                @"windowID": windowID,
                @"ownerName": item[(id)kCGWindowOwnerName] ?: @"",
                @"windowName": item[(id)kCGWindowName] ?: @"",
                @"layer": item[(id)kCGWindowLayer] ?: @0,
            }];
        }

        NSError *error = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:result options:0 error:&error];
        if (data == nil) {
            fprintf(
                stderr,
                "NSJSONSerialization failed: %s\\n",
                error.localizedDescription.UTF8String ?: "unknown error"
            );
            return 1;
        }

        if (fwrite(data.bytes, 1, data.length, stdout) != data.length) {
            fprintf(stderr, "fwrite failed\\n");
            return 1;
        }
        return 0;
    }
}
"""
_WINDOW_QUERY_HELPER_HASH = hashlib.sha256(
    _WINDOW_QUERY_HELPER_SOURCE.encode("utf-8")
).hexdigest()[:12]
_WINDOW_QUERY_HELPER_SOURCE_PATH = (
    _WINDOW_QUERY_HELPER_DIR / f"window-query-{_WINDOW_QUERY_HELPER_HASH}.m"
)
_WINDOW_QUERY_HELPER_BINARY_PATH = (
    _WINDOW_QUERY_HELPER_DIR / f"window-query-{_WINDOW_QUERY_HELPER_HASH}"
)


@dataclass(frozen=True)
class _MacOSWindowInfo:
    window_id: int
    owner_name: str
    window_name: str | None
    layer: int


class MacOSWindowScreenCapture(ScreenCaptureEngine):
    name = "macos-screencapture"

    def __init__(
        self,
        window_name: str,
        timeout_seconds: float = 5.0,
        resize_max_edge_px: int | None = 1280,
    ) -> None:
        self.window_name = window_name
        self.timeout_seconds = timeout_seconds
        self.resize_max_edge_px = resize_max_edge_px
        self._cached_window_id: int | None = None
        self._window_query_helper_path: Path | None = None
        self._window_query_helper_lock = asyncio.Lock()

    async def capture(
        self,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> ConversationInlineDataPart:
        del context
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        cached_window_id = self._cached_window_id
        if cached_window_id is not None:
            try:
                return await self._capture_window(
                    window_id=cached_window_id,
                    cancellation=cancellation,
                )
            except RuntimeError:
                self._cached_window_id = None

        window_id = await self._resolve_window_id(cancellation=cancellation)
        self._cached_window_id = window_id
        return await self._capture_window(
            window_id=window_id,
            cancellation=cancellation,
        )

    async def _resolve_window_id(
        self,
        cancellation: CancellationToken | None = None,
    ) -> int:
        helper_path = await self._ensure_window_query_helper(cancellation=cancellation)
        try:
            process = await asyncio.create_subprocess_exec(
                str(helper_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await _communicate_with_cancellation(
                process=process,
                cancellation=cancellation,
                timeout_seconds=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("macOS window lookup helper is unavailable") from exc
        except asyncio.TimeoutError as exc:
            raise RuntimeError("macOS window lookup timed out") from exc

        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            detail = f": {stderr_text}" if stderr_text else ""
            raise RuntimeError(f"macOS window lookup failed{detail}")

        try:
            raw_windows = json.loads(stdout.decode("utf-8") or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError("macOS window lookup returned invalid JSON") from exc

        windows = tuple(_coerce_window_info(entry) for entry in raw_windows if isinstance(entry, dict))
        selected_window = _select_window(windows, self.window_name)
        if selected_window is None:
            raise RuntimeError(
                f"No on-screen window matched VOCALIVE_SCREEN_WINDOW_NAME={self.window_name!r}"
            )
        return selected_window.window_id

    async def _ensure_window_query_helper(
        self,
        cancellation: CancellationToken | None = None,
    ) -> Path:
        cached_path = self._window_query_helper_path
        if cached_path is not None and cached_path.exists():
            return cached_path

        async with self._window_query_helper_lock:
            cached_path = self._window_query_helper_path
            if cached_path is not None and cached_path.exists():
                return cached_path
            if _WINDOW_QUERY_HELPER_BINARY_PATH.exists():
                self._window_query_helper_path = _WINDOW_QUERY_HELPER_BINARY_PATH
                return _WINDOW_QUERY_HELPER_BINARY_PATH

            _WINDOW_QUERY_HELPER_DIR.mkdir(parents=True, exist_ok=True)
            _WINDOW_QUERY_HELPER_SOURCE_PATH.write_text(
                _WINDOW_QUERY_HELPER_SOURCE,
                encoding="utf-8",
            )
            with tempfile.NamedTemporaryFile(
                dir=_WINDOW_QUERY_HELPER_DIR,
                prefix="window-query-",
                delete=False,
            ) as handle:
                temporary_binary_path = Path(handle.name)

            try:
                try:
                    process = await asyncio.create_subprocess_exec(
                        "/usr/bin/clang",
                        "-fobjc-arc",
                        "-framework",
                        "CoreGraphics",
                        "-framework",
                        "Foundation",
                        str(_WINDOW_QUERY_HELPER_SOURCE_PATH),
                        "-o",
                        str(temporary_binary_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await _communicate_with_cancellation(
                        process=process,
                        cancellation=cancellation,
                        timeout_seconds=max(
                            self.timeout_seconds,
                            _WINDOW_QUERY_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS,
                        ),
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError("clang is unavailable for macOS window lookup") from exc
                except asyncio.TimeoutError as exc:
                    raise RuntimeError("macOS window lookup helper build timed out") from exc

                if process.returncode != 0:
                    stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                    stdout_text = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
                    detail_text = stderr_text or stdout_text
                    detail = f": {detail_text}" if detail_text else ""
                    raise RuntimeError(f"macOS window lookup helper build failed{detail}")

                temporary_binary_path.replace(_WINDOW_QUERY_HELPER_BINARY_PATH)
            finally:
                temporary_binary_path.unlink(missing_ok=True)

            self._window_query_helper_path = _WINDOW_QUERY_HELPER_BINARY_PATH
            return _WINDOW_QUERY_HELPER_BINARY_PATH

    async def _capture_window(
        self,
        window_id: int,
        cancellation: CancellationToken | None = None,
    ) -> ConversationInlineDataPart:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            output_path = Path(handle.name)

        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    "screencapture",
                    "-x",
                    "-o",
                    "-t",
                    "png",
                    f"-l{window_id}",
                    str(output_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await _communicate_with_cancellation(
                    process=process,
                    cancellation=cancellation,
                    timeout_seconds=self.timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("macOS screencapture command is unavailable") from exc
            except asyncio.TimeoutError as exc:
                raise RuntimeError("macOS screencapture timed out") from exc

            if process.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                detail = f": {stderr_text}" if stderr_text else ""
                raise RuntimeError(f"macOS screencapture failed{detail}")

            await self._resize_image(output_path, cancellation=cancellation)
            image_bytes = output_path.read_bytes()
            if not image_bytes:
                raise RuntimeError("macOS screencapture produced an empty image")

            return ConversationInlineDataPart(mime_type="image/png", data=image_bytes)
        finally:
            output_path.unlink(missing_ok=True)

    async def _resize_image(
        self,
        image_path: Path,
        cancellation: CancellationToken | None = None,
    ) -> None:
        max_edge_px = self.resize_max_edge_px
        if max_edge_px is None:
            return

        with tempfile.NamedTemporaryFile(suffix=image_path.suffix, delete=False) as handle:
            resized_output_path = Path(handle.name)

        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    "sips",
                    "--resampleHeightWidthMax",
                    str(max_edge_px),
                    str(image_path),
                    "--out",
                    str(resized_output_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await _communicate_with_cancellation(
                    process=process,
                    cancellation=cancellation,
                    timeout_seconds=self.timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("macOS sips command is unavailable") from exc
            except asyncio.TimeoutError as exc:
                raise RuntimeError("macOS image resize timed out") from exc

            if process.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                stdout_text = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
                detail_text = stderr_text or stdout_text
                detail = f": {detail_text}" if detail_text else ""
                raise RuntimeError(f"macOS image resize failed{detail}")

            resized_image_bytes = resized_output_path.read_bytes()
            if not resized_image_bytes:
                raise RuntimeError("macOS image resize produced an empty image")
            image_path.write_bytes(resized_image_bytes)
        finally:
            resized_output_path.unlink(missing_ok=True)


def _coerce_window_info(raw_window: dict[str, object]) -> _MacOSWindowInfo:
    window_name = raw_window.get("windowName")
    owner_name = raw_window.get("ownerName")
    return _MacOSWindowInfo(
        window_id=int(raw_window["windowID"]),
        owner_name=str(owner_name or ""),
        window_name=str(window_name) if window_name else None,
        layer=int(raw_window.get("layer", 0)),
    )


def _select_window(
    windows: tuple[_MacOSWindowInfo, ...],
    target_name: str,
) -> _MacOSWindowInfo | None:
    normalized_target = _normalize_window_match_text(target_name)
    if not normalized_target:
        return None

    for extractor in (
        lambda window: window.window_name or "",
        lambda window: window.owner_name,
    ):
        for window in windows:
            if window.layer != 0:
                continue
            if normalized_target in _normalize_window_match_text(extractor(window)):
                return window
    return None


def _normalize_window_match_text(value: str) -> str:
    return " ".join(value.lower().split())


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    cancellation: CancellationToken | None,
    timeout_seconds: float,
) -> tuple[bytes, bytes]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    communicate_task = asyncio.create_task(process.communicate())
    try:
        while True:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            done, _ = await asyncio.wait(
                {communicate_task},
                timeout=min(remaining, 0.1),
            )
            if communicate_task in done:
                return await communicate_task
    except BaseException:
        await _terminate_process(process)
        if not communicate_task.done():
            communicate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await communicate_task
        raise


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.kill()
    with contextlib.suppress(ProcessLookupError):
        await process.wait()
