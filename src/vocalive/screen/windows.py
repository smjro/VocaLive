from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vocalive.models import ConversationInlineDataPart, TurnContext
from vocalive.pipeline.interruption import CancellationToken
from vocalive.screen.base import ScreenCaptureEngine
from vocalive.util.windows_csharp import communicate_with_cancellation, ensure_csharp_helper


_WINDOW_CAPTURE_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS = 10.0
_WINDOW_CAPTURE_HELPER_DIR = Path(tempfile.gettempdir()) / "vocalive-screen-capture"
_WINDOW_CAPTURE_HELPER_SOURCE = r"""
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;

[DataContract]
internal sealed class WindowInfo {
    [DataMember(Name = "windowID")]
    public long WindowId { get; set; }

    [DataMember(Name = "ownerName")]
    public string OwnerName { get; set; }

    [DataMember(Name = "windowName")]
    public string WindowName { get; set; }

    [DataMember(Name = "layer")]
    public int Layer { get; set; }
}

internal static class NativeMethods {
    internal delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [StructLayout(LayoutKind.Sequential)]
    internal struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    internal static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    internal static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    internal static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    internal static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);

    [DllImport("user32.dll")]
    internal static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    internal static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    internal static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint flags);

    [DllImport("user32.dll")]
    internal static extern IntPtr GetWindowDC(IntPtr hWnd);

    [DllImport("user32.dll")]
    internal static extern int ReleaseDC(IntPtr hWnd, IntPtr hdc);

    [DllImport("gdi32.dll")]
    internal static extern bool BitBlt(
        IntPtr hdcDest,
        int xDest,
        int yDest,
        int width,
        int height,
        IntPtr hdcSrc,
        int xSrc,
        int ySrc,
        int rop
    );
}

internal static class Program {
    private const int SRCCOPY = 0x00CC0020;
    private const uint PW_RENDERFULLCONTENT = 0x00000002;

    private static int Main(string[] args) {
        try {
            if (args.Length == 0) {
                throw new InvalidOperationException("expected --list-windows or --capture-window");
            }

            if (string.Equals(args[0], "--list-windows", StringComparison.OrdinalIgnoreCase)) {
                WriteWindowsJson();
                return 0;
            }

            if (string.Equals(args[0], "--capture-window", StringComparison.OrdinalIgnoreCase)) {
                long windowId = 0;
                string outputPath = null;
                int maxEdgePx = 0;
                bool hasMaxEdge = false;
                for (int index = 1; index < args.Length; index += 2) {
                    if (index + 1 >= args.Length) {
                        throw new InvalidOperationException("missing argument value");
                    }
                    string option = args[index];
                    string value = args[index + 1];
                    if (string.Equals(option, "--window-id", StringComparison.OrdinalIgnoreCase)) {
                        windowId = long.Parse(value);
                    } else if (string.Equals(option, "--output-path", StringComparison.OrdinalIgnoreCase)) {
                        outputPath = value;
                    } else if (string.Equals(option, "--max-edge-px", StringComparison.OrdinalIgnoreCase)) {
                        maxEdgePx = int.Parse(value);
                        hasMaxEdge = true;
                    } else {
                        throw new InvalidOperationException("unsupported option: " + option);
                    }
                }
                if (windowId == 0 || string.IsNullOrWhiteSpace(outputPath)) {
                    throw new InvalidOperationException("--capture-window requires --window-id and --output-path");
                }
                CaptureWindow(new IntPtr(windowId), outputPath, hasMaxEdge ? (int?)maxEdgePx : null);
                return 0;
            }

            throw new InvalidOperationException("unsupported command: " + args[0]);
        } catch (Exception ex) {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static void WriteWindowsJson() {
        var windows = new List<WindowInfo>();
        NativeMethods.EnumWindows((hWnd, _) => {
            if (!NativeMethods.IsWindowVisible(hWnd)) {
                return true;
            }
            string windowTitle = ReadWindowText(hWnd);
            uint processId;
            NativeMethods.GetWindowThreadProcessId(hWnd, out processId);
            string ownerName = string.Empty;
            try {
                ownerName = Process.GetProcessById((int)processId).ProcessName;
            } catch {
                ownerName = string.Empty;
            }
            if (string.IsNullOrWhiteSpace(windowTitle) && string.IsNullOrWhiteSpace(ownerName)) {
                return true;
            }
            windows.Add(new WindowInfo {
                WindowId = hWnd.ToInt64(),
                OwnerName = ownerName ?? string.Empty,
                WindowName = string.IsNullOrWhiteSpace(windowTitle) ? null : windowTitle,
                Layer = 0,
            });
            return true;
        }, IntPtr.Zero);

        var serializer = new DataContractJsonSerializer(typeof(List<WindowInfo>));
        serializer.WriteObject(Console.OpenStandardOutput(), windows);
    }

    private static string ReadWindowText(IntPtr hWnd) {
        int length = NativeMethods.GetWindowTextLength(hWnd);
        if (length <= 0) {
            return string.Empty;
        }
        var builder = new StringBuilder(length + 1);
        NativeMethods.GetWindowText(hWnd, builder, builder.Capacity);
        return builder.ToString();
    }

    private static void CaptureWindow(IntPtr hWnd, string outputPath, int? maxEdgePx) {
        NativeMethods.RECT rect;
        if (!NativeMethods.GetWindowRect(hWnd, out rect)) {
            throw new InvalidOperationException("GetWindowRect failed");
        }
        int width = Math.Max(1, rect.Right - rect.Left);
        int height = Math.Max(1, rect.Bottom - rect.Top);

        using (var bitmap = new Bitmap(width, height))
        using (var graphics = Graphics.FromImage(bitmap)) {
            graphics.Clear(Color.Black);
            bool rendered = TryPrintWindow(hWnd, graphics, PW_RENDERFULLCONTENT)
                || TryPrintWindow(hWnd, graphics, 0);
            if (!rendered) {
                IntPtr sourceDc = NativeMethods.GetWindowDC(hWnd);
                if (sourceDc == IntPtr.Zero) {
                    throw new InvalidOperationException("GetWindowDC failed");
                }
                IntPtr targetDc = graphics.GetHdc();
                try {
                    if (!NativeMethods.BitBlt(targetDc, 0, 0, width, height, sourceDc, 0, 0, SRCCOPY)) {
                        throw new InvalidOperationException("BitBlt failed");
                    }
                } finally {
                    graphics.ReleaseHdc(targetDc);
                    NativeMethods.ReleaseDC(hWnd, sourceDc);
                }
            }

            using (var outputBitmap = ResizeIfNeeded(bitmap, maxEdgePx)) {
                outputBitmap.Save(outputPath, ImageFormat.Png);
            }
        }
    }

    private static bool TryPrintWindow(IntPtr hWnd, Graphics graphics, uint flags) {
        IntPtr targetDc = graphics.GetHdc();
        try {
            return NativeMethods.PrintWindow(hWnd, targetDc, flags);
        } finally {
            graphics.ReleaseHdc(targetDc);
        }
    }

    private static Bitmap ResizeIfNeeded(Bitmap bitmap, int? maxEdgePx) {
        if (!maxEdgePx.HasValue) {
            return (Bitmap)bitmap.Clone();
        }
        int maxEdge = maxEdgePx.Value;
        if (maxEdge <= 0 || (bitmap.Width <= maxEdge && bitmap.Height <= maxEdge)) {
            return (Bitmap)bitmap.Clone();
        }
        double scale = Math.Min((double)maxEdge / bitmap.Width, (double)maxEdge / bitmap.Height);
        int resizedWidth = Math.Max(1, (int)Math.Round(bitmap.Width * scale));
        int resizedHeight = Math.Max(1, (int)Math.Round(bitmap.Height * scale));
        var resized = new Bitmap(resizedWidth, resizedHeight);
        using (var graphics = Graphics.FromImage(resized)) {
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
            graphics.SmoothingMode = SmoothingMode.HighQuality;
            graphics.DrawImage(bitmap, 0, 0, resizedWidth, resizedHeight);
        }
        return resized;
    }
}
"""
_WINDOW_CAPTURE_HELPER_HASH = hashlib.sha256(
    _WINDOW_CAPTURE_HELPER_SOURCE.encode("utf-8")
).hexdigest()[:12]
_WINDOW_CAPTURE_HELPER_SOURCE_PATH = (
    _WINDOW_CAPTURE_HELPER_DIR / f"windows-window-capture-{_WINDOW_CAPTURE_HELPER_HASH}.cs"
)
_WINDOW_CAPTURE_HELPER_BINARY_PATH = (
    _WINDOW_CAPTURE_HELPER_DIR / f"windows-window-capture-{_WINDOW_CAPTURE_HELPER_HASH}.exe"
)


@dataclass(frozen=True)
class _WindowsWindowInfo:
    window_id: int
    owner_name: str
    window_name: str | None
    layer: int


class WindowsWindowScreenCapture(ScreenCaptureEngine):
    name = "windows-window-capture"

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
        self._window_capture_helper_path: Path | None = None
        self._window_capture_helper_lock = asyncio.Lock()

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
        helper_path = await self._ensure_window_capture_helper(cancellation=cancellation)
        try:
            process = await asyncio.create_subprocess_exec(
                str(helper_path),
                "--list-windows",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await communicate_with_cancellation(
                process=process,
                cancellation=cancellation,
                timeout_seconds=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Windows window capture helper is unavailable") from exc
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Windows window lookup timed out") from exc
        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            detail = f": {stderr_text}" if stderr_text else ""
            raise RuntimeError(f"Windows window lookup failed{detail}")
        try:
            raw_windows = json.loads(stdout.decode("utf-8") or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Windows window lookup returned invalid JSON") from exc
        windows = tuple(_coerce_window_info(entry) for entry in raw_windows if isinstance(entry, dict))
        selected_window = _select_window(windows, self.window_name)
        if selected_window is None:
            raise RuntimeError(
                f"No on-screen window matched VOCALIVE_SCREEN_WINDOW_NAME={self.window_name!r}"
            )
        return selected_window.window_id

    async def _ensure_window_capture_helper(
        self,
        cancellation: CancellationToken | None = None,
    ) -> Path:
        cached_path = self._window_capture_helper_path
        if cached_path is not None and cached_path.exists():
            return cached_path

        async with self._window_capture_helper_lock:
            cached_path = self._window_capture_helper_path
            if cached_path is not None and cached_path.exists():
                return cached_path
            if _WINDOW_CAPTURE_HELPER_BINARY_PATH.exists():
                self._window_capture_helper_path = _WINDOW_CAPTURE_HELPER_BINARY_PATH
                return _WINDOW_CAPTURE_HELPER_BINARY_PATH
            helper_path = await ensure_csharp_helper(
                source=_WINDOW_CAPTURE_HELPER_SOURCE,
                source_path=_WINDOW_CAPTURE_HELPER_SOURCE_PATH,
                output_path=_WINDOW_CAPTURE_HELPER_BINARY_PATH,
                references=("System.Drawing.dll", "System.Runtime.Serialization.dll"),
                timeout_seconds=self.timeout_seconds,
                build_timeout_floor_seconds=_WINDOW_CAPTURE_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS,
                cancellation=cancellation,
                unavailable_message="csc.exe is unavailable for Windows window capture",
                timeout_message="Windows window capture helper build timed out",
                failure_message="Windows window capture helper build failed",
            )
            self._window_capture_helper_path = helper_path
            return helper_path

    async def _capture_window(
        self,
        window_id: int,
        cancellation: CancellationToken | None = None,
    ) -> ConversationInlineDataPart:
        helper_path = await self._ensure_window_capture_helper(cancellation=cancellation)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            output_path = Path(handle.name)

        try:
            command = [
                str(helper_path),
                "--capture-window",
                "--window-id",
                str(window_id),
                "--output-path",
                str(output_path),
            ]
            if self.resize_max_edge_px is not None:
                command.extend(["--max-edge-px", str(self.resize_max_edge_px)])
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await communicate_with_cancellation(
                    process=process,
                    cancellation=cancellation,
                    timeout_seconds=self.timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("Windows window capture helper is unavailable") from exc
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Windows window capture timed out") from exc
            if process.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                stdout_text = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
                detail_text = stderr_text or stdout_text
                detail = f": {detail_text}" if detail_text else ""
                raise RuntimeError(f"Windows window capture failed{detail}")
            image_bytes = output_path.read_bytes()
            if not image_bytes:
                raise RuntimeError("Windows window capture produced an empty image")
            return ConversationInlineDataPart(mime_type="image/png", data=image_bytes)
        finally:
            output_path.unlink(missing_ok=True)


def _coerce_window_info(raw_window: dict[str, object]) -> _WindowsWindowInfo:
    window_name = raw_window.get("windowName")
    owner_name = raw_window.get("ownerName")
    return _WindowsWindowInfo(
        window_id=int(raw_window["windowID"]),
        owner_name=str(owner_name or ""),
        window_name=str(window_name) if window_name else None,
        layer=int(raw_window.get("layer", 0)),
    )


def _select_window(
    windows: tuple[_WindowsWindowInfo, ...],
    target_name: str,
) -> _WindowsWindowInfo | None:
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
