from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import tempfile
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from vocalive.audio.input import AudioInput, UtteranceAccumulator
from vocalive.audio.speech_detection import AdaptiveEnergySpeechDetector
from vocalive.audio.vad import FixedSilenceTurnDetector
from vocalive.models import AudioSegment
from vocalive.util.logging import get_logger, log_event
from vocalive.util.windows_csharp import (
    communicate_with_cancellation,
    ensure_csharp_helper,
    terminate_process,
)


logger = get_logger(__name__)

_APPLICATION_AUDIO_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS = 20.0
_APPLICATION_AUDIO_HELPER_DIR = Path(tempfile.gettempdir()) / "vocalive-application-audio"
_APPLICATION_AUDIO_HELPER_SOURCE = r"""
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Threading;

[DataContract]
internal sealed class ApplicationInfo {
    [DataMember(Name = "processId")]
    public int ProcessId { get; set; }

    [DataMember(Name = "applicationName")]
    public string ApplicationName { get; set; }

    [DataMember(Name = "bundleIdentifier")]
    public string BundleIdentifier { get; set; }

    [DataMember(Name = "windowTitle")]
    public string WindowTitle { get; set; }
}

[Flags]
internal enum AudioClientStreamFlags : uint {
    Loopback = 0x00020000,
    EventCallback = 0x00040000,
}

[Flags]
internal enum AudioClientBufferFlags : uint {
    Silent = 0x2,
}

internal enum AudioClientShareMode {
    Shared = 0,
}

internal enum AUDIOCLIENT_ACTIVATION_TYPE {
    AUDIOCLIENT_ACTIVATION_TYPE_DEFAULT = 0,
    AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1,
}

internal enum PROCESS_LOOPBACK_MODE {
    PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0,
    PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE = 1,
}

[StructLayout(LayoutKind.Sequential)]
internal struct AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
    public uint TargetProcessId;
    public PROCESS_LOOPBACK_MODE ProcessLoopbackMode;
}

[StructLayout(LayoutKind.Sequential)]
internal struct AUDIOCLIENT_ACTIVATION_PARAMS {
    public AUDIOCLIENT_ACTIVATION_TYPE ActivationType;
    public AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS ProcessLoopbackParams;
}

[StructLayout(LayoutKind.Sequential)]
internal struct BLOB {
    public uint cbSize;
    public IntPtr pBlobData;
}

[StructLayout(LayoutKind.Explicit)]
internal struct PROPVARIANT {
    [FieldOffset(0)]
    public ushort vt;

    [FieldOffset(2)]
    public ushort wReserved1;

    [FieldOffset(4)]
    public ushort wReserved2;

    [FieldOffset(6)]
    public ushort wReserved3;

    [FieldOffset(8)]
    public BLOB blob;
}

[StructLayout(LayoutKind.Sequential)]
internal struct WAVEFORMATEX {
    public ushort wFormatTag;
    public ushort nChannels;
    public uint nSamplesPerSec;
    public uint nAvgBytesPerSec;
    public ushort nBlockAlign;
    public ushort wBitsPerSample;
    public ushort cbSize;
}

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2")]
internal interface IAudioClient {
    int Initialize(AudioClientShareMode shareMode, AudioClientStreamFlags streamFlags, long hnsBufferDuration, long hnsPeriodicity, IntPtr format, IntPtr audioSessionGuid);
    int GetBufferSize(out uint bufferSize);
    int GetStreamLatency(out long streamLatency);
    int GetCurrentPadding(out uint currentPadding);
    int IsFormatSupported(AudioClientShareMode shareMode, IntPtr format, IntPtr closestMatchFormat);
    int GetMixFormat(out IntPtr deviceFormatPointer);
    int GetDevicePeriod(out long defaultDevicePeriod, out long minimumDevicePeriod);
    int Start();
    int Stop();
    int Reset();
    int SetEventHandle(IntPtr eventHandle);
    int GetService(ref Guid riid, [MarshalAs(UnmanagedType.Interface)] out IAudioCaptureClient captureClient);
}

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("C8ADBD64-E71E-48a0-A4DE-185C395CD317")]
internal interface IAudioCaptureClient {
    int GetBuffer(out IntPtr data, out uint frameCount, out AudioClientBufferFlags flags, out ulong devicePosition, out ulong qpcPosition);
    int ReleaseBuffer(uint frameCount);
    int GetNextPacketSize(out uint packetSize);
}

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("41D949AB-9862-444A-80F6-C261334DA5EB")]
internal interface IActivateAudioInterfaceCompletionHandler {
    int ActivateCompleted(IActivateAudioInterfaceAsyncOperation activateOperation);
}

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("72A22D78-CDE4-431D-B8CC-843A71199B6D")]
internal interface IActivateAudioInterfaceAsyncOperation {
    int GetActivateResult(out int activateResult, [MarshalAs(UnmanagedType.IUnknown)] out object activatedInterface);
}

[ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90")]
internal interface IAgileObject {
}

[ComVisible(true)]
internal sealed class ActivateAudioInterfaceCompletionHandler : IActivateAudioInterfaceCompletionHandler, IAgileObject, IDisposable {
    private readonly AutoResetEvent completedEvent = new AutoResetEvent(false);
    private int activateResult = unchecked((int)0x80004005);
    private object activatedInterface;

    public int ActivateCompleted(IActivateAudioInterfaceAsyncOperation activateOperation) {
        try {
            int activationHr;
            object activationInterface;
            int operationHr = activateOperation.GetActivateResult(out activationHr, out activationInterface);
            activateResult = operationHr < 0 ? operationHr : activationHr;
            if (activateResult >= 0) {
                activatedInterface = activationInterface;
            }
        } catch (Exception ex) {
            activateResult = Marshal.GetHRForException(ex);
        } finally {
            completedEvent.Set();
        }
        return 0;
    }

    public IAudioClient WaitForClient(int timeoutMilliseconds) {
        if (!completedEvent.WaitOne(timeoutMilliseconds)) {
            throw new TimeoutException("ActivateAudioInterfaceAsync timed out");
        }
        Check(activateResult, "GetActivateResult");
        if (activatedInterface == null) {
            throw new InvalidOperationException("ActivateAudioInterfaceAsync returned no audio client");
        }
        IAudioClient audioClient = (IAudioClient)activatedInterface;
        activatedInterface = null;
        return audioClient;
    }

    public void Dispose() {
        completedEvent.Dispose();
    }

    private static void Check(int hr, string operation) {
        if (hr < 0) {
            Marshal.ThrowExceptionForHR(hr);
        }
    }
}

internal static class Program {
    private const ushort VT_BLOB = 0x0041;
    private const string VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback";
    private const int ActivateAudioInterfaceTimeoutMilliseconds = 10000;

    [DllImport("Mmdevapi.dll", ExactSpelling = true, CharSet = CharSet.Unicode)]
    private static extern int ActivateAudioInterfaceAsync(
        [MarshalAs(UnmanagedType.LPWStr)] string deviceInterfacePath,
        ref Guid riid,
        ref PROPVARIANT activationParams,
        IActivateAudioInterfaceCompletionHandler completionHandler,
        out IActivateAudioInterfaceAsyncOperation activationOperation
    );

    [MTAThread]
    private static int Main(string[] args) {
        try {
            if (args.Length == 0) {
                throw new InvalidOperationException("expected --list-applications or --capture-audio");
            }

            if (string.Equals(args[0], "--list-applications", StringComparison.OrdinalIgnoreCase)) {
                WriteApplicationsJson();
                return 0;
            }

            if (string.Equals(args[0], "--capture-audio", StringComparison.OrdinalIgnoreCase)) {
                int processId = 0;
                int sampleRateHz = 16000;
                int channels = 1;
                for (int index = 1; index < args.Length; index += 2) {
                    if (index + 1 >= args.Length) {
                        throw new InvalidOperationException("missing argument value");
                    }
                    string option = args[index];
                    string value = args[index + 1];
                    if (string.Equals(option, "--process-id", StringComparison.OrdinalIgnoreCase)) {
                        processId = int.Parse(value);
                    } else if (string.Equals(option, "--sample-rate-hz", StringComparison.OrdinalIgnoreCase)) {
                        sampleRateHz = int.Parse(value);
                    } else if (string.Equals(option, "--channels", StringComparison.OrdinalIgnoreCase)) {
                        channels = int.Parse(value);
                    } else {
                        throw new InvalidOperationException("unsupported option: " + option);
                    }
                }
                if (processId <= 0) {
                    throw new InvalidOperationException("--capture-audio requires --process-id");
                }
                CaptureLoopback(processId, sampleRateHz, channels);
                return 0;
            }

            throw new InvalidOperationException("unsupported command: " + args[0]);
        } catch (Exception ex) {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static void WriteApplicationsJson() {
        var applications = new List<ApplicationInfo>();
        foreach (Process process in Process.GetProcesses()) {
            try {
                string processName = process.ProcessName;
                string executablePath = null;
                string windowTitle = null;
                try {
                    executablePath = process.MainModule != null ? process.MainModule.FileName : null;
                } catch {
                    executablePath = null;
                }
                try {
                    windowTitle = process.MainWindowTitle;
                } catch {
                    windowTitle = null;
                }
                if (string.IsNullOrWhiteSpace(processName) && string.IsNullOrWhiteSpace(windowTitle)) {
                    continue;
                }
                applications.Add(new ApplicationInfo {
                    ProcessId = process.Id,
                    ApplicationName = processName,
                    BundleIdentifier = string.IsNullOrWhiteSpace(executablePath) ? null : executablePath,
                    WindowTitle = string.IsNullOrWhiteSpace(windowTitle) ? null : windowTitle,
                });
            } catch {
            } finally {
                process.Dispose();
            }
        }
        applications.Sort((left, right) => {
            int byName = string.Compare(left.ApplicationName, right.ApplicationName, StringComparison.OrdinalIgnoreCase);
            if (byName != 0) {
                return byName;
            }
            int byWindowTitle = string.Compare(left.WindowTitle, right.WindowTitle, StringComparison.OrdinalIgnoreCase);
            if (byWindowTitle != 0) {
                return byWindowTitle;
            }
            return left.ProcessId.CompareTo(right.ProcessId);
        });
        var serializer = new DataContractJsonSerializer(typeof(List<ApplicationInfo>));
        serializer.WriteObject(Console.OpenStandardOutput(), applications);
    }

    private static void CaptureLoopback(int processId, int sampleRateHz, int channels) {
        Process targetProcess = Process.GetProcessById(processId);
        IAudioClient audioClient = null;
        IAudioCaptureClient captureClient = null;
        IntPtr formatPtr = IntPtr.Zero;
        try {
            audioClient = ActivateProcessLoopbackAudioClient(processId);
            var format = new WAVEFORMATEX();
            format.wFormatTag = 1;
            format.nChannels = (ushort)Math.Max(1, channels);
            format.nSamplesPerSec = (uint)Math.Max(1, sampleRateHz);
            format.wBitsPerSample = 16;
            format.nBlockAlign = (ushort)(format.nChannels * (format.wBitsPerSample / 8));
            format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
            format.cbSize = 0;
            formatPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(WAVEFORMATEX)));
            Marshal.StructureToPtr(format, formatPtr, false);

            Check(
                audioClient.Initialize(
                    AudioClientShareMode.Shared,
                    AudioClientStreamFlags.Loopback | AudioClientStreamFlags.EventCallback,
                    0,
                    0,
                    formatPtr,
                    IntPtr.Zero
                ),
                "Initialize"
            );

            Guid captureClientGuid = typeof(IAudioCaptureClient).GUID;
            Check(audioClient.GetService(ref captureClientGuid, out captureClient), "GetService");

            using (var readyEvent = new AutoResetEvent(false)) {
                Check(audioClient.SetEventHandle(readyEvent.SafeWaitHandle.DangerousGetHandle()), "SetEventHandle");
                Check(audioClient.Start(), "Start");
                try {
                    Stream stdout = Console.OpenStandardOutput();
                    while (true) {
                        targetProcess.Refresh();
                        if (targetProcess.HasExited) {
                            return;
                        }
                        readyEvent.WaitOne(250);
                        uint packetSize;
                        while (captureClient.GetNextPacketSize(out packetSize) == 0 && packetSize > 0) {
                            IntPtr data;
                            uint frameCount;
                            AudioClientBufferFlags flags;
                            ulong devicePosition;
                            ulong qpcPosition;
                            Check(
                                captureClient.GetBuffer(
                                    out data,
                                    out frameCount,
                                    out flags,
                                    out devicePosition,
                                    out qpcPosition
                                ),
                                "GetBuffer"
                            );
                            try {
                                int byteCount = checked((int)(frameCount * format.nBlockAlign));
                                byte[] buffer = new byte[byteCount];
                                if ((flags & AudioClientBufferFlags.Silent) != AudioClientBufferFlags.Silent && byteCount > 0) {
                                    Marshal.Copy(data, buffer, 0, byteCount);
                                }
                                stdout.Write(buffer, 0, buffer.Length);
                            } finally {
                                Check(captureClient.ReleaseBuffer(frameCount), "ReleaseBuffer");
                            }
                        }
                    }
                } finally {
                    audioClient.Stop();
                }
            }
        } finally {
            if (formatPtr != IntPtr.Zero) {
                Marshal.FreeHGlobal(formatPtr);
            }
            if (captureClient != null && Marshal.IsComObject(captureClient)) {
                Marshal.ReleaseComObject(captureClient);
            }
            if (audioClient != null && Marshal.IsComObject(audioClient)) {
                Marshal.ReleaseComObject(audioClient);
            }
            targetProcess.Dispose();
        }
    }

    private static IAudioClient ActivateProcessLoopbackAudioClient(int processId) {
        var activationParams = new AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType = AUDIOCLIENT_ACTIVATION_TYPE.AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            ProcessLoopbackParams = new AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                TargetProcessId = (uint)processId,
                ProcessLoopbackMode = PROCESS_LOOPBACK_MODE.PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
            },
        };
        IntPtr activationParamsPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(AUDIOCLIENT_ACTIVATION_PARAMS)));
        Marshal.StructureToPtr(activationParams, activationParamsPtr, false);
        var propVariant = new PROPVARIANT {
            vt = VT_BLOB,
            blob = new BLOB {
                cbSize = (uint)Marshal.SizeOf(typeof(AUDIOCLIENT_ACTIVATION_PARAMS)),
                pBlobData = activationParamsPtr,
            },
        };
        Guid audioClientGuid = typeof(IAudioClient).GUID;
        using (var completionHandler = new ActivateAudioInterfaceCompletionHandler()) {
            IActivateAudioInterfaceAsyncOperation activationOperation = null;
            try {
                Check(
                    ActivateAudioInterfaceAsync(
                        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
                        ref audioClientGuid,
                        ref propVariant,
                        completionHandler,
                        out activationOperation
                    ),
                    "ActivateAudioInterfaceAsync"
                );
                return completionHandler.WaitForClient(ActivateAudioInterfaceTimeoutMilliseconds);
            } finally {
                if (activationOperation != null && Marshal.IsComObject(activationOperation)) {
                    Marshal.ReleaseComObject(activationOperation);
                }
                Marshal.FreeHGlobal(activationParamsPtr);
            }
        }
    }

    private static void Check(int hr, string operation) {
        if (hr < 0) {
            Marshal.ThrowExceptionForHR(hr);
        }
    }
}
"""
_APPLICATION_AUDIO_HELPER_HASH = hashlib.sha256(
    _APPLICATION_AUDIO_HELPER_SOURCE.encode("utf-8")
).hexdigest()[:12]
_APPLICATION_AUDIO_HELPER_SOURCE_PATH = (
    _APPLICATION_AUDIO_HELPER_DIR / f"windows-application-audio-{_APPLICATION_AUDIO_HELPER_HASH}.cs"
)
_APPLICATION_AUDIO_HELPER_BINARY_PATH = (
    _APPLICATION_AUDIO_HELPER_DIR / f"windows-application-audio-{_APPLICATION_AUDIO_HELPER_HASH}.exe"
)


@dataclass(frozen=True)
class _WindowsApplicationInfo:
    process_id: int
    application_name: str
    bundle_identifier: str | None
    window_title: str | None


class WindowsApplicationAudioInput(AudioInput):
    def __init__(
        self,
        target: str,
        sample_rate_hz: int = 16_000,
        channels: int = 1,
        sample_width_bytes: int = 2,
        block_duration_ms: float = 40.0,
        speech_threshold: float = 0.02,
        pre_speech_ms: float = 200.0,
        speech_hold_ms: float = 320.0,
        silence_threshold_ms: float = 650.0,
        min_utterance_ms: float = 250.0,
        max_utterance_ms: float = 15_000.0,
        timeout_seconds: float = 10.0,
        adaptive_vad_enabled: bool = True,
        speech_start_events_enabled: bool = True,
    ) -> None:
        self.target = target
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.block_duration_ms = block_duration_ms
        self.frames_per_block = max(1, int(sample_rate_hz * block_duration_ms / 1000.0))
        self.bytes_per_block = (
            self.frames_per_block * self.channels * self.sample_width_bytes
        )
        self.timeout_seconds = timeout_seconds
        self.adaptive_vad_enabled = adaptive_vad_enabled
        self.speech_start_events_enabled = speech_start_events_enabled
        self._on_speech_start: Callable[[], Awaitable[None] | None] | None = None
        self._background_tasks: set[asyncio.Future[object]] = set()
        self._accumulator = UtteranceAccumulator(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
            speech_threshold=speech_threshold,
            pre_speech_ms=pre_speech_ms,
            speech_hold_ms=speech_hold_ms,
            min_utterance_ms=min_utterance_ms,
            max_utterance_ms=max_utterance_ms,
            segment_source="application_audio",
            segment_source_label=None,
            turn_detector=FixedSilenceTurnDetector(silence_threshold_ms=silence_threshold_ms),
            speech_detector=(
                AdaptiveEnergySpeechDetector(speech_threshold=speech_threshold)
                if adaptive_vad_enabled
                else None
            ),
            on_speech_start=self._emit_speech_start,
        )
        self._selected_application: _WindowsApplicationInfo | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=8)
        self._helper_path: Path | None = None
        self._helper_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> str:
        await self._ensure_process()
        return self.selected_application_label

    def set_speech_start_handler(
        self,
        handler: Callable[[], Awaitable[None] | None] | None,
    ) -> None:
        if not self.speech_start_events_enabled:
            self._on_speech_start = None
            return
        self._on_speech_start = handler

    @property
    def selected_application_label(self) -> str:
        if self._selected_application is None:
            return f"application audio target {self.target!r}"
        label = (
            f"application audio {self._selected_application.application_name} "
            f"(pid={self._selected_application.process_id})"
        )
        if self._selected_application.window_title:
            return f"{label} {self._selected_application.window_title!r}"
        return label

    async def read(self) -> AudioSegment | None:
        if self._closed:
            return None
        process = await self._ensure_process()
        while not self._closed:
            chunk = await self._read_chunk(process)
            if not chunk:
                return self._flush_after_process_end()
            segment = self._accumulator.add_chunk(chunk)
            if segment is not None:
                return segment
        return self._accumulator.flush()

    async def close(self) -> None:
        self._closed = True
        process = self._process
        self._process = None
        if process is not None:
            await terminate_process(process)
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
        if self._background_tasks:
            await asyncio.gather(*tuple(self._background_tasks), return_exceptions=True)
        if self._selected_application is not None:
            log_event(
                logger,
                "application_audio_stream_closed",
                application_name=self._selected_application.application_name,
                executable_path=self._selected_application.bundle_identifier,
                process_id=self._selected_application.process_id,
                window_title=self._selected_application.window_title,
            )

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is not None:
            return process
        helper_path = await self._ensure_helper()
        selected_application = await self._resolve_target_application(helper_path)
        try:
            process = await asyncio.create_subprocess_exec(
                str(helper_path),
                "--capture-audio",
                "--process-id",
                str(selected_application.process_id),
                "--sample-rate-hz",
                str(self.sample_rate_hz),
                "--channels",
                str(self.channels),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Windows application audio helper is unavailable") from exc
        self._selected_application = selected_application
        self._accumulator.segment_source_label = selected_application.application_name
        self._process = process
        self._stderr_tail.clear()
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(process),
            name="vocalive-windows-application-audio-stderr",
        )
        log_event(
            logger,
            "application_audio_stream_started",
            application_name=selected_application.application_name,
            executable_path=selected_application.bundle_identifier,
            process_id=selected_application.process_id,
            window_title=selected_application.window_title,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            speech_threshold=self._accumulator.speech_threshold,
            adaptive_vad=self.adaptive_vad_enabled,
            backend="windows_process_loopback",
        )
        return process

    async def _read_chunk(self, process: asyncio.subprocess.Process) -> bytes:
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("Windows application audio capture stdout is unavailable")
        try:
            return await stdout.readexactly(self.bytes_per_block)
        except asyncio.IncompleteReadError as exc:
            if exc.partial:
                return bytes(exc.partial)
            await process.wait()
            self._raise_if_process_failed(process)
            return b""

    def _flush_after_process_end(self) -> AudioSegment | None:
        self._closed = True
        return self._accumulator.flush()

    async def _resolve_target_application(
        self,
        helper_path: Path,
    ) -> _WindowsApplicationInfo:
        try:
            process = await asyncio.create_subprocess_exec(
                str(helper_path),
                "--list-applications",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await communicate_with_cancellation(
                process=process,
                cancellation=None,
                timeout_seconds=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Windows application audio helper is unavailable") from exc
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Windows application audio app lookup timed out") from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"Windows application audio app lookup failed{suffix}")
        try:
            raw_applications = json.loads(stdout.decode("utf-8") or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Windows application audio app lookup returned invalid JSON") from exc
        applications = tuple(
            _coerce_application_info(entry)
            for entry in raw_applications
            if isinstance(entry, dict)
        )
        selected_application = _select_application(applications, self.target)
        if selected_application is None:
            raise RuntimeError(
                f"No running Windows application matched VOCALIVE_APP_AUDIO_TARGET={self.target!r}"
            )
        return selected_application

    async def _ensure_helper(self) -> Path:
        cached_path = self._helper_path
        if cached_path is not None and cached_path.exists():
            return cached_path
        async with self._helper_lock:
            cached_path = self._helper_path
            if cached_path is not None and cached_path.exists():
                return cached_path
            if _APPLICATION_AUDIO_HELPER_BINARY_PATH.exists():
                self._helper_path = _APPLICATION_AUDIO_HELPER_BINARY_PATH
                return _APPLICATION_AUDIO_HELPER_BINARY_PATH
            helper_path = await ensure_csharp_helper(
                source=_APPLICATION_AUDIO_HELPER_SOURCE,
                source_path=_APPLICATION_AUDIO_HELPER_SOURCE_PATH,
                output_path=_APPLICATION_AUDIO_HELPER_BINARY_PATH,
                references=("System.Runtime.Serialization.dll",),
                timeout_seconds=self.timeout_seconds,
                build_timeout_floor_seconds=_APPLICATION_AUDIO_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS,
                cancellation=None,
                unavailable_message="csc.exe is unavailable for Windows application audio capture",
                timeout_message="Windows application audio helper build timed out",
                failure_message="Windows application audio helper build failed",
            )
            self._helper_path = helper_path
            return helper_path

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

    def _raise_if_process_failed(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode in {None, 0}:
            return
        detail = "; ".join(self._stderr_tail)
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Windows application audio capture failed{suffix}")

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

    def _finalize_background_task(self, task: asyncio.Future[object]) -> None:
        self._background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log_event(
                logger,
                "application_audio_speech_start_handler_failed",
                error=str(exc),
            )


def _coerce_application_info(raw_application: dict[str, object]) -> _WindowsApplicationInfo:
    application_name = str(raw_application.get("applicationName") or "").strip()
    bundle_identifier = str(raw_application.get("bundleIdentifier") or "").strip() or None
    window_title = str(raw_application.get("windowTitle") or "").strip() or None
    return _WindowsApplicationInfo(
        process_id=int(raw_application["processId"]),
        application_name=application_name,
        bundle_identifier=bundle_identifier,
        window_title=window_title,
    )


def _select_application(
    applications: tuple[_WindowsApplicationInfo, ...],
    target: str,
) -> _WindowsApplicationInfo | None:
    normalized_target = _normalize_application_match_text(target)
    if not normalized_target:
        return None
    matchers = (
        lambda application: _normalize_application_match_text(application.application_name),
        lambda application: _normalize_application_match_text(application.bundle_identifier or ""),
        lambda application: _normalize_application_match_text(application.window_title or ""),
    )
    for matcher in matchers:
        for application in applications:
            if matcher(application) == normalized_target:
                return application
    for matcher in matchers:
        for application in applications:
            normalized_value = matcher(application)
            if normalized_value and normalized_target in normalized_value:
                return application
    return None


def _normalize_application_match_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())
