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


logger = get_logger(__name__)

_APPLICATION_AUDIO_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS = 20.0
_APPLICATION_AUDIO_HELPER_DIR = Path(tempfile.gettempdir()) / "vocalive-application-audio"
_APPLICATION_AUDIO_HELPER_SOURCE = r"""
import AVFoundation
import CoreMedia
import Darwin
import Foundation
import ScreenCaptureKit

struct ShareableApplication: Codable {
    let applicationName: String
    let bundleIdentifier: String?
}

enum HelperMode {
    case listApplications
    case captureAudio(target: String, sampleRateHz: Double, channels: Int)
}

enum HelperError: Error {
    case invalidArguments(String)
    case missingDisplay
    case applicationNotFound(String)
    case invalidAudioFormat
    case audioConversionFailed(String)
}

@main
struct ApplicationAudioCaptureHelper {
    static func main() async {
        do {
            let mode = try parseMode(arguments: Array(CommandLine.arguments.dropFirst()))
            switch mode {
            case .listApplications:
                try await listApplications()
            case let .captureAudio(target, sampleRateHz, channels):
                try await captureAudio(target: target, sampleRateHz: sampleRateHz, channels: channels)
            }
        } catch {
            let message = String(describing: error)
            FileHandle.standardError.write(Data((message + "\n").utf8))
            exit(1)
        }
    }

    static func parseMode(arguments: [String]) throws -> HelperMode {
        guard let command = arguments.first else {
            throw HelperError.invalidArguments("expected --list-applications or --capture-audio")
        }
        switch command {
        case "--list-applications":
            return .listApplications
        case "--capture-audio":
            var target: String?
            var sampleRateHz = 16_000.0
            var channels = 1
            var index = 1
            while index < arguments.count {
                let option = arguments[index]
                index += 1
                guard index <= arguments.count else {
                    break
                }
                switch option {
                case "--target":
                    guard index < arguments.count else {
                        throw HelperError.invalidArguments("missing value for --target")
                    }
                    target = arguments[index]
                    index += 1
                case "--sample-rate-hz":
                    guard index < arguments.count, let value = Double(arguments[index]) else {
                        throw HelperError.invalidArguments("invalid value for --sample-rate-hz")
                    }
                    sampleRateHz = value
                    index += 1
                case "--channels":
                    guard index < arguments.count, let value = Int(arguments[index]) else {
                        throw HelperError.invalidArguments("invalid value for --channels")
                    }
                    channels = value
                    index += 1
                default:
                    throw HelperError.invalidArguments("unsupported option: \(option)")
                }
            }
            guard let target else {
                throw HelperError.invalidArguments("--capture-audio requires --target")
            }
            return .captureAudio(target: target, sampleRateHz: sampleRateHz, channels: channels)
        default:
            throw HelperError.invalidArguments("unsupported command: \(command)")
        }
    }

    static func listApplications() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        let applications = content.applications
            .map { ShareableApplication(applicationName: $0.applicationName, bundleIdentifier: $0.bundleIdentifier) }
            .sorted { lhs, rhs in
                lhs.applicationName.localizedCaseInsensitiveCompare(rhs.applicationName) == .orderedAscending
            }
        let data = try JSONEncoder().encode(applications)
        FileHandle.standardOutput.write(data)
    }

    static func captureAudio(
        target: String,
        sampleRateHz: Double,
        channels: Int,
    ) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            throw HelperError.missingDisplay
        }
        guard let application = content.applications.first(where: {
            $0.applicationName == target || $0.bundleIdentifier == target
        }) else {
            throw HelperError.applicationNotFound(target)
        }

        let filter = SCContentFilter(
            display: display,
            including: [application],
            exceptingWindows: []
        )
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = Int(sampleRateHz)
        configuration.channelCount = channels
        configuration.queueDepth = 1
        configuration.width = 2
        configuration.height = 2

        let output = AudioCaptureOutput(sampleRateHz: sampleRateHz, channels: channels)
        let stream = SCStream(filter: filter, configuration: configuration, delegate: output)
        try stream.addStreamOutput(
            output,
            type: .audio,
            sampleHandlerQueue: DispatchQueue(label: "vocalive.app-audio")
        )
        try await stream.startCapture()
        while true {
            try await Task.sleep(nanoseconds: 60_000_000_000)
        }
    }
}

final class AudioCaptureOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    private let outputFormat: AVAudioFormat
    private let outputHandle = FileHandle.standardOutput
    private var converter: AVAudioConverter?

    init(sampleRateHz: Double, channels: Int) {
        self.outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRateHz,
            channels: AVAudioChannelCount(channels),
            interleaved: true
        )!
        super.init()
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write(Data((String(describing: error) + "\n").utf8))
        exit(1)
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .audio else {
            return
        }
        guard sampleBuffer.isValid, CMSampleBufferDataIsReady(sampleBuffer) else {
            return
        }

        do {
            let inputBuffer = try Self.makePCMBuffer(from: sampleBuffer)
            guard inputBuffer.frameLength > 0 else {
                return
            }
            if converter == nil || converter?.inputFormat != inputBuffer.format {
                converter = AVAudioConverter(from: inputBuffer.format, to: outputFormat)
            }
            guard let converter else {
                throw HelperError.invalidAudioFormat
            }
            let ratio = outputFormat.sampleRate / inputBuffer.format.sampleRate
            let outputFrameCapacity = max(
                AVAudioFrameCount(1),
                AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio))
            )
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outputFrameCapacity
            ) else {
                throw HelperError.invalidAudioFormat
            }

            var didProvideInput = false
            var conversionError: NSError?
            let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
                if didProvideInput {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                didProvideInput = true
                outStatus.pointee = .haveData
                return inputBuffer
            }
            if status == .error {
                throw HelperError.audioConversionFailed(
                    conversionError?.localizedDescription ?? "unknown converter error"
                )
            }
            guard outputBuffer.frameLength > 0 else {
                return
            }
            let bytesPerFrame = Int(outputFormat.streamDescription.pointee.mBytesPerFrame)
            let byteCount = Int(outputBuffer.frameLength) * bytesPerFrame
            guard
                let audioBuffer = outputBuffer.audioBufferList.pointee.mBuffers.mData,
                byteCount > 0
            else {
                return
            }
            outputHandle.write(Data(bytes: audioBuffer, count: byteCount))
        } catch {
            FileHandle.standardError.write(Data((String(describing: error) + "\n").utf8))
        }
    }

    private static func makePCMBuffer(from sampleBuffer: CMSampleBuffer) throws -> AVAudioPCMBuffer {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else {
            throw HelperError.invalidAudioFormat
        }
        guard let basicDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
            throw HelperError.invalidAudioFormat
        }
        var streamDescription = basicDescription.pointee
        guard let inputFormat = AVAudioFormat(streamDescription: &streamDescription) else {
            throw HelperError.invalidAudioFormat
        }
        let frameCount = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCount) else {
            throw HelperError.invalidAudioFormat
        }
        buffer.frameLength = frameCount
        try sampleBuffer.copyPCMData(fromRange: 0 ..< Int(frameCount), into: buffer.mutableAudioBufferList)
        return buffer
    }
}
"""
_APPLICATION_AUDIO_HELPER_HASH = hashlib.sha256(
    _APPLICATION_AUDIO_HELPER_SOURCE.encode("utf-8")
).hexdigest()[:12]
_APPLICATION_AUDIO_HELPER_SOURCE_PATH = (
    _APPLICATION_AUDIO_HELPER_DIR / f"application-audio-{_APPLICATION_AUDIO_HELPER_HASH}.swift"
)
_APPLICATION_AUDIO_HELPER_BINARY_PATH = (
    _APPLICATION_AUDIO_HELPER_DIR / f"application-audio-{_APPLICATION_AUDIO_HELPER_HASH}"
)
_APPLICATION_AUDIO_HELPER_MODULE_CACHE_PATH = (
    _APPLICATION_AUDIO_HELPER_DIR / "swift-module-cache"
)


@dataclass(frozen=True)
class _MacOSApplicationInfo:
    application_name: str
    bundle_identifier: str | None


class MacOSApplicationAudioInput(AudioInput):
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
        self._selected_application: _MacOSApplicationInfo | None = None
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
        if self._selected_application.bundle_identifier:
            return (
                f"application audio {self._selected_application.application_name} "
                f"({self._selected_application.bundle_identifier})"
            )
        return f"application audio {self._selected_application.application_name}"

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
            await _terminate_process(process)
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
                bundle_identifier=self._selected_application.bundle_identifier,
            )

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is not None:
            return process
        helper_path = await self._ensure_helper()
        selected_application = await self._resolve_target_application(helper_path)
        selected_target = (
            selected_application.bundle_identifier or selected_application.application_name
        )
        try:
            process = await asyncio.create_subprocess_exec(
                str(helper_path),
                "--capture-audio",
                "--target",
                selected_target,
                "--sample-rate-hz",
                str(self.sample_rate_hz),
                "--channels",
                str(self.channels),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("macOS application audio helper is unavailable") from exc
        self._selected_application = selected_application
        self._accumulator.segment_source_label = selected_application.application_name
        self._process = process
        self._stderr_tail.clear()
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(process),
            name="vocalive-application-audio-stderr",
        )
        log_event(
            logger,
            "application_audio_stream_started",
            application_name=selected_application.application_name,
            bundle_identifier=selected_application.bundle_identifier,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            speech_threshold=self._accumulator.speech_threshold,
            adaptive_vad=self.adaptive_vad_enabled,
        )
        return process

    async def _read_chunk(self, process: asyncio.subprocess.Process) -> bytes:
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("macOS application audio capture stdout is unavailable")
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
    ) -> _MacOSApplicationInfo:
        try:
            process = await asyncio.create_subprocess_exec(
                str(helper_path),
                "--list-applications",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("macOS application audio helper is unavailable") from exc
        except asyncio.TimeoutError as exc:
            raise RuntimeError("macOS application audio app lookup timed out") from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"macOS application audio app lookup failed{suffix}")
        try:
            raw_applications = json.loads(stdout.decode("utf-8") or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError("macOS application audio app lookup returned invalid JSON") from exc
        applications = tuple(
            _coerce_application_info(entry)
            for entry in raw_applications
            if isinstance(entry, dict)
        )
        selected_application = _select_application(applications, self.target)
        if selected_application is None:
            raise RuntimeError(
                f"No running macOS application matched VOCALIVE_APP_AUDIO_TARGET={self.target!r}"
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
            _APPLICATION_AUDIO_HELPER_DIR.mkdir(parents=True, exist_ok=True)
            _APPLICATION_AUDIO_HELPER_MODULE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
            _APPLICATION_AUDIO_HELPER_SOURCE_PATH.write_text(
                _APPLICATION_AUDIO_HELPER_SOURCE,
                encoding="utf-8",
            )
            with tempfile.NamedTemporaryFile(
                dir=_APPLICATION_AUDIO_HELPER_DIR,
                prefix="application-audio-",
                delete=False,
            ) as handle:
                temporary_binary_path = Path(handle.name)
            try:
                try:
                    process = await asyncio.create_subprocess_exec(
                        "/usr/bin/xcrun",
                        "swiftc",
                        "-parse-as-library",
                        "-module-cache-path",
                        str(_APPLICATION_AUDIO_HELPER_MODULE_CACHE_PATH),
                        str(_APPLICATION_AUDIO_HELPER_SOURCE_PATH),
                        "-o",
                        str(temporary_binary_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=max(
                            self.timeout_seconds,
                            _APPLICATION_AUDIO_HELPER_BUILD_TIMEOUT_FLOOR_SECONDS,
                        ),
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError("xcrun is unavailable for macOS application audio capture") from exc
                except asyncio.TimeoutError as exc:
                    raise RuntimeError("macOS application audio helper build timed out") from exc
                if process.returncode != 0:
                    stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                    stdout_text = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
                    detail_text = stderr_text or stdout_text
                    detail = f": {detail_text}" if detail_text else ""
                    raise RuntimeError(f"macOS application audio helper build failed{detail}")
                temporary_binary_path.replace(_APPLICATION_AUDIO_HELPER_BINARY_PATH)
            finally:
                temporary_binary_path.unlink(missing_ok=True)
            self._helper_path = _APPLICATION_AUDIO_HELPER_BINARY_PATH
            return _APPLICATION_AUDIO_HELPER_BINARY_PATH

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
        raise RuntimeError(f"macOS application audio capture failed{suffix}")

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


def _coerce_application_info(raw_application: dict[str, object]) -> _MacOSApplicationInfo:
    application_name = str(raw_application.get("applicationName") or "").strip()
    bundle_identifier = str(raw_application.get("bundleIdentifier") or "").strip() or None
    return _MacOSApplicationInfo(
        application_name=application_name,
        bundle_identifier=bundle_identifier,
    )


def _select_application(
    applications: tuple[_MacOSApplicationInfo, ...],
    target: str,
) -> _MacOSApplicationInfo | None:
    normalized_target = _normalize_application_match_text(target)
    if not normalized_target:
        return None
    matchers = (
        lambda application: _normalize_application_match_text(application.bundle_identifier or ""),
        lambda application: _normalize_application_match_text(application.application_name),
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


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except asyncio.TimeoutError:
        process.kill()
    with contextlib.suppress(ProcessLookupError):
        await process.wait()
