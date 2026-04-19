from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from vocalive.config.settings import ScreenCaptureSettings
from vocalive.llm.base import LanguageModel
from vocalive.models import (
    AudioSegment,
    ConversationInlineDataPart,
    ConversationRequestPart,
    TurnContext,
)
from vocalive.pipeline.interruption import CancellationToken, TurnCancelledError
from vocalive.pipeline.request_building import (
    build_screen_capture_parts,
    classify_screen_capture_request,
    passive_screen_capture_is_on_cooldown,
    screen_capture_fingerprint,
)
from vocalive.screen.base import ScreenCaptureEngine
from vocalive.util.logging import log_event
from vocalive.util.metrics import MetricsRecorder


@dataclass(frozen=True)
class _ScreenCaptureResult:
    completed_ms: float
    screenshot: ConversationInlineDataPart | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class _PendingScreenCapture:
    started_ms: float
    screen_capture_engine: ScreenCaptureEngine
    task: asyncio.Task[_ScreenCaptureResult]


class CurrentTurnScreenCaptureCoordinator:
    def __init__(
        self,
        *,
        settings: ScreenCaptureSettings,
        get_screen_capture_engine: Callable[[], ScreenCaptureEngine | None],
        get_language_model: Callable[[], LanguageModel],
        now_ms: Callable[[], float],
        set_active_stage: Callable[[str | None], None],
        logger: logging.Logger,
        metrics: MetricsRecorder,
    ) -> None:
        self.settings = settings
        self._get_screen_capture_engine = get_screen_capture_engine
        self._get_language_model = get_language_model
        self._now_ms = now_ms
        self._set_active_stage = set_active_stage
        self._logger = logger
        self._metrics = metrics
        self._last_observation_ms: float | None = None
        self._last_capture_fingerprint: str | None = None

    def reset(self) -> None:
        self._last_observation_ms = None
        self._last_capture_fingerprint = None

    def start_pending_capture(
        self,
        *,
        segment: AudioSegment,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> _PendingScreenCapture | None:
        if segment.source != "user":
            return None
        screen_capture_engine = self._get_screen_capture_engine()
        if (
            not self.settings.enabled
            or screen_capture_engine is None
            or not self._get_language_model().supports_multimodal_input
        ):
            return None
        user_text_hint = (segment.transcript_hint or "").strip()
        if user_text_hint:
            capture_mode = self._classify_capture_mode(user_text_hint)
            if capture_mode is None:
                return None
            if capture_mode == "passive" and passive_screen_capture_is_on_cooldown(
                now_ms=self._now_ms(),
                last_observation_ms=self._last_observation_ms,
                cooldown_seconds=self.settings.passive_cooldown_seconds,
            ):
                return None
        started_ms = self._now_ms()
        return _PendingScreenCapture(
            started_ms=started_ms,
            screen_capture_engine=screen_capture_engine,
            task=asyncio.create_task(
                self._capture_screenshot(
                    screen_capture_engine=screen_capture_engine,
                    context=context,
                    cancellation=cancellation,
                ),
                name=f"vocalive-screen-capture-{context.turn_id}",
            ),
        )

    async def discard_pending_capture(
        self,
        pending_capture: _PendingScreenCapture | None,
    ) -> None:
        if pending_capture is None:
            return
        if not pending_capture.task.done():
            pending_capture.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pending_capture.task

    async def maybe_capture_current_user_parts(
        self,
        *,
        user_text: str,
        context: TurnContext,
        cancellation: CancellationToken,
        pending_capture: _PendingScreenCapture | None = None,
    ) -> tuple[ConversationRequestPart, ...]:
        screen_capture_engine = (
            pending_capture.screen_capture_engine
            if pending_capture is not None
            else self._get_screen_capture_engine()
        )
        if (
            not self.settings.enabled
            or screen_capture_engine is None
            or not self._get_language_model().supports_multimodal_input
        ):
            await self.discard_pending_capture(pending_capture)
            return tuple()
        capture_mode = self._classify_capture_mode(user_text)
        if capture_mode is None:
            await self.discard_pending_capture(pending_capture)
            return tuple()

        capture_timestamp_ms: float | None = None
        if capture_mode == "passive":
            capture_timestamp_ms = (
                pending_capture.started_ms
                if pending_capture is not None
                else self._now_ms()
            )
            if passive_screen_capture_is_on_cooldown(
                now_ms=capture_timestamp_ms,
                last_observation_ms=self._last_observation_ms,
                cooldown_seconds=self.settings.passive_cooldown_seconds,
            ):
                log_event(
                    self._logger,
                    "screen_capture_skipped",
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    screen_capture_provider=screen_capture_engine.name,
                    screen_window_name=self.settings.window_name,
                    reason="passive_cooldown",
                    trigger_mode=capture_mode,
                )
                await self.discard_pending_capture(pending_capture)
                return tuple()

        self._set_active_stage("screen_capture")
        started_ms = (
            pending_capture.started_ms if pending_capture is not None else self._now_ms()
        )
        if pending_capture is not None:
            result = await pending_capture.task
        else:
            result = await self._capture_screenshot(
                screen_capture_engine=screen_capture_engine,
                context=context,
                cancellation=cancellation,
            )
        self._metrics.record_duration(
            stage="screen_capture",
            duration_ms=max(0.0, result.completed_ms - started_ms),
            context=context,
        )

        if result.error is not None:
            if isinstance(result.error, TurnCancelledError):
                raise result.error
            log_event(
                self._logger,
                "screen_capture_failed",
                session_id=context.session_id,
                turn_id=context.turn_id,
                screen_capture_provider=screen_capture_engine.name,
                screen_window_name=self.settings.window_name,
                error=str(result.error),
            )
            return tuple()

        screenshot = result.screenshot
        if screenshot is None:
            return tuple()
        if capture_timestamp_ms is None:
            capture_timestamp_ms = result.completed_ms
        screenshot_fingerprint = screen_capture_fingerprint(screenshot)
        self._last_observation_ms = capture_timestamp_ms
        if (
            capture_mode == "passive"
            and screenshot_fingerprint == self._last_capture_fingerprint
        ):
            log_event(
                self._logger,
                "screen_capture_skipped",
                session_id=context.session_id,
                turn_id=context.turn_id,
                screen_capture_provider=screen_capture_engine.name,
                screen_window_name=self.settings.window_name,
                reason="passive_unchanged",
                trigger_mode=capture_mode,
                mime_type=screenshot.mime_type,
                byte_length=len(screenshot.data),
            )
            return tuple()

        self._last_capture_fingerprint = screenshot_fingerprint
        log_event(
            self._logger,
            "screen_capture_ready",
            session_id=context.session_id,
            turn_id=context.turn_id,
            screen_capture_provider=screen_capture_engine.name,
            screen_window_name=self.settings.window_name,
            trigger_mode=capture_mode,
            mime_type=screenshot.mime_type,
            byte_length=len(screenshot.data),
        )
        return build_screen_capture_parts(
            screenshot,
            self.settings.window_name,
            capture_mode=capture_mode,
        )

    async def _capture_screenshot(
        self,
        *,
        screen_capture_engine: ScreenCaptureEngine,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> _ScreenCaptureResult:
        try:
            screenshot = await screen_capture_engine.capture(
                context=context,
                cancellation=cancellation,
            )
        except Exception as exc:
            return _ScreenCaptureResult(
                completed_ms=self._now_ms(),
                error=exc,
            )
        return _ScreenCaptureResult(
            completed_ms=self._now_ms(),
            screenshot=screenshot,
        )

    def _classify_capture_mode(
        self,
        user_text: str,
    ) -> str | None:
        return classify_screen_capture_request(
            user_text,
            trigger_phrases=self.settings.trigger_phrases,
            always_attach=self.settings.always_attach,
            passive_enabled=self.settings.passive_enabled,
            passive_trigger_phrases=self.settings.passive_trigger_phrases,
        )

    def capture_supported(self) -> bool:
        return (
            self.settings.enabled
            and self._get_screen_capture_engine() is not None
            and self._get_language_model().supports_multimodal_input
        )
