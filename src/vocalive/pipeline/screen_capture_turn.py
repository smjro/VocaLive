from __future__ import annotations

import logging
from collections.abc import Callable

from vocalive.config.settings import ScreenCaptureSettings
from vocalive.llm.base import LanguageModel
from vocalive.models import ConversationRequestPart, TurnContext
from vocalive.pipeline.interruption import CancellationToken, TurnCancelledError
from vocalive.pipeline.request_building import (
    build_screen_capture_parts,
    classify_screen_capture_request,
    passive_screen_capture_is_on_cooldown,
    screen_capture_fingerprint,
)
from vocalive.screen.base import ScreenCaptureEngine
from vocalive.util.logging import log_event
from vocalive.util.metrics import MetricsRecorder, timed_stage


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

    async def maybe_capture_current_user_parts(
        self,
        *,
        user_text: str,
        context: TurnContext,
        cancellation: CancellationToken,
    ) -> tuple[ConversationRequestPart, ...]:
        if not self.capture_supported():
            return tuple()
        capture_mode = classify_screen_capture_request(
            user_text,
            trigger_phrases=self.settings.trigger_phrases,
            always_attach=self.settings.always_attach,
            passive_enabled=self.settings.passive_enabled,
            passive_trigger_phrases=self.settings.passive_trigger_phrases,
        )
        if capture_mode is None:
            return tuple()
        screen_capture_engine = self._get_screen_capture_engine()
        if screen_capture_engine is None:
            return tuple()

        capture_timestamp_ms: float | None = None
        if capture_mode == "passive":
            capture_timestamp_ms = self._now_ms()
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
                return tuple()

        self._set_active_stage("screen_capture")
        try:
            with timed_stage(self._metrics, "screen_capture", context):
                screenshot = await screen_capture_engine.capture(
                    context=context,
                    cancellation=cancellation,
                )
        except TurnCancelledError:
            raise
        except Exception as exc:
            log_event(
                self._logger,
                "screen_capture_failed",
                session_id=context.session_id,
                turn_id=context.turn_id,
                screen_capture_provider=screen_capture_engine.name,
                screen_window_name=self.settings.window_name,
                error=str(exc),
            )
            return tuple()

        if capture_timestamp_ms is None:
            capture_timestamp_ms = self._now_ms()
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

    def capture_supported(self) -> bool:
        return (
            self.settings.enabled
            and self._get_screen_capture_engine() is not None
            and self._get_language_model().supports_multimodal_input
        )
