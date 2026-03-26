from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from vocalive.config.settings import AppSettings, ApplicationAudioMode
from vocalive.llm.base import LanguageModel
from vocalive.models import (
    AudioSegment,
    ConversationInlineDataPart,
    ConversationRequestPart,
    TurnContext,
)
from vocalive.pipeline.queues import BoundedAsyncQueue
from vocalive.pipeline.request_building import (
    build_proactive_current_user_parts,
    screen_capture_fingerprint,
)
from vocalive.screen.base import ScreenCaptureEngine
from vocalive.util.logging import log_event


_PROACTIVE_MONITOR_INTERVAL_SECONDS = 0.05
_PROACTIVE_SCREEN_FAILURE_BACKOFF_SECONDS = 60.0


@dataclass(frozen=True)
class ProactiveTurnRequest:
    observation_version: int


class ProactiveCoordinator:
    def __init__(
        self,
        *,
        settings: AppSettings,
        proactive_queue: BoundedAsyncQueue[ProactiveTurnRequest],
        idle_event: asyncio.Event,
        work_available: asyncio.Event,
        get_language_model: Callable[[], LanguageModel],
        get_screen_capture_engine: Callable[[], ScreenCaptureEngine | None],
        now_ms: Callable[[], float],
        get_session_id: Callable[[], str],
        build_poll_context: Callable[[], TurnContext],
        is_assistant_idle_for_proactive_work: Callable[[], bool],
        is_microphone_user_segment: Callable[[AudioSegment], bool],
        set_idle_if_drained: Callable[[], bool],
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self._proactive_queue = proactive_queue
        self._idle_event = idle_event
        self._work_available = work_available
        self._get_language_model = get_language_model
        self._get_screen_capture_engine = get_screen_capture_engine
        self._now_ms = now_ms
        self._get_session_id = get_session_id
        self._build_poll_context = build_poll_context
        self._is_assistant_idle_for_proactive_work = (
            is_assistant_idle_for_proactive_work
        )
        self._is_microphone_user_segment = is_microphone_user_segment
        self._set_idle_if_drained = set_idle_if_drained
        self._logger = logger
        self._last_user_activity_ms: float | None = None
        self._last_proactive_response_ms: float | None = None
        self._last_proactive_screen_poll_ms: float | None = None
        self._proactive_screen_failure_backoff_until_ms: float | None = None
        self._latest_proactive_screen_capture: ConversationInlineDataPart | None = None
        self._latest_proactive_screen_capture_fingerprint: str | None = None
        self._has_new_proactive_observation = False
        self._proactive_observation_version = 0
        self._queued_proactive_observation_version: int | None = None
        self._active_proactive_observation_version: int | None = None

    @property
    def last_user_activity_ms(self) -> float | None:
        return self._last_user_activity_ms

    @last_user_activity_ms.setter
    def last_user_activity_ms(self, value: float | None) -> None:
        self._last_user_activity_ms = value

    @property
    def last_proactive_response_ms(self) -> float | None:
        return self._last_proactive_response_ms

    @last_proactive_response_ms.setter
    def last_proactive_response_ms(self, value: float | None) -> None:
        self._last_proactive_response_ms = value

    def initialize_user_activity(self) -> None:
        if self.settings.proactive.enabled and self._last_user_activity_ms is None:
            self._last_user_activity_ms = self._now_ms()

    def reset_session_state(self) -> None:
        self._last_user_activity_ms = (
            self._now_ms() if self.settings.proactive.enabled else None
        )
        self._last_proactive_response_ms = None
        self.clear_observations()

    def record_live_user_activity(self) -> None:
        self._last_user_activity_ms = self._now_ms()

    def record_proactive_response(self, committed_at_ms: float) -> None:
        self._last_proactive_response_ms = committed_at_ms

    def record_microphone_observation(self, segment: AudioSegment) -> None:
        if not self.settings.proactive.enabled or not self.settings.proactive.microphone_enabled:
            return
        if not self._is_microphone_user_segment(segment):
            return
        self.record_observation()

    def record_application_audio_observation(self, segment: AudioSegment) -> None:
        if (
            not self.settings.proactive.enabled
            or not self.settings.proactive.application_audio_enabled
        ):
            return
        if segment.source != "application_audio":
            return
        if self.settings.application_audio.mode is not ApplicationAudioMode.CONTEXT_ONLY:
            return
        self.record_observation()

    def record_observation(self) -> None:
        self._proactive_observation_version += 1
        self._has_new_proactive_observation = True

    def consume_observation(self, observation_version: int) -> None:
        if self._proactive_observation_version == observation_version:
            self._has_new_proactive_observation = False

    def clear_observations(self) -> None:
        self.discard_pending_turns()
        self._last_proactive_screen_poll_ms = None
        self._proactive_screen_failure_backoff_until_ms = None
        self._latest_proactive_screen_capture = None
        self._latest_proactive_screen_capture_fingerprint = None
        self._has_new_proactive_observation = False
        self._proactive_observation_version = 0
        self._active_proactive_observation_version = None

    def discard_pending_turns(self) -> int:
        drained_count = _drain_queue(self._proactive_queue)
        self._queued_proactive_observation_version = None
        return drained_count

    def is_current_observation(self, observation_version: int) -> bool:
        return (
            self.settings.proactive.enabled
            and self._has_new_proactive_observation
            and observation_version == self._proactive_observation_version
        )

    def can_start_turn(self, *, now_ms: float) -> bool:
        if not self.settings.proactive.enabled or not self._has_new_proactive_observation:
            return False
        if not self._is_assistant_idle_for_proactive_work():
            return False
        if self._last_user_activity_ms is None:
            return False
        if (
            now_ms - self._last_user_activity_ms
        ) < (self.settings.proactive.idle_seconds * 1000.0):
            return False
        if self._last_proactive_response_ms is None:
            return True
        return (
            now_ms - self._last_proactive_response_ms
        ) >= (self.settings.proactive.cooldown_seconds * 1000.0)

    def should_run_request(self, request: ProactiveTurnRequest) -> bool:
        if not self.is_current_observation(request.observation_version):
            return False
        return self.can_start_turn(now_ms=self._now_ms())

    def mark_request_dequeued(self, request: ProactiveTurnRequest) -> None:
        if self._queued_proactive_observation_version == request.observation_version:
            self._queued_proactive_observation_version = None

    def mark_request_started(self, request: ProactiveTurnRequest) -> None:
        self._active_proactive_observation_version = request.observation_version

    def mark_request_finished(self, request: ProactiveTurnRequest) -> None:
        if self._active_proactive_observation_version == request.observation_version:
            self._active_proactive_observation_version = None

    async def run_monitor(self) -> None:
        try:
            while True:
                await self.poll_screen_if_due()
                await self.maybe_enqueue_turn()
                await asyncio.sleep(_PROACTIVE_MONITOR_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    async def maybe_enqueue_turn(self) -> None:
        if not self.settings.proactive.enabled:
            return
        if not self.can_start_turn(now_ms=self._now_ms()):
            return
        observation_version = self._proactive_observation_version
        if self._queued_proactive_observation_version == observation_version:
            return
        if self._active_proactive_observation_version == observation_version:
            return
        self._idle_event.clear()
        accepted = await self._proactive_queue.put(
            ProactiveTurnRequest(observation_version=observation_version)
        )
        if not accepted:
            self._set_idle_if_drained()
            return
        self._queued_proactive_observation_version = observation_version
        self._work_available.set()

    async def poll_screen_if_due(self) -> None:
        if not self.screen_capture_supported():
            return
        if not self._is_assistant_idle_for_proactive_work():
            return
        now_ms = self._now_ms()
        if self._last_user_activity_ms is None:
            return
        if (
            now_ms - self._last_user_activity_ms
        ) < (self.settings.proactive.idle_seconds * 1000.0):
            return
        backoff_until_ms = self._proactive_screen_failure_backoff_until_ms
        if backoff_until_ms is not None and now_ms < backoff_until_ms:
            return
        last_poll_ms = self._last_proactive_screen_poll_ms
        if (
            last_poll_ms is not None
            and (now_ms - last_poll_ms)
            < (self.settings.proactive.screen_poll_seconds * 1000.0)
        ):
            return
        screen_capture_engine = self._get_screen_capture_engine()
        if screen_capture_engine is None:
            return
        self._last_proactive_screen_poll_ms = now_ms
        try:
            screenshot = await screen_capture_engine.capture(
                context=self._build_poll_context(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._proactive_screen_failure_backoff_until_ms = (
                now_ms + (_PROACTIVE_SCREEN_FAILURE_BACKOFF_SECONDS * 1000.0)
            )
            log_event(
                self._logger,
                "proactive_screen_capture_failed",
                session_id=self._get_session_id(),
                screen_capture_provider=screen_capture_engine.name,
                screen_window_name=self.settings.screen_capture.window_name,
                error=str(exc),
                retry_after_seconds=_PROACTIVE_SCREEN_FAILURE_BACKOFF_SECONDS,
            )
            return
        self._proactive_screen_failure_backoff_until_ms = None
        screenshot_fingerprint = screen_capture_fingerprint(screenshot)
        if screenshot_fingerprint == self._latest_proactive_screen_capture_fingerprint:
            return
        self._latest_proactive_screen_capture = screenshot
        self._latest_proactive_screen_capture_fingerprint = screenshot_fingerprint
        self.record_observation()

    def screen_capture_supported(self) -> bool:
        screen_capture_engine = self._get_screen_capture_engine()
        return (
            self.settings.proactive.enabled
            and self.settings.proactive.screen_enabled
            and self.settings.screen_capture.enabled
            and screen_capture_engine is not None
            and self._get_language_model().supports_multimodal_input
        )

    def build_current_user_parts(self) -> tuple[ConversationRequestPart, ...]:
        if not self.screen_capture_supported():
            return tuple()
        return build_proactive_current_user_parts(
            self._latest_proactive_screen_capture,
            self.settings.screen_capture.window_name,
        )


def _drain_queue(queue: BoundedAsyncQueue[object]) -> int:
    drained_count = 0
    while True:
        item = queue.get_nowait()
        if item is None:
            return drained_count
        queue.task_done()
        drained_count += 1
