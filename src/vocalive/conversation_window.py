from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable

from vocalive.config.settings import ConversationWindowSettings
from vocalive.models import AudioSegment, AudioSource
from vocalive.util.logging import get_logger, log_event
from vocalive.util.time import monotonic_ms


logger = get_logger(__name__)


class ConversationWindowGate:
    def __init__(
        self,
        settings: ConversationWindowSettings,
        *,
        session_id: str | None = None,
        now_ms: Callable[[], float] | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.session_id = session_id
        self._now_ms = now_ms or monotonic_ms
        self._logger = logger_ or logger
        initial_now_ms = self._now_ms()
        self._open_started_ms: float | None = None
        self._closed_started_ms: float | None = None
        self._awaiting_user_reopen = False
        if self.enabled:
            if self.settings.start_open or self.settings.closed_duration_seconds <= 0.0:
                self._open_started_ms = initial_now_ms
            else:
                self._closed_started_ms = initial_now_ms
        self._drop_next_segment_sources: set[AudioSource] = set()
        self._last_reported_is_open: bool | None = None
        self._history_reset_pending = False
        self._resume_summary_capture_pending = False

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def summary(self) -> str | None:
        if not self.enabled:
            return None
        start_label = "starts open" if self.settings.start_open else "starts closed"
        target_label = (
            "microphone + app audio"
            if self.settings.apply_to_application_audio
            else "microphone only"
        )
        return (
            "conversation window: "
            f"{start_label}, "
            f"{_format_seconds(self.settings.open_duration_seconds)} open, "
            f"{_format_seconds(self.settings.closed_duration_seconds)} closed, "
            "reopens on user speech, "
            f"{target_label}"
        )

    def wrap_speech_start_handler(
        self,
        handler: Callable[[AudioSource], Awaitable[None] | None] | None,
    ) -> Callable[[AudioSource], Awaitable[None] | None] | None:
        if handler is None:
            return None

        async def wrapped(source: AudioSource) -> None:
            if not self._applies_to_source(source):
                await _await_if_needed(handler(source))
                return
            now_ms = self._now_ms()
            is_open = self._window_is_open(now_ms)
            if not is_open and self._should_reopen_on_speech_start(source):
                self._start_open_window(now_ms)
                is_open = self._window_is_open(now_ms)
            if not is_open:
                self._drop_next_segment_sources.add(source)
                self._log_skip(
                    event_name="conversation_window_speech_ignored",
                    source=source,
                    reason="window_closed",
                )
                return
            await _await_if_needed(handler(source))

        return wrapped

    def should_forward_segment(self, segment: AudioSegment) -> bool:
        if not self._applies_to_source(segment.source):
            return True
        now_ms = self._now_ms()
        if segment.source in self._drop_next_segment_sources:
            self._drop_next_segment_sources.discard(segment.source)
            self._window_is_open(now_ms)
            self._log_skip(
                event_name="conversation_window_segment_skipped",
                source=segment.source,
                reason="started_while_closed",
            )
            return False
        if self._window_is_open(now_ms):
            return True
        self._log_skip(
            event_name="conversation_window_segment_skipped",
            source=segment.source,
            reason="window_closed",
        )
        return False

    def consume_history_reset_request(self) -> bool:
        if not self._history_reset_pending:
            return False
        self._history_reset_pending = False
        return True

    def consume_resume_summary_capture_request(self) -> bool:
        if not self._resume_summary_capture_pending:
            return False
        self._resume_summary_capture_pending = False
        return True

    def poll_state(self) -> bool:
        return self._window_is_open(self._now_ms())

    def _applies_to_source(self, source: AudioSource) -> bool:
        if not self.enabled:
            return False
        if source == "user":
            return True
        return self.settings.apply_to_application_audio and source == "application_audio"

    def _window_is_open(self, now_ms: float) -> bool:
        self._refresh_state(now_ms)
        is_open = not self.enabled or self._open_started_ms is not None
        if self._last_reported_is_open is None:
            self._last_reported_is_open = is_open
            return is_open
        if is_open != self._last_reported_is_open:
            self._last_reported_is_open = is_open
            if is_open:
                self._history_reset_pending = True
            else:
                self._resume_summary_capture_pending = True
            log_event(
                self._logger,
                "conversation_window_state_changed",
                session_id=self.session_id,
                state="open" if is_open else "closed",
                open_duration_seconds=self.settings.open_duration_seconds,
                closed_duration_seconds=self.settings.closed_duration_seconds,
            )
        return is_open

    def _refresh_state(self, now_ms: float) -> None:
        if not self.enabled:
            return
        open_duration_ms = self.settings.open_duration_seconds * 1000.0
        closed_duration_ms = self.settings.closed_duration_seconds * 1000.0
        if closed_duration_ms <= 0.0:
            if self._open_started_ms is None:
                self._open_started_ms = now_ms
            self._closed_started_ms = None
            self._awaiting_user_reopen = False
            return
        if self._open_started_ms is not None:
            elapsed_open_ms = max(0.0, now_ms - self._open_started_ms)
            if elapsed_open_ms >= open_duration_ms:
                self._closed_started_ms = self._open_started_ms + open_duration_ms
                self._open_started_ms = None
                self._awaiting_user_reopen = False
        if self._open_started_ms is not None:
            return
        if self._closed_started_ms is None:
            self._closed_started_ms = now_ms
        if self._awaiting_user_reopen:
            return
        elapsed_closed_ms = max(0.0, now_ms - self._closed_started_ms)
        if elapsed_closed_ms >= closed_duration_ms:
            self._awaiting_user_reopen = True

    def _should_reopen_on_speech_start(self, source: AudioSource) -> bool:
        return source == "user" and self._awaiting_user_reopen

    def _start_open_window(self, now_ms: float) -> None:
        self._open_started_ms = now_ms
        self._closed_started_ms = None
        self._awaiting_user_reopen = False

    def _log_skip(
        self,
        *,
        event_name: str,
        source: AudioSource,
        reason: str,
    ) -> None:
        log_event(
            self._logger,
            event_name,
            session_id=self.session_id,
            source=source,
            reason=reason,
            open_duration_seconds=self.settings.open_duration_seconds,
            closed_duration_seconds=self.settings.closed_duration_seconds,
        )


async def _await_if_needed(result: Awaitable[None] | None) -> None:
    if inspect.isawaitable(result):
        await result


def _format_seconds(value: float) -> str:
    normalized = f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{normalized}s"
