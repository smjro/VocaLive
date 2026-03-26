from __future__ import annotations

from datetime import datetime, timezone

from vocalive.models import ConversationMessage


_SUMMARY_ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
}
_OMITTED_LINE = "- Earlier summarized turns omitted for brevity."
_APPLICATION_PREFIX = "Application audio ("


def build_compacted_messages(
    messages: tuple[ConversationMessage, ...],
    *,
    recent_message_count: int,
    conversation_summary_max_chars: int,
    application_recent_message_count: int = 0,
    application_summary_max_chars: int = 0,
    application_summary_min_message_chars: int = 0,
    active_message_max_age_seconds: float = 0.0,
    now_utc: datetime | None = None,
) -> tuple[ConversationMessage, ...]:
    resolved_now_utc = _resolve_now_utc(
        now_utc,
        active_message_max_age_seconds=active_message_max_age_seconds,
    )
    recent_conversation_indexes = _select_recent_indexes(
        messages,
        roles={"user", "assistant"},
        recent_message_count=max(0, recent_message_count),
        active_message_max_age_seconds=active_message_max_age_seconds,
        now_utc=resolved_now_utc,
    )
    recent_application_indexes = _select_recent_indexes(
        messages,
        roles={"application"},
        recent_message_count=max(0, application_recent_message_count),
        active_message_max_age_seconds=active_message_max_age_seconds,
        now_utc=resolved_now_utc,
    )
    older_conversation_messages: list[ConversationMessage] = []
    older_application_messages: list[ConversationMessage] = []
    compacted_messages: list[ConversationMessage] = []

    for index, message in enumerate(messages):
        if message.role in {"user", "assistant"} and index not in recent_conversation_indexes:
            older_conversation_messages.append(message)
            continue
        if message.role == "application" and index not in recent_application_indexes:
            older_application_messages.append(message)
            continue
        compacted_messages.append(message)

    summary_messages: list[ConversationMessage] = []
    conversation_summary = _build_conversation_summary(
        older_conversation_messages,
        max_chars=conversation_summary_max_chars,
        active_message_max_age_seconds=active_message_max_age_seconds,
        now_utc=resolved_now_utc,
    )
    if conversation_summary is not None:
        summary_messages.append(
            ConversationMessage(role="system", content=conversation_summary)
        )
    application_summary = _build_application_summary(
        older_application_messages,
        max_chars=application_summary_max_chars,
        min_message_chars=application_summary_min_message_chars,
        active_message_max_age_seconds=active_message_max_age_seconds,
        now_utc=resolved_now_utc,
    )
    if application_summary is not None:
        summary_messages.append(
            ConversationMessage(role="system", content=application_summary)
        )
    return tuple(summary_messages + compacted_messages)


def _select_recent_indexes(
    messages: tuple[ConversationMessage, ...],
    *,
    roles: set[str],
    recent_message_count: int,
    active_message_max_age_seconds: float,
    now_utc: datetime | None,
) -> set[int]:
    if recent_message_count <= 0:
        return set()
    recent_indexes: list[int] = []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role not in roles:
            continue
        if _message_is_reference_only(
            message,
            active_message_max_age_seconds=active_message_max_age_seconds,
            now_utc=now_utc,
        ):
            continue
        recent_indexes.append(index)
        if len(recent_indexes) >= recent_message_count:
            break
    return set(recent_indexes)


def _build_conversation_summary(
    messages: list[ConversationMessage],
    *,
    max_chars: int,
    active_message_max_age_seconds: float,
    now_utc: datetime | None,
) -> str | None:
    if max_chars <= 0 or not messages:
        return None
    header = "Earlier conversation summary:"
    line_char_limit = max(32, min(120, max_chars // 4))
    lines: list[str] = []
    if _has_reference_only_messages(
        messages,
        active_message_max_age_seconds=active_message_max_age_seconds,
        now_utc=now_utc,
    ):
        lines.append(
            _build_reference_only_note(
                active_message_max_age_seconds=active_message_max_age_seconds
            )
        )
    for message in messages:
        lines.append(
            _format_summary_line(
                message,
                max_chars=line_char_limit,
                reference_only=_message_is_reference_only(
                    message,
                    active_message_max_age_seconds=active_message_max_age_seconds,
                    now_utc=now_utc,
                ),
            )
        )
    compacted_lines = _fit_summary_lines(lines, header=header, max_chars=max_chars)
    if not compacted_lines:
        return None
    return header + "\n" + "\n".join(compacted_lines)


def _build_application_summary(
    messages: list[ConversationMessage],
    *,
    max_chars: int,
    min_message_chars: int,
    active_message_max_age_seconds: float,
    now_utc: datetime | None,
) -> str | None:
    if max_chars <= 0 or not messages:
        return None
    header = "Earlier application audio summary:"
    line_char_limit = max(32, min(140, max_chars // 4))
    lines: list[str] = []
    if _has_reference_only_messages(
        messages,
        active_message_max_age_seconds=active_message_max_age_seconds,
        now_utc=now_utc,
    ):
        lines.append(
            _build_reference_only_note(
                active_message_max_age_seconds=active_message_max_age_seconds
            )
        )
    previous_key: tuple[str, str, bool] | None = None
    previous_count = 0

    def flush_previous() -> None:
        nonlocal previous_key, previous_count
        if previous_key is None:
            return
        source_label, content, reference_only = previous_key
        suffix = f" (x{previous_count})" if previous_count > 1 else ""
        prefix = f"- {source_label}"
        if reference_only:
            prefix += " [reference only]"
        prefix += ": "
        lines.append(
            prefix
            + _trim_line(
                content,
                max_chars=max(12, line_char_limit - len(prefix) - len(suffix)),
            )
            + suffix
        )
        previous_key = None
        previous_count = 0

    for message in messages:
        source_label, content = _parse_application_audio_message(message.content)
        normalized_content = " ".join(content.split())
        if len(normalized_content) < max(0, min_message_chars):
            continue
        reference_only = _message_is_reference_only(
            message,
            active_message_max_age_seconds=active_message_max_age_seconds,
            now_utc=now_utc,
        )
        key = (source_label, normalized_content, reference_only)
        if key == previous_key:
            previous_count += 1
            continue
        flush_previous()
        previous_key = key
        previous_count = 1
    flush_previous()

    compacted_lines = _fit_summary_lines(lines, header=header, max_chars=max_chars)
    if not compacted_lines:
        return None
    return header + "\n" + "\n".join(compacted_lines)


def _fit_summary_lines(
    lines: list[str],
    *,
    header: str,
    max_chars: int,
) -> list[str]:
    if not lines:
        return []
    if len(_render_summary(header, lines)) <= max_chars:
        return lines

    head_lines = lines[:2]
    if len(_render_summary(header, head_lines)) > max_chars:
        return [_trim_line(lines[-1], max_chars=max(16, max_chars - len(header) - 1))]

    tail_lines: list[str] = []
    remaining_lines = lines[2:]
    for line in reversed(remaining_lines):
        candidate_tail = [line, *tail_lines]
        candidate_lines = head_lines + [_OMITTED_LINE] + candidate_tail
        if len(_render_summary(header, candidate_lines)) > max_chars:
            continue
        tail_lines = candidate_tail
    if len(head_lines) + len(tail_lines) >= len(lines):
        return head_lines + tail_lines
    return head_lines + [_OMITTED_LINE] + tail_lines


def _format_summary_line(
    message: ConversationMessage,
    *,
    max_chars: int,
    reference_only: bool,
) -> str:
    role_label = _SUMMARY_ROLE_LABELS.get(message.role, message.role.title())
    content = " ".join(message.content.split())
    prefix = f"- {role_label}"
    if reference_only:
        prefix += " [reference only]"
    prefix += ": "
    return prefix + _trim_line(
        content,
        max_chars=max(12, max_chars - len(prefix)),
    )


def _trim_line(value: str, *, max_chars: int) -> str:
    if max_chars <= 1 or len(value) <= max_chars:
        return value[:max_chars]
    return value[: max_chars - 1].rstrip() + "..."


def _render_summary(header: str, lines: list[str]) -> str:
    return header + "\n" + "\n".join(lines)


def _parse_application_audio_message(value: str) -> tuple[str, str]:
    if value.startswith(_APPLICATION_PREFIX) and "):" in value:
        raw_label, content = value[len(_APPLICATION_PREFIX) :].split("):", maxsplit=1)
        return (raw_label.strip() or "application", content.strip())
    return ("application", value.strip())


def _has_reference_only_messages(
    messages: list[ConversationMessage],
    *,
    active_message_max_age_seconds: float,
    now_utc: datetime | None,
) -> bool:
    return any(
        _message_is_reference_only(
            message,
            active_message_max_age_seconds=active_message_max_age_seconds,
            now_utc=now_utc,
        )
        for message in messages
    )


def _build_reference_only_note(*, active_message_max_age_seconds: float) -> str:
    return (
        "- Messages marked [reference only] are older than about "
        f"{_format_age_label(active_message_max_age_seconds)}; use them as background only "
        "and verify time-sensitive details before relying on them."
    )


def _message_is_reference_only(
    message: ConversationMessage,
    *,
    active_message_max_age_seconds: float,
    now_utc: datetime | None,
) -> bool:
    if active_message_max_age_seconds <= 0:
        return False
    if now_utc is None:
        return False
    created_at = _parse_created_at(message.created_at)
    if created_at is None:
        return True
    return (now_utc - created_at).total_seconds() > active_message_max_age_seconds


def _parse_created_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_now_utc(
    now_utc: datetime | None,
    *,
    active_message_max_age_seconds: float,
) -> datetime | None:
    if active_message_max_age_seconds <= 0:
        return None
    if now_utc is not None:
        return now_utc.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _format_age_label(value: float) -> str:
    rounded_value = round(value)
    if rounded_value > 0 and rounded_value % 60 == 0:
        minutes = rounded_value // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit}"
    unit = "second" if rounded_value == 1 else "seconds"
    return f"{rounded_value:g} {unit}"
