from __future__ import annotations

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
) -> tuple[ConversationMessage, ...]:
    recent_conversation_indexes = _select_recent_indexes(
        messages,
        roles={"user", "assistant"},
        recent_message_count=max(0, recent_message_count),
    )
    recent_application_indexes = _select_recent_indexes(
        messages,
        roles={"application"},
        recent_message_count=max(0, application_recent_message_count),
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
    )
    if conversation_summary is not None:
        summary_messages.append(
            ConversationMessage(role="system", content=conversation_summary)
        )
    application_summary = _build_application_summary(
        older_application_messages,
        max_chars=application_summary_max_chars,
        min_message_chars=application_summary_min_message_chars,
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
) -> set[int]:
    if recent_message_count <= 0:
        return set()
    recent_indexes: list[int] = []
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role not in roles:
            continue
        recent_indexes.append(index)
        if len(recent_indexes) >= recent_message_count:
            break
    return set(recent_indexes)


def _build_conversation_summary(
    messages: list[ConversationMessage],
    *,
    max_chars: int,
) -> str | None:
    if max_chars <= 0 or not messages:
        return None
    header = "Earlier conversation summary:"
    line_char_limit = max(32, min(120, max_chars // 4))
    lines = [
        _format_summary_line(message, max_chars=line_char_limit)
        for message in messages
    ]
    compacted_lines = _fit_summary_lines(lines, header=header, max_chars=max_chars)
    if not compacted_lines:
        return None
    return header + "\n" + "\n".join(compacted_lines)


def _build_application_summary(
    messages: list[ConversationMessage],
    *,
    max_chars: int,
    min_message_chars: int,
) -> str | None:
    if max_chars <= 0 or not messages:
        return None
    header = "Earlier application audio summary:"
    line_char_limit = max(32, min(140, max_chars // 4))
    lines: list[str] = []
    previous_key: tuple[str, str] | None = None
    previous_count = 0

    def flush_previous() -> None:
        nonlocal previous_key, previous_count
        if previous_key is None:
            return
        source_label, content = previous_key
        suffix = f" (x{previous_count})" if previous_count > 1 else ""
        lines.append(
            f"- {source_label}: {_trim_line(content, max_chars=line_char_limit)}{suffix}"
        )
        previous_key = None
        previous_count = 0

    for message in messages:
        source_label, content = _parse_application_audio_message(message.content)
        normalized_content = " ".join(content.split())
        if len(normalized_content) < max(0, min_message_chars):
            continue
        key = (source_label, normalized_content)
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


def _format_summary_line(message: ConversationMessage, *, max_chars: int) -> str:
    role_label = _SUMMARY_ROLE_LABELS.get(message.role, message.role.title())
    content = " ".join(message.content.split())
    return f"- {role_label}: {_trim_line(content, max_chars=max_chars)}"


def _trim_line(value: str, *, max_chars: int) -> str:
    if max_chars <= 1 or len(value) <= max_chars:
        return value[:max_chars]
    return value[: max_chars - 1].rstrip() + "…"


def _render_summary(header: str, lines: list[str]) -> str:
    return header + "\n" + "\n".join(lines)


def _parse_application_audio_message(value: str) -> tuple[str, str]:
    if value.startswith(_APPLICATION_PREFIX) and "):" in value:
        raw_label, content = value[len(_APPLICATION_PREFIX) :].split("):", maxsplit=1)
        return (raw_label.strip() or "application", content.strip())
    return ("application", value.strip())
