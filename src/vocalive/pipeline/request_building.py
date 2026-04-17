from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from vocalive.config.settings import AppSettings
from vocalive.models import (
    AudioSegment,
    ConversationInlineDataPart,
    ConversationMessage,
    ConversationRequestPart,
    ConversationTextPart,
)
from vocalive.pipeline.context import build_compacted_messages


def build_request_messages(
    messages: tuple[ConversationMessage, ...],
    *,
    settings: AppSettings,
    conversation_language: str | None,
    transient_system_messages: tuple[str, ...] = (),
    now_utc: datetime | None = None,
) -> tuple[ConversationMessage, ...]:
    compacted_messages = list(
        build_compacted_messages(
            messages,
            recent_message_count=settings.context.recent_message_count,
            conversation_summary_max_chars=(
                settings.context.conversation_summary_max_chars
            ),
            application_recent_message_count=(
                settings.context.application_recent_message_count
            ),
            application_summary_max_chars=(
                settings.context.application_summary_max_chars
            ),
            application_summary_min_message_chars=(
                settings.context.application_summary_min_message_chars
            ),
            active_message_max_age_seconds=(
                settings.context.active_message_max_age_seconds
            ),
            now_utc=now_utc,
        )
    )
    compacted_messages = _focus_latest_reply_target(compacted_messages)
    identity_instruction = build_participant_identity_instruction(settings)
    if identity_instruction is not None:
        compacted_messages.insert(
            0,
            ConversationMessage(role="system", content=identity_instruction),
        )
    language_instruction = build_conversation_language_instruction(
        conversation_language
    )
    insertion_index = 1 if identity_instruction is not None else 0
    if language_instruction is not None:
        compacted_messages.insert(
            insertion_index,
            ConversationMessage(role="system", content=language_instruction),
        )
        insertion_index += 1
    for transient_message in transient_system_messages:
        normalized_message = transient_message.strip()
        if not normalized_message:
            continue
        compacted_messages.insert(
            insertion_index,
            ConversationMessage(role="system", content=normalized_message),
        )
        insertion_index += 1
    return tuple(compacted_messages)


def build_proactive_request_messages(
    messages: tuple[ConversationMessage, ...],
    *,
    settings: AppSettings,
    now_utc: datetime | None = None,
) -> tuple[ConversationMessage, ...]:
    proactive_messages = list(
        build_request_messages(
            messages,
            settings=settings,
            conversation_language=settings.conversation.language,
            now_utc=now_utc,
        )
    )
    proactive_messages.insert(
        0,
        ConversationMessage(
            role="system",
            content=build_proactive_system_instruction(),
        ),
    )
    return tuple(proactive_messages)


def build_participant_identity_instruction(settings: AppSettings) -> str:
    user_name = (settings.conversation.user_name or "").strip()
    if user_name:
        return (
            "Your name is コハク. "
            f"The current user's name is {user_name}. "
            "If the user asks what their name is or who you are speaking with, answer with that name. "
            "Do not begin replies by addressing the user by name unless the user asks for that "
            "or the name is genuinely needed for clarity."
        )
    return (
        "Your name is コハク. "
        "You are speaking directly with the current user. "
        "Do not begin replies by addressing the user by name unless the user asks for that "
        "or the name is genuinely needed for clarity."
    )


def build_conversation_language_instruction(language: str | None) -> str | None:
    normalized_language = _normalize_language(language)
    if normalized_language is None:
        return None
    language_name = _language_name(normalized_language)
    return (
        f"The conversation language is {language_name}. "
        f"Reply in {language_name} unless the user explicitly asks to switch languages."
    )


def build_proactive_system_instruction() -> str:
    return (
        "The user has not made a direct request right now. "
        "If you speak, keep it to one or two short sentences. "
        "Do not start a full back-and-forth or narrate continuously. "
        "Only react to immediate live context that is already present in the conversation "
        "history or attached turn data."
    )


def build_reply_target_focus_instruction() -> str:
    return (
        "This turn has one reply_target. "
        "The latest utterance below is the only reply_target for this turn. "
        "Any recent_context provided in system messages is interpretation-only background. "
        "Reply only to reply_target. Do not directly answer, continue, or summarize "
        "recent_context unless that is necessary to resolve reply_target."
    )


def build_recent_audible_assistant_instruction() -> str:
    return (
        "If this request includes a transient assistant message immediately before the latest "
        "user turn, that message was already heard before an interruption. Treat it as already "
        "said context. Do not restart, restate, or continue it unless the user explicitly asks "
        "you to repeat or continue it."
    )


def inject_recent_audible_assistant_message(
    messages: tuple[ConversationMessage, ...],
    text: str,
) -> tuple[ConversationMessage, ...]:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return messages
    transient_message = ConversationMessage(role="assistant", content=normalized_text)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return messages[:index] + (transient_message,) + messages[index:]
    return messages + (transient_message,)


def _focus_latest_reply_target(
    messages: list[ConversationMessage],
) -> list[ConversationMessage]:
    last_assistant_index = -1
    for index, message in enumerate(messages):
        if message.role == "assistant":
            last_assistant_index = index

    utterance_indexes = [
        index
        for index, message in enumerate(messages)
        if index > last_assistant_index and message.role in {"user", "application"}
    ]
    if len(utterance_indexes) <= 1:
        return messages

    target_index = utterance_indexes[-1]
    recent_context_indexes = utterance_indexes[-3:-1]
    omitted_utterance_count = max(
        0,
        len(utterance_indexes) - 1 - len(recent_context_indexes),
    )
    skipped_indexes = set(utterance_indexes[:-1])
    focused_messages: list[ConversationMessage] = []

    for index, message in enumerate(messages):
        if index in skipped_indexes:
            continue
        if index == target_index:
            focused_messages.append(
                ConversationMessage(
                    role="system",
                    content=build_reply_target_focus_instruction(),
                )
            )
            focused_messages.append(
                ConversationMessage(
                    role="system",
                    content=_build_recent_context_message(
                        recent_context_messages=tuple(
                            messages[recent_index]
                            for recent_index in recent_context_indexes
                        ),
                        omitted_utterance_count=omitted_utterance_count,
                    ),
                )
            )
            focused_messages.append(_build_reply_target_message(message))
            continue
        focused_messages.append(message)
    return focused_messages


def _build_recent_context_message(
    *,
    recent_context_messages: tuple[ConversationMessage, ...],
    omitted_utterance_count: int,
) -> str:
    lines = ["recent_context:"]
    if omitted_utterance_count > 0:
        lines.append(
            f"- {omitted_utterance_count} older utterance(s) from this same unresolved turn were omitted."
        )
    for message in recent_context_messages:
        lines.append(_format_recent_context_line(message))
    return "\n".join(lines)


def _format_recent_context_line(message: ConversationMessage) -> str:
    normalized_content = " ".join(message.content.split())
    if message.role == "application":
        return f"- {normalized_content}"
    if message.role == "user":
        return f"- User: {normalized_content}"
    return f"- {message.role.title()}: {normalized_content}"


def _build_reply_target_message(message: ConversationMessage) -> ConversationMessage:
    normalized_content = " ".join(message.content.split())
    return ConversationMessage(
        role=message.role,
        content=f"reply_target: {normalized_content}",
        created_at=message.created_at,
    )


def build_proactive_current_user_parts(
    screenshot: ConversationInlineDataPart | None,
    window_name: str | None,
) -> tuple[ConversationRequestPart, ...]:
    if screenshot is None:
        return tuple()
    if window_name:
        context_text = (
            f"Configured target window: {window_name}. "
            "The attached image is the latest changed screenshot observed while the user "
            "was quiet."
        )
    else:
        context_text = (
            "The attached image is the latest changed screenshot observed while the user "
            "was quiet."
        )
    return (
        ConversationTextPart(text=context_text),
        screenshot,
    )


def build_session_message_text(segment: AudioSegment, transcription_text: str) -> str:
    normalized_text = transcription_text.strip()
    if segment.source != "application_audio":
        return normalized_text
    source_label = (segment.source_label or "unknown application").strip() or "unknown application"
    return f"Application audio ({source_label}): {normalized_text}"


def build_screen_capture_parts(
    screenshot: ConversationInlineDataPart,
    window_name: str | None,
    capture_mode: Literal["explicit", "passive"],
) -> tuple[ConversationRequestPart, ...]:
    if capture_mode == "passive" and window_name:
        context_text = (
            f"Configured target window: {window_name}. "
            "The attached image is a screenshot of that window for this turn because the user "
            "appears to be referring to the current screen."
        )
    elif capture_mode == "passive":
        context_text = (
            "The attached image is a screenshot of the requested window for this turn because "
            "the user appears to be referring to the current screen."
        )
    elif window_name:
        context_text = (
            f"Configured target window: {window_name}. "
            "The attached image is a screenshot of that window for this turn."
        )
    else:
        context_text = "The attached image is a screenshot of the requested window for this turn."
    return (
        ConversationTextPart(text=context_text),
        screenshot,
    )


def classify_screen_capture_request(
    user_text: str,
    trigger_phrases: tuple[str, ...],
    passive_enabled: bool,
    passive_trigger_phrases: tuple[str, ...],
) -> Literal["explicit", "passive"] | None:
    if _matches_screen_trigger(user_text, trigger_phrases):
        return "explicit"
    if passive_enabled and _matches_screen_trigger(user_text, passive_trigger_phrases):
        return "passive"
    return None


def passive_screen_capture_is_on_cooldown(
    now_ms: float,
    last_observation_ms: float | None,
    cooldown_seconds: float,
) -> bool:
    if last_observation_ms is None or cooldown_seconds <= 0:
        return False
    return (now_ms - last_observation_ms) < (cooldown_seconds * 1000.0)


def screen_capture_fingerprint(screenshot: ConversationInlineDataPart) -> str:
    digest = hashlib.sha256()
    digest.update(screenshot.mime_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(screenshot.data)
    return digest.hexdigest()


def _matches_screen_trigger(user_text: str, trigger_phrases: tuple[str, ...]) -> bool:
    normalized_user_text = _normalize_screen_trigger_text(user_text)
    if not normalized_user_text:
        return False
    for trigger_phrase in trigger_phrases:
        normalized_trigger = _normalize_screen_trigger_text(trigger_phrase)
        if normalized_trigger and normalized_trigger in normalized_user_text:
            return True
    return False


def _normalize_screen_trigger_text(value: str) -> str:
    return "".join(value.lower().split())


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized_language = language.strip()
    if not normalized_language:
        return None
    return normalized_language


def _language_name(language: str) -> str:
    normalized_language = language.lower()
    language_names = {
        "ja": "Japanese",
        "ja-jp": "Japanese",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
    }
    return language_names.get(normalized_language, language)
