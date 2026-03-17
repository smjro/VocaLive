from __future__ import annotations

from collections.abc import Iterable

from vocalive.llm.base import LanguageModel
from vocalive.models import ConversationMessage, ConversationRequest, TurnContext


_RESUME_NOTE_HEADER = "Conversation window resume note:"
_SUMMARY_SYSTEM_PROMPT = """
You create concise hidden resume notes for an assistant when one conversation window ends and the next one may reopen after a gap.

Keep:
- durable user goals, constraints, urgency, preferences, named tasks, named games/projects, and still-relevant unfinished threads
- guidance about how the assistant should answer next time if that preference is clear

Drop:
- current screen contents, exact current location, exact remaining timers, app/audio state that may have changed, and vague deictic references like this/that/here/now
- facts that were only true in the immediate moment unless they are explicitly stable

If the user is rushing because of a timed goal, preserve the goal and urgency, but not an exact remaining countdown unless it is a fixed external deadline that should still matter later.

Return plain text only using exactly this format:
Carry forward:
- ...
Assistant approach:
- ...
Freshness cautions:
- ...

Use 1 to 3 bullets per section. Write "- None." when a section is empty.
Do not add commentary before or after the note.
""".strip()


class ConversationResumeSummarizer:
    def __init__(self, language_model: LanguageModel) -> None:
        self.language_model = language_model

    async def summarize(
        self,
        *,
        session_id: str,
        messages: tuple[ConversationMessage, ...],
        closed_duration_seconds: float,
    ) -> str | None:
        transcript = _build_summary_transcript(messages)
        if transcript is None:
            return None
        request = ConversationRequest(
            context=TurnContext(session_id=session_id, turn_id=0),
            messages=(
                ConversationMessage(role="system", content=_SUMMARY_SYSTEM_PROMPT),
                ConversationMessage(
                    role="user",
                    content=_build_summary_request_content(
                        transcript=transcript,
                        closed_duration_seconds=closed_duration_seconds,
                    ),
                ),
            ),
        )
        response = await self.language_model.generate(request)
        normalized_text = _normalize_summary_text(response.text)
        if not normalized_text:
            return None
        return normalized_text


def build_resume_system_message(
    summary_text: str,
    *,
    closed_duration_seconds: float,
) -> str:
    gap_label = _format_gap_label(closed_duration_seconds)
    return (
        f"{_RESUME_NOTE_HEADER}\n"
        f"This note was generated from the previous conversation window after about {gap_label} "
        "of closed-window time. Use it to avoid repetitive questions and preserve durable "
        "goals, constraints, and preferences. Do not treat it as proof that the current "
        "screen, application audio, exact remaining timer, or deictic references such as "
        "\"this\", \"that\", \"here\", and \"now\" are still valid.\n"
        f"{summary_text}"
    )


def is_resume_system_message(message: ConversationMessage) -> bool:
    return message.role == "system" and message.content.startswith(_RESUME_NOTE_HEADER)


def _build_summary_request_content(
    *,
    transcript: str,
    closed_duration_seconds: float,
) -> str:
    gap_label = _format_gap_label(closed_duration_seconds)
    return (
        "Prepare a carry-forward note for the next conversation window.\n"
        f"The next window may resume after about {gap_label}.\n\n"
        f"{transcript}"
    )


def _build_summary_transcript(messages: tuple[ConversationMessage, ...]) -> str | None:
    dialogue_lines = list(_render_dialogue_lines(messages))
    if not dialogue_lines:
        return None
    previous_resume_note = _latest_resume_note(messages)
    sections: list[str] = []
    if previous_resume_note is not None:
        sections.append(
            "Previous carry-forward note from the earlier window "
            "(keep only if it still fits the latest dialogue):\n"
            f"{previous_resume_note}"
        )
    sections.append("Latest window dialogue:\n" + "\n".join(dialogue_lines))
    return "\n\n".join(sections)


def _latest_resume_note(messages: tuple[ConversationMessage, ...]) -> str | None:
    for message in reversed(messages):
        if is_resume_system_message(message):
            return message.content
    return None


def _render_dialogue_lines(
    messages: tuple[ConversationMessage, ...],
) -> Iterable[str]:
    for index, message in enumerate(messages, start=1):
        if message.role == "system":
            continue
        normalized_content = " ".join(message.content.split())
        if not normalized_content:
            continue
        yield f"{index}. [{_label_for_role(message.role)}] {normalized_content}"


def _label_for_role(role: str) -> str:
    if role == "user":
        return "User"
    if role == "assistant":
        return "Assistant"
    if role == "application":
        return "Application"
    return role.title()


def _normalize_summary_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        stripped = text[3:]
        newline_index = stripped.find("\n")
        if newline_index != -1:
            stripped = stripped[newline_index + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        text = stripped.strip()
    return text


def _format_gap_label(value: float) -> str:
    if value <= 0:
        return "0 seconds"
    rounded_value = round(value)
    if rounded_value and rounded_value % 60 == 0:
        minutes = rounded_value // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit}"
    unit = "second" if rounded_value == 1 else "seconds"
    return f"{rounded_value:g} {unit}"
