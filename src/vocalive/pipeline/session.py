from __future__ import annotations

from dataclasses import dataclass, field

from vocalive.models import ConversationMessage


@dataclass
class ConversationSession:
    session_id: str
    messages: list[ConversationMessage] = field(default_factory=list)

    def append_user_message(self, text: str) -> None:
        self.messages.append(ConversationMessage(role="user", content=text))

    def append_assistant_message(self, text: str) -> None:
        self.messages.append(ConversationMessage(role="assistant", content=text))

    def snapshot(self) -> tuple[ConversationMessage, ...]:
        return tuple(self.messages)

    def last_assistant_message(self) -> ConversationMessage | None:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message
        return None
