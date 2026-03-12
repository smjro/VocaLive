from __future__ import annotations

import asyncio

from vocalive.llm.base import LanguageModel
from vocalive.models import AssistantResponse, ConversationRequest
from vocalive.pipeline.interruption import CancellationToken


class EchoLanguageModel(LanguageModel):
    name = "mock-echo"

    def __init__(self, delay_seconds: float = 0.01, prefix: str = "Assistant") -> None:
        self.delay_seconds = delay_seconds
        self.prefix = prefix

    async def generate(
        self,
        request: ConversationRequest,
        cancellation: CancellationToken | None = None,
    ) -> AssistantResponse:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        latest_user_message = next(
            message.content
            for message in reversed(request.messages)
            if message.role in {"user", "application"}
        )
        return AssistantResponse(
            text=f"{self.prefix}: {latest_user_message}",
            provider=self.name,
        )
