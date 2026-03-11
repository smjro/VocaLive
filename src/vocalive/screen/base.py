from __future__ import annotations

from abc import ABC, abstractmethod

from vocalive.models import ConversationInlineDataPart, TurnContext
from vocalive.pipeline.interruption import CancellationToken


class ScreenCaptureEngine(ABC):
    name: str

    @abstractmethod
    async def capture(
        self,
        context: TurnContext,
        cancellation: CancellationToken | None = None,
    ) -> ConversationInlineDataPart:
        """Capture one screen image for the current turn."""
