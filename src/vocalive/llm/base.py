from __future__ import annotations

from abc import ABC, abstractmethod

from vocalive.models import AssistantResponse, ConversationRequest
from vocalive.pipeline.interruption import CancellationToken


class LanguageModel(ABC):
    name: str
    supports_multimodal_input: bool = False

    @abstractmethod
    async def generate(
        self,
        request: ConversationRequest,
        cancellation: CancellationToken | None = None,
    ) -> AssistantResponse:
        """Generate an assistant response."""
