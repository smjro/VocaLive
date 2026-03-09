from __future__ import annotations

from abc import ABC, abstractmethod


class TurnDetector(ABC):
    @abstractmethod
    def is_end_of_turn(self, buffered_ms: float, silence_ms: float) -> bool:
        """Return True when the current speech buffer should be emitted as a turn."""


class FixedSilenceTurnDetector(TurnDetector):
    def __init__(self, silence_threshold_ms: float = 500.0) -> None:
        self.silence_threshold_ms = silence_threshold_ms

    def is_end_of_turn(self, buffered_ms: float, silence_ms: float) -> bool:
        return buffered_ms > 0 and silence_ms >= self.silence_threshold_ms
