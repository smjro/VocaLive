from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol

from vocalive.models import TurnContext
from vocalive.util.time import monotonic_ms


@dataclass(frozen=True)
class DurationMetric:
    stage: str
    duration_ms: float
    context: TurnContext


class MetricsRecorder(Protocol):
    def record_duration(self, stage: str, duration_ms: float, context: TurnContext) -> None:
        """Record a latency measurement."""


class InMemoryMetricsRecorder:
    def __init__(self) -> None:
        self.events: list[DurationMetric] = []

    def record_duration(self, stage: str, duration_ms: float, context: TurnContext) -> None:
        self.events.append(DurationMetric(stage=stage, duration_ms=duration_ms, context=context))


@contextmanager
def timed_stage(
    recorder: MetricsRecorder,
    stage: str,
    context: TurnContext,
) -> Iterator[None]:
    started_ms = monotonic_ms()
    try:
        yield
    finally:
        recorder.record_duration(
            stage=stage,
            duration_ms=monotonic_ms() - started_ms,
            context=context,
        )
