from __future__ import annotations

import asyncio


class TurnCancelledError(Exception):
    """Raised when a turn is interrupted by a newer user utterance or shutdown."""


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = asyncio.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def wait(self) -> None:
        await self._cancelled.wait()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise TurnCancelledError("turn cancelled")


class InterruptionController:
    def __init__(self) -> None:
        self._current_token: CancellationToken | None = None

    def begin_turn(self) -> CancellationToken:
        self.interrupt_active_turn()
        token = CancellationToken()
        self._current_token = token
        return token

    def interrupt_active_turn(self) -> None:
        if self._current_token is not None:
            self._current_token.cancel()

    def clear_if_current(self, token: CancellationToken) -> None:
        if self._current_token is token:
            self._current_token = None

    @property
    def has_active_turn(self) -> bool:
        return self._current_token is not None
