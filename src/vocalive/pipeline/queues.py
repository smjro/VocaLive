from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

from vocalive.config.settings import QueueOverflowStrategy


T = TypeVar("T")


class BoundedAsyncQueue(Generic[T]):
    def __init__(self, maxsize: int, overflow_strategy: QueueOverflowStrategy) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._overflow_strategy = overflow_strategy

    async def put(self, item: T) -> bool:
        if not self._queue.full():
            await self._queue.put(item)
            return True

        if self._overflow_strategy is QueueOverflowStrategy.REJECT_NEW:
            return False

        dropped_item = self._queue.get_nowait()
        self._queue.task_done()
        del dropped_item
        await self._queue.put(item)
        return True

    async def get(self) -> T:
        return await self._queue.get()

    def get_nowait(self) -> T | None:
        if self._queue.empty():
            return None
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
