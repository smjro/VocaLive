from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import QueueOverflowStrategy
from vocalive.pipeline.queues import BoundedAsyncQueue


class BoundedAsyncQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_drop_oldest_keeps_latest_items(self) -> None:
        queue = BoundedAsyncQueue[int](maxsize=2, overflow_strategy=QueueOverflowStrategy.DROP_OLDEST)

        self.assertTrue(await queue.put(1))
        self.assertTrue(await queue.put(2))
        self.assertTrue(await queue.put(3))

        self.assertEqual(await queue.get(), 2)
        queue.task_done()
        self.assertEqual(await queue.get(), 3)
        queue.task_done()

    async def test_reject_new_preserves_existing_items(self) -> None:
        queue = BoundedAsyncQueue[int](maxsize=1, overflow_strategy=QueueOverflowStrategy.REJECT_NEW)

        self.assertTrue(await queue.put(1))
        self.assertFalse(await queue.put(2))
        self.assertEqual(await queue.get(), 1)
        queue.task_done()
