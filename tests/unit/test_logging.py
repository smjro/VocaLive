from __future__ import annotations

import io
import json
import logging
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.util.logging import log_event


class StructuredLoggingTests(unittest.TestCase):
    def test_log_event_preserves_unicode_text_in_log_output(self) -> None:
        stream = io.StringIO()
        logger = logging.getLogger("tests.logging")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False

        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        try:
            log_event(logger, "response_ready", text="はい、どのようなことでしょうか。")
        finally:
            logger.removeHandler(handler)

        output = stream.getvalue().strip()
        self.assertIn("はい、どのようなことでしょうか。", output)
        self.assertNotIn("\\u306f\\u3044", output)
        self.assertEqual(json.loads(output)["text"], "はい、どのようなことでしょうか。")
