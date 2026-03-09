from __future__ import annotations

import time
from datetime import datetime, timezone


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
