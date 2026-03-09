from __future__ import annotations

import json
import logging
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **{key: _normalize(value) for key, value in fields.items()}}
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _normalize(value: Any) -> Any:
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value
