from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vocalive.config.settings import (
    controller_default_values,
    controller_secret_env_names,
    normalize_controller_values,
    sanitize_persisted_controller_values,
)


CONTROLLER_CONFIG_VERSION = 1
_CONTROLLER_SECRET_ENV_NAMES = frozenset(controller_secret_env_names())


@dataclass(frozen=True)
class ControllerConfig:
    version: int
    values: dict[str, str | None]


def default_controller_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / ".vocalive" / "controller-config.json"


class ControllerConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_controller_config_path()

    def load(self) -> ControllerConfig:
        defaults = controller_default_values()
        if not self.path.is_file():
            return ControllerConfig(version=CONTROLLER_CONFIG_VERSION, values=defaults)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Controller config must be a JSON object: {self.path}")
        version = payload.get("version", CONTROLLER_CONFIG_VERSION)
        if version != CONTROLLER_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported controller config version {version!r}: {self.path}"
            )
        raw_values = payload.get("values", {})
        if not isinstance(raw_values, dict):
            raise ValueError(f"Controller config values must be a JSON object: {self.path}")
        normalized = normalize_controller_values(
            {str(key): _coerce_raw_value(value) for key, value in raw_values.items()}
        )
        sanitized = sanitize_persisted_controller_values(normalized)
        if sanitized != normalized:
            self.save_values(sanitized)
        return ControllerConfig(version=CONTROLLER_CONFIG_VERSION, values=sanitized)

    def load_values(self) -> dict[str, str | None]:
        return dict(self.load().values)

    def save_values(self, values: dict[str, str | None]) -> dict[str, str | None]:
        normalized = sanitize_persisted_controller_values(values)
        payload = {
            "version": CONTROLLER_CONFIG_VERSION,
            "values": {
                env_name: value
                for env_name, value in normalized.items()
                if env_name not in _CONTROLLER_SECRET_ENV_NAMES
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self.path)
        return normalized


def _coerce_raw_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
