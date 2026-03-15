from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Mapping

from vocalive.config.controller_store import ControllerConfigStore
from vocalive.config.settings import AppSettings, normalize_controller_values
from vocalive.runtime import (
    build_audio_input,
    build_managed_aivis_engine,
    build_orchestrator,
    build_overlay,
    run_headless,
    run_microphone_loop as _run_microphone_loop,
)
from vocalive.ui.controller import run_controller
from vocalive.util.logging import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vocalive")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run",),
        help="Start the saved configuration directly without the GUI controller.",
    )
    return parser


def load_headless_settings(
    store: ControllerConfigStore | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    config_store = store or ControllerConfigStore()
    saved_values = config_store.load_values()
    current_environ = environ or os.environ
    env_overrides = normalize_controller_values(current_environ, include_defaults=False)
    if "GEMINI_API_KEY" in current_environ:
        env_overrides["GEMINI_API_KEY"] = current_environ.get("GEMINI_API_KEY")
    if "OPENAI_API_KEY" in current_environ:
        env_overrides["OPENAI_API_KEY"] = current_environ.get("OPENAI_API_KEY")
    merged_values = dict(saved_values)
    merged_values.update(env_overrides)
    return AppSettings.from_mapping(merged_values)


async def _run_headless() -> int:
    settings = load_headless_settings()
    configure_logging(settings.log_level)
    return await run_headless(settings)


async def _run_controller() -> int:
    configure_logging("INFO")
    return await run_controller()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run_headless())
        return asyncio.run(_run_controller())
    except KeyboardInterrupt:
        return 130
