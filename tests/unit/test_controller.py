from __future__ import annotations

import asyncio
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.controller_store import ControllerConfigStore
from vocalive.config.settings import controller_default_values
from vocalive.ui.controller import ControllerRuntimeManager, ControllerServer


class _IdleAudioInput:
    def __init__(self) -> None:
        self.closed = threading.Event()
        self.speech_start_handler = None

    async def start(self) -> str:
        return "fake microphone"

    def set_speech_start_handler(self, handler) -> None:
        self.speech_start_handler = handler

    async def read(self):
        await asyncio.to_thread(self.closed.wait)
        return None

    async def close(self) -> None:
        self.closed.set()


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.submitted = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def submit_utterance(self, segment) -> bool:
        self.submitted.append(segment)
        return True

    async def handle_user_speech_start(self) -> None:
        return None


class _FakeOverlay:
    def __init__(self) -> None:
        self.url = "http://127.0.0.1:9999/"
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class ControllerConfigStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory() as tempdir:
            store = ControllerConfigStore(Path(tempdir) / "controller-config.json")

            saved = store.save_values(
                {
                    "VOCALIVE_INPUT_PROVIDER": "microphone",
                    "VOCALIVE_GEMINI_API_KEY": "secret",
                }
            )
            loaded = store.load_values()

        self.assertEqual(saved["VOCALIVE_INPUT_PROVIDER"], "microphone")
        self.assertEqual(loaded["VOCALIVE_INPUT_PROVIDER"], "microphone")
        self.assertEqual(loaded["VOCALIVE_GEMINI_API_KEY"], "secret")
        self.assertEqual(loaded["VOCALIVE_MODEL_PROVIDER"], "mock")


class ControllerRuntimeManagerTests(unittest.TestCase):
    def test_start_and_stop_runtime_transitions(self) -> None:
        manager = ControllerRuntimeManager()
        values = controller_default_values()
        values["VOCALIVE_INPUT_PROVIDER"] = "microphone"
        audio_input = _IdleAudioInput()
        orchestrator = _FakeOrchestrator()
        overlay = _FakeOverlay()

        with patch("vocalive.ui.controller.build_audio_input", return_value=audio_input), patch(
            "vocalive.ui.controller.build_orchestrator",
            return_value=orchestrator,
        ), patch("vocalive.ui.controller.build_overlay", return_value=overlay), patch(
            "vocalive.ui.controller.configure_logging"
        ):
            started = manager.start_runtime(values)
            stopped = manager.stop_runtime()

        self.assertEqual(started["status"], "running")
        self.assertEqual(started["input_label"], "fake microphone")
        self.assertEqual(started["overlay_url"], overlay.url)
        self.assertEqual(stopped["status"], "stopped")
        self.assertTrue(orchestrator.started)
        self.assertTrue(orchestrator.stopped)
        self.assertTrue(overlay.started)
        self.assertTrue(overlay.stopped)
        self.assertIsNotNone(audio_input.speech_start_handler)
        manager.close()

    def test_runtime_manager_rejects_stdin_mode(self) -> None:
        manager = ControllerRuntimeManager()
        with self.assertRaisesRegex(
            ValueError,
            "python -m vocalive run",
        ):
            manager.start_runtime(controller_default_values())
        manager.close()


class ControllerServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = TemporaryDirectory()
        self.store = ControllerConfigStore(Path(self._tempdir.name) / "controller-config.json")
        self.server = ControllerServer(store=self.store, auto_open=False)
        self.addCleanup(self._tempdir.cleanup)

    def test_validate_values_and_store_round_trip_secret_values(self) -> None:
        values = controller_default_values()
        values["VOCALIVE_INPUT_PROVIDER"] = "microphone"
        values["VOCALIVE_GEMINI_API_KEY"] = "top-secret"

        normalized = self.server._validate_values(values)
        saved = self.store.save_values(normalized)
        loaded = self.store.load_values()

        self.assertEqual(
            saved["VOCALIVE_GEMINI_API_KEY"],
            "top-secret",
        )
        self.assertEqual(
            loaded["VOCALIVE_GEMINI_API_KEY"],
            "top-secret",
        )
        self.assertEqual(
            loaded["VOCALIVE_INPUT_PROVIDER"],
            "microphone",
        )

    def test_load_values_with_warning_falls_back_to_defaults_for_invalid_json(self) -> None:
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        self.store.path.write_text("{invalid", encoding="utf-8")

        values, warning = self.server._load_values_with_warning()

        self.assertIsNotNone(warning)
        self.assertEqual(values["VOCALIVE_INPUT_PROVIDER"], "stdin")

    def test_extract_values_requires_values_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "`values` object"):
            self.server._extract_values({"values": ["not", "a", "dict"]})
