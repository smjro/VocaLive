from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.audio.output import _default_playback_command


class DefaultPlaybackCommandTests(unittest.TestCase):
    def test_defaults_to_afplay_on_macos(self) -> None:
        with patch("vocalive.audio.output.sys.platform", "darwin"), patch(
            "vocalive.audio.output.shutil.which",
            side_effect=lambda command: "/usr/bin/afplay" if command == "afplay" else None,
        ):
            command = _default_playback_command()

        self.assertEqual(command, ["/usr/bin/afplay", "{path}"])

    def test_defaults_to_powershell_soundplayer_on_windows(self) -> None:
        with patch("vocalive.audio.output.sys.platform", "win32"), patch(
            "vocalive.audio.output.shutil.which",
            side_effect=lambda command: (
                "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
                if command == "powershell.exe"
                else None
            ),
        ):
            command = _default_playback_command()

        self.assertEqual(
            command,
            [
                "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(New-Object System.Media.SoundPlayer '{path}').PlaySync()",
            ],
        )

    def test_raises_when_no_platform_default_is_available(self) -> None:
        with patch("vocalive.audio.output.sys.platform", "linux"), patch(
            "vocalive.audio.output.shutil.which",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "speaker output requires a playback command"):
                _default_playback_command()
