from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import OverlaySettings
from vocalive.pipeline.events import ConversationEvent
from vocalive.ui.overlay import OverlayServer, character_image_path, render_overlay_page


class OverlayRenderingTests(unittest.TestCase):
    def test_render_overlay_page_includes_title_and_character_name(self) -> None:
        html = render_overlay_page(
            OverlaySettings(
                enabled=True,
                auto_open=False,
                title="Overlay Test",
                character_name="Stripe",
            )
        )

        self.assertIn("<title>Overlay Test</title>", html)
        self.assertIn("Stripe", html)
        self.assertIn("show-comment", html)
        self.assertIn("assistant-text", html)
        self.assertIn("height: clamp(140px, 58vmin, 840px);", html)
        self.assertIn("aspect-ratio: 2 / 3;", html)

    def test_render_overlay_page_uses_image_when_character_png_exists(self) -> None:
        with patch("vocalive.ui.overlay._CHARACTER_IMAGE_PATH", Path(__file__)):
            html = render_overlay_page(
                OverlaySettings(
                    enabled=True,
                    auto_open=False,
                )
            )

        self.assertIn('src="/assets/character.png"', html)
        self.assertNotIn("<svg", html)

    def test_character_image_path_points_to_assets_directory(self) -> None:
        self.assertEqual(character_image_path().name, "character.png")
        self.assertEqual(character_image_path().parent.name, "assets")

    def test_emit_updates_snapshot_for_caption_flow(self) -> None:
        overlay = OverlayServer(
            OverlaySettings(
                enabled=True,
                auto_open=False,
            )
        )

        overlay.emit(
            ConversationEvent(
                type="transcription_ready",
                session_id="session-1",
                turn_id=1,
                text="hello",
            )
        )
        overlay.emit(
            ConversationEvent(
                type="assistant_chunk_started",
                session_id="session-1",
                turn_id=1,
                text="partial answer",
                duration_ms=600.0,
            )
        )
        overlay.emit(
            ConversationEvent(
                type="assistant_message_committed",
                session_id="session-1",
                turn_id=1,
                text="partial answer",
            )
        )

        snapshot = overlay._snapshot_payload()
        self.assertEqual(snapshot["session_id"], "session-1")
        self.assertEqual(snapshot["turn_id"], 1)
        self.assertEqual(snapshot["user_text"], "hello")
        self.assertEqual(snapshot["assistant_text"], "partial answer")
        self.assertEqual(snapshot["status"], "idle")
