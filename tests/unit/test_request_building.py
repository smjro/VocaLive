from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import AppSettings, ConversationSettings
from vocalive.models import AudioSegment, ConversationInlineDataPart, ConversationMessage
from vocalive.pipeline.request_building import (
    build_proactive_current_user_parts,
    build_proactive_request_messages,
    build_request_messages,
    build_session_message_text,
    classify_screen_capture_request,
)


class RequestBuildingTests(unittest.TestCase):
    def test_build_request_messages_includes_identity_and_language_instructions(self) -> None:
        settings = AppSettings(
            conversation=ConversationSettings(user_name="Taketo", language="ja")
        )

        messages = build_request_messages(
            (ConversationMessage(role="user", content="hello"),),
            settings=settings,
            conversation_language="en",
        )

        self.assertEqual(messages[0].role, "system")
        self.assertIn("Your name is", messages[0].content)
        self.assertIn("Taketo", messages[0].content)
        self.assertEqual(messages[1].role, "system")
        self.assertIn("Reply in English", messages[1].content)
        self.assertEqual(messages[-1].content, "hello")

    def test_build_proactive_request_messages_prepends_proactive_instruction(self) -> None:
        messages = build_proactive_request_messages(
            (ConversationMessage(role="user", content="hello"),),
            settings=AppSettings(),
        )

        self.assertEqual(messages[0].role, "system")
        self.assertIn("The user has not made a direct request", messages[0].content)
        self.assertEqual(messages[-1].content, "hello")

    def test_build_proactive_current_user_parts_returns_empty_without_screenshot(self) -> None:
        self.assertEqual(
            build_proactive_current_user_parts(None, "Steam"),
            (),
        )

    def test_build_proactive_current_user_parts_mentions_window_name(self) -> None:
        screenshot = ConversationInlineDataPart(mime_type="image/png", data=b"png")

        current_user_parts = build_proactive_current_user_parts(
            screenshot,
            "Steam",
        )

        self.assertEqual(len(current_user_parts), 2)
        self.assertIn("Configured target window: Steam.", current_user_parts[0].text)
        self.assertEqual(current_user_parts[1], screenshot)

    def test_build_session_message_text_formats_application_audio_labels(self) -> None:
        message = build_session_message_text(
            AudioSegment.from_text(
                "boss fight",
                source="application_audio",
                source_label="Steam",
            ),
            "boss fight",
        )

        self.assertEqual(message, "Application audio (Steam): boss fight")

    def test_classify_screen_capture_request_prefers_explicit_trigger(self) -> None:
        capture_mode = classify_screen_capture_request(
            "この画面みて",
            trigger_phrases=("みて",),
            passive_enabled=True,
            passive_trigger_phrases=("この画面",),
        )

        self.assertEqual(capture_mode, "explicit")


if __name__ == "__main__":
    unittest.main()
