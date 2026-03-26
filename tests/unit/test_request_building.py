from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.config.settings import AppSettings, ContextSettings, ConversationSettings
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

    def test_build_request_messages_moves_stale_turns_to_reference_only_summary(self) -> None:
        now_utc = datetime(2026, 3, 26, 15, 0, tzinfo=timezone.utc)
        settings = AppSettings(
            context=ContextSettings(
                recent_message_count=8,
                active_message_max_age_seconds=90.0,
                conversation_summary_max_chars=320,
            )
        )

        messages = build_request_messages(
            (
                ConversationMessage(
                    role="assistant",
                    content="The remaining time was about 2 minutes.",
                    created_at=(now_utc - timedelta(seconds=95)).isoformat(),
                ),
                ConversationMessage(
                    role="user",
                    content="Are we still talking about that?",
                    created_at=now_utc.isoformat(),
                ),
            ),
            settings=settings,
            conversation_language="en",
            now_utc=now_utc,
        )

        self.assertEqual(messages[2].role, "system")
        self.assertIn("Earlier conversation summary:", messages[2].content)
        self.assertIn("[reference only]", messages[2].content)
        self.assertIn("remaining time", messages[2].content)
        self.assertEqual(
            [(message.role, message.content) for message in messages[3:]],
            [("user", "Are we still talking about that?")],
        )

    def test_build_request_messages_keeps_fresh_compacted_turns_without_reference_marker(
        self,
    ) -> None:
        now_utc = datetime(2026, 3, 26, 15, 0, tzinfo=timezone.utc)
        settings = AppSettings(
            context=ContextSettings(
                recent_message_count=1,
                active_message_max_age_seconds=90.0,
                conversation_summary_max_chars=320,
            )
        )

        messages = build_request_messages(
            (
                ConversationMessage(
                    role="assistant",
                    content="First reply",
                    created_at=(now_utc - timedelta(seconds=40)).isoformat(),
                ),
                ConversationMessage(
                    role="user",
                    content="Second question",
                    created_at=(now_utc - timedelta(seconds=20)).isoformat(),
                ),
            ),
            settings=settings,
            conversation_language="en",
            now_utc=now_utc,
        )

        self.assertEqual(messages[2].role, "system")
        self.assertIn("Earlier conversation summary:", messages[2].content)
        self.assertIn("First reply", messages[2].content)
        self.assertNotIn("[reference only]", messages[2].content)
        self.assertEqual(
            [(message.role, message.content) for message in messages[3:]],
            [("user", "Second question")],
        )

    def test_build_request_messages_marks_stale_application_audio_as_reference_only(
        self,
    ) -> None:
        now_utc = datetime(2026, 3, 26, 15, 0, tzinfo=timezone.utc)
        settings = AppSettings(
            context=ContextSettings(
                recent_message_count=8,
                active_message_max_age_seconds=90.0,
                application_recent_message_count=1,
                application_summary_max_chars=320,
                application_summary_min_message_chars=1,
            )
        )

        messages = build_request_messages(
            (
                ConversationMessage(
                    role="application",
                    content="Application audio (Steam): boss incoming",
                    created_at=(now_utc - timedelta(seconds=120)).isoformat(),
                ),
                ConversationMessage(
                    role="user",
                    content="What should we do?",
                    created_at=now_utc.isoformat(),
                ),
            ),
            settings=settings,
            conversation_language="en",
            now_utc=now_utc,
        )

        self.assertEqual(messages[2].role, "system")
        self.assertIn("Earlier application audio summary:", messages[2].content)
        self.assertIn("[reference only]", messages[2].content)
        self.assertIn("boss incoming", messages[2].content)
        self.assertEqual(
            [(message.role, message.content) for message in messages[3:]],
            [("user", "What should we do?")],
        )

    def test_classify_screen_capture_request_prefers_explicit_trigger(self) -> None:
        capture_mode = classify_screen_capture_request(
            "please show this screen",
            trigger_phrases=("show this",),
            passive_enabled=True,
            passive_trigger_phrases=("this screen",),
        )

        self.assertEqual(capture_mode, "explicit")


if __name__ == "__main__":
    unittest.main()
