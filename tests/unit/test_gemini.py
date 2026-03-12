from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.llm.gemini import _build_generate_content_payload, _extract_response_text
from vocalive.models import (
    ConversationInlineDataPart,
    ConversationMessage,
    ConversationRequest,
    ConversationTextPart,
    TurnContext,
)


class GeminiPayloadTests(unittest.TestCase):
    def test_payload_coalesces_consecutive_messages_and_system_instruction(self) -> None:
        request = ConversationRequest(
            context=TurnContext(session_id="session", turn_id=1),
            messages=(
                ConversationMessage(role="system", content="s1"),
                ConversationMessage(role="user", content="u1"),
                ConversationMessage(role="user", content="u2"),
                ConversationMessage(role="assistant", content="a1"),
            ),
        )

        payload = _build_generate_content_payload(
            request=request,
            model_name="gemini-2.5-flash",
            system_instruction="external",
            temperature=0.4,
        )

        self.assertEqual(
            payload["systemInstruction"],
            {"parts": [{"text": "external"}, {"text": "s1"}]},
        )
        self.assertEqual(
            payload["contents"],
            [
                {"role": "user", "parts": [{"text": "u1"}, {"text": "u2"}]},
                {"role": "model", "parts": [{"text": "a1"}]},
            ],
        )
        self.assertEqual(
            payload["generationConfig"],
            {
                "temperature": 0.4,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        )

    def test_payload_omits_thinking_config_for_non_25_models(self) -> None:
        request = ConversationRequest(
            context=TurnContext(session_id="session", turn_id=1),
            messages=(ConversationMessage(role="user", content="u1"),),
        )

        payload = _build_generate_content_payload(
            request=request,
            model_name="gemini-2.0-flash",
            thinking_budget=0,
        )

        self.assertNotIn("generationConfig", payload)

    def test_extract_response_text_joins_text_parts(self) -> None:
        response_body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "hello"},
                            {"text": " world"},
                        ]
                    }
                }
            ]
        }

        self.assertEqual(_extract_response_text(response_body), "hello world")

    def test_payload_appends_current_user_text_and_image_parts(self) -> None:
        request = ConversationRequest(
            context=TurnContext(session_id="session", turn_id=1),
            messages=(ConversationMessage(role="user", content="画面見て"),),
            current_user_parts=(
                ConversationTextPart(text="The attached image is the current full-screen screenshot."),
                ConversationInlineDataPart(mime_type="image/png", data=b"\x89PNG"),
            ),
        )

        payload = _build_generate_content_payload(
            request=request,
            model_name="gemini-2.5-flash",
        )

        self.assertEqual(
            payload["contents"],
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": "画面見て"},
                        {"text": "The attached image is the current full-screen screenshot."},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": "iVBORw==",
                            }
                        },
                    ],
                }
            ],
        )

    def test_payload_keeps_application_messages_separate_from_user_messages(self) -> None:
        request = ConversationRequest(
            context=TurnContext(session_id="session", turn_id=1),
            messages=(
                ConversationMessage(
                    role="application",
                    content="Application audio (Steam): enemy spotted",
                ),
                ConversationMessage(role="user", content="何て言ってた?"),
            ),
        )

        payload = _build_generate_content_payload(
            request=request,
            model_name="gemini-2.5-flash",
        )

        self.assertEqual(
            payload["contents"],
            [
                {
                    "role": "user",
                    "parts": [{"text": "Application audio (Steam): enemy spotted"}],
                },
                {
                    "role": "user",
                    "parts": [{"text": "何て言ってた?"}],
                },
            ],
        )
