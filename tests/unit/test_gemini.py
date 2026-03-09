from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vocalive.llm.gemini import _build_generate_content_payload, _extract_response_text
from vocalive.models import ConversationMessage, ConversationRequest, TurnContext


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
