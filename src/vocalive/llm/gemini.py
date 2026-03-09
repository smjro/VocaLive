from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request

from vocalive.llm.base import LanguageModel
from vocalive.models import AssistantResponse, ConversationMessage, ConversationRequest
from vocalive.pipeline.interruption import CancellationToken


class GeminiLanguageModel(LanguageModel):
    name = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: str | None,
        model_name: str = "gemini-2.5-flash",
        timeout_seconds: float = 30.0,
        temperature: float | None = None,
        thinking_budget: int | None = 0,
        system_instruction: str | None = None,
        api_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.system_instruction = system_instruction
        self.api_base_url = api_base_url.rstrip("/")

    async def generate(
        self,
        request: ConversationRequest,
        cancellation: CancellationToken | None = None,
    ) -> AssistantResponse:
        if not self.api_key:
            raise RuntimeError(
                "Gemini adapter requires VOCALIVE_GEMINI_API_KEY or GEMINI_API_KEY."
            )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        payload = _build_generate_content_payload(
            request=request,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            temperature=self.temperature,
            thinking_budget=self.thinking_budget,
        )
        response_body = await asyncio.to_thread(self._post_json, payload)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        text = _extract_response_text(response_body)
        return AssistantResponse(text=text, provider=self.model_name)

    def _post_json(self, payload: dict[str, object]) -> dict[str, object]:
        url = (
            f"{self.api_base_url}/models/"
            f"{urllib.parse.quote(self.model_name, safe='')}:generateContent"
        )
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key or "",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini API request failed with status {exc.code}: {raw_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc


def _build_generate_content_payload(
    request: ConversationRequest,
    model_name: str,
    system_instruction: str | None = None,
    temperature: float | None = None,
    thinking_budget: int | None = 0,
) -> dict[str, object]:
    instruction_parts = []
    if system_instruction:
        instruction_parts.append({"text": system_instruction})
    for message in request.messages:
        if message.role == "system":
            instruction_parts.append({"text": message.content})

    contents = _coalesce_messages(request.messages)
    if not contents:
        raise ValueError("Gemini request must include at least one non-system message")

    payload: dict[str, object] = {"contents": contents}
    if instruction_parts:
        payload["systemInstruction"] = {"parts": instruction_parts}
    generation_config = _build_generation_config(
        model_name=model_name,
        temperature=temperature,
        thinking_budget=thinking_budget,
    )
    if generation_config:
        payload["generationConfig"] = generation_config
    return payload


def _build_generation_config(
    model_name: str,
    temperature: float | None,
    thinking_budget: int | None,
) -> dict[str, object]:
    generation_config: dict[str, object] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    thinking_config = _build_thinking_config(model_name=model_name, thinking_budget=thinking_budget)
    if thinking_config is not None:
        generation_config["thinkingConfig"] = thinking_config
    return generation_config


def _build_thinking_config(
    model_name: str,
    thinking_budget: int | None,
) -> dict[str, object] | None:
    if thinking_budget is None:
        return None
    normalized_model_name = model_name.strip().lower()
    if normalized_model_name.startswith("gemini-2.5"):
        return {"thinkingBudget": thinking_budget}
    return None


def _coalesce_messages(messages: tuple[ConversationMessage, ...]) -> list[dict[str, object]]:
    contents: list[dict[str, object]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            parts = contents[-1]["parts"]
            assert isinstance(parts, list)
            parts.append({"text": message.content})
            continue
        contents.append({"role": role, "parts": [{"text": message.content}]})
    return contents


def _extract_response_text(response_body: dict[str, object]) -> str:
    candidates = response_body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response_body}")
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise RuntimeError(f"Gemini returned an invalid candidate: {first_candidate!r}")
    content = first_candidate.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Gemini returned no content: {response_body}")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError(f"Gemini returned no text parts: {response_body}")
    text_parts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    text = "".join(text_parts).strip()
    if not text:
        raise RuntimeError(f"Gemini response did not include text: {response_body}")
    return text
