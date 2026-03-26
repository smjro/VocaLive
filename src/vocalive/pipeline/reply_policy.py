from __future__ import annotations

from dataclasses import dataclass

from vocalive.config.settings import ReplySettings


_EXPLICIT_REQUEST_MARKERS = (
    "?",
    "？",
    "教えて",
    "おしえて",
    "どうすれば",
    "どうやって",
    "見て",
    "みて",
)
_EXPLICIT_REQUEST_ENDINGS = (
    "どうする",
    "どうしよう",
    "どうかな",
    "どう思う",
    "なんで",
    "なぜ",
    "なに",
    "何",
    "どこ",
    "いつ",
)
_SHORT_EXPLICIT_REQUEST_MARKERS = (
    "なんで",
    "なぜ",
    "なに",
    "何",
    "どこ",
    "いつ",
    "どう",
)
_SHORT_EXPLICIT_REQUEST_MAX_CHARS = 8
_GREETING_MARKERS = {
    "こんにちは",
    "こんばんは",
    "おはよう",
    "hello",
    "hi",
    "hey",
}
_ASSISTANT_INVOCATION_MARKERS = {
    "assistant",
    "アシスタント",
}
_SHORT_REACTION_MARKERS = {
    "あ",
    "あっ",
    "あー",
    "え",
    "えっ",
    "うわ",
    "うわっ",
    "うわー",
    "おっ",
    "おー",
    "よし",
    "やばい",
    "やば",
    "まじ",
    "なるほど",
    "くそ",
    "くっ",
    "痛い",
    "惜しい",
}


@dataclass(frozen=True)
class ReplyDecision:
    should_reply: bool
    reason: str


def decide_reply(
    text: str,
    *,
    settings: ReplySettings,
    last_assistant_response_ms: float | None,
    now_ms: float,
    assistant_names: tuple[str, ...] = (),
) -> ReplyDecision:
    if not settings.policy_enabled:
        return ReplyDecision(should_reply=True, reason="policy_disabled")

    normalized_text = "".join(text.lower().split())
    if not normalized_text:
        return ReplyDecision(should_reply=False, reason="empty_text")

    if looks_like_explicit_assistant_address(text, assistant_names=assistant_names):
        return ReplyDecision(should_reply=True, reason="assistant_addressed")

    if _looks_like_explicit_request(normalized_text):
        return ReplyDecision(should_reply=True, reason="explicit_request")

    if last_assistant_response_ms is None and normalized_text in _GREETING_MARKERS:
        return ReplyDecision(should_reply=True, reason="greeting")

    if _looks_like_short_reaction(
        normalized_text,
        short_utterance_max_chars=settings.short_utterance_max_chars,
    ):
        return ReplyDecision(should_reply=False, reason="short_reaction")

    if settings.require_explicit_trigger:
        return ReplyDecision(should_reply=False, reason="explicit_trigger_required")

    if (
        last_assistant_response_ms is not None
        and (now_ms - last_assistant_response_ms) < settings.min_gap_ms
        and len(normalized_text) <= max(24, settings.short_utterance_max_chars * 2)
    ):
        return ReplyDecision(should_reply=False, reason="cooldown")

    return ReplyDecision(should_reply=True, reason="default")


def looks_like_explicit_assistant_address(
    text: str,
    *,
    assistant_names: tuple[str, ...] = (),
) -> bool:
    normalized_text = "".join(text.lower().split())
    if not normalized_text:
        return False
    for assistant_name in assistant_names:
        normalized_name = "".join(assistant_name.lower().split())
        if normalized_name and normalized_name in normalized_text:
            return True
    return any(marker in normalized_text for marker in _ASSISTANT_INVOCATION_MARKERS)


def _looks_like_explicit_request(normalized_text: str) -> bool:
    if any(marker in normalized_text for marker in _EXPLICIT_REQUEST_MARKERS):
        return True
    if any(normalized_text.endswith(marker) for marker in _EXPLICIT_REQUEST_ENDINGS):
        return True
    return (
        len(normalized_text) <= _SHORT_EXPLICIT_REQUEST_MAX_CHARS
        and any(marker in normalized_text for marker in _SHORT_EXPLICIT_REQUEST_MARKERS)
    )


def _looks_like_short_reaction(
    normalized_text: str,
    *,
    short_utterance_max_chars: int,
) -> bool:
    if len(normalized_text) > max(0, short_utterance_max_chars):
        return False
    return (
        normalized_text in _SHORT_REACTION_MARKERS
        or normalized_text.endswith("!")
        or normalized_text.endswith("！")
    )
