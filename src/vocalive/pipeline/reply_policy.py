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
    "なんで",
    "なぜ",
    "なに",
    "何",
    "どこ",
    "いつ",
    "help",
    "please",
    "can you",
    "could you",
    "what",
    "why",
    "how",
)
_GREETING_MARKERS = {
    "こんにちは",
    "こんばんは",
    "おはよう",
    "hello",
    "hi",
    "hey",
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
) -> ReplyDecision:
    if not settings.policy_enabled:
        return ReplyDecision(should_reply=True, reason="policy_disabled")

    normalized_text = "".join(text.lower().split())
    if not normalized_text:
        return ReplyDecision(should_reply=False, reason="empty_text")

    if _looks_like_explicit_request(normalized_text):
        return ReplyDecision(should_reply=True, reason="explicit_request")

    if last_assistant_response_ms is None and normalized_text in _GREETING_MARKERS:
        return ReplyDecision(should_reply=True, reason="greeting")

    if _looks_like_short_reaction(
        normalized_text,
        short_utterance_max_chars=settings.short_utterance_max_chars,
    ):
        return ReplyDecision(should_reply=False, reason="short_reaction")

    if (
        last_assistant_response_ms is not None
        and (now_ms - last_assistant_response_ms) < settings.min_gap_ms
        and len(normalized_text) <= max(24, settings.short_utterance_max_chars * 2)
    ):
        return ReplyDecision(should_reply=False, reason="cooldown")

    return ReplyDecision(should_reply=True, reason="default")


def _looks_like_explicit_request(normalized_text: str) -> bool:
    return any(marker in normalized_text for marker in _EXPLICIT_REQUEST_MARKERS)


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
