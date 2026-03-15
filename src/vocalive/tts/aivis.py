from __future__ import annotations

from typing import Any

from vocalive.tts.style_api import (
    HttpStyleTextToSpeechEngine,
    _read_wave_duration_ms,
    _read_wave_metadata,
    _select_style as _shared_select_style,
)


class AivisSpeechTextToSpeechEngine(HttpStyleTextToSpeechEngine):
    name = "aivisspeech"
    provider_label = "AivisSpeech"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:10101",
        speaker_id: int | None = None,
        speaker_name: str | None = None,
        style_name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            style_name=style_name,
            timeout_seconds=timeout_seconds,
        )


def _select_style(
    speakers: list[Any],
    speaker_name: str | None,
    style_name: str | None,
) -> int:
    return _shared_select_style(
        speakers=speakers,
        speaker_name=speaker_name,
        style_name=style_name,
        provider_label="AivisSpeech",
    )
