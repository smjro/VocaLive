from __future__ import annotations

from vocalive.tts.style_api import HttpStyleTextToSpeechEngine


class VoicevoxTextToSpeechEngine(HttpStyleTextToSpeechEngine):
    name = "voicevox"
    provider_label = "VOICEVOX"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:50021",
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
