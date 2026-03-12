# VocaLive

VocaLive is an adapter-based local voice conversation runtime focused on low latency, explicit interruption handling, and replaceable STT / LLM / TTS providers.

The repository currently ships with:

- a stdin shell for local development
- optional live microphone capture via `sounddevice`
- optional macOS application-audio capture for one named running app
- mock providers for end-to-end local testing
- Moonshine STT, Gemini LLM, and AivisSpeech TTS adapters
- in-memory output for tests and speaker playback through an external command
- queue-based orchestration with bounded backlog, stale-turn cancellation, and latency logging

## Current status

- Implemented: bounded queue orchestration with explicit overflow handling
- Implemented: stale-turn interruption on new utterances
- Implemented: microphone speech-start barge-in before turn-end emission
- Implemented: stdin shell and microphone capture with preroll / hold / silence-based utterance detection
- Implemented: automatic preference for headset-like external microphones when the system default input is built-in
- Implemented: optional macOS application-audio capture that feeds STT and stores transcripts as application context instead of user speech, with `context_only` as the default mode
- Implemented: adaptive VAD and low-frequency-preserving STT-side speech enhancement for application audio
- Implemented: mock STT, echo LLM, mock TTS, and in-memory playback for local development
- Implemented: Moonshine STT via `moonshine-voice`
- Implemented: Gemini `generateContent` integration over HTTPS
- Implemented: trigger-based named-window screenshot capture for Gemini input on macOS
- Implemented: AivisSpeech synthesis over the local HTTP API
- Implemented: sentence-by-sentence TTS playback with one-sentence-ahead prefetch
- Implemented: bounded LLM request compaction with an earlier-conversation summary plus a recent raw-message window
- Implemented: microphone-only reply debounce that merges closely spaced live utterances before one LLM turn
- Implemented: conservative microphone reply suppression for short reactions and cooldown-period chatter while preserving explicit questions/requests
- Implemented: separate application-audio summary compaction so older app context does not stay fully verbatim in every LLM request
- Implemented: structured JSON logging and in-memory stage latency metrics
- Implemented: unit tests for settings, device resolution, utterance accumulation, provider payload/selection logic, queue behavior, and orchestration
- Not implemented yet: streaming partial STT / LLM / TTS
- Not implemented yet: echo cancellation for full-duplex microphone + speaker use
- Not implemented yet: separate worker processes, persistent metrics export, or generic retry policies

## Quick start

Run directly from the source tree:

```bash
PYTHONPATH=src python3 -m vocalive
```

Or install the package in editable mode:

```bash
python3 -m pip install -e .
python3 -m vocalive
```

Install the optional voice dependencies when you want live microphone capture or Moonshine STT:

```bash
python3 -m pip install -e '.[voice]'
```

Run the full local voice path against Moonshine, Gemini, AivisSpeech, and speaker playback:

```bash
export VOCALIVE_INPUT_PROVIDER=microphone
export VOCALIVE_APP_AUDIO_ENABLED=true
export VOCALIVE_APP_AUDIO_TARGET="Google Chrome"
export VOCALIVE_STT_PROVIDER=moonshine
export VOCALIVE_MODEL_PROVIDER=gemini
export VOCALIVE_TTS_PROVIDER=aivis
export VOCALIVE_OUTPUT_PROVIDER=speaker
export VOCALIVE_CONVERSATION_LANGUAGE=ja
export VOCALIVE_SCREEN_CAPTURE_ENABLED=true
export VOCALIVE_SCREEN_WINDOW_NAME="Steam"
export VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX=1280
export VOCALIVE_AIVIS_BASE_URL=http://127.0.0.1:10101
export VOCALIVE_GEMINI_API_KEY=...
PYTHONPATH=src python3 -m vocalive
```

Current runtime constraints:

- live microphone or application-audio input currently requires `VOCALIVE_STT_PROVIDER=moonshine`
- `VOCALIVE_APP_AUDIO_ENABLED=true` currently requires macOS, `VOCALIVE_APP_AUDIO_TARGET`, and Screen Recording permission
- `VOCALIVE_OUTPUT_PROVIDER=speaker` currently requires `VOCALIVE_TTS_PROVIDER=aivis`
- `VOCALIVE_SCREEN_CAPTURE_ENABLED=true` currently requires `VOCALIVE_MODEL_PROVIDER=gemini`, macOS, and Screen Recording permission
- speaker playback uses `afplay {path}` by default on macOS; on other platforms set `VOCALIVE_SPEAKER_COMMAND`
- Gemini accepts either `VOCALIVE_GEMINI_API_KEY` or `GEMINI_API_KEY`
- Gemini defaults to a surreal, deadpan conversation persona inspired by the vibe of Kamiusagi Rope; set `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` to override it, or set it to an empty string to disable it

Microphone tuning notes:

- `VOCALIVE_MIC_DEVICE=external` tells VocaLive to search for a currently connected headset-like external microphone
- when `VOCALIVE_MIC_DEVICE` is unset and `VOCALIVE_MIC_PREFER_EXTERNAL=true`, VocaLive will switch away from a built-in default mic if it finds a better external input
- if phrase starts are clipped, increase `VOCALIVE_MIC_PRE_SPEECH_MS`
- if mid-sentence pauses cause early cuts, increase `VOCALIVE_MIC_SPEECH_HOLD_MS` and `VOCALIVE_MIC_SILENCE_MS`
- after one live utterance is emitted, VocaLive waits briefly before queueing the LLM turn so closely spaced microphone utterances can merge; tune this with `VOCALIVE_REPLY_DEBOUNCE_MS`
- microphone reply suppression is enabled by default for live user speech so short reactions such as `やばい` are more likely to stay silent unless they are clear questions/requests; tune this with the `VOCALIVE_REPLY_*` settings
- in microphone mode, local speech onset interrupts stale assistant playback before the next utterance is fully emitted

Application-audio notes:

- application audio can be enabled alongside either `stdin` or `microphone` input
- `VOCALIVE_APP_AUDIO_MODE=context_only` is the default; app audio is transcribed and appended to session history as labeled application context, but it does not immediately trigger LLM/TTS or interrupt active playback
- older application-audio entries are compacted into a separate bounded summary while the newest configured app-context messages stay verbatim in the LLM request window
- set `VOCALIVE_APP_AUDIO_MODE=respond` when you want application-audio utterances to behave like live turns and trigger immediate assistant replies
- the configured target is matched against the running macOS application name first and bundle identifier second
- application audio uses adaptive energy-based VAD by default so steady BGM is more likely to stay in the background; set `VOCALIVE_APP_AUDIO_ADAPTIVE_VAD=false` to fall back to fixed thresholding
- application-audio utterances go through STT like live user audio, but session history stores them as labeled application context such as `Application audio (Steam): ...`
- default application-audio tuning keeps more preroll and trailing context so phrase starts and endings are less likely to clip; raise `VOCALIVE_APP_AUDIO_PRE_SPEECH_MS`, `VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS`, or `VOCALIVE_APP_AUDIO_SILENCE_MS` further if one app still cuts too aggressively
- Moonshine applies low-frequency-preserving enhancement with a gentle presence boost, soft gate, short edge padding, and normalization to application audio before STT by default; set `VOCALIVE_APP_AUDIO_STT_ENHANCEMENT=false` to disable it
- in `respond` mode, application-audio speech start also interrupts stale assistant playback before the buffered utterance is fully emitted
- app lookup and capture rely on a small ScreenCaptureKit helper that is built on first use
- if macOS Screen Recording permission is missing, app lookup or capture will time out/fail and the error is logged

Screen-capture notes:

- screen capture is request-scoped, not persistent session history
- older user/assistant turns are compacted into one system summary when the request window grows past the configured raw-message count
- capture is triggered only when the normalized user utterance contains one of the configured trigger phrases
- the current implementation resolves the first on-screen window whose title or owner name matches `VOCALIVE_SCREEN_WINDOW_NAME`
- captured screenshots are downscaled so the longest edge is at most `1280px` before they are attached to Gemini; set `VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX` empty to disable that
- the resolved window id is cached and reused until capture fails, then looked up again
- `VOCALIVE_SCREEN_WINDOW_NAME` is required when screen capture is enabled
- if macOS screen recording permission is missing, the turn falls back to text-only input and logs `screen_capture_failed`

The stdin shell can also exercise the Gemini and Aivis wiring without a microphone. Typed input is stored as `AudioSegment.transcript_hint`, so a `moonshine` configuration can still run cleanly before you switch to live microphone mode. The first real Moonshine transcription downloads and caches the selected model files.

## Development commands

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Verify imports compile:

```bash
python3 -m compileall src tests
```

## Configuration

All runtime configuration is environment-driven.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOCALIVE_SESSION_ID` | random UUID | Stable identifier for one conversation session |
| `VOCALIVE_LOG_LEVEL` | `INFO` | Python logging level |
| `VOCALIVE_INPUT_PROVIDER` | `stdin` | `stdin` or `microphone` |
| `VOCALIVE_MIC_SAMPLE_RATE` | `16000` | Microphone capture sample rate |
| `VOCALIVE_MIC_CHANNELS` | `1` | Captured microphone channel count |
| `VOCALIVE_MIC_BLOCK_MS` | `40` | Duration of each captured PCM block |
| `VOCALIVE_MIC_DEVICE` | unset | Optional input device id, device name, `default`, or `external` |
| `VOCALIVE_MIC_PREFER_EXTERNAL` | `true` | Prefer a connected headset-like external mic when the default input looks built-in |
| `VOCALIVE_MIC_SPEECH_THRESHOLD` | `0.02` | RMS threshold for treating a block as speech |
| `VOCALIVE_MIC_PRE_SPEECH_MS` | `200` | Audio kept before speech starts so utterance onsets are not clipped |
| `VOCALIVE_MIC_SPEECH_HOLD_MS` | `200` | Keep an utterance in the speech state briefly after the threshold drops |
| `VOCALIVE_MIC_SILENCE_MS` | `500` | Silence required before emitting the buffered utterance |
| `VOCALIVE_MIC_MIN_UTTERANCE_MS` | `250` | Minimum buffered audio before end-of-turn detection may emit |
| `VOCALIVE_MIC_MAX_UTTERANCE_MS` | `15000` | Hard cap for one buffered utterance |
| `VOCALIVE_APP_AUDIO_ENABLED` | `false` | Enables macOS application-audio capture as an additional live input |
| `VOCALIVE_APP_AUDIO_MODE` | `context_only` | `context_only` stores app transcripts in session without immediate assistant replies; `respond` makes app audio behave like normal live turns |
| `VOCALIVE_APP_AUDIO_TARGET` | unset | Required application selector; matches running application name first, then bundle identifier |
| `VOCALIVE_APP_AUDIO_SAMPLE_RATE` | `16000` | Application-audio capture sample rate after helper-side conversion |
| `VOCALIVE_APP_AUDIO_CHANNELS` | `1` | Captured application-audio channel count |
| `VOCALIVE_APP_AUDIO_BLOCK_MS` | `40` | Duration of each buffered application-audio PCM block |
| `VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD` | `0.02` | Minimum floor for application-audio speech detection; adaptive VAD treats it as a fallback absolute threshold |
| `VOCALIVE_APP_AUDIO_PRE_SPEECH_MS` | `200` | Buffered application audio kept before speech onset |
| `VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS` | `320` | Keeps application audio in the speech state briefly after the threshold drops |
| `VOCALIVE_APP_AUDIO_SILENCE_MS` | `650` | Silence required before emitting a buffered application-audio utterance |
| `VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS` | `250` | Minimum buffered application audio before end-of-turn detection may emit |
| `VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS` | `15000` | Hard cap for one buffered application-audio utterance |
| `VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS` | `10` | Timeout for macOS app lookup, helper startup, and helper build floor |
| `VOCALIVE_APP_AUDIO_ADAPTIVE_VAD` | `true` | Enables adaptive energy-based VAD for application audio; `false` falls back to fixed thresholding |
| `VOCALIVE_APP_AUDIO_STT_ENHANCEMENT` | `true` | Enables lightweight application-audio speech enhancement before Moonshine STT |
| `VOCALIVE_STT_PROVIDER` | `mock` | STT adapter; accepts `moonshine` and aliases such as `moonshine voice` |
| `VOCALIVE_MODEL_PROVIDER` | `mock` | LLM adapter; accepts `gemini` and aliases such as `google gemini` |
| `VOCALIVE_TTS_PROVIDER` | `mock` | TTS adapter; accepts `aivis` and aliases such as `aivis speech` |
| `VOCALIVE_OUTPUT_PROVIDER` | `memory` | `memory` or `speaker` |
| `VOCALIVE_CONVERSATION_LANGUAGE` | `ja` | Per-turn language instruction injected before the LLM call; set empty to disable |
| `VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT` | `8` | Number of recent user/assistant messages kept verbatim in Gemini requests before older dialogue is compacted |
| `VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS` | `1200` | Character budget for the earlier-conversation summary injected ahead of the recent raw-message window |
| `VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT` | `4` | Number of recent application-audio messages kept verbatim in Gemini requests before older app context is compacted |
| `VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS` | `900` | Character budget for the earlier application-audio summary injected ahead of the recent raw app-context window |
| `VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS` | `8` | Minimum normalized application-audio message length kept in the older app-context summary |
| `VOCALIVE_REPLY_DEBOUNCE_MS` | `1000` | Delay before a microphone user utterance is queued for the LLM so nearby follow-up utterances can merge into one turn |
| `VOCALIVE_REPLY_POLICY_ENABLED` | `true` | Enables conservative microphone reply suppression for low-value live chatter |
| `VOCALIVE_REPLY_MIN_GAP_MS` | `6000` | Minimum time after a completed assistant reply during which short microphone chatter is more likely to be suppressed |
| `VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS` | `12` | Maximum normalized length treated as a short microphone reaction for suppression heuristics |
| `VOCALIVE_GEMINI_API_KEY` | unset | Gemini API key; `GEMINI_API_KEY` is also accepted |
| `VOCALIVE_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name used for `generateContent` |
| `VOCALIVE_GEMINI_TIMEOUT_SECONDS` | `30` | Gemini HTTP timeout |
| `VOCALIVE_GEMINI_TEMPERATURE` | unset | Optional Gemini generation temperature |
| `VOCALIVE_GEMINI_THINKING_BUDGET` | `0` | Gemini 2.5 thinking budget; empty unsets it |
| `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` | surreal deadpan persona prompt | Overrides the default Gemini character prompt; set empty to disable it entirely |
| `VOCALIVE_SCREEN_CAPTURE_ENABLED` | `false` | Enables request-scoped named-window screenshot capture for Gemini turns |
| `VOCALIVE_SCREEN_WINDOW_NAME` | unset | Required window selector; matches on-screen window title first, then owner name |
| `VOCALIVE_SCREEN_TRIGGER_PHRASES` | `画面みて,画面見て,画面をみて,画面を見て,スクショみて,スクショ見て` | Comma-separated trigger phrases that cause a screenshot to be attached |
| `VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS` | `5` | Timeout for macOS window lookup and `screencapture` |
| `VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX` | `1280` | Resizes captured screenshots so their longest edge stays within this many pixels; empty disables resizing |
| `VOCALIVE_MOONSHINE_MODEL` | `base` | Moonshine model architecture such as `base` / `tiny`, or a concrete model id such as `base-ja` |
| `VOCALIVE_AIVIS_BASE_URL` | `http://127.0.0.1:10101` | AivisSpeech engine base URL |
| `VOCALIVE_AIVIS_SPEAKER_ID` | unset | Explicit AivisSpeech style ID |
| `VOCALIVE_AIVIS_SPEAKER_NAME` | unset | Speaker name to resolve via `/speakers` |
| `VOCALIVE_AIVIS_STYLE_NAME` | unset | Style name to resolve via `/speakers` |
| `VOCALIVE_AIVIS_TIMEOUT_SECONDS` | `30` | AivisSpeech API timeout |
| `VOCALIVE_SPEAKER_COMMAND` | `afplay {path}` | Override playback command; must include `{path}` |
| `VOCALIVE_QUEUE_MAXSIZE` | `4` | Maximum queued utterances waiting to run |
| `VOCALIVE_QUEUE_OVERFLOW` | `drop_oldest` | Overflow strategy: `drop_oldest` or `reject_new` |

Current provider support:

- `mock` STT returns `transcript_hint` or decodes UTF-8 PCM bytes for local tests
- `mock` model uses `EchoLanguageModel` and replies with `Assistant: <latest user message>`
- `mock` TTS returns synthetic audio bytes for exercising the pipeline without a real engine
- `moonshine` uses the optional `moonshine-voice` package for STT and applies low-frequency-preserving enhancement to application audio before transcription by default
- `VOCALIVE_MOONSHINE_MODEL=base` resolves a language-specific Moonshine model from `VOCALIVE_CONVERSATION_LANGUAGE`, so the default Japanese configuration resolves to `base-ja`
- `gemini` uses the Gemini `generateContent` API over HTTPS; the default config sets `thinkingBudget=0` to reduce latency
- optional application-audio capture resolves one running macOS app, segments its audio into utterances, and by default submits those transcripts as labeled application context without immediate assistant replies
- older application-audio context is compacted into one bounded system summary while the newest configured app-audio messages remain verbatim in requests
- optional screen capture resolves a named on-screen window on macOS and attaches one PNG of that window to the current Gemini turn when a trigger phrase matches
- older user/assistant dialogue is compacted into one bounded system summary before Gemini requests so long sessions do not resend the entire raw conversation every turn
- `aivis` uses the local AivisSpeech engine API and resolves a style id from `/speakers` when needed
- `speaker` output plays synthesized audio through the configured external command
- provider names are normalized case-insensitively, so values such as `Moonshine Voice` and `Aivis Speech` resolve to the supported adapters

## Repository layout

```text
src/vocalive/
  audio/       audio input, output, turn detection, and device selection
  config/      environment-driven runtime configuration
  llm/         language model interface and adapters
  pipeline/    orchestration, cancellation, queues, and session state
  screen/      optional named-window screen capture adapters
  stt/         speech-to-text interface and adapters
  tts/         text-to-speech interface and adapters
  util/        logging, metrics, and time helpers
  main.py      CLI entry point and adapter assembly

tests/unit/    unit coverage for settings, adapters, audio, and orchestration
docs/          architecture, development, and documentation maintenance rules
AGENT.md       implementation brief for coding work in this repository
```

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Development guide](docs/development.md)

`AGENT.md` is the repository-specific engineering brief for implementation work. `README.md` and `docs/` are the maintained human-facing documentation set.
