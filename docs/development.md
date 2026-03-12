# Development Guide

## Local workflow

Recommended local commands:

```bash
PYTHONPATH=src python3 -m vocalive
python3 -m unittest discover -s tests -v
python3 -m compileall src tests
```

Editable install is optional:

```bash
python3 -m pip install -e .
```

Install the optional voice dependencies when you want live microphone capture or Moonshine STT:

```bash
python3 -m pip install -e '.[voice]'
```

The repository currently requires Python 3.10+.

## CLI behavior

The current entry point is `src/vocalive/main.py`.

- `VOCALIVE_INPUT_PROVIDER=stdin` keeps the text shell
- `VOCALIVE_INPUT_PROVIDER=microphone` uses `sounddevice` and local utterance detection
- live microphone or application-audio input currently requires `VOCALIVE_STT_PROVIDER=moonshine`
- `VOCALIVE_APP_AUDIO_ENABLED=true` layers macOS application-audio capture on top of either `stdin` or `microphone`
- application-audio turns are stored in session history as labeled application context, not as user messages
- application-audio capture currently requires macOS, `VOCALIVE_APP_AUDIO_TARGET`, and Screen Recording permission
- `VOCALIVE_OUTPUT_PROVIDER=speaker` currently requires `VOCALIVE_TTS_PROVIDER=aivis`
- `/quit`, `quit`, and `exit` stop the stdin shell
- the stdin shell waits for the orchestrator to become idle, then prints the last committed assistant message
- the microphone loop keeps reading while the assistant is speaking, so speech onset can stop stale playback immediately
- the application-audio loop also keeps reading while the assistant is speaking, so new app dialogue can stop stale playback immediately
- `VOCALIVE_MIC_DEVICE=external` forces selection of a connected headset-like external mic
- when `VOCALIVE_MIC_DEVICE` is unset and `VOCALIVE_MIC_PREFER_EXTERNAL=true`, VocaLive prefers a connected external mic over a built-in default input
- the microphone path uses local RMS thresholding plus silence timing, not a production VAD
- the stdin shell sets `AudioSegment.transcript_hint`, so `moonshine`-selected configs can still exercise Gemini and Aivis wiring before switching to live microphone capture
- when screen capture is enabled, configured trigger phrases attach one screenshot of the configured window to the current Gemini turn only

Speaker playback uses `afplay {path}` by default on macOS. On other platforms, set `VOCALIVE_SPEAKER_COMMAND` to a command template that includes `{path}`.

Screen capture currently uses the macOS `screencapture` command, resolves the configured on-screen window by title or owner name, and requires Screen Recording permission.

Application-audio capture currently uses a small ScreenCaptureKit helper compiled on first use, resolves one running app by name or bundle identifier, and also requires Screen Recording permission.

## Configuration

Runtime settings are loaded from `AppSettings.from_env()` in `src/vocalive/config/settings.py`.

| Variable | Default | Notes |
| --- | --- | --- |
| `VOCALIVE_SESSION_ID` | random UUID | Useful for correlating logs across one session |
| `VOCALIVE_LOG_LEVEL` | `INFO` | Passed to Python logging |
| `VOCALIVE_INPUT_PROVIDER` | `stdin` | `stdin` or `microphone` |
| `VOCALIVE_MIC_SAMPLE_RATE` | `16000` | Microphone capture sample rate |
| `VOCALIVE_MIC_CHANNELS` | `1` | Captured microphone channel count |
| `VOCALIVE_MIC_BLOCK_MS` | `40` | Duration of each captured PCM block |
| `VOCALIVE_MIC_DEVICE` | unset | Optional input device id, device name, `default`, or `external` |
| `VOCALIVE_MIC_PREFER_EXTERNAL` | `true` | Prefer a connected headset-like external mic when the default input looks built-in |
| `VOCALIVE_MIC_SPEECH_THRESHOLD` | `0.02` | RMS threshold for treating a block as speech |
| `VOCALIVE_MIC_PRE_SPEECH_MS` | `200` | Audio kept before speech starts so utterance onsets are not clipped |
| `VOCALIVE_MIC_SPEECH_HOLD_MS` | `200` | Keeps an utterance in the speech state briefly after the threshold drops |
| `VOCALIVE_MIC_SILENCE_MS` | `500` | Silence required before the live path emits a buffered utterance |
| `VOCALIVE_MIC_MIN_UTTERANCE_MS` | `250` | Minimum buffered audio before turn-end emission is allowed |
| `VOCALIVE_MIC_MAX_UTTERANCE_MS` | `15000` | Hard cap for one buffered utterance |
| `VOCALIVE_APP_AUDIO_ENABLED` | `false` | Enables macOS application-audio capture as an extra live source |
| `VOCALIVE_APP_AUDIO_TARGET` | unset | Required selector matched against running application name first, then bundle identifier |
| `VOCALIVE_APP_AUDIO_SAMPLE_RATE` | `16000` | Application-audio sample rate after helper-side conversion |
| `VOCALIVE_APP_AUDIO_CHANNELS` | `1` | Captured application-audio channel count |
| `VOCALIVE_APP_AUDIO_BLOCK_MS` | `40` | Duration of each buffered application-audio PCM block |
| `VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD` | `0.02` | RMS threshold for treating an application-audio block as speech |
| `VOCALIVE_APP_AUDIO_PRE_SPEECH_MS` | `120` | Preroll retained before application-audio speech starts |
| `VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS` | `180` | Keeps an application-audio utterance in the speech state briefly after the threshold drops |
| `VOCALIVE_APP_AUDIO_SILENCE_MS` | `450` | Silence required before emitting buffered application audio |
| `VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS` | `250` | Minimum buffered application audio before turn-end emission is allowed |
| `VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS` | `15000` | Hard cap for one buffered application-audio utterance |
| `VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS` | `10` | Timeout for app lookup and helper startup/build |
| `VOCALIVE_STT_PROVIDER` | `mock` | `moonshine` is supported; aliases such as `moonshine voice` are accepted |
| `VOCALIVE_MODEL_PROVIDER` | `mock` | `gemini` is supported; aliases such as `google gemini` are accepted |
| `VOCALIVE_TTS_PROVIDER` | `mock` | `aivis` is supported; aliases such as `aivis speech` are accepted |
| `VOCALIVE_OUTPUT_PROVIDER` | `memory` | `memory` or `speaker` |
| `VOCALIVE_CONVERSATION_LANGUAGE` | `ja` | Injects a per-turn language instruction before the LLM call; set empty to disable |
| `VOCALIVE_GEMINI_API_KEY` | unset | Required for `gemini`; `GEMINI_API_KEY` is also accepted |
| `VOCALIVE_GEMINI_MODEL` | `gemini-2.5-flash` | Model name passed to `generateContent` |
| `VOCALIVE_GEMINI_TIMEOUT_SECONDS` | `30` | Gemini HTTP timeout |
| `VOCALIVE_GEMINI_TEMPERATURE` | unset | Optional Gemini generation temperature |
| `VOCALIVE_GEMINI_THINKING_BUDGET` | `0` | Gemini 2.5 thinking budget; empty unsets it |
| `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` | surreal deadpan persona prompt | Overrides the default Gemini character prompt; set empty to disable it |
| `VOCALIVE_SCREEN_CAPTURE_ENABLED` | `false` | Enables request-scoped named-window screenshot capture for Gemini turns |
| `VOCALIVE_SCREEN_WINDOW_NAME` | unset | Required selector matched against on-screen window title first, then owner name |
| `VOCALIVE_SCREEN_TRIGGER_PHRASES` | `画面みて,画面見て,画面をみて,画面を見て,スクショみて,スクショ見て` | Comma-separated trigger phrases matched against the normalized utterance |
| `VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS` | `5` | Timeout for macOS window lookup and `screencapture` |
| `VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX` | `1280` | Resizes captured screenshots so their longest edge stays within this many pixels; empty disables resizing |
| `VOCALIVE_MOONSHINE_MODEL` | `base` | Moonshine architecture such as `base` / `tiny`, or a concrete model id such as `base-ja` |
| `VOCALIVE_AIVIS_BASE_URL` | `http://127.0.0.1:10101` | Local AivisSpeech engine base URL |
| `VOCALIVE_AIVIS_SPEAKER_ID` | unset | Preferred explicit AivisSpeech style ID |
| `VOCALIVE_AIVIS_SPEAKER_NAME` | unset | Optional speaker name for `/speakers` lookup |
| `VOCALIVE_AIVIS_STYLE_NAME` | unset | Optional style name for `/speakers` lookup |
| `VOCALIVE_AIVIS_TIMEOUT_SECONDS` | `30` | AivisSpeech API timeout |
| `VOCALIVE_SPEAKER_COMMAND` | `afplay {path}` | Override playback command; must include `{path}` |
| `VOCALIVE_QUEUE_MAXSIZE` | `4` | Bounded utterance backlog |
| `VOCALIVE_QUEUE_OVERFLOW` | `drop_oldest` | `drop_oldest` or `reject_new` |

If you add a new setting, update this table and the root `README.md`.

## Current provider combinations

Useful working combinations today:

1. Default local shell
   `stdin` + `mock` STT + `mock` model + `mock` TTS + `memory`
2. Real remote/local providers without microphone
   `stdin` + `moonshine` STT + `gemini` + `aivis` + `memory` or `speaker`
3. Full live voice path
   `microphone` + `moonshine` + `gemini` + `aivis` + `speaker`
4. Game/video commentary path
   `microphone` or `stdin` + `VOCALIVE_APP_AUDIO_ENABLED=true` + `moonshine` + `gemini` + `aivis` + `memory` or `speaker`

Screen capture can be layered on combinations 2 and 3 when:

- `VOCALIVE_SCREEN_CAPTURE_ENABLED=true`
- `VOCALIVE_MODEL_PROVIDER=gemini`
- the app is running on macOS with Screen Recording permission

The second combination is useful because the stdin shell supplies `transcript_hint`, which lets the real-provider assembly come up before the live microphone path is enabled.

## Adding a new adapter

When integrating a provider:

1. Implement the relevant base interface in `stt/`, `llm/`, `tts/`, or `audio/`.
2. Keep provider SDK or HTTP logic inside the adapter module only.
3. Accept `TurnContext` and `CancellationToken` where the interface expects them.
4. Wire the adapter in `build_orchestrator()` or a future assembly layer.
5. Add tests for the control-flow behavior or request-shaping logic the adapter changes.
6. Update `README.md`, `docs/architecture.md`, and this guide if support becomes user-visible.

Do not push provider-specific logic into `pipeline/orchestrator.py`.

## Testing focus

Current unit tests cover:

- settings parsing, provider alias normalization, and language/config edge cases
- queue overflow semantics
- microphone utterance accumulation, combined live-input fan-in, and external-device selection
- Moonshine model resolution and transcript-hint behavior
- Gemini payload construction and response extraction
- Aivis speaker/style resolution and WAV metadata parsing
- single-turn orchestration flow
- interruption of in-flight playback by newer speech
- session history rules for interrupted turns and application-audio context commits
- sentence-by-sentence playback chunking and TTS prefetch behavior
- structured logging output

High-value next tests:

- adapter error propagation and recovery behavior
- cancellation during STT, LLM, and TTS stages
- smoke tests for live microphone and speaker backends
- broader platform checks for playback command compatibility

## Logging and metrics

Structured logs are emitted through `util/logging.py`.

- log payloads are JSON strings written through standard Python logging
- metrics are recorded through `MetricsRecorder`
- the default recorder is in-memory and suitable for local inspection and unit tests

If you add a new pipeline stage, add both logs and latency tracking for it.

## Documentation maintenance

Development changes should update docs in the same commit.

Minimum rule:

- startup changes: update `README.md`
- runtime flow changes: update `docs/architecture.md`
- new config or command changes: update `docs/development.md`
- coding constraints or maintenance expectations: update `AGENT.md`

For the full rule set, see `docs/README.md`.
