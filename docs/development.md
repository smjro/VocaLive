# Development Guide

## Local workflow

Recommended local commands:

```bash
PYTHONPATH=src python3 -m vocalive
PYTHONPATH=src python3 -m vocalive run
python3 -m unittest discover -s tests -v
python3 -m compileall src tests
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m vocalive
python -m vocalive run
python -m unittest discover -s tests -v
python -m compileall src tests
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

## Launch behavior

The current entry point is `src/vocalive/main.py`.

- `python -m vocalive` starts the local browser controller, loads `.vocalive/controller-config.json`, and opens the controller page on localhost
- the controller edits and persists the non-secret env-shaped runtime config, then starts or stops the live runtime from the browser UI
- `python -m vocalive run` starts the runtime directly using the saved config plus current environment-variable overrides
- `VOCALIVE_INPUT_PROVIDER=stdin` keeps the text shell, but only in explicit `run` mode
- `VOCALIVE_INPUT_PROVIDER=microphone` uses `sounddevice` and local utterance detection
- live microphone or application-audio input currently requires a real STT adapter such as `moonshine` or `openai`
- `VOCALIVE_APP_AUDIO_ENABLED=true` layers application-audio capture on top of either `stdin` or `microphone`
- `VOCALIVE_APP_AUDIO_MODE=context_only` is the default; application-audio segments are transcribed and appended to session history without immediately triggering LLM/TTS
- set `VOCALIVE_APP_AUDIO_MODE=respond` when you want application audio to behave like a normal live turn and interrupt stale playback
- application-audio turns are stored in session history as labeled application context, not as user messages
- application-audio capture currently requires `VOCALIVE_APP_AUDIO_TARGET`; on macOS it also requires Screen Recording permission, and on Windows it requires `csc.exe` plus a Windows build with WASAPI process-loopback support
- application-audio capture uses adaptive energy-based VAD by default and can fall back to fixed thresholding with `VOCALIVE_APP_AUDIO_ADAPTIVE_VAD=false`
- Moonshine applies low-frequency-preserving speech enhancement to application-audio segments before transcription unless `VOCALIVE_APP_AUDIO_STT_ENHANCEMENT=false`; this path is only used when `VOCALIVE_STT_PROVIDER=moonshine`
- `VOCALIVE_OUTPUT_PROVIDER=speaker` currently requires `VOCALIVE_TTS_PROVIDER=aivis`
- `VOCALIVE_AIVIS_ENGINE_MODE=cpu` or `gpu` makes VocaLive launch the local AivisSpeech engine automatically; GPU mode uses `run(.exe) --use_gpu`
- `VOCALIVE_OVERLAY_ENABLED=true` starts a local transparent browser overlay that shows the character and assistant speech text
- when `VOCALIVE_OVERLAY_AUTO_OPEN=true`, startup asks the system browser to open the overlay automatically
- the overlay shows captions only while audio is actively playing and clears them on completion or interruption
- place custom character art at `src/vocalive/ui/assets/character.png`; when the file is missing the built-in vector character is used
- controller mode rejects `VOCALIVE_INPUT_PROVIDER=stdin`; use `python -m vocalive run` when you want the stdin shell
- `/quit`, `quit`, and `exit` stop the stdin shell
- the stdin shell waits for the orchestrator to become idle, then prints the last committed assistant message
- the microphone loop keeps reading while the assistant is speaking; `VOCALIVE_MIC_INTERRUPT_MODE=always` stops stale playback on speech onset, while `explicit` waits for a finalized utterance that directly addresses the assistant
- in `respond` mode, the application-audio loop also keeps reading while the assistant is speaking, so new app dialogue can stop stale playback immediately
- older user/assistant turns are compacted into one bounded summary before Gemini requests once the configured recent raw-message window is exceeded
- microphone user utterances wait for `VOCALIVE_REPLY_DEBOUNCE_MS` before queueing so closely spaced follow-up speech can merge into one LLM turn
- microphone reply suppression is enabled by default for low-value live chatter; explicit questions/requests still bypass the policy
- older application-audio context is compacted into its own bounded summary while the newest configured app-audio messages stay verbatim in Gemini requests
- `VOCALIVE_MIC_DEVICE=external` forces selection of a connected external mic, including Bluetooth hands-free inputs when they are the best match
- when `VOCALIVE_MIC_DEVICE` is unset and `VOCALIVE_MIC_PREFER_EXTERNAL=true`, VocaLive prefers a connected higher-fidelity external mic over a built-in default input and skips Bluetooth hands-free / AG Audio inputs during auto-selection
- the microphone path uses local RMS thresholding plus silence timing, not a production VAD
- the stdin shell sets `AudioSegment.transcript_hint`, so real STT configs such as `moonshine` and `openai` can still exercise Gemini and Aivis wiring before switching to live microphone capture
- when screen capture is enabled, explicit trigger phrases attach one screenshot of the configured window to the current Gemini turn only; optional passive screen-reference phrases can also attach one, but passive sends are rate-limited and unchanged screenshots are skipped

Speaker playback uses `afplay {path}` by default on macOS and PowerShell `SoundPlayer` on Windows. On other platforms, set `VOCALIVE_SPEAKER_COMMAND` to a command template that includes `{path}`.

Windows supports the full `stdin` / `microphone` / `application audio` / `screen capture` / overlay / speaker path. On Windows, application audio records WASAPI process loopback scoped to the selected process tree while the selected process remains alive.

On Windows, Bluetooth hands-free microphone profiles often force headset playback into low-fidelity call mode while the mic is active. VocaLive therefore avoids auto-selecting those inputs unless `VOCALIVE_MIC_DEVICE` is set explicitly.

Screen capture resolves the configured on-screen window by title or owner name. macOS uses `screencapture` and requires Screen Recording permission; Windows uses a small C# helper built with `csc.exe`.

Application-audio capture uses a small helper compiled on first use. macOS uses ScreenCaptureKit and resolves one running app by name or bundle identifier; Windows resolves a process name, executable path, or window title and captures WASAPI process loopback for that process tree while the selected process stays alive.

## Configuration

Runtime settings are parsed through `AppSettings.from_mapping()` in `src/vocalive/config/settings.py`.

- controller mode persists recognized non-secret `VOCALIVE_*` values in `.vocalive/controller-config.json`
- `python -m vocalive run` loads that file first, then overlays current environment variables
- secret values such as `VOCALIVE_GEMINI_API_KEY` are intentionally excluded from the controller config and must be supplied per start or via the current environment
- the runtime still accepts the same env variable names shown below

| Variable | Default | Notes |
| --- | --- | --- |
| `VOCALIVE_SESSION_ID` | random UUID | Useful for correlating logs across one session |
| `VOCALIVE_LOG_LEVEL` | `INFO` | Passed to Python logging |
| `VOCALIVE_INPUT_PROVIDER` | `stdin` | `stdin` or `microphone` |
| `VOCALIVE_MIC_SAMPLE_RATE` | `16000` | Microphone capture sample rate |
| `VOCALIVE_MIC_CHANNELS` | `1` | Captured microphone channel count |
| `VOCALIVE_MIC_BLOCK_MS` | `40` | Duration of each captured PCM block |
| `VOCALIVE_MIC_DEVICE` | unset | Optional input device id, device name, `default`, or `external` |
| `VOCALIVE_MIC_PREFER_EXTERNAL` | `true` | Prefer a connected higher-fidelity external mic when the default input looks built-in; auto-selection skips Bluetooth hands-free inputs |
| `VOCALIVE_MIC_INTERRUPT_MODE` | `always` | Microphone barge-in policy: `always`, `explicit`, or `disabled` |
| `VOCALIVE_MIC_SPEECH_THRESHOLD` | `0.02` | RMS threshold for treating a block as speech |
| `VOCALIVE_MIC_PRE_SPEECH_MS` | `200` | Audio kept before speech starts so utterance onsets are not clipped |
| `VOCALIVE_MIC_SPEECH_HOLD_MS` | `200` | Keeps an utterance in the speech state briefly after the threshold drops |
| `VOCALIVE_MIC_SILENCE_MS` | `500` | Silence required before the live path emits a buffered utterance |
| `VOCALIVE_MIC_MIN_UTTERANCE_MS` | `250` | Minimum buffered audio before turn-end emission is allowed |
| `VOCALIVE_MIC_MAX_UTTERANCE_MS` | `15000` | Hard cap for one buffered utterance |
| `VOCALIVE_APP_AUDIO_ENABLED` | `false` | Enables application-audio capture as an extra live source |
| `VOCALIVE_APP_AUDIO_MODE` | `context_only` | `context_only` stores app transcripts as session context only; `respond` makes app audio trigger live assistant turns |
| `VOCALIVE_APP_AUDIO_TARGET` | unset | Required selector. macOS matches application name first then bundle identifier; Windows matches process name, executable path, or main window title |
| `VOCALIVE_APP_AUDIO_SAMPLE_RATE` | `16000` | Application-audio sample rate after helper-side conversion |
| `VOCALIVE_APP_AUDIO_CHANNELS` | `1` | Captured application-audio channel count |
| `VOCALIVE_APP_AUDIO_BLOCK_MS` | `40` | Duration of each buffered application-audio PCM block |
| `VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD` | `0.02` | Minimum floor for application-audio speech detection; adaptive VAD treats it as a fallback absolute threshold |
| `VOCALIVE_APP_AUDIO_PRE_SPEECH_MS` | `200` | Preroll retained before application-audio speech starts |
| `VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS` | `320` | Keeps an application-audio utterance in the speech state briefly after the threshold drops |
| `VOCALIVE_APP_AUDIO_SILENCE_MS` | `650` | Silence required before emitting buffered application audio |
| `VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS` | `250` | Minimum buffered application audio before turn-end emission is allowed |
| `VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS` | `15000` | Hard cap for one buffered application-audio utterance |
| `VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS` | `10` | Timeout for app lookup and helper startup/build |
| `VOCALIVE_APP_AUDIO_ADAPTIVE_VAD` | `true` | Enables adaptive energy-based VAD for application audio; `false` falls back to fixed thresholding |
| `VOCALIVE_APP_AUDIO_STT_ENHANCEMENT` | `true` | Enables lightweight application-audio speech enhancement before Moonshine STT |
| `VOCALIVE_STT_PROVIDER` | `mock` | `moonshine` and `openai` are supported; aliases such as `moonshine voice` and `gpt-4o-mini-transcribe` are accepted |
| `VOCALIVE_MODEL_PROVIDER` | `mock` | `gemini` is supported; aliases such as `google gemini` are accepted |
| `VOCALIVE_TTS_PROVIDER` | `mock` | `aivis` is supported; aliases such as `aivis speech` are accepted |
| `VOCALIVE_OUTPUT_PROVIDER` | `memory` | `memory` or `speaker` |
| `VOCALIVE_OVERLAY_ENABLED` | `false` | Starts the local transparent browser overlay with speech-only captions |
| `VOCALIVE_OVERLAY_HOST` | `127.0.0.1` | Bind host/interface for the overlay server |
| `VOCALIVE_OVERLAY_PORT` | `8765` | Bind port for the overlay server |
| `VOCALIVE_OVERLAY_AUTO_OPEN` | `true` | Opens the overlay page in the default browser on startup |
| `VOCALIVE_OVERLAY_TITLE` | `VocaLive Overlay` | Browser page title |
| `VOCALIVE_OVERLAY_CHARACTER_NAME` | `Tora` | Accessibility label and page text for the overlay character |
| `VOCALIVE_CONVERSATION_LANGUAGE` | `ja` | Injects a per-turn language instruction before the LLM call; set empty to disable |
| `VOCALIVE_USER_NAME` | unset | Optional user name injected before the LLM call so the assistant can answer who it is speaking with without defaulting to name-based greetings |
| `VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT` | `8` | Number of recent user/assistant messages kept raw in LLM requests before older conversation is compacted |
| `VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS` | `1200` | Character budget for the bounded earlier-conversation summary inserted ahead of the recent raw window |
| `VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT` | `4` | Number of recent application-audio messages kept raw in LLM requests before older app context is compacted |
| `VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS` | `900` | Character budget for the bounded earlier application-audio summary inserted ahead of the recent raw app-context window |
| `VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS` | `8` | Minimum normalized application-audio message length kept in the older app-context summary |
| `VOCALIVE_REPLY_DEBOUNCE_MS` | `1000` | Delay before a microphone user utterance is queued so nearby follow-up utterances can merge into one turn |
| `VOCALIVE_REPLY_POLICY_ENABLED` | `true` | Enables conservative reply suppression for low-value live microphone chatter |
| `VOCALIVE_REPLY_MIN_GAP_MS` | `6000` | Minimum time after a completed assistant reply during which short microphone chatter is more likely to be suppressed |
| `VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS` | `12` | Maximum normalized length treated as a short microphone reaction by the suppression heuristics |
| `VOCALIVE_GEMINI_API_KEY` | unset | Required for `gemini`; `GEMINI_API_KEY` is also accepted |
| `VOCALIVE_GEMINI_MODEL` | `gemini-2.5-flash` | Model name passed to `generateContent` |
| `VOCALIVE_GEMINI_TIMEOUT_SECONDS` | `30` | Gemini HTTP timeout |
| `VOCALIVE_GEMINI_TEMPERATURE` | unset | Optional Gemini generation temperature |
| `VOCALIVE_GEMINI_THINKING_BUDGET` | `0` | Gemini 2.5 thinking budget; empty unsets it |
| `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` | Kohaku surreal deadpan persona prompt | Overrides the default Gemini character prompt; set empty to disable it |
| `VOCALIVE_SCREEN_CAPTURE_ENABLED` | `false` | Enables request-scoped named-window screenshot capture for Gemini turns |
| `VOCALIVE_SCREEN_WINDOW_NAME` | unset | Required selector matched against on-screen window title first, then owner name |
| `VOCALIVE_SCREEN_TRIGGER_PHRASES` | `画面みて,画面見て,画面をみて,画面を見て,スクショみて,スクショ見て` | Comma-separated trigger phrases matched against the normalized utterance |
| `VOCALIVE_SCREEN_PASSIVE_ENABLED` | `false` | Allows screen-reference phrases to attach a screenshot opportunistically during normal conversation |
| `VOCALIVE_SCREEN_PASSIVE_TRIGGER_PHRASES` | `この画面,今の画面,いまの画面,見えてる,見えてます` | Comma-separated screen-reference phrases checked only when passive capture is enabled |
| `VOCALIVE_SCREEN_PASSIVE_COOLDOWN_SECONDS` | `30.0` | Minimum delay between passive screenshot sends; unchanged passive screenshots are also skipped |
| `VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS` | `5` | Timeout for window lookup and platform capture helpers |
| `VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX` | `1280` | Resizes captured screenshots so their longest edge stays within this many pixels; empty disables resizing |
| `VOCALIVE_MOONSHINE_MODEL` | `base` | Moonshine architecture such as `base` / `tiny`, or a concrete model id such as `base-ja` |
| `VOCALIVE_OPENAI_API_KEY` | unset | Required for `openai`; `OPENAI_API_KEY` is also accepted |
| `VOCALIVE_OPENAI_MODEL` | `gpt-4o-mini-transcribe` | Model name passed to the OpenAI audio transcription API |
| `VOCALIVE_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL for audio transcription |
| `VOCALIVE_OPENAI_TIMEOUT_SECONDS` | `30.0` | OpenAI audio transcription HTTP timeout |
| `VOCALIVE_AIVIS_BASE_URL` | `http://127.0.0.1:10101` | Local AivisSpeech engine base URL |
| `VOCALIVE_AIVIS_ENGINE_MODE` | `external` | AivisSpeech engine startup mode: `external`, `cpu`, or `gpu` |
| `VOCALIVE_AIVIS_ENGINE_PATH` | unset | Optional path to the AivisSpeech Engine `run(.exe)` file for managed startup |
| `VOCALIVE_AIVIS_STARTUP_TIMEOUT_SECONDS` | `60.0` | How long VocaLive waits for a managed AivisSpeech engine to become ready |
| `VOCALIVE_AIVIS_SPEAKER_ID` | unset | Preferred explicit AivisSpeech style ID |
| `VOCALIVE_AIVIS_SPEAKER_NAME` | unset | Optional speaker name for `/speakers` lookup |
| `VOCALIVE_AIVIS_STYLE_NAME` | unset | Optional style name for `/speakers` lookup |
| `VOCALIVE_AIVIS_TIMEOUT_SECONDS` | `30` | AivisSpeech API timeout |
| `VOCALIVE_SPEAKER_COMMAND` | platform default | Override playback command; must include `{path}`. Defaults to `afplay {path}` on macOS and PowerShell `SoundPlayer` on Windows |
| `VOCALIVE_QUEUE_MAXSIZE` | `4` | Bounded utterance backlog |
| `VOCALIVE_QUEUE_OVERFLOW` | `drop_oldest` | `drop_oldest` or `reject_new` |

If you add a new setting, update this table and the root `README.md`.

## Current provider combinations

Useful working combinations today:

1. Default local shell
   `python -m vocalive run` with `stdin` + `mock` STT + `mock` model + `mock` TTS + `memory`
2. Real remote/local providers without microphone
   `python -m vocalive run` with `stdin` + `moonshine` or `openai` STT + `gemini` + `aivis` + `memory` or `speaker`
3. Full live voice path
   controller or `run` mode with `microphone` + `moonshine` or `openai` + `gemini` + `aivis` + `speaker`
4. Live voice path with overlay
   controller or `run` mode with `microphone` + `moonshine` or `openai` + `gemini` + `aivis` + `speaker` + `VOCALIVE_OVERLAY_ENABLED=true`
5. Game/video commentary path
   `microphone` or `stdin` + `VOCALIVE_APP_AUDIO_ENABLED=true` + `moonshine` or `openai` + `gemini` + `aivis` + `memory` or `speaker`
   default app-audio behavior is `VOCALIVE_APP_AUDIO_MODE=context_only`; set `VOCALIVE_APP_AUDIO_MODE=respond` only when immediate replies to app dialogue are desired

Screen capture can be layered on combinations 2 and 3 when:

- `VOCALIVE_SCREEN_CAPTURE_ENABLED=true`
- `VOCALIVE_MODEL_PROVIDER=gemini`
- the app is running on macOS with Screen Recording permission, or on Windows with `csc.exe` available

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
- UI-event emission for caption overlays and interruption handling
- overlay rendering and character-asset fallback
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
