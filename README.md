# VocaLive

VocaLive is an adapter-based local voice conversation runtime focused on low latency, explicit interruption handling, and replaceable STT / LLM / TTS providers.

The repository currently ships with:

- a local browser controller that saves non-secret runtime configuration
- an explicit headless `run` mode with the legacy stdin shell for local development
- optional live microphone capture via `sounddevice`
- optional application-audio capture for one named running app on macOS or Windows
- mock providers for end-to-end local testing
- Moonshine STT, Gemini LLM, and AivisSpeech TTS adapters
- in-memory output for tests and speaker playback through an external command
- queue-based orchestration with bounded backlog, stale-turn cancellation, and latency logging

## Current status

- Implemented: bounded queue orchestration with explicit overflow handling
- Implemented: stale-turn interruption on new utterances
- Implemented: configurable microphone barge-in modes including explicit assistant-address interruption
- Implemented: local browser GUI controller with saved non-secret config editing, runtime start/stop, and explicit headless `run` mode
- Implemented: stdin shell and microphone capture with preroll / hold / silence-based utterance detection
- Implemented: automatic preference for higher-fidelity external microphones when the system default input is built-in, while avoiding Bluetooth hands-free inputs unless explicitly requested
- Implemented: optional macOS per-app application-audio capture and Windows process-loopback application-audio capture, both feeding STT and storing transcripts as application context by default
- Implemented: adaptive VAD and low-frequency-preserving STT-side speech enhancement for application audio
- Implemented: mock STT, echo LLM, mock TTS, and in-memory playback for local development
- Implemented: Moonshine STT via `moonshine-voice`
- Implemented: OpenAI STT via the audio transcription API with `gpt-4o-mini-transcribe`
- Implemented: Gemini `generateContent` integration over HTTPS
- Implemented: trigger-based named-window screenshot capture for Gemini input on macOS and Windows
- Implemented: AivisSpeech synthesis over the local HTTP API
- Implemented: default speaker playback via `afplay` on macOS and PowerShell `SoundPlayer` on Windows
- Implemented: sentence-by-sentence TTS playback with one-sentence-ahead prefetch
- Implemented: optional transparent browser overlay with a character image, speech-only captions, and per-chunk text reveal timed to playback
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

Start the local browser controller from the source tree:

```bash
PYTHONPATH=src python3 -m vocalive
```

This opens a controller page in your default browser, stores non-secret settings in `.vocalive/controller-config.json`, and lets you start or stop the live runtime. Secret fields such as `VOCALIVE_GEMINI_API_KEY` are never written to disk and must be entered each time.

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m vocalive
```

Or install the package in editable mode:

```bash
python3 -m pip install -e .
python3 -m vocalive
```

Run the saved configuration directly without the controller:

```bash
PYTHONPATH=src python3 -m vocalive run
```

`run` loads `.vocalive/controller-config.json` first, then applies any current environment-variable overrides on top. Use this mode when you want the legacy stdin shell or scriptable env-driven startup.

Install the optional voice dependencies when you want live microphone capture or Moonshine STT:

```bash
python3 -m pip install -e '.[voice]'
```

Run the full local voice path against Moonshine, Gemini, AivisSpeech, and speaker playback in explicit headless mode:

```bash
export VOCALIVE_INPUT_PROVIDER=microphone
export VOCALIVE_APP_AUDIO_ENABLED=true
export VOCALIVE_APP_AUDIO_TARGET="Google Chrome"
export VOCALIVE_STT_PROVIDER=moonshine
export VOCALIVE_MODEL_PROVIDER=gemini
export VOCALIVE_TTS_PROVIDER=aivis
export VOCALIVE_OUTPUT_PROVIDER=speaker
export VOCALIVE_OVERLAY_ENABLED=true
export VOCALIVE_CONVERSATION_LANGUAGE=ja
export VOCALIVE_SCREEN_CAPTURE_ENABLED=true
export VOCALIVE_SCREEN_WINDOW_NAME="Steam"
export VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX=1280
export VOCALIVE_AIVIS_BASE_URL=http://127.0.0.1:10101
export VOCALIVE_AIVIS_ENGINE_MODE=gpu
export VOCALIVE_GEMINI_API_KEY=...
PYTHONPATH=src python3 -m vocalive run
```

On Windows PowerShell, set the same variables with `$env:NAME = "value"` and run `python -m vocalive run`.

To use OpenAI STT instead of Moonshine, switch `VOCALIVE_STT_PROVIDER=openai` and set
`VOCALIVE_OPENAI_API_KEY` or `OPENAI_API_KEY`. The default OpenAI STT model is
`gpt-4o-mini-transcribe`.

If you prefer the controller, launch `python -m vocalive` once and enter the same values in the browser UI; subsequent runs reuse the saved non-secret config, while secret fields must be re-entered.

When AivisSpeech is installed locally, you can let VocaLive launch it directly by setting `VOCALIVE_AIVIS_ENGINE_MODE=cpu` or `gpu`. GPU mode starts the engine with `run(.exe) --use_gpu`. If your install lives outside the standard Windows/macOS path, set `VOCALIVE_AIVIS_ENGINE_PATH` to the engine `run(.exe)` path.
If managed Aivis startup drives CPU usage too high, set `VOCALIVE_AIVIS_CPU_NUM_THREADS` to cap the engine worker threads.

When `VOCALIVE_OVERLAY_ENABLED=true`, VocaLive starts a local overlay server and prints its URL. By default it also asks the system browser to open the page automatically. The overlay is transparent, renders the character on the right, and shows assistant text only while the assistant is actively speaking. Each sentence-sized chunk is revealed progressively to match playback timing, then cleared when playback finishes or is interrupted.

Current runtime constraints:

- live microphone or application-audio input currently requires a real STT adapter such as `moonshine` or `openai`
- `VOCALIVE_APP_AUDIO_ENABLED=true` currently requires `VOCALIVE_APP_AUDIO_TARGET`; on macOS it also requires Screen Recording permission, and on Windows it requires `csc.exe` plus a Windows build with WASAPI process-loopback support
- `VOCALIVE_OUTPUT_PROVIDER=speaker` currently requires `VOCALIVE_TTS_PROVIDER=aivis`
- `VOCALIVE_AIVIS_ENGINE_MODE=cpu` or `gpu` starts the local AivisSpeech engine automatically; managed startup requires a bare local HTTP `VOCALIVE_AIVIS_BASE_URL` and either a standard install path or `VOCALIVE_AIVIS_ENGINE_PATH`
- `VOCALIVE_SCREEN_CAPTURE_ENABLED=true` currently requires `VOCALIVE_MODEL_PROVIDER=gemini`; on macOS it also requires Screen Recording permission, and on Windows it requires `csc.exe`
- speaker playback uses `afplay {path}` by default on macOS and PowerShell `SoundPlayer` on Windows; on other platforms set `VOCALIVE_SPEAKER_COMMAND`
- the overlay is local-only and driven by sentence playback events; it is not token streaming from the LLM
- Gemini accepts either `VOCALIVE_GEMINI_API_KEY` or `GEMINI_API_KEY`
- OpenAI STT accepts either `VOCALIVE_OPENAI_API_KEY` or `OPENAI_API_KEY`
- Gemini defaults to a surreal, deadpan conversation persona inspired by the vibe of Kamiusagi Rope; set `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` to override it, or set it to an empty string to disable it

Windows supports the full `stdin` / `microphone` / `application audio` / `screen capture` / overlay / speaker path. On Windows, application audio uses WASAPI process loopback scoped to the selected process tree while the selected process remains alive.

When microphone auto-selection is enabled, VocaLive avoids Bluetooth hands-free / AG Audio inputs by default because Windows often drops playback quality while those microphone profiles are active. If you need one anyway, set `VOCALIVE_MIC_DEVICE` explicitly.

Microphone tuning notes:

- `VOCALIVE_MIC_DEVICE=external` tells VocaLive to search for a currently connected external microphone, including Bluetooth hands-free inputs when they are the only match
- when `VOCALIVE_MIC_DEVICE` is unset and `VOCALIVE_MIC_PREFER_EXTERNAL=true`, VocaLive will switch away from a built-in default mic if it finds a higher-fidelity external input; Bluetooth hands-free / AG Audio inputs are skipped by auto-selection because they commonly degrade Windows playback quality
- if phrase starts are clipped, increase `VOCALIVE_MIC_PRE_SPEECH_MS`
- if mid-sentence pauses cause early cuts, increase `VOCALIVE_MIC_SPEECH_HOLD_MS` and `VOCALIVE_MIC_SILENCE_MS`
- after one live utterance is emitted, VocaLive waits briefly before queueing the LLM turn so closely spaced microphone utterances can merge; tune this with `VOCALIVE_REPLY_DEBOUNCE_MS`
- microphone reply suppression is enabled by default for live user speech so short reactions such as `やばい` are more likely to stay silent unless they are clear questions/requests; tune this with the `VOCALIVE_REPLY_*` settings
- in microphone mode, `VOCALIVE_MIC_INTERRUPT_MODE=always` keeps the legacy speech-start interruption, `explicit` waits for a finalized utterance that directly addresses the assistant, and `disabled` turns early microphone barge-in off

Application-audio notes:

- application audio can be enabled alongside either `stdin` or `microphone` input
- `VOCALIVE_APP_AUDIO_MODE=context_only` is the default; app audio is transcribed and appended to session history as labeled application context, but it does not immediately trigger LLM/TTS or interrupt active playback
- older application-audio entries are compacted into a separate bounded summary while the newest configured app-context messages stay verbatim in the LLM request window
- set `VOCALIVE_APP_AUDIO_MODE=respond` when you want application-audio utterances to behave like live turns and trigger immediate assistant replies
- on macOS, the configured target is matched against the running application name first and bundle identifier second
- on Windows, the configured target is matched against process name first, executable path second, and main window title third
- application audio uses adaptive energy-based VAD by default so steady BGM is more likely to stay in the background; set `VOCALIVE_APP_AUDIO_ADAPTIVE_VAD=false` to fall back to fixed thresholding
- application-audio utterances go through STT like live user audio, but session history stores them as labeled application context such as `Application audio (Steam): ...`
- default application-audio tuning keeps more preroll and trailing context so phrase starts and endings are less likely to clip; raise `VOCALIVE_APP_AUDIO_PRE_SPEECH_MS`, `VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS`, or `VOCALIVE_APP_AUDIO_SILENCE_MS` further if one app still cuts too aggressively
- Moonshine applies low-frequency-preserving enhancement with a gentle presence boost, soft gate, short edge padding, and normalization to application audio before STT by default; this is only used when `VOCALIVE_STT_PROVIDER=moonshine`, and `VOCALIVE_APP_AUDIO_STT_ENHANCEMENT=false` disables it
- if continuous app audio keeps Moonshine busy, set `VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS` to accept app utterances less often
- in `respond` mode, application-audio speech start also interrupts stale assistant playback before the buffered utterance is fully emitted
- app lookup and capture rely on a small platform helper that is built on first use; macOS uses ScreenCaptureKit via `swiftc`, and Windows uses a C# helper via `csc.exe`
- on Windows, capture uses WASAPI process loopback for the selected process tree, so other audible apps or VocaLive speaker playback on the same output device are excluded; this requires a Windows build with process-loopback support
- if macOS Screen Recording permission is missing, app lookup or capture will time out/fail and the error is logged

Screen-capture notes:

- screen capture is request-scoped, not persistent session history
- older user/assistant turns are compacted into one system summary when the request window grows past the configured raw-message count
- explicit capture is triggered when the normalized user utterance contains one of the configured trigger phrases
- optional passive capture can also watch for configured screen-reference phrases during normal conversation; passive sends are rate-limited and unchanged screenshots are skipped
- the current implementation resolves the first on-screen window whose title or owner name matches `VOCALIVE_SCREEN_WINDOW_NAME` on macOS and Windows
- captured screenshots are downscaled so the longest edge is at most `1280px` before they are attached to Gemini; set `VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX` empty to disable that
- the resolved window id is cached and reused until capture fails, then looked up again
- `VOCALIVE_SCREEN_WINDOW_NAME` is required when screen capture is enabled
- macOS uses a small Objective-C helper plus `screencapture`; Windows uses a C# helper that captures the target window with `PrintWindow` and a `BitBlt` fallback
- if macOS screen recording permission is missing, the turn falls back to text-only input and logs `screen_capture_failed`

The stdin shell is still available in `python -m vocalive run` when `VOCALIVE_INPUT_PROVIDER=stdin`. Typed input is stored as `AudioSegment.transcript_hint`, so real STT configurations such as `moonshine` and `openai` can still run cleanly before you switch to live microphone mode. The first real Moonshine transcription downloads and caches the selected model files.

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

The default workflow is controller-driven: `python -m vocalive` edits and persists the non-secret env-shaped config in `.vocalive/controller-config.json`.

`python -m vocalive run` keeps the runtime env-compatible by loading the saved config first and then applying current environment variables as overrides. Secret values such as `VOCALIVE_GEMINI_API_KEY` are not loaded from the controller config file, so provide them via the current environment when needed.

The controller UI exposes the same per-setting descriptions through each field's `Info` toggle.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOCALIVE_SESSION_ID` | random UUID | Stable identifier for one conversation session |
| `VOCALIVE_LOG_LEVEL` | `INFO` | Python logging level |
| `VOCALIVE_STT_PROVIDER` | `mock` | STT adapter; accepts `moonshine`, `openai`, and aliases such as `moonshine voice` or `gpt-4o-mini-transcribe` |
| `VOCALIVE_MODEL_PROVIDER` | `mock` | LLM adapter; accepts `gemini` and aliases such as `google gemini` |
| `VOCALIVE_TTS_PROVIDER` | `mock` | TTS adapter; accepts `aivis` and aliases such as `aivis speech` |
| `VOCALIVE_CONVERSATION_LANGUAGE` | `ja` | Per-turn language instruction injected before the LLM call; set empty to disable |
| `VOCALIVE_USER_NAME` | unset | Optional user name injected before the LLM call so the assistant can answer who it is speaking with without defaulting to name-based greetings |
| `VOCALIVE_CONTEXT_RECENT_MESSAGE_COUNT` | `8` | Number of recent user/assistant messages kept verbatim in Gemini requests before older dialogue is compacted |
| `VOCALIVE_CONTEXT_CONVERSATION_SUMMARY_MAX_CHARS` | `1200` | Character budget for the earlier-conversation summary injected ahead of the recent raw-message window |
| `VOCALIVE_CONTEXT_APPLICATION_RECENT_MESSAGE_COUNT` | `4` | Number of recent application-audio messages kept verbatim in Gemini requests before older app context is compacted |
| `VOCALIVE_CONTEXT_APPLICATION_SUMMARY_MAX_CHARS` | `900` | Character budget for the earlier application-audio summary injected ahead of the recent raw app-context window |
| `VOCALIVE_CONTEXT_APPLICATION_MIN_MESSAGE_CHARS` | `8` | Minimum normalized application-audio message length kept in the older app-context summary |
| `VOCALIVE_QUEUE_MAXSIZE` | `4` | Maximum queued utterances waiting to run |
| `VOCALIVE_QUEUE_OVERFLOW` | `drop_oldest` | Overflow strategy: `drop_oldest` or `reject_new` |
| `VOCALIVE_INPUT_PROVIDER` | `stdin` | `stdin` or `microphone` |
| `VOCALIVE_MIC_SAMPLE_RATE` | `16000` | Microphone capture sample rate |
| `VOCALIVE_MIC_CHANNELS` | `1` | Captured microphone channel count |
| `VOCALIVE_MIC_BLOCK_MS` | `40.0` | Duration of each captured PCM block |
| `VOCALIVE_MIC_SPEECH_THRESHOLD` | `0.02` | RMS threshold for treating a block as speech |
| `VOCALIVE_MIC_PRE_SPEECH_MS` | `200.0` | Audio kept before speech starts so utterance onsets are not clipped |
| `VOCALIVE_MIC_SPEECH_HOLD_MS` | `200.0` | Keep an utterance in the speech state briefly after the threshold drops |
| `VOCALIVE_MIC_SILENCE_MS` | `500.0` | Silence required before emitting the buffered utterance |
| `VOCALIVE_MIC_MIN_UTTERANCE_MS` | `250.0` | Minimum buffered audio before end-of-turn detection may emit |
| `VOCALIVE_MIC_MAX_UTTERANCE_MS` | `15000.0` | Hard cap for one buffered utterance |
| `VOCALIVE_MIC_DEVICE` | unset | Optional input device id, device name, `default`, or `external` |
| `VOCALIVE_MIC_PREFER_EXTERNAL` | `true` | Prefer a connected higher-fidelity external mic when the default input looks built-in; auto-selection avoids Bluetooth hands-free inputs |
| `VOCALIVE_MIC_INTERRUPT_MODE` | `always` | Microphone barge-in policy: `always` interrupts active assistant speech on new user speech, `explicit` waits for a finalized utterance that directly calls the assistant, and `disabled` never interrupts early |
| `VOCALIVE_APP_AUDIO_ENABLED` | `false` | Enables application-audio capture as an additional live input |
| `VOCALIVE_APP_AUDIO_MODE` | `context_only` | `context_only` stores app transcripts in session without immediate assistant replies; `respond` makes app audio behave like normal live turns |
| `VOCALIVE_APP_AUDIO_TARGET` | unset | Required application selector; on macOS it matches application name first then bundle identifier, and on Windows it matches process name, executable path, or main window title |
| `VOCALIVE_APP_AUDIO_SAMPLE_RATE` | `16000` | Application-audio capture sample rate after helper-side conversion |
| `VOCALIVE_APP_AUDIO_CHANNELS` | `1` | Captured application-audio channel count |
| `VOCALIVE_APP_AUDIO_BLOCK_MS` | `40.0` | Duration of each buffered application-audio PCM block |
| `VOCALIVE_APP_AUDIO_SPEECH_THRESHOLD` | `0.02` | Minimum floor for application-audio speech detection; adaptive VAD treats it as a fallback absolute threshold |
| `VOCALIVE_APP_AUDIO_PRE_SPEECH_MS` | `200.0` | Buffered application audio kept before speech onset |
| `VOCALIVE_APP_AUDIO_SPEECH_HOLD_MS` | `320.0` | Keeps application audio in the speech state briefly after the threshold drops |
| `VOCALIVE_APP_AUDIO_SILENCE_MS` | `650.0` | Silence required before emitting a buffered application-audio utterance |
| `VOCALIVE_APP_AUDIO_MIN_UTTERANCE_MS` | `250.0` | Minimum buffered application audio before end-of-turn detection may emit |
| `VOCALIVE_APP_AUDIO_MAX_UTTERANCE_MS` | `15000.0` | Hard cap for one buffered application-audio utterance |
| `VOCALIVE_APP_AUDIO_TIMEOUT_SECONDS` | `10.0` | Timeout for application lookup, helper startup, and helper build floor |
| `VOCALIVE_APP_AUDIO_TRANSCRIPTION_COOLDOWN_SECONDS` | `0.0` | Minimum delay between accepted application-audio utterances before STT runs again; increase this to reduce Moonshine CPU load on continuous media |
| `VOCALIVE_APP_AUDIO_ADAPTIVE_VAD` | `true` | Enables adaptive energy-based VAD for application audio; `false` falls back to fixed thresholding |
| `VOCALIVE_APP_AUDIO_STT_ENHANCEMENT` | `true` | Enables lightweight application-audio speech enhancement before Moonshine STT |
| `VOCALIVE_OUTPUT_PROVIDER` | `memory` | `memory` or `speaker` |
| `VOCALIVE_SPEAKER_COMMAND` | platform default | Override playback command; must include `{path}`. Defaults to `afplay {path}` on macOS and PowerShell `SoundPlayer` on Windows |
| `VOCALIVE_OVERLAY_ENABLED` | `false` | Start the local transparent browser overlay with speech-only captions |
| `VOCALIVE_OVERLAY_HOST` | `127.0.0.1` | Host/interface used by the overlay HTTP server |
| `VOCALIVE_OVERLAY_PORT` | `8765` | Port used by the overlay HTTP server |
| `VOCALIVE_OVERLAY_AUTO_OPEN` | `true` | Ask the system browser to open the overlay page automatically |
| `VOCALIVE_OVERLAY_TITLE` | `VocaLive Overlay` | Browser page title for the overlay |
| `VOCALIVE_OVERLAY_CHARACTER_NAME` | `Tora` | Accessibility label and page text for the overlay character |
| `VOCALIVE_REPLY_DEBOUNCE_MS` | `1000.0` | Delay before a microphone user utterance is queued for the LLM so nearby follow-up utterances can merge into one turn |
| `VOCALIVE_REPLY_POLICY_ENABLED` | `true` | Enables conservative microphone reply suppression for low-value live chatter |
| `VOCALIVE_REPLY_MIN_GAP_MS` | `6000.0` | Minimum time after a completed assistant reply during which short microphone chatter is more likely to be suppressed |
| `VOCALIVE_REPLY_SHORT_UTTERANCE_MAX_CHARS` | `12` | Maximum normalized length treated as a short microphone reaction for suppression heuristics |
| `VOCALIVE_GEMINI_API_KEY` | unset | Gemini API key; `GEMINI_API_KEY` is also accepted |
| `VOCALIVE_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name used for `generateContent` |
| `VOCALIVE_GEMINI_TIMEOUT_SECONDS` | `30.0` | Gemini HTTP timeout |
| `VOCALIVE_GEMINI_TEMPERATURE` | unset | Optional Gemini generation temperature |
| `VOCALIVE_GEMINI_THINKING_BUDGET` | `0` | Gemini 2.5 thinking budget; empty unsets it |
| `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` | Kohaku surreal deadpan persona prompt | Overrides the default Gemini character prompt; set empty to disable it entirely |
| `VOCALIVE_SCREEN_CAPTURE_ENABLED` | `false` | Enables request-scoped named-window screenshot capture for Gemini turns |
| `VOCALIVE_SCREEN_WINDOW_NAME` | unset | Required window selector; matches on-screen window title first, then owner name |
| `VOCALIVE_SCREEN_TRIGGER_PHRASES` | `画面みて,画面見て,画面をみて,画面を見て,スクショみて,スクショ見て` | Comma-separated trigger phrases that cause a screenshot to be attached |
| `VOCALIVE_SCREEN_PASSIVE_ENABLED` | `false` | Allows screen-reference phrases to attach a screenshot opportunistically during normal conversation |
| `VOCALIVE_SCREEN_PASSIVE_TRIGGER_PHRASES` | `この画面,今の画面,いまの画面,見えてる,見えてます` | Comma-separated screen-reference phrases checked only when passive capture is enabled |
| `VOCALIVE_SCREEN_PASSIVE_COOLDOWN_SECONDS` | `30.0` | Minimum delay between passive screenshot sends; unchanged passive screenshots are also skipped |
| `VOCALIVE_SCREEN_CAPTURE_TIMEOUT_SECONDS` | `5.0` | Timeout for window lookup and platform capture helpers |
| `VOCALIVE_SCREEN_RESIZE_MAX_EDGE_PX` | `1280` | Resizes captured screenshots so their longest edge stays within this many pixels; empty disables resizing |
| `VOCALIVE_MOONSHINE_MODEL` | `base` | Moonshine model architecture such as `base` / `tiny`, or a concrete model id such as `base-ja` |
| `VOCALIVE_OPENAI_API_KEY` | unset | OpenAI API key for STT; `OPENAI_API_KEY` is also accepted |
| `VOCALIVE_OPENAI_MODEL` | `gpt-4o-mini-transcribe` | OpenAI transcription model name |
| `VOCALIVE_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL for audio transcription |
| `VOCALIVE_OPENAI_TIMEOUT_SECONDS` | `30.0` | OpenAI audio transcription HTTP timeout |
| `VOCALIVE_AIVIS_BASE_URL` | `http://127.0.0.1:10101` | AivisSpeech engine base URL |
| `VOCALIVE_AIVIS_ENGINE_MODE` | `external` | AivisSpeech engine startup mode: `external`, `cpu`, or `gpu` |
| `VOCALIVE_AIVIS_ENGINE_PATH` | unset | Optional path to AivisSpeech Engine `run(.exe)` for managed CPU/GPU startup |
| `VOCALIVE_AIVIS_CPU_NUM_THREADS` | unset | Optional managed-startup CPU thread limit passed to AivisSpeech Engine as `--cpu_num_threads`; lower values can reduce CPU load |
| `VOCALIVE_AIVIS_STARTUP_TIMEOUT_SECONDS` | `60.0` | How long VocaLive waits for a managed AivisSpeech engine to become ready |
| `VOCALIVE_AIVIS_SPEAKER_ID` | unset | Explicit AivisSpeech style ID |
| `VOCALIVE_AIVIS_SPEAKER_NAME` | unset | Speaker name to resolve via `/speakers` |
| `VOCALIVE_AIVIS_STYLE_NAME` | unset | Style name to resolve via `/speakers` |
| `VOCALIVE_AIVIS_TIMEOUT_SECONDS` | `30.0` | AivisSpeech API timeout |

Current provider support:

- `mock` STT returns `transcript_hint` or decodes UTF-8 PCM bytes for local tests
- `mock` model uses `EchoLanguageModel` and replies with `Assistant: <latest user message>`
- `mock` TTS returns synthetic audio bytes for exercising the pipeline without a real engine
- `moonshine` uses the optional `moonshine-voice` package for STT and applies low-frequency-preserving enhancement to application audio before transcription by default
- `openai` uploads utterance WAV audio to the OpenAI audio transcription API and defaults to `gpt-4o-mini-transcribe`
- `VOCALIVE_MOONSHINE_MODEL=base` resolves a language-specific Moonshine model from `VOCALIVE_CONVERSATION_LANGUAGE`, so the default Japanese configuration resolves to `base-ja`
- `gemini` uses the Gemini `generateContent` API over HTTPS; the default config sets `thinkingBudget=0` to reduce latency
- optional application-audio capture resolves one named running app, segments audio into utterances, and by default submits those transcripts as labeled application context without immediate assistant replies
- older application-audio context is compacted into one bounded system summary while the newest configured app-audio messages remain verbatim in requests
- on macOS, application-audio capture uses ScreenCaptureKit to isolate the selected app; on Windows, application-audio capture uses WASAPI process loopback for the selected process tree while the selected process remains alive
- optional screen capture resolves a named on-screen window on macOS or Windows and attaches one PNG of that window to the current Gemini turn when an explicit trigger phrase or an eligible passive screen-reference phrase matches
- older user/assistant dialogue is compacted into one bounded system summary before Gemini requests so long sessions do not resend the entire raw conversation every turn
- `aivis` uses the local AivisSpeech engine API, resolves a style id from `/speakers` when needed, and can optionally launch the local engine in `cpu` or `gpu` mode before the runtime starts
- `speaker` output plays synthesized audio through the configured external command or the platform default playback command
- `overlay` is an optional local browser UI fed by orchestrator events and chunk-level playback timing
- the overlay loads character art from `src/vocalive/ui/assets/character.png` when present, and otherwise falls back to the built-in vector character
- provider names are normalized case-insensitively, so values such as `Moonshine Voice`, `gpt-4o-mini-transcribe`, and `Aivis Speech` resolve to the supported adapters

## Repository layout

```text
src/vocalive/
  audio/       audio input, output, turn detection, and device selection
  config/      controller-backed and env-compatible runtime configuration
  llm/         language model interface and adapters
  pipeline/    orchestration, cancellation, queues, and session state
  runtime.py   shared runtime assembly for controller and headless modes
  screen/      optional named-window screen capture adapters
  stt/         speech-to-text interface and adapters
  tts/         text-to-speech interface and adapters
  ui/          local browser controller, overlay server, and UI assets
  util/        logging, metrics, and time helpers
  main.py      entry point for controller mode and explicit `run` mode

tests/unit/    unit coverage for settings, adapters, audio, and orchestration
docs/          architecture, development, and documentation maintenance rules
AGENT.md       implementation brief for coding work in this repository
```

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Development guide](docs/development.md)

`AGENT.md` is the repository-specific engineering brief for implementation work. `README.md` and `docs/` are the maintained human-facing documentation set.
