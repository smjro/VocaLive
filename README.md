# VocaLive

VocaLive is an adapter-based local voice conversation runtime focused on low latency, explicit interruption handling, and replaceable STT / LLM / TTS providers.

The repository currently ships with:

- a stdin shell for local development
- optional live microphone capture via `sounddevice`
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
- Implemented: mock STT, echo LLM, mock TTS, and in-memory playback for local development
- Implemented: Moonshine STT via `moonshine-voice`
- Implemented: Gemini `generateContent` integration over HTTPS
- Implemented: AivisSpeech synthesis over the local HTTP API
- Implemented: sentence-by-sentence TTS playback with one-sentence-ahead prefetch
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
export VOCALIVE_STT_PROVIDER=moonshine
export VOCALIVE_MODEL_PROVIDER=gemini
export VOCALIVE_TTS_PROVIDER=aivis
export VOCALIVE_OUTPUT_PROVIDER=speaker
export VOCALIVE_CONVERSATION_LANGUAGE=ja
export VOCALIVE_AIVIS_BASE_URL=http://127.0.0.1:10101
export VOCALIVE_GEMINI_API_KEY=...
PYTHONPATH=src python3 -m vocalive
```

Current runtime constraints:

- `VOCALIVE_INPUT_PROVIDER=microphone` currently requires `VOCALIVE_STT_PROVIDER=moonshine`
- `VOCALIVE_OUTPUT_PROVIDER=speaker` currently requires `VOCALIVE_TTS_PROVIDER=aivis`
- speaker playback uses `afplay {path}` by default on macOS; on other platforms set `VOCALIVE_SPEAKER_COMMAND`
- Gemini accepts either `VOCALIVE_GEMINI_API_KEY` or `GEMINI_API_KEY`
- Gemini defaults to a surreal, deadpan conversation persona inspired by the vibe of Kamiusagi Rope; set `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` to override it, or set it to an empty string to disable it

Microphone tuning notes:

- `VOCALIVE_MIC_DEVICE=external` tells VocaLive to search for a currently connected headset-like external microphone
- when `VOCALIVE_MIC_DEVICE` is unset and `VOCALIVE_MIC_PREFER_EXTERNAL=true`, VocaLive will switch away from a built-in default mic if it finds a better external input
- if phrase starts are clipped, increase `VOCALIVE_MIC_PRE_SPEECH_MS`
- if mid-sentence pauses cause early cuts, increase `VOCALIVE_MIC_SPEECH_HOLD_MS` and `VOCALIVE_MIC_SILENCE_MS`
- in microphone mode, local speech onset interrupts stale assistant playback before the next utterance is fully emitted

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
| `VOCALIVE_STT_PROVIDER` | `mock` | STT adapter; accepts `moonshine` and aliases such as `moonshine voice` |
| `VOCALIVE_MODEL_PROVIDER` | `mock` | LLM adapter; accepts `gemini` and aliases such as `google gemini` |
| `VOCALIVE_TTS_PROVIDER` | `mock` | TTS adapter; accepts `aivis` and aliases such as `aivis speech` |
| `VOCALIVE_OUTPUT_PROVIDER` | `memory` | `memory` or `speaker` |
| `VOCALIVE_CONVERSATION_LANGUAGE` | `ja` | Per-turn language instruction injected before the LLM call; set empty to disable |
| `VOCALIVE_GEMINI_API_KEY` | unset | Gemini API key; `GEMINI_API_KEY` is also accepted |
| `VOCALIVE_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name used for `generateContent` |
| `VOCALIVE_GEMINI_TIMEOUT_SECONDS` | `30` | Gemini HTTP timeout |
| `VOCALIVE_GEMINI_TEMPERATURE` | unset | Optional Gemini generation temperature |
| `VOCALIVE_GEMINI_THINKING_BUDGET` | `0` | Gemini 2.5 thinking budget; empty unsets it |
| `VOCALIVE_GEMINI_SYSTEM_INSTRUCTION` | surreal deadpan persona prompt | Overrides the default Gemini character prompt; set empty to disable it entirely |
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
- `moonshine` uses the optional `moonshine-voice` package for STT
- `VOCALIVE_MOONSHINE_MODEL=base` resolves a language-specific Moonshine model from `VOCALIVE_CONVERSATION_LANGUAGE`, so the default Japanese configuration resolves to `base-ja`
- `gemini` uses the Gemini `generateContent` API over HTTPS; the default config sets `thinkingBudget=0` to reduce latency
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
