# VocaLive

VocaLive is an adapter-based foundation for a low-latency local voice conversation system.

The current repository contains the core orchestration layer, mock providers for local development, and a CLI that can run either from stdin or from a live microphone. Real STT, LLM, TTS, and device integrations stay behind replaceable interfaces.

## Current status

- Implemented: queue-driven conversation orchestration
- Implemented: interruption and cancellation for stale turns
- Implemented: structured logging and in-memory latency metrics
- Implemented: mock STT, mock LLM, mock TTS, in-memory playback
- Implemented: optional microphone input via `sounddevice`
- Implemented: Moonshine STT adapter via `moonshine-voice`
- Implemented: Gemini API adapter via HTTP
- Implemented: AivisSpeech TTS adapter via local HTTP API
- Implemented: sentence-by-sentence TTS playback with background prefetch
- Implemented: speaker playback via `afplay`
- Implemented: microphone speech-start barge-in for stale reply cancellation
- Implemented: unit tests for queue overflow and interruption behavior
- Not implemented yet: streaming partial STT / LLM / TTS
- Not implemented yet: echo cancellation for full-duplex mic + speaker use

## Quick start

Run directly from the source tree:

```bash
PYTHONPATH=src python3 -m vocalive
```

Or install the package in editable mode first:

```bash
python3 -m pip install -e .
python3 -m vocalive
```

To enable real microphone capture with Moonshine STT, install the optional extras.
The current Moonshine adapter expects Python 3.10+ because it uses `moonshine-voice`:

```bash
python3 -m pip install -e .'[voice]'
```

To run the live voice path against a local AivisSpeech engine:

```bash
export VOCALIVE_INPUT_PROVIDER=microphone
export VOCALIVE_MIC_DEVICE=default
export VOCALIVE_STT_PROVIDER=moonshine
export VOCALIVE_MODEL_PROVIDER=gemini
export VOCALIVE_TTS_PROVIDER=aivis
export VOCALIVE_OUTPUT_PROVIDER=speaker
export VOCALIVE_CONVERSATION_LANGUAGE=ja
export VOCALIVE_AIVIS_BASE_URL=http://127.0.0.1:10101
export VOCALIVE_GEMINI_API_KEY=...
PYTHONPATH=src python3 -m vocalive
```

`VOCALIVE_MIC_DEVICE=external` tells VocaLive to look for a currently connected
headset or other external microphone instead of staying on the built-in mic.
If `VOCALIVE_MIC_DEVICE` is unset, VocaLive keeps the system default input but,
by default, will still switch away from a built-in microphone when it detects a
headset-like external input.
If `external` does not resolve on your machine, use `default` or a concrete
device name from `python3 -c "import sounddevice as sd; print(sd.query_devices())"`.
If STT drops the beginning of a phrase, first increase `VOCALIVE_MIC_PRE_SPEECH_MS`.
If it still cuts off while you are mid-sentence, then increase
`VOCALIVE_MIC_SPEECH_HOLD_MS` and `VOCALIVE_MIC_SILENCE_MS`.
In microphone mode, VocaLive keeps capturing while the assistant is speaking and
stops the active reply as soon as local speech crosses the configured threshold.

The stdin shell can also exercise the Gemini and Aivis wiring without a microphone.
When `VOCALIVE_STT_PROVIDER=moonshine`, typed shell input is reused as a transcript hint
so the real-provider assembly still starts cleanly before you switch to microphone mode.
The first real Moonshine transcription downloads and caches the selected model files.

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

Environment variables are optional.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOCALIVE_SESSION_ID` | random UUID | Stable identifier for one conversation session |
| `VOCALIVE_LOG_LEVEL` | `INFO` | Python logging level |
| `VOCALIVE_INPUT_PROVIDER` | `stdin` | `stdin` or `microphone` |
| `VOCALIVE_MIC_SAMPLE_RATE` | `16000` | Microphone capture sample rate |
| `VOCALIVE_MIC_CHANNELS` | `1` | Captured microphone channels |
| `VOCALIVE_MIC_BLOCK_MS` | `40` | Duration of each captured PCM block |
| `VOCALIVE_MIC_DEVICE` | unset | Optional input device id, device name, `default`, or `external` |
| `VOCALIVE_MIC_PREFER_EXTERNAL` | `true` | When `VOCALIVE_MIC_DEVICE` is unset, prefer a connected headset-like external mic over a built-in default input |
| `VOCALIVE_MIC_SPEECH_THRESHOLD` | `0.02` | RMS threshold for detecting speech in a block |
| `VOCALIVE_MIC_PRE_SPEECH_MS` | `200` | Keep this much audio before the first detected speech block so utterance onsets are not clipped |
| `VOCALIVE_MIC_SPEECH_HOLD_MS` | `200` | Keep an utterance "in speech" briefly after audio dips below the threshold |
| `VOCALIVE_MIC_SILENCE_MS` | `500` | Silence required before VocaLive emits the buffered utterance |
| `VOCALIVE_MIC_MIN_UTTERANCE_MS` | `250` | Minimum buffered audio before end-of-turn detection can emit |
| `VOCALIVE_MIC_MAX_UTTERANCE_MS` | `15000` | Hard cap for one buffered utterance |
| `VOCALIVE_STT_PROVIDER` | `mock` | Selected STT adapter; accepts `moonshine` and aliases such as `moonshine voice` |
| `VOCALIVE_MODEL_PROVIDER` | `mock` | Selected LLM adapter; accepts `gemini` and aliases such as `google gemini` |
| `VOCALIVE_TTS_PROVIDER` | `mock` | Selected TTS adapter; accepts `aivis` and aliases such as `aivis speech` |
| `VOCALIVE_OUTPUT_PROVIDER` | `memory` | `memory` or `speaker` |
| `VOCALIVE_CONVERSATION_LANGUAGE` | `ja` | Conversation language instruction injected before each LLM turn; set empty to disable |
| `VOCALIVE_GEMINI_API_KEY` | unset | Gemini API key |
| `VOCALIVE_GEMINI_THINKING_BUDGET` | `0` | Gemini 2.5 thinking budget; `0` disables reasoning, empty unsets |
| `VOCALIVE_MOONSHINE_MODEL` | `base` | Moonshine model arch such as `base` / `tiny`, or a concrete model id such as `base-ja` |
| `VOCALIVE_AIVIS_BASE_URL` | `http://127.0.0.1:10101` | AivisSpeech engine base URL |
| `VOCALIVE_AIVIS_SPEAKER_ID` | unset | Explicit AivisSpeech style ID |
| `VOCALIVE_AIVIS_SPEAKER_NAME` | unset | Speaker name to resolve via `/speakers` |
| `VOCALIVE_AIVIS_STYLE_NAME` | unset | Style name to resolve via `/speakers` |
| `VOCALIVE_SPEAKER_COMMAND` | `afplay {path}` | Override playback command; must include `{path}` |
| `VOCALIVE_QUEUE_MAXSIZE` | `4` | Max queued utterances waiting to run |
| `VOCALIVE_QUEUE_OVERFLOW` | `drop_oldest` | Overflow strategy: `drop_oldest` or `reject_new` |

Current provider support:

- `mock` works end-to-end for local development
- `moonshine` uses the optional `moonshine-voice` package for STT
- `VOCALIVE_MOONSHINE_MODEL=base` resolves a language-specific Moonshine model from
  `VOCALIVE_CONVERSATION_LANGUAGE`; the default configuration therefore uses `base-ja`
- `moonshine-voice` downloads the selected Moonshine model on first use and reuses the local cache after that
- `gemini` uses the Gemini `generateContent` API over HTTPS
  Current default wiring sets Gemini 2.5 `thinkingBudget=0` to reduce response latency.
- `aivis` uses the local AivisSpeech engine API at `http://127.0.0.1:10101` by default
- `speaker` output plays synthesized `.wav` / `.aiff` audio through `afplay`
- live microphone input can barge in on an active reply before the new turn is fully emitted
- Provider names are normalized case-insensitively, so values such as `Moonshine Voice`
  and `Aivis Speech` resolve to the supported adapters

## Repository layout

```text
src/vocalive/
  audio/       audio input, output, and turn detection abstractions
  config/      environment-driven runtime configuration
  llm/         language model interface and adapters
  pipeline/    orchestration, cancellation, queues, session state
  stt/         speech-to-text interface and adapters
  tts/         text-to-speech interface and adapters
  util/        logging, metrics, time helpers
  main.py      CLI entry point and default wiring

tests/unit/    queue and orchestrator coverage
docs/          architecture, development, and documentation process
AGENT.md       implementation constraints and repository intent
```

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Development guide](docs/development.md)

`AGENT.md` is the engineering brief for Codex and implementation work. `README.md` and `docs/` are the maintained documentation set for humans.
