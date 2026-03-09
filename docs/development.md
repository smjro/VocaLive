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

`moonshine-voice` is part of the `voice` extra and currently requires Python 3.10+ in this repository's runtime path.

## CLI behavior

The current entry point is `src/vocalive/main.py`.

- `VOCALIVE_INPUT_PROVIDER=stdin` keeps the text shell.
- `VOCALIVE_INPUT_PROVIDER=microphone` uses `sounddevice` and local silence detection.
- `VOCALIVE_MIC_DEVICE=external` forces VocaLive to use a connected headset-like external mic.
- `/quit`, `quit`, and `exit` stop the stdin shell.
- Assistant output is printed after the orchestrator becomes idle in both modes.
- The stdin shell sets `AudioSegment.transcript_hint`, so `moonshine`-selected configs can
  still exercise Gemini and Aivis wiring before switching to live microphone capture.

The microphone path is intentionally simple: it uses a local RMS threshold and silence timer, not a production VAD.

## Configuration

Runtime settings are loaded from `AppSettings.from_env()` in `src/vocalive/config/settings.py`.

| Variable | Default | Notes |
| --- | --- | --- |
| `VOCALIVE_SESSION_ID` | random UUID | Useful for correlating logs across one session |
| `VOCALIVE_LOG_LEVEL` | `INFO` | Passed to Python logging |
| `VOCALIVE_INPUT_PROVIDER` | `stdin` | `stdin` or `microphone` |
| `VOCALIVE_MIC_DEVICE` | unset | Optional input device id, device name, `default`, or `external` |
| `VOCALIVE_MIC_PREFER_EXTERNAL` | `true` | When no device is set explicitly, prefer a connected headset-like external mic over a built-in default input |
| `VOCALIVE_STT_PROVIDER` | `mock` | `moonshine` requires the optional `moonshine-voice` package; aliases such as `moonshine voice` are accepted |
| `VOCALIVE_MODEL_PROVIDER` | `mock` | `gemini` uses the remote Gemini API; aliases such as `google gemini` are accepted |
| `VOCALIVE_TTS_PROVIDER` | `mock` | `aivis` uses the local AivisSpeech HTTP API; aliases such as `aivis speech` are accepted |
| `VOCALIVE_OUTPUT_PROVIDER` | `memory` | `speaker` uses `afplay` |
| `VOCALIVE_CONVERSATION_LANGUAGE` | `ja` | Injects a per-turn language instruction before the LLM call; set empty to disable |
| `VOCALIVE_GEMINI_API_KEY` | unset | Required when `VOCALIVE_MODEL_PROVIDER=gemini` |
| `VOCALIVE_MOONSHINE_MODEL` | `base` | Moonshine model arch such as `base` / `tiny`, or a concrete model id such as `base-ja`; generic arch values use `VOCALIVE_CONVERSATION_LANGUAGE` |
| `VOCALIVE_AIVIS_BASE_URL` | `http://127.0.0.1:10101` | Local AivisSpeech engine base URL |
| `VOCALIVE_AIVIS_SPEAKER_ID` | unset | Preferred explicit AivisSpeech style ID |
| `VOCALIVE_AIVIS_SPEAKER_NAME` | unset | Optional speaker name for `/speakers` lookup |
| `VOCALIVE_AIVIS_STYLE_NAME` | unset | Optional style name for `/speakers` lookup |
| `VOCALIVE_SPEAKER_COMMAND` | `afplay {path}` | Override playback command; must include `{path}` |
| `VOCALIVE_QUEUE_MAXSIZE` | `4` | Bounded utterance backlog |
| `VOCALIVE_QUEUE_OVERFLOW` | `drop_oldest` | `drop_oldest` or `reject_new` |

If you add a new setting, update this table and the root `README.md`.

## Adding a new adapter

When integrating a real provider:

1. Implement the relevant base interface in `stt/`, `llm/`, `tts/`, or `audio/`.
2. Keep provider SDK calls inside the adapter module only.
3. Accept `TurnContext` and cancellation where the interface expects them.
4. Wire the adapter in `build_orchestrator()` or a future dependency assembly layer.
5. Add tests for the control-flow behavior the adapter changes.
6. Update `README.md` and `docs/architecture.md` to reflect new support.

Do not push provider-specific logic into `pipeline/orchestrator.py`.

## Testing focus

Current tests cover:

- queue overflow semantics
- single-turn orchestration flow
- interruption of an in-flight turn by a newer utterance

High-value next tests:

- adapter error propagation and recovery behavior
- cancellation during STT, LLM, and TTS stages
- session history rules for interrupted turns
- configuration parsing failures
- live device smoke tests for microphone and speaker backends

## Logging and metrics

Structured logs are emitted through `util/logging.py`.

- Log payloads are JSON strings written through standard Python logging.
- Metrics are recorded through `MetricsRecorder`.
- The default recorder is in-memory and suitable for local inspection and tests.

If you add a new pipeline stage, add both logs and latency tracking for it.

## Documentation maintenance

Development changes should update docs in the same commit.

Minimum rule:

- User-visible startup changes: update `README.md`
- Runtime flow changes: update `docs/architecture.md`
- New config or commands: update `docs/development.md`

For the full rule set, see `docs/README.md`.
