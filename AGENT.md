# AGENT.md

## Project snapshot

VocaLive is a local, adapter-based voice conversation runtime.

Current entry point:

- `src/vocalive/main.py`

Current runtime modes:

- `stdin` shell for local development and provider wiring checks
- `microphone` capture via `sounddevice` with local utterance detection

Current default assembly:

- STT: `MockSpeechToTextEngine`
- LLM: `EchoLanguageModel`
- TTS: `MockTextToSpeechEngine`
- Output: `MemoryAudioOutput`

Current optional real adapters:

- STT: `MoonshineSpeechToTextEngine`
- LLM: `GeminiLanguageModel`
- TTS: `AivisSpeechTextToSpeechEngine`
- Output: `SpeakerAudioOutput`

Current hard constraints in the shipped app assembly:

- `VOCALIVE_INPUT_PROVIDER=microphone` requires a real STT adapter, currently `moonshine`
- `VOCALIVE_OUTPUT_PROVIDER=speaker` currently requires `VOCALIVE_TTS_PROVIDER=aivis`
- speaker playback uses `afplay` by default on macOS unless `VOCALIVE_SPEAKER_COMMAND` is set

## Current behavior that changes must preserve

The repository is not a blank prototype. These behaviors already exist and should stay stable unless the task explicitly changes them:

1. New utterances interrupt the currently active turn.
2. The ingress queue is bounded and overflow behavior is explicit.
3. Microphone speech start can interrupt stale playback before end-of-turn emission.
4. Assistant responses are split into sentence-sized chunks for playback.
5. TTS for the next sentence is prefetched while the current sentence is playing.
6. User messages are committed to session history after STT.
7. Assistant messages are committed only after playback completes.
8. Interrupted assistant replies do not become committed history.
9. Structured logs and per-stage latency metrics are emitted around the pipeline.

## Working priorities

Optimize for these in order:

1. Low latency for live conversation
2. Correct interruption and cancellation behavior
3. Long-session stability
4. Clear adapter boundaries
5. Debuggability and observability
6. Small, maintainable changes over clever rewrites

## Architecture guardrails

Keep these concerns separated:

- audio input and device selection
- utterance detection / turn detection
- STT adapters
- conversation session state
- LLM adapters
- TTS adapters
- audio output / playback
- orchestration and interruption control
- configuration
- logging and metrics

Do not move provider-specific HTTP or SDK logic into `pipeline/orchestrator.py`.

Prefer extending the existing boundaries in:

- `src/vocalive/audio/`
- `src/vocalive/stt/`
- `src/vocalive/llm/`
- `src/vocalive/tts/`
- `src/vocalive/pipeline/`
- `src/vocalive/config/settings.py`

## Configuration rules

Runtime configuration is environment-driven through `AppSettings.from_env()`.

When adding or changing behavior:

- add a typed setting instead of scattering constants
- normalize provider aliases in `settings.py` if a new alias is meant to be supported
- keep defaults aligned with the actual local development path
- document any new compatibility constraint between providers

If a config name, default, or meaning changes, update:

- `README.md`
- `docs/development.md`
- any architecture text affected by the behavior change

## Provider rules

Adapters should remain swappable.

When adding a provider:

1. implement the relevant base interface
2. keep all provider-specific logic inside the adapter module
3. respect `CancellationToken` where the interface expects it
4. wire the adapter in `build_orchestrator()` or a future assembly layer
5. add tests for the control-flow changes introduced by the provider

Do not document a provider as supported until it is actually wired from the current entry point.

## Observability expectations

The current pipeline records:

- metrics for `stt`, `llm`, `tts`, `playback`, and `turn_total`
- structured logs for queue overflow, interruption, cancellation, transcription, response completion, and failures

If you add a new pipeline stage or a new long-lived resource, add:

- a log event for success and failure when appropriate
- latency measurement if the stage affects responsiveness
- clean startup and shutdown behavior

Never log secrets or raw credentials.

## Testing expectations

For non-trivial changes, update or add tests.

Current high-value areas already covered by unit tests:

- settings parsing and provider alias normalization
- queue overflow behavior
- microphone utterance accumulation and device resolution
- Moonshine model resolution and transcript-hint behavior
- Gemini payload shaping
- Aivis speaker/style selection
- orchestrator interruption, playback chunking, and session rules
- structured logging serialization

Add regression coverage when fixing bugs in latency-sensitive or cancellation-sensitive code.

## Dependency policy

Prefer the standard library unless a new dependency clearly improves the implementation.

Be conservative about adding packages that:

- complicate packaging for local desktop use
- introduce background services or heavy frameworks
- make provider swapping harder

## Current limitations

These are not implemented today and should not be described as already available:

- streaming partial STT results
- streaming token output from the LLM
- streaming audio output from TTS
- echo cancellation / full-duplex coordination
- persistent metrics export
- separate restartable worker processes
- generic retry/backoff infrastructure

## Documentation contract

Keep docs aligned with repository reality.

Update docs in the same change when you modify:

- startup flow or CLI behavior
- environment variables or defaults
- supported provider combinations
- queue, interruption, or session semantics
- architecture or module boundaries

Future ideas are fine, but label them clearly as future work rather than current behavior.
