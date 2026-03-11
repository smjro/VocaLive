# Architecture

## Goals

VocaLive is organized around a low-latency local conversation pipeline with explicit cancellation and replaceable providers.

Current design priorities:

1. Keep provider-specific logic behind adapter interfaces.
2. Keep queueing, interruption, session state, and observability explicit.
3. Preserve responsiveness under repeated turn interruption.
4. Make it possible to swap mock and real adapters without rewriting the orchestration layer.

## Runtime modes

The current entry point supports two input modes:

1. `stdin`
   Typed text is converted to `AudioSegment.from_text()` and submitted through the same orchestrator used by the live path.
2. `microphone`
   `MicrophoneAudioInput` captures `int16` PCM through `sounddevice`, applies preroll / speech-hold / silence thresholds, and emits utterance-sized `AudioSegment` instances.

## Current runtime flow

```text
stdin text
  -> AudioSegment(transcript_hint)
  -> ConversationOrchestrator.submit_utterance()

microphone PCM
  -> UtteranceAccumulator
  -> AudioSegment
  -> ConversationOrchestrator.submit_utterance()

shared pipeline
  -> interrupt current turn
  -> bounded ingress queue
  -> STT adapter
  -> append user message to session
  -> prepend conversation-language system instruction when configured
  -> LLM adapter
  -> split assistant text into sentence-sized chunks
  -> emit conversation events for optional UI sinks
  -> synthesize first chunk
  -> prefetch next chunk while current chunk is playing
  -> AudioOutput.play()
  -> optional overlay reveals the active chunk while playback runs
  -> overlay hides captions again when playback completes or is interrupted
  -> append assistant message to session after playback completes
```

The orchestration logic lives in `src/vocalive/pipeline/orchestrator.py`.

## Core modules

| Area | Responsibility |
| --- | --- |
| `audio/devices.py` | Input device resolution, default-device lookup, and headset/external microphone preference |
| `audio/input.py` | Stdin-like queue input, microphone capture, utterance accumulation, and speech-start callbacks |
| `audio/output.py` | Playback abstraction, in-memory output, and external speaker command playback |
| `audio/vad.py` | Turn detection abstraction; current live path uses fixed-silence detection |
| `stt/` | Speech-to-text interface and adapters |
| `llm/` | Language model interface and adapters |
| `tts/` | Text-to-speech interface and adapters |
| `pipeline/queues.py` | Bounded ingress queue with explicit overflow policy |
| `pipeline/interruption.py` | Cancellation token and active-turn interruption control |
| `pipeline/session.py` | Ordered conversation history for one session |
| `pipeline/events.py` | Conversation lifecycle events emitted to optional presentation sinks |
| `pipeline/orchestrator.py` | End-to-end turn execution and playback chunk prefetch |
| `config/settings.py` | Environment-driven settings, defaults, aliases, and provider normalization |
| `ui/` | Local HTTP/SSE overlay server, transparent character overlay, and asset loading |
| `util/logging.py` | Structured JSON log helpers |
| `util/metrics.py` | In-memory stage latency recording |

## Domain model

`src/vocalive/models.py` defines the provider-agnostic objects shared across modules:

- `AudioSegment`: utterance-sized audio payload plus metadata, including optional `transcript_hint`
- `Transcription`: normalized STT output
- `ConversationMessage`: stored session history item
- `ConversationRequest`: snapshot of the current conversation passed to the LLM
- `AssistantResponse`: normalized LLM output
- `SynthesizedSpeech`: TTS output for playback
- `TurnContext`: session and turn identifiers for logs and metrics

## Queueing and interruption

The pipeline is built around a bounded ingress queue.

- queue capacity comes from `QueueSettings.ingress_maxsize`
- overflow policy is explicit: `drop_oldest` or `reject_new`
- `submit_utterance()` interrupts the currently active turn before queue insertion
- in microphone mode, speech onset calls `handle_user_speech_start()` so stale playback can stop before the next utterance is fully emitted
- playback backends receive a `CancellationToken` and must stop quickly when a turn is cancelled
- `drop_oldest` keeps the newest utterance by discarding the oldest queued item
- `reject_new` preserves queued work and refuses the new utterance

This prevents unbounded backlog growth and avoids finishing obsolete replies after the user has moved on.

## Session handling

`ConversationSession` stores the ordered conversation history for one session.

- user messages are appended after STT completes
- the LLM receives a snapshot of the current session plus an optional conversation-language system instruction
- assistant messages are appended only after the full reply has been synthesized and played
- interrupted assistant replies are therefore not committed to session history

This commit policy is important. Changes that alter it should be treated as behavior changes and documented.

## Provider assembly

The current adapter assembly happens in `build_orchestrator()` inside `src/vocalive/main.py`.

Default development assembly:

- `MockSpeechToTextEngine`
- `EchoLanguageModel`
- `MockTextToSpeechEngine`
- `MemoryAudioOutput`

Optional real adapters:

- `MoonshineSpeechToTextEngine`
- `GeminiLanguageModel`
- `AivisSpeechTextToSpeechEngine`
- `SpeakerAudioOutput`

Optional presentation path:

- `OverlayServer` subscribes to orchestrator events and serves a local browser overlay
- the overlay reveals each playback chunk progressively over the chunk's estimated playback duration
- the overlay currently renders only the character and assistant speech text; captions are shown in front of the lower body so the face stays visible
- captions are visible only while the assistant is actively speaking and clear on completion or interruption
- the overlay is driven by sentence-sized playback chunks, not token streaming from the model

Current compatibility constraints:

- microphone input is rejected when STT is still `mock`
- speaker output is rejected unless TTS is `aivis`
- speaker playback depends on an external playback command and defaults to `afplay` on macOS

The stdin shell still works with the real-provider assembly because `AudioSegment.from_text()` sets `transcript_hint`, and the Moonshine adapter short-circuits to that hint before touching the backend.

## Observability

Structured logs are emitted for:

- `microphone_stream_started`
- `microphone_stream_closed`
- `queue_overflow`
- `turn_interrupted`
- `turn_cancelled`
- `turn_failed`
- `transcription_ready`
- `response_ready`
- `event_sink_failed`

The orchestrator records latency for:

- `stt`
- `llm`
- `tts`
- `playback`
- `turn_total`

The current metrics recorder is in-memory and suitable for local development and tests.

## Current limitations

The following are not implemented today:

- streaming partial STT results
- streaming token output from the LLM
- streaming audio output from TTS
- token-accurate text reveal; the current overlay is synchronized at sentence/chunk granularity
- echo cancellation / duplex coordination between microphone and speaker
- persistent metrics export
- restartable worker isolation between pipeline stages
- generic retry / backoff infrastructure

The current codebase is therefore a solid local runtime foundation, not a finished production voice stack.
