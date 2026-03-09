# Architecture

## Goals

The repository is organized around a low-latency, replaceable, long-running conversation pipeline.

Current design priorities:

1. Keep provider-specific code behind adapter interfaces.
2. Keep queueing, interruption, session state, and observability explicit.
3. Make it possible to swap mock implementations for real ones without rewriting the pipeline.

## Current runtime flow

The current runtime supports either stdin simulation or live microphone capture:

```text
stdin text or microphone PCM
  -> AudioSegment
  -> ConversationOrchestrator.submit_utterance()
  -> bounded ingress queue
  -> STT adapter
  -> session state update
  -> conversation-language system instruction
  -> LLM adapter
  -> sentence-level TTS synthesis with next-sentence prefetch
  -> AudioOutput.play()
```

The orchestrator is implemented in `src/vocalive/pipeline/orchestrator.py` and is the main coordination point.

## Core modules

| Area | Responsibility |
| --- | --- |
| `audio/devices.py` | Input device resolution and external microphone selection |
| `audio/input.py` | Stdin queue input and live microphone capture |
| `audio/output.py` | Playback abstraction, memory output, and speaker playback |
| `audio/vad.py` | Turn detection abstraction |
| `stt/` | Speech-to-text interface and adapters |
| `llm/` | Language model interface and adapters |
| `tts/` | Text-to-speech interface and adapters |
| `pipeline/queues.py` | Bounded queue with explicit overflow policy |
| `pipeline/interruption.py` | Turn cancellation token and interruption control |
| `pipeline/session.py` | Conversation history for the current session |
| `pipeline/orchestrator.py` | End-to-end turn execution |
| `util/logging.py` | Structured log emission |
| `util/metrics.py` | Per-stage latency recording |

## Domain model

`src/vocalive/models.py` defines the shared objects passed between modules:

- `AudioSegment`: utterance-sized audio payload plus metadata
  including optional `transcript_hint` for stdin-driven simulation
- `Transcription`: normalized STT output
- `ConversationMessage`: stored conversation history item
- `ConversationRequest`: snapshot of the current session for the LLM
- `AssistantResponse`: LLM output
- `SynthesizedSpeech`: TTS output for playback
- `TurnContext`: session and turn identifier for logs and metrics

These models are intentionally provider-agnostic.

## Queueing and interruption

The pipeline is built around a bounded ingress queue.

- Queue capacity is configured through `QueueSettings.ingress_maxsize`.
- Overflow policy is explicit: `drop_oldest` or `reject_new`.
- Every new utterance interrupts the currently active turn.
- In microphone mode, local speech onset can interrupt the active turn before
  silence-based end-of-turn detection finishes emitting the new utterance.
- Playback receives a cancellation token so stale speech can stop quickly.
- Assistant responses are split into sentence-sized playback chunks when possible.
- While one synthesized sentence is playing, the orchestrator prepares the next one in the background.

This prevents unbounded backlog growth and avoids finishing old responses after the user has moved on.

## Session handling

`ConversationSession` stores the ordered list of user and assistant messages for one session.

- User messages are appended after STT completes.
- Assistant messages are appended only after TTS playback completes.
- A snapshot is passed to the LLM for each turn.
- When `VOCALIVE_CONVERSATION_LANGUAGE` is set, the orchestrator prepends a system
  instruction so the LLM keeps replies in the configured language unless the user asks to switch.

This means interrupted assistant responses do not become committed session history.

## Observability

The orchestrator records latency for:

- `stt`
- `llm`
- `tts`
- `playback`
- `turn_total`

Structured log events are emitted for:

- transcription completion
- response completion
- queue overflow
- turn interruption request
- turn cancellation
- turn failure

The current metrics recorder is in-memory and suitable for tests and local development. A future persistent or external recorder can be injected without changing pipeline control flow.

Live microphone capture resolves the input device at startup. The current logic supports:

- explicit input device ids or names
- `external` selection for headset-like or other external microphones
- automatic fallback from a built-in default mic to a connected headset-like external mic

## Provider boundaries

Current adapter interfaces:

- `SpeechToTextEngine`
- `LanguageModel`
- `TextToSpeechEngine`
- `AudioOutput`

Current concrete implementations:

- `MockSpeechToTextEngine`
- `MoonshineSpeechToTextEngine`
- `EchoLanguageModel`
- `GeminiLanguageModel`
- `MockTextToSpeechEngine`
- `AivisSpeechTextToSpeechEngine`
- `MemoryAudioOutput`
- `SpeakerAudioOutput`

`MoonshineSpeechToTextEngine` keeps the STT backend behind the shared interface while resolving a language-specific `moonshine-voice` model from configuration.

## Known limitations

The following are not implemented yet:

- streaming partial STT results
- streaming token output from the LLM
- streaming audio output from TTS
- restartable component workers
- persistent metrics export
- echo cancellation / duplex coordination between microphone and speaker

The current code is therefore a foundation, not a finished real-time voice product.
