# Architecture

## Goals

VocaLive is organized around a low-latency local conversation pipeline with explicit cancellation and replaceable providers.

Current design priorities:

1. Keep provider-specific logic behind adapter interfaces.
2. Keep queueing, interruption, session state, and observability explicit.
3. Preserve responsiveness under repeated turn interruption.
4. Make it possible to swap mock and real adapters without rewriting the orchestration layer.

## Runtime modes

The current entry point supports two primary input modes plus an optional extra live source:

1. `stdin`
   Typed text is converted to `AudioSegment.from_text()` and submitted through the same orchestrator used by the live path.
2. `microphone`
   `MicrophoneAudioInput` captures `int16` PCM through `sounddevice`, applies preroll / speech-hold / silence thresholds, and emits utterance-sized `AudioSegment` instances.
3. `application audio` (optional)
   `MacOSApplicationAudioInput` captures one running macOS app through a ScreenCaptureKit helper, applies adaptive speech detection by default plus local utterance segmentation, and emits utterance-sized `AudioSegment` instances tagged as application audio. The default `context_only` mode stores those transcripts as session context without immediate assistant replies; `respond` opt-in restores live-turn behavior.

## Current runtime flow

```text
stdin text
  -> AudioSegment(transcript_hint)
  -> ConversationOrchestrator.submit_utterance()

microphone PCM
  -> UtteranceAccumulator
  -> AudioSegment
  -> ConversationOrchestrator.submit_utterance()

application audio PCM
  -> SpeechDetector (adaptive by default)
  -> UtteranceAccumulator
  -> AudioSegment(source=application_audio)
  -> if app-audio mode is context_only:
       low-priority application-context queue
       -> STT adapter
       -> append application-context message to session
     else:
       ConversationOrchestrator.submit_utterance()

shared pipeline
  -> interrupt current turn
  -> bounded ingress queue
  -> briefly debounce microphone user turns and merge compatible follow-up utterances
  -> STT adapter
  -> append user or application-context message to session
  -> optionally suppress low-value microphone chatter before screen capture / LLM / TTS
  -> optionally capture the configured window for the current turn when a trigger phrase matches
  -> compact older user/assistant history into one bounded summary while keeping a recent raw-message window
  -> compact older application-audio context into a separate bounded summary while keeping a recent raw app-context window
  -> prepend conversation-language system instruction when configured
  -> LLM adapter
  -> split assistant text into sentence-sized chunks
  -> synthesize first chunk
  -> prefetch next chunk while current chunk is playing
  -> AudioOutput.play()
  -> append assistant message to session after playback completes
```

The orchestration logic lives in `src/vocalive/pipeline/orchestrator.py`.

## Core modules

| Area | Responsibility |
| --- | --- |
| `audio/devices.py` | Input device resolution, default-device lookup, and headset/external microphone preference |
| `audio/input.py` | Stdin-like queue input, microphone capture, combined live-input fan-in, utterance accumulation, and speech-start callbacks |
| `audio/macos_application.py` | macOS application-audio helper build, app lookup, capture, and utterance emission |
| `audio/speech_detection.py` | Fixed-threshold and adaptive speech detectors used before utterance segmentation |
| `audio/output.py` | Playback abstraction, in-memory output, and external speaker command playback |
| `audio/vad.py` | Turn detection abstraction; current live path uses fixed-silence detection |
| `stt/` | Speech-to-text interface and adapters, including Moonshine application-audio enhancement |
| `llm/` | Language model interface and adapters |
| `screen/` | Optional named-window screenshot capture adapters |
| `tts/` | Text-to-speech interface and adapters |
| `pipeline/queues.py` | Bounded ingress queue with explicit overflow policy |
| `pipeline/interruption.py` | Cancellation token and active-turn interruption control |
| `pipeline/session.py` | Ordered conversation history for one session |
| `pipeline/orchestrator.py` | End-to-end turn execution and playback chunk prefetch |
| `config/settings.py` | Environment-driven settings, defaults, aliases, and provider normalization |
| `util/logging.py` | Structured JSON log helpers |
| `util/metrics.py` | In-memory stage latency recording |

## Domain model

`src/vocalive/models.py` defines the provider-agnostic objects shared across modules:

- `AudioSegment`: utterance-sized audio payload plus metadata, including optional `transcript_hint` and source tagging
- `Transcription`: normalized STT output
- `ConversationMessage`: stored session history item, including `application` role entries for app-audio context
- `ConversationRequest`: snapshot of the current conversation plus optional current-turn multimodal parts passed to the LLM
- `AssistantResponse`: normalized LLM output
- `SynthesizedSpeech`: TTS output for playback
- `TurnContext`: session and turn identifiers for logs and metrics

## Queueing and interruption

The pipeline is built around a bounded user-turn queue plus a low-priority application-context queue used by app audio in `context_only` mode.

- queue capacity comes from `QueueSettings.ingress_maxsize`
- overflow policy is explicit: `drop_oldest` or `reject_new`
- `submit_utterance()` interrupts the currently active turn before queue insertion
- application audio in `context_only` mode is routed into the low-priority queue and does not interrupt active playback or trigger LLM/TTS
- microphone speech onset calls `handle_user_speech_start()` so stale playback can stop before the next utterance is fully emitted
- application-audio speech onset calls `handle_user_speech_start()` only in `respond` mode
- playback backends receive a `CancellationToken` and must stop quickly when a turn is cancelled
- `drop_oldest` keeps the newest utterance by discarding the oldest queued item
- `reject_new` preserves queued work and refuses the new utterance

This prevents unbounded backlog growth and avoids finishing obsolete replies after the user has moved on.

## Session handling

`ConversationSession` stores the ordered conversation history for one session.

- user messages are appended after STT completes
- application-audio transcripts are appended after STT completes as labeled `application` context messages, not user messages
- in the default `context_only` mode, those application messages do not immediately trigger LLM/TTS; they are consumed on the next user-driven turn
- the LLM receives a compacted session view: recent user/assistant raw messages, one bounded earlier-conversation summary, recent application-audio raw messages, one bounded earlier application-audio summary, and an optional conversation-language system instruction
- screen captures are request-scoped extras for the current user turn only and are not persisted in session history
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
- `MacOSFullscreenScreenCapture`
- `AivisSpeechTextToSpeechEngine`
- `SpeakerAudioOutput`

Current compatibility constraints:

- microphone or application-audio input is rejected when STT is still `mock`
- application-audio input is rejected unless `VOCALIVE_APP_AUDIO_TARGET` is configured
- application-audio input currently supports macOS only and depends on Screen Recording permission plus a first-run helper build
- speaker output is rejected unless TTS is `aivis`
- screen capture is rejected unless the model provider is `gemini`
- screen capture is rejected unless `VOCALIVE_SCREEN_WINDOW_NAME` is configured
- screen capture currently supports macOS only and resolves the first on-screen window that matches the configured title or owner name
- speaker playback depends on an external playback command and defaults to `afplay` on macOS

The stdin shell still works with the real-provider assembly because `AudioSegment.from_text()` sets `transcript_hint`, and the Moonshine adapter short-circuits to that hint before touching the backend.

When `segment.source == "application_audio"`, the current Moonshine adapter applies low-frequency-preserving enhancement with a gentle presence boost, soft gate, short edge padding, and normalization before transcription. This is intentionally scoped to application audio so the microphone path stays unchanged.

## Observability

Structured logs are emitted for:

- `microphone_stream_started`
- `microphone_stream_closed`
- `application_audio_stream_started`
- `application_audio_stream_closed`
- `queue_overflow`
- `turn_interrupted`
- `turn_cancelled`
- `turn_failed`
- `response_suppressed`
- `screen_capture_ready`
- `screen_capture_failed`
- `transcription_ready`
- `response_ready`

The orchestrator records latency for:

- `screen_capture` when the feature is triggered
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
- echo cancellation / duplex coordination between microphone and speaker
- persistent metrics export
- restartable worker isolation between pipeline stages
- generic retry / backoff infrastructure

The current codebase is therefore a solid local runtime foundation, not a finished production voice stack.
