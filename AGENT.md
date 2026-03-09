# AGENT.md

## Project overview

This repository is a real-time voice conversation system for talking with an AI.

Primary conversation flow:

1. Human speaks
2. STT converts speech to text
3. Gemini 2.5 Flash generates a response
4. TTS synthesizes voice
5. Audio is played back immediately
6. Repeat with low latency

Current intended components mentioned by the owner:

- STT: configurable speech-to-text engine
- LLM: Gemini 2.5 Flash
- TTS: AivisSpeech
- Runtime target: local PC, real-time conversational use
- Main goal: natural, responsive, low-latency voice interaction

Important:
- The repository may evolve and component choices may change.
- Do not hardcode assumptions that make STT/TTS/LLM vendors impossible to swap.
- Prefer an adapter-based architecture.

---

## Mission for Codex

When working on this repository, optimize for the following in order:

1. **Low latency**
2. **Reliability during long-running sessions**
3. **Simple debuggability**
4. **Clear module boundaries**
5. **Ease of swapping STT / TTS / LLM providers**
6. **Maintainable code over clever code**

This is not a prototype-only repo.
Implement changes as if this project will continue to grow.

---

## Product goals

The desired user experience is:

- The AI responds quickly enough to feel conversational
- Long sessions do not degrade badly over time
- Audio interruptions, backlog, and queue explosions are controlled
- Failures are visible and recoverable
- Individual components can be restarted or replaced independently
- Logs are sufficient to diagnose latency and quality issues

---

## Non-goals

Unless explicitly requested, do not:

- Add heavy frameworks without strong reason
- Introduce complex distributed infrastructure
- Add cloud dependencies beyond the required model providers
- Rewrite the whole project when a local improvement is sufficient
- Optimize prematurely at the cost of readability
- Change public behavior drastically without documenting it

---

## Core architectural principles

### 1. Keep modules separated

Structure the code so these concerns remain isolated:

- audio input capture
- VAD / turn detection
- STT adapter
- conversation/session state
- LLM adapter
- TTS adapter
- audio output playback
- orchestration / pipeline control
- metrics / logging
- configuration

### 2. Prefer adapters over vendor-specific logic

Use interfaces or abstract base classes where appropriate.

Examples:

- `SpeechToTextEngine`
- `LanguageModel`
- `TextToSpeechEngine`
- `AudioInput`
- `AudioOutput`

Vendor-specific SDK logic should stay inside adapter modules.

### 3. Design for real-time flow, not batch flow

Assume the app is long-running and interactive.
Prioritize:

- streaming where possible
- bounded queues
- cancellation support
- interruption handling
- backpressure
- partial output support when feasible

### 4. Make latency observable

Every major stage should be measurable.

Track at least:

- microphone capture start/end
- utterance end detection time
- STT start/end
- LLM request start/first token/end
- TTS request start/first audio/end
- playback start/end
- total turn latency

If a metrics system does not exist yet, add lightweight structured logging.

---

## Preferred repository direction

Codex should guide the project toward this shape if practical:

```text
src/
  audio/
    input.py
    output.py
    vad.py
    devices.py
  stt/
    base.py
    <provider>.py
  llm/
    base.py
    gemini.py
  tts/
    base.py
    aivis.py
  pipeline/
    orchestrator.py
    session.py
    queues.py
    interruption.py
  config/
    settings.py
  util/
    logging.py
    metrics.py
    retry.py
    time.py
  main.py

tests/
  unit/
  integration/

This exact layout is not mandatory, but separation of concerns is.

⸻

Coding standards

General
	•	Prefer Python 3.11+ style if the repo allows it
	•	Use type hints on all new or modified public functions
	•	Write small functions with clear responsibilities
	•	Avoid giant files and giant classes
	•	Avoid hidden global state
	•	Use dependency injection where practical
	•	Prefer explicit configuration over magic constants

Readability
	•	Write code for future maintenance
	•	Use clear names, not abbreviations unless standard
	•	Add comments only where the intent is non-obvious
	•	Do not add decorative comments
	•	Keep control flow straightforward

Error handling
	•	Fail loudly in development
	•	Fail gracefully in runtime paths where recovery is possible
	•	Never silently swallow exceptions unless explicitly justified
	•	Include context in raised/logged errors
	•	Distinguish retryable vs non-retryable failures

Async/concurrency

If the project uses async code:
	•	Stay consistent with the existing async model
	•	Do not mix threading/async/processes unnecessarily
	•	Be careful with blocking SDK calls inside async code
	•	Offload blocking work explicitly when needed
	•	Ensure shutdown and cancellation are handled cleanly

If the project is sync-based:
	•	Do not introduce async casually
	•	Only introduce concurrency where it clearly improves latency or isolation

⸻

Audio pipeline requirements

This project is audio-first. Protect responsiveness.

Must-haves
	•	bounded queues for audio/text tasks
	•	queue overflow strategy must be explicit
	•	interruption/cancel support for stale responses
	•	prevention of unbounded memory growth
	•	clear ownership of audio resources
	•	clean release of microphone/speaker handles

Strongly preferred
	•	VAD or turn-end detection abstraction
	•	ability to stop TTS playback when a new user turn starts
	•	optional streaming response path
	•	optional partial transcript handling

Avoid
	•	storing raw audio forever in memory
	•	letting old TTS continue after user interruption
	•	single giant loop with all responsibilities mixed together

⸻

Configuration rules

Configuration should live outside business logic as much as possible.

Use env vars, config files, or a typed settings layer for things like:
	•	API keys
	•	model names
	•	audio device names or ids
	•	sample rates
	•	timeout values
	•	retry counts
	•	queue sizes
	•	log levels
	•	VAD thresholds
	•	streaming toggles

Do not scatter constants across many files.

⸻

Logging and metrics

Use structured logs when possible.

Every important operation should log enough to answer:
	•	What happened?
	•	How long did it take?
	•	Which provider/component was involved?
	•	Was the result partial, complete, cancelled, or failed?

At minimum, log:
	•	startup config summary without secrets
	•	provider initialization
	•	turn lifecycle
	•	retries
	•	cancellations
	•	playback interruptions
	•	provider failures
	•	degraded mode fallbacks

Never log secrets.

⸻

Testing expectations

For non-trivial changes, add or update tests.

Favor these test types
	1.	Unit tests
	•	parsing
	•	config loading
	•	queue behavior
	•	interruption logic
	•	retry logic
	•	adapter contract behavior
	2.	Integration tests
	•	end-to-end turn orchestration with mocked providers
	•	timeout behavior
	•	cancellation behavior
	•	queue overflow handling

Testing rules
	•	Do not require real cloud APIs in normal tests
	•	Mock provider SDK/network calls
	•	Keep tests deterministic
	•	Add regression tests for bugs you fix

⸻

Performance guidance

When improving performance, prefer this order:
	1.	remove redundant work
	2.	reduce blocking points
	3.	stream results earlier
	4.	bound queues and stale work
	5.	reduce serialization/copying
	6.	optimize hot paths only after measuring

Do not claim performance improvement without either:
	•	measurement, or
	•	a very obvious architectural reason

If you change latency-sensitive code, add timing logs or benchmarks where reasonable.

⸻

Long-session stability guidance

The owner specifically cares about degradation over long runs.

When touching long-lived flows, check for:
	•	memory leaks
	•	queue buildup
	•	unreleased audio buffers
	•	repeated model/session reinitialization
	•	reconnection behavior
	•	task accumulation
	•	duplicated callbacks/listeners
	•	file descriptor / handle leaks

Favor designs that remain stable over hours, not just minutes.

⸻

Dependency policy

Before adding a new dependency, ask:
	•	Is the standard library enough?
	•	Is this dependency actively maintained?
	•	Does it materially simplify the implementation?
	•	Does it add startup/runtime overhead?
	•	Does it complicate packaging on Windows?

Avoid adding dependencies for trivial helpers.

⸻

Documentation expectations

When making a meaningful architectural or behavior change, also update:
	•	README
	•	example config/env documentation
	•	module docstrings if needed
	•	migration notes if behavior changed

Do not leave the repo in a state where the code and docs disagree.

⸻

Secrets and security
	•	Never hardcode API keys or tokens
	•	Never commit secrets
	•	Use environment variables or secret loading mechanisms
	•	Redact sensitive values from logs
	•	Treat microphone/audio data as sensitive user data
	•	Do not add telemetry without explicit request

⸻

How to make changes

Implementation planning and progress tracking

Before making non-trivial changes:
	•	create a short implementation plan with concrete steps
	•	make progress visible while working
	•	update step status as work advances
	•	note scope changes or newly discovered blockers
	•	close the task by stating what is complete and what remains, if anything

For small changes, keep the plan lightweight, but still make progress explicit.

⸻

When asked to implement something, follow this process:
	1.	Understand the current architecture first
	2.	Create a short implementation plan with explicit steps
	3.	Track progress clearly and keep statuses updated while working
	4.	Identify the smallest clean change that solves the problem
	5.	Preserve working behavior unless change is requested
	6.	Implement with clear module boundaries
	7.	Add or update tests
	8.	Update docs if behavior/config changes
	9.	Summarize what changed, progress completed, and any tradeoffs

When making multi-file changes, keep them coherent and minimal.

⸻

When fixing bugs

Always try to identify:
	•	root cause
	•	symptom
	•	reproduction conditions
	•	whether the bug is timing-related, state-related, or provider-related

Do not apply a band-aid fix if a small structural fix is possible.

Add a regression test when practical.

⸻

When implementing new providers

For STT / LLM / TTS provider additions:
	•	conform to existing adapter interfaces
	•	keep provider-specific config isolated
	•	normalize outputs into internal data structures
	•	map provider errors into consistent internal exceptions if useful
	•	document required env vars and model names
	•	avoid leaking SDK-specific types outside the adapter

⸻

Definition of done

A change is considered done when:
	•	it solves the requested problem
	•	code is understandable
	•	logs/errors are sufficient
	•	tests pass or are updated appropriately
	•	docs/config examples are updated if needed
	•	no obvious architectural damage was introduced

⸻

Preferred response style from Codex

When proposing or applying changes, be:
	•	concise
	•	concrete
	•	technically honest
	•	explicit about tradeoffs
	•	explicit about assumptions

Do not give vague reassurance.
Do not pretend something was verified if it was not.

⸻

Things Codex should proactively improve

If relevant to the requested task, Codex may proactively improve:
	•	missing type hints in touched code
	•	poor error messages
	•	missing timeout handling
	•	missing cancellation hooks
	•	queue bound enforcement
	•	logging around latency-sensitive stages
	•	docstrings for non-obvious interfaces
	•	tests around touched behavior

Keep scope reasonable.

⸻

Things Codex should not do without explicit request

Do not do these unless asked:
	•	large-scale renames across the whole repo
	•	switching frameworks
	•	replacing the entire concurrency model
	•	changing all formatting tooling
	•	moving every file to a new architecture at once
	•	adding Docker/Kubernetes/CI from scratch unless clearly useful to the current task

⸻

If repository state is inconsistent

If you notice contradictions between docs, code, and config:
	1.	trust the running code path more than stale docs
	2.	mention the inconsistency
	3.	fix nearby documentation if part of the requested work
	4.	avoid speculative refactors without evidence

⸻

Recommended internal priorities for this repository

When in doubt, optimize for:
	•	conversational responsiveness
	•	clean interruption behavior
	•	stable long-session operation
	•	provider swappability
	•	developer clarity

⸻

Maintainer intent summary

This project is intended to become a practical real-time AI voice conversation system.
Treat it like a maintainable product, not a throwaway demo.

The most valuable improvements are the ones that:
	•	reduce perceived latency
	•	prevent degradation over time
	•	make debugging easier
	•	keep the architecture modular
