# Documentation Index

This directory is the maintained documentation set for the repository.

## Document map

- `../README.md`: project overview, quick start, current status
- `architecture.md`: runtime design, module boundaries, and extension points
- `development.md`: local workflow, test commands, configuration, and adapter integration notes
- `../AGENT.md`: implementation constraints and project direction used during coding work

## Source of truth

- Public repository overview lives in `../README.md`
- Runtime behavior and module responsibilities live in `architecture.md`
- Developer workflow and setup live in `development.md`
- Product and architectural intent for implementation work lives in `../AGENT.md`

Do not duplicate the same detail in multiple files unless it is necessary for onboarding. Prefer linking to the detailed document instead.

## Update rules

Update the documentation in the same change when any of the following happens:

| Change type | Required document updates |
| --- | --- |
| Conversation flow changes | `architecture.md`, and `../README.md` if behavior is user-visible |
| New environment variable or config semantics | `development.md`, and `../README.md` if needed for startup |
| New provider support or removal | `../README.md`, `architecture.md`, `development.md` |
| New entry point or CLI behavior | `../README.md`, `development.md` |
| Test command changes | `development.md` |
| Change to long-term project intent | `../AGENT.md` and any impacted human-facing docs |

## Review checklist

Before closing a documentation-related change, verify:

1. The quick start in `../README.md` still works as written.
2. Provider support matches the actual code paths.
3. Architecture text matches the runtime flow in `src/vocalive/pipeline/`.
4. Configuration names in docs match `src/vocalive/config/settings.py`.
5. Any placeholder or unimplemented integration is clearly labeled as such.

## Scope boundary

Keep the documents focused on repository reality. Do not describe microphone, speaker, Gemini, or Moonshine behavior as implemented until code exists for those paths.
