# Documentation Index

This directory is the maintained documentation set for the current repository state.

## Document map

- `../README.md`: project overview, supported runtime combinations, startup examples, and configuration matrix
- `architecture.md`: runtime flow, module boundaries, interruption model, and provider assembly
- `development.md`: local workflow, CLI behavior, configuration details, and testing guidance
- `../AGENT.md`: repository-specific implementation constraints and maintenance rules for coding work

## Source of truth

Use the code first, then keep the docs aligned with it:

- startup wiring and provider compatibility constraints: `src/vocalive/main.py`
- runtime settings, defaults, and aliases: `src/vocalive/config/settings.py`
- orchestration, interruption, and session semantics: `src/vocalive/pipeline/`
- human-facing project overview and startup guidance: `../README.md`
- repository-specific implementation brief: `../AGENT.md`

Do not duplicate low-level detail across multiple files unless it materially helps onboarding.

## Update rules

Update documentation in the same change when any of the following happens:

| Change type | Required document updates |
| --- | --- |
| Conversation flow or interruption semantics change | `architecture.md`, and `../README.md` if behavior is user-visible |
| New environment variable, default, or config meaning | `development.md`, and `../README.md` when startup behavior changes |
| New provider support, removal, or compatibility constraint | `../README.md`, `architecture.md`, `development.md` |
| Entry point or CLI behavior changes | `../README.md`, `development.md` |
| Test command or current coverage statement changes | `development.md` |
| Long-term implementation constraints change | `../AGENT.md` and any affected human-facing docs |

## Review checklist

Before closing a documentation-related change, verify:

1. The startup examples still match `src/vocalive/main.py`.
2. Provider support and compatibility constraints match the actual assembly code.
3. Configuration names and defaults match `src/vocalive/config/settings.py`.
4. Coverage statements still reflect what exists in `tests/unit/`.
5. Platform-specific or optional requirements such as `afplay` and `moonshine-voice` are called out explicitly.
6. Future ideas are labeled as future work, not present-tense behavior.

## Scope boundary

Keep docs focused on repository reality.

- Describe a provider or runtime mode in present tense only when it is actually wired from the current entry point.
- Label optional, local-only, or platform-specific behavior explicitly.
- If a path exists only as a base interface or future direction, say so directly instead of implying it is production-ready.
