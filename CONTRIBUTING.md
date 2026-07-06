# Contributing to specllm

## Design Philosophy

**Make it as simple as possible for users.** Every decision starts here. If a feature requires boilerplate, understanding internals, or configuring something that could be inferred — the design is wrong.

**Less code is better code.** Every line is debt. A feature that makes the codebase smaller is better than one that makes it larger. Before adding, ask: can I delete something instead?

The ideal experience: write an OpenAPI spec, run one line of Python, get a working LLM-powered API.

## Principles

### How we design

**1. Start from the user's pain, not the technical solution.**
Before writing code, describe what the user currently suffers through. Then find the smallest change that eliminates that pain.

**2. The spec is the single source of truth.**
All configuration lives in the OpenAPI spec via `x-specllm-*` fields. Python arguments exist only as overrides. If it can be in the spec, it must be.

**3. Flexibility belongs at the API level.**
When users need control (which model, which constraints), express it in the spec — not in application code. The spec is what they already write. Meet them there.

**4. Infer, don't ask.**
If information can be derived, don't make the user provide it. The model name tells you the provider. The request body tells you the constraints.

**5. One mechanism per concept.**
If there are two ways to do the same thing, pick one. Spec-declarative wins over programmatic.

### How we code

**6. Every line must justify its existence.**
After building a feature, go through every line and ask: what breaks if I delete this? If nothing breaks — delete it.

**7. No aspirational code.**
Don't add parameters, classes, fields, or error codes for features that don't exist yet. Add them the day they're needed.

**8. No single-function modules.**
If a module contains one function, inline it into its consumer. A new file needs distinct, non-trivial logic.

**9. Shared logic belongs in one place.**
If three code paths do the same thing, extract a helper. Copy-paste is a design failure.

**10. Every stored field must have a reader.**
If `self.x` is only used during `__init__`, use a local variable.

**11. Dependencies: zero required.**
stdlib only. Optional deps (PyYAML, anthropic, openai) gated behind try/except with install instructions.

### How we test

**12. Integration tests are the source of truth.**
Before adding a test file, check if `test_integration.py` already covers the behavior. Separate unit tests only for complex pure functions (validator, parser).

**13. Never duplicate test coverage.**
If test A already proves a behavior, don't write test B that proves the same thing with a different spec shape. Each test must prove exactly one thing no other test covers.

**14. Validate at the boundary.**
Mock external SDKs at the SDK boundary (not our code). This exercises our full stack while isolating from network/credentials.

### How we ship

**15. The server must use the app's pipeline.**
Never re-parse or create separate pipelines in server code. `app._pipeline.handle()` is the single path. Decorators, cost limits, fallback — all work identically in test and HTTP mode.

**16. README: show each concept once.**
One code example per idea. If it's shown in the diagram AND a YAML block AND an explanation — cut two.

**17. Iterate until you can't remove more.**
After a feature is "done," review it 10 times asking "can this be simpler?" Ship when the answer is finally no.

## Structure

```
specllm/
├── __init__.py       → Entry point (SpecLLM, TestClient, CostTracker)
├── spec/             → Parse + validate (pure functions, no I/O)
├── pipeline/         → Runtime orchestration (cache, constraints, request handling)
├── llm/              → Provider abstraction (Anthropic, OpenAI, Mock)
├── server/           → HTTP serving (optional, lazy-imported)
└── testing/          → Test utilities (optional, lazy-imported)
```
