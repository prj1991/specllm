# Contributing to specllm

## Coding Guidelines

These guidelines exist to keep the codebase minimal and maintainable. Every line is tech debt.

### 1. No single-function modules

If a module contains only one function or class, inline it into its consumer. A new file must justify its existence with distinct, non-trivial logic that would clutter its consumer.

### 2. No aspirational code

Don't add code for features that don't exist yet. No unused error codes, unused parameters, unused classes, or unused fields. Add them when they're needed, not before.

### 3. One mechanism per concept

If there are two ways to do the same thing (e.g., a decorator AND a spec annotation for the same feature), pick one. The spec-declarative approach (`x-constrain-from`) wins over programmatic decorators when both are possible.

### 4. Tests must not duplicate coverage

Before adding a test file, check if the behavior is already tested through `test_integration.py`. Unit tests are only justified for:
- Complex pure functions with many edge cases (validator, parser)
- Logic that's hard to reach through the integration path

Never create a separate test file for a feature that's fully exercised by integration tests.

### 5. Every stored field must have a reader

If `self.x = value` is written in `__init__`, something must read `self.x` after construction. If it's only used during init, use a local variable instead.

### 6. The server must use the app's pipeline

Never re-parse specs or create a separate pipeline in server code. The server receives the `SpecLLM` app instance and calls `app._pipeline.handle()` directly. This ensures decorators, cost limits, and fallback providers work identically in test and HTTP mode.

### 7. README: show, don't repeat

Each concept gets ONE code example. If the same idea (e.g., "schema validation works") is shown in the architecture diagram, the feature list, AND a full YAML example — cut two of them.

### 8. Shared logic belongs in one place

If three code paths do the same thing with different parameters, extract a helper. The error handling pattern (try fallback → set metadata → return error) is one function, not three copy-pasted blocks.

### 9. Dependencies: zero required

The library uses only Python stdlib. Optional dependencies (PyYAML) are gated behind try/except with clear install instructions in the error message.

### 10. Structure mirrors the execution flow

```
specllm/
├── __init__.py       → Entry point (SpecLLM, TestClient, CostTracker)
├── spec/             → Parse + validate (pure functions, no I/O)
├── pipeline/         → Runtime orchestration (cache, constraints, request handling)
├── llm/              → Provider abstraction
├── server/           → HTTP serving (optional, lazy-imported)
└── testing/          → Test utilities (optional)
```

Packages that are lazy-imported (`server/`, `testing/`) don't load at `import specllm` time.
