# core — Infrastructure Improvements

## 2026-06-22 — Extend reasoning tag handling for `<think>`/`</think>`

Target: `feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`
Files:
- `src/rotator_library/client/streaming.py`
- `src/rotator_library/client/executor.py`

Working commits before autosquash:
- `24876cfb fixup! feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`

Final stack commit after autosquash:
- `039836ab feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`

Verification:
- `uv run python3 -m py_compile src/rotator_library/client/streaming.py` — passed
- `uv run python3 -m py_compile src/rotator_library/client/executor.py` — passed
- `uv run ruff check src/rotator_library/client/streaming.py --select F401,F811,F821,E9` — passed
- `uv run ruff check src/rotator_library/client/executor.py --select F401,F811,F821,E9` — passed
- Live streaming test (umans/kimi-k2.6, max_tokens=300): reasoning properly captured, content clean
- Live streaming test (umans/kimi-k2.6, max_tokens=4096): full reasoning separated, haiku content only
- Live non-streaming test: reasoning extracted via regex, content clean
- Non-kimi model (mistral): unaffected, no false reasoning detection

Notes:
- Kimi K2 via umans streams reasoning as plain content with only a bare `</think>`
  (no opening `<think>`) to mark the transition to real content.
- Extended `_split_thought_tags` to handle both `<thought>...</thought>` (Gemma-4)
  and `<think>...</think>` tag pairs, plus the bare-close-tag pattern.
- Extended `_extract_thought_tags_from_response` (non-streaming) with matching
  regex patterns and bare-close detection.

## 2026-06-22 — Fix: remove _model_has_implicit_thinking (double-handling bug)

Target: `feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`
Files:
- `src/rotator_library/client/streaming.py`

Working commits before autosquash:
- `9811c72c fixup! feat(core): ...`

Verification:
- `uv run python3 -m py_compile src/rotator_library/client/streaming.py` — passed
- `uv run ruff check src/rotator_library/client/streaming.py --select F401,F811,F821,E9` — passed
- Live streaming test (umans/kimi-k2.6): content no longer misclassified as reasoning

Notes:
- Removed `_IMPLICIT_THINKING_MODELS` tuple and `_model_has_implicit_thinking()`.
- LiteLLM already handles `<think>` tag separation during streaming for providers
  that emit them. Starting `in_thought_block=True` caused double-handling where
  post-thinking content was suppressed as reasoning_content.
- The tag-based detection in `_split_thought_tags` remains for providers that emit
  tags but where LiteLLM does not strip them (e.g. `<thought>` from Gemma-4).

## 2026-06-22 — Multi-segment display keys for PROVIDER_MODELS

Target: `feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`
Files:
- `src/rotator_library/client/models.py`
- `src/rotator_library/model_definitions.py`
- `src/rotator_library/providers/openai_compatible_provider.py`
- `.fork/stack.yml` (added file ownership for model_definitions.py, openai_compatible_provider.py)

Verification:
- `uv run python3 -m py_compile` — passed (all 3 files)
- `uv run ruff check --select F401,F811,F821,E9` — passed (all 3 files)
- Hot-patched llm-proxy-dev: models listed as `umans/moonshot/kimi-k2.6` etc.
- Pricing fuzzy-matched correctly for kimi-k2.6, glm-5.1, glm-5.2
- Routing test: `umans/moonshot/kimi-k2` resolved to `umans-kimi-k2.6` upstream, response received

Notes:
- Enables aggregation providers (like umans) to present canonical `provider/org/model`
  display names via `PROVIDER_MODELS` dict keys containing slashes, while routing
  the `"id"` value to the upstream API.
- `ModelResolver.resolve_model_id`: changed `split("/")[-1]` to `split("/", 1)[1]`
  so multi-segment keys like `moonshot/kimi-k2.6` are preserved during lookup.
- `ModelDefinitions.get_model_definition`: added last-segment fallback so existing
  single-segment configs still work when the resolver passes multi-segment names.
- `OpenAICompatibleProvider.get_models`: dedup now checks both display-key suffixes
  and explicit `"id"` values from static definitions, preventing dynamic discovery
  from re-adding models that have a static display mapping.
- `OpenAICompatibleProvider.get_model_options`: strips provider prefix correctly
  instead of taking only the last segment.

## 2026-06-22 — Sub-provider alias resolution and upstream context overrides

Target: `feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`
Files:
- `src/rotator_library/model_info_service.py`
- `src/proxy_app/main.py`

Working commits before autosquash:
- `b9966964 fixup! feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`

Changes:
- `model_info_service._get_alias_candidates()`: for multi-segment model IDs
  (e.g. `umans/moonshot/kimi-k2.6`), generate alias candidates for the
  sub-provider segment using `PROVIDER_ALIASES` (moonshot→moonshotai,
  z-ai→zai/zhipuai, qwen→alibaba).
- `model_info_service._resolve_model()` Step 3: fuzzy index search now includes
  alias candidates alongside the raw model ID for better matching.
- `PROVIDER_ALIASES` expanded with `moonshot`, `z-ai`, `qwen` entries.
- `main.py /v1/models`: generic loop applies upstream-authoritative context
  window overrides from providers (like Umans) that fetch context data during
  model discovery, overriding models.dev enrichment values.

Verification:
- `uv run python3 -m py_compile` — passed (both files)
- `uv run ruff check --select F401,F811,F821,E9` — passed (both files)
- Hot-patched llm-proxy-dev: pricing resolved correctly for all umans models
- `/v1/models` context_window values match upstream API (e.g. glm-5.2: 405504)

Notes:
- The sub-provider aliasing is generic and will benefit any future aggregation
  provider that uses multi-segment display keys.
- Context window override loop is guarded with try/except and only runs for
  providers known to supply `get_model_context_overrides()`.

## 2026-06-25 — Remove hardcoded thinking whitelist from request_sanitizer

Target: `feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`
Files:
- `src/rotator_library/request_sanitizer.py`
- `tests/test_request_sanitizer.py`

PR: b3nw/LLM-API-Key-Proxy#79

Root cause:
- Fork-added code in this commit replaced upstream's narrow check (only stripped
  `thinking` when exactly `{"type": "enabled", "budget_tokens": -1}` for non-Gemini)
  with a broad model-name whitelist: `anthropic/`, `claude-`, `gemini-2.0-`, `gemini-2.5-`.
- This silently stripped `thinking` for all other providers (Lightning AI, OpenAI,
  Chutes, etc.), preventing clients from enabling reasoning via the Anthropic-style
  `thinking` parameter.
- Also stripped `extra_body.thinking` for non-whitelisted models, which cleaned up
  the `_guard_thinking_tool_calls` injection — but that guard runs AFTER the sanitizer
  (executor.py line 502 transforms, line 512 sanitize), so the sanitizer's
  `extra_body.thinking` stripping only affected client-sent values, not guard injections.

Fix:
- Removed the `_supports_thinking` whitelist entirely.
- Thinking parameter filtering is now delegated to each provider's `acompletion()`
  method. Providers that don't support `thinking` strip it via their own
  `SUPPORTED_PARAMS` filtering (e.g. Lightning AI) or litellm's param handling.
- Updated 13 tests in `test_request_sanitizer.py` to verify `thinking` is passed
  through (not stripped) + new `extra_body.thinking` preservation test.

Verification:
- `uv run python3 -m py_compile src/rotator_library/request_sanitizer.py` — passed
- `uv run ruff check src/rotator_library/request_sanitizer.py --select F401,F811,F821,E9` — passed
- `uv run pytest tests/test_request_sanitizer.py -v` — 13 passed
- Full suite: 394 passed, 1 pre-existing failure (test_umans_quota_tracker, unrelated)
