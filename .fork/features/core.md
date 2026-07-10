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

## 2026-07-09 — Add SSE keepalives for quiet stream gaps

Target: `feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`
Files:
- `src/proxy_app/main.py`
- `src/proxy_app/sse_keepalive.py`
- `tests/test_sse_keepalive.py`

Working commits before autosquash:
- PR branch commit for `fix(core): emit SSE keepalives during quiet stream gaps`.

Verification:
- `uv run python3 -m py_compile src/proxy_app/main.py src/proxy_app/sse_keepalive.py tests/test_sse_keepalive.py` — passed
- `uv run ruff check src/proxy_app/main.py src/proxy_app/sse_keepalive.py tests/test_sse_keepalive.py --select F401,F811,F821,E9` — passed
- `uv run --with pytest --with pytest-asyncio pytest tests/test_sse_keepalive.py -q` — passed (`2 passed`)
- `uv run python .fork/check-stack.py` — failed on pre-existing current-base webui stack metadata, unrelated to this PR's touched files

Notes:
- Long OpenAI-compatible SSE streams can legitimately have quiet periods while an upstream model reasons or waits on tool work.
- Some clients abort a stream if no bytes arrive within their read-idle window; emitting SSE comment keepalives keeps the connection byte-active without changing parsed model events.
- The default interval is 30 seconds and can be disabled with `SSE_KEEPALIVE_INTERVAL_SECONDS=0`.
