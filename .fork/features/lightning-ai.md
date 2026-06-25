## 2026-06-22 — Recover Lightning AI provider and archive retired fork providers

Target: `feat(lightning-ai): dollar credit quotas and date parsing`
Files:
- `src/rotator_library/providers/lightning_ai_provider.py` (restored)
- `src/rotator_library/providers/utilities/lightning_ai_quota_tracker.py` (restored)
- `src/rotator_library/providers/_retired/cursor_provider.py` (archived from git history)
- `src/rotator_library/providers/_retired/cursor_quota_tracker.py` (archived from git history)
- `src/rotator_library/providers/_retired/gemini_a2a_provider.py` (archived from git history)
- `src/rotator_library/providers/_retired/a2a_client.py` (archived from git history)
- `src/rotator_library/providers/_retired/a2a_session_manager.py` (archived from git history)
- `src/rotator_library/providers/_retired/a2a_sidecar_manager.py` (archived from git history)
- `src/rotator_library/providers/_retired/a2a_translator.py` (archived from git history)
- `src/rotator_library/providers/_retired/bedrock_provider.py` (archived from git history)
- `src/rotator_library/providers/_retired/gemini_file_logger.py` (archived from git history)
- `src/rotator_library/providers/_retired/README.md` (updated with full inventory)

Recovery source commits:
- Lightning AI: `47a3bb26` (dangling — dropped during linear rebase ~3-6 months ago)
- Cursor + Gemini A2A: `2033342a^` (parent of cleanup commit, dangling)
- Bedrock: `1ce8eba8^` (parent of refactor that removed it, reachable from main)
- Gemini file logger: `e82f08a2^` (parent of unified transaction logger commit)

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/lightning_ai_provider.py` — passed
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/lightning_ai_quota_tracker.py` — passed
- `uv run ruff check ... --select F401,F811,F821,E9` — passed (both files)
- Provider auto-discovery: `__init__.py` iterates `providers/` and skips `_`-prefixed dirs,
  so `_retired/` is ignored and `lightning_ai_provider.py` registers as `lightning_ai`.

Notes:
- The Lightning AI provider supports dollar-based monthly credit quota tracking via
  the /v1/memberships API. Balance is tracked in cents (×100) for integer quota
  compatibility. The TUI detects the `credits($)` group suffix and formats as dollars.
- Retired providers are archived with retirement headers and documented in _retired/README.md.
- All archived code was recovered before `git gc` could prune the dangling objects.

## 2026-06-23 — Add Lightning AI documentation to .env.example and README.md

Target: `feat(lightning-ai): dollar credit quotas and date parsing`
Files:
- `.env.example` (added Lightning AI section with env var reference)
- `README.md` (added provider to Additional Providers table and quick-start env block)

Verification:
- llm-proxy-dev stack confirmed Lightning AI active with 48 models discovered
- Health endpoint shows `lightning_ai` in active providers list
- Provider routes requests to `https://lightning.ai/api/v1/chat/completions` and `/responses`

Notes:
- Resolves GitHub issue #70 (documentation gap for Lightning AI provider)
- Documents: `LIGHTNING_AI_API_KEY_1` (UUID format), `LIGHTNING_AI_API_BASE`,
  `LIGHTNING_AI_MONTHLY_GRANT` (plan tiers: free=$15, pro=$20, teams=$50),
  `LIGHTNING_AI_QUOTA_REFRESH_INTERVAL` (default 300s)
- Documents billing routing suffix (`UUID/ORG_NAME/TEAMSPACE_NAME`)
- Notes that quota uses the API key directly (no separate session token like KiloCode)

## 2026-06-23 — Fix 405 on /v1/responses with GPT-5 tools + reasoning

Target: `feat(lightning-ai): dollar credit quotas and date parsing`
Files:
- `src/rotator_library/providers/lightning_ai_provider.py` (added has_custom_logic + acompletion override)

Root cause:
- litellm 1.85+ `responses_api_bridge_check()` routes GPT-5.4+ models to the
  `/responses` endpoint when `reasoning_effort` + `tools` are present.
- Lightning AI only supports `/chat/completions` → 405.
- The bridge fires because `ProviderConfig.convert_for_litellm` sets
  `custom_llm_provider="openai"` (Lightning AI is not a known litellm provider),
  and the bridge check matches `custom_llm_provider in ("openai", "azure")`.
- xAI and Codex providers avoid this via `custom_llm_provider="xai"` / custom
  Responses API calls, but Lightning AI needs OpenAI-compatible routing.

Fix:
- Override `has_custom_logic()` → True so the executor calls our `acompletion()`
  directly instead of `litellm.acompletion()`.
- `acompletion()` uses `openai.AsyncOpenAI.chat.completions.create()` directly,
  bypassing litellm entirely. This avoids the bridge check completely.
- Response objects (OpenAI SDK `ChatCompletion` / `AsyncStream`) are compatible
  with the executor's duck-typed usage extraction and the streaming handler's
  `AsyncIterator[Any]` contract.
- Normalizes `reasoning` dict from /v1/responses format to `reasoning_effort`
  string for the OpenAI SDK.

Verification:
- `uv run python3 -m py_compile ...` — passed
- `uv run ruff check ... --select F401,F811,F821,E9` — passed
- `uv run python3 -m pytest tests/ -q` — 414 passed, 1 pre-existing failure
  (test_umans_quota_tracker, unrelated)
- End-to-end /v1/responses with tools + reasoning — pending live deployment test

Notes:
- Problem 1 (missing docs) was already resolved by commit 290ca46 in the
  2026-06-23 force-push (.env.example + README.md sections added).
- Problem 3 (quota verification) requires live deployment testing after the
  405 fix is deployed to llm-proxy-dev.
- The fixup! commit will be autosquashed into the owning commit before merge.

## 2026-06-25 — Add thinking→reasoning_effort conversion and thinking param handling

Target: `feat(lightning-ai): dollar credit quotas and date parsing`
Files:
- `src/rotator_library/providers/lightning_ai_provider.py`
- `tests/test_lightning_ai_thinking.py` (new)

PR: b3nw/LLM-API-Key-Proxy#79

Problem:
- The provider's `acompletion()` only handled `reasoning` dicts (Responses API format)
  but not the Anthropic-style `thinking` parameter (`{"type": "enabled", "budget_tokens": N}`).
- Even after the sanitizer whitelist was removed (see core.md entry), `thinking` would
  be stripped by the provider's own `SUPPORTED_PARAMS` filter because it wasn't in the
  allowed set and wasn't converted to `reasoning_effort`.

Fix:
- Added `thinking` → `reasoning_effort` conversion in `acompletion()`, mirroring the
  existing `reasoning` dict conversion:
  - `thinking: {"type": "enabled"}` → `reasoning_effort: "high"`
  - `thinking: {"type": "disabled"}` → no reasoning_effort set
- Added cleanup of `extra_body.thinking` injected by `_guard_thinking_tool_calls`:
  - Pops `thinking` from `extra_body` to avoid sending unknown fields to Lightning AI
  - When guard set `type: "disabled"`, takes precedence over client's `type: "enabled"`
    (correct for multi-turn tool-call safety — prevents 400s when reasoning_content
    was dropped from assistant tool-call turns)
  - Cleans up empty `extra_body` dict after removing `thinking`
- Added 9 tests in `test_lightning_ai_thinking.py` covering:
  - thinking enabled/disabled → reasoning_effort conversion
  - reasoning dict/string conversion (existing behavior, regression coverage)
  - explicit reasoning_effort not overridden by thinking (setdefault)
  - guard disabled overrides client enabled
  - guard disabled without client thinking
  - extra_body other keys preserved when thinking is removed

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/lightning_ai_provider.py` — passed
- `uv run ruff check src/rotator_library/providers/lightning_ai_provider.py --select F401,F811,F821,E9` — passed
- `uv run pytest tests/test_lightning_ai_thinking.py -v` — 9 passed
- Live test on llm-proxy-dev.ext.ben.io (40 Lightning AI models):
  - `reasoning_effort=high` reaches Lightning AI, models reason internally
    (reasoning_tokens in usage stats confirmed)
  - Lightning AI's Chat Completions API does not return `reasoning_content` in
    responses — this is an upstream API limitation, not a proxy bug

Notes:
- `stream_options: {"include_reasoning": true}` was also tested — no effect on
  Lightning AI's endpoint.
- The upstream API limitation (no `reasoning_content` in responses) is documented
  in the PR description but cannot be fixed in the proxy.
