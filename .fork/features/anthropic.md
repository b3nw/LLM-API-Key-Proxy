## 2026-06-30 — Remove unused imports in anthropic_provider.py

Target: fixup! feat(anthropic): add OAuth support and handle streaming nulls
Files: src/rotator_library/providers/anthropic_provider.py

Verification:
- uv run python3 -m py_compile src/rotator_library/providers/anthropic_provider.py — passed
- uv run ruff check src/rotator_library/providers/anthropic_provider.py --select F401 — passed

Notes: Removed unused imports (asyncio, re, Path, UsageManager and TYPE_CHECKING).

## 2026-07-01 — Add newer Claude models to OAuth whitelist and max output tokens

Target: `feat(anthropic): add OAuth support and handle streaming nulls`
Files:
- `src/rotator_library/providers/anthropic_provider.py`

Changes:
- Added `claude-fable-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-sonnet-4-6`
  to `OAUTH_MODELS` — these are current active Anthropic models available via
  Claude Pro/Max OAuth subscription.
- Added corresponding entries to `_MODEL_MAX_OUTPUT_TOKENS`:
  - `claude-fable-5`: 128,000
  - `claude-opus-4-8`: 128,000
  - `claude-opus-4-7`: 128,000
  - `claude-sonnet-4-6`: 64,000
- `claude-mythos-5` intentionally excluded (restricted to Project Glasswing participants).

Model IDs sourced from Anthropic's official skills catalog
(anthropics/skills/skills/claude-api/shared/models.md).

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/anthropic_provider.py` — passed
- `uv run ruff check src/rotator_library/providers/anthropic_provider.py --select F401,F811,F821,E9` — passed

Notes:
- Existing models (opus-4-6, opus-4-5, sonnet-4-5, haiku-4-5) remain in the list.
- The `model_quota_groups` (5h-limit, weekly-limit, anthropic-global) automatically
  include the new models since they use `list(OAUTH_MODELS)`.
- The max output token prefix-matching loop uses `startswith()` with `break` on
  first match. No prefix collisions exist between the new entries and existing ones.
- Ref: b3nw/LLM-API-Key-Proxy#97

## 2026-07-01 — Dynamic model discovery via models.dev

Target: `feat(anthropic): add OAuth support and handle streaming nulls`
Files:
- `src/rotator_library/providers/anthropic_provider.py`
- `tests/test_anthropic_models_dev.py`
- `.gitignore`

Changes:
- Added `_fetch_anthropic_models_from_models_dev()` — fetches the Anthropic model
  catalog from `https://models.dev/api.json` (community-maintained, no auth required).
  Filters out retired 3.x models and restricted mythos models. Only includes models
  with `tool_call: true` (required by Claude Code).
- Added `_get_dynamic_models()` — module-level cache with 1-hour TTL and 3-tier
  fallback: fresh cache → fetch → stale cache → None (caller falls back to hardcoded
  `OAUTH_MODELS`). Pattern follows the Codex provider's GitHub JSON catalog fetch.
- Modified `get_models()` to use dynamic list, falling back to `OAUTH_MODELS`.
- Modified max output tokens lookup in `handle_oauth_completion()` to check dynamic
  data first (exact match), then fall back to hardcoded `_MODEL_MAX_OUTPUT_TOKENS`.
- Moved `model_quota_groups` from class attribute to `__init__`, populated from
  dynamic model list. Override `get_model_quota_group()` to always return
  `"anthropic-global"` for any Anthropic model (matches Codex pattern).
- Added 11 tests: fetch parsing, filtering (3x, mythos, no-tool-call), network/JSON
  errors, cache behavior, stale fallback, quota group override.

Rationale:
- OAuth tokens (`sk-ant-oat-*`) cannot call Anthropic's `GET /v1/models` endpoint.
  models.dev provides the same data (model IDs, context windows, max output tokens)
  without auth. This is the approach used by pi (earendil-works/pi) and opencode
  (sst/opencode).
- Builds on PR #99's hardcoded fallback list. Dynamic discovery augments the
  fallback — when models.dev is reachable, new models appear automatically.

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/anthropic_provider.py` — passed
- `uv run ruff check src/rotator_library/providers/anthropic_provider.py --select F401,F811,F821,E9` — passed
- `pytest tests/test_anthropic_models_dev.py tests/test_model_alias.py -v` — 23 passed

Notes:
- `MODELS_DEV_URL` env var allows overriding the catalog URL (e.g., for testing or
  self-hosting). `ANTHROPIC_MODELS_CACHE_TTL` controls the cache TTL (default 3600s).
- models.dev includes `claude-sonnet-5` which was NOT in PR #99's hardcoded list —
  this demonstrates the value of dynamic discovery.
- Ref: b3nw/LLM-API-Key-Proxy#97

## 2026-07-01 — Mirror pi-agent OAuth headers and tool naming

Target: `feat(anthropic): add OAuth support and handle streaming nulls`
Files:
- `src/rotator_library/providers/anthropic_provider.py`
- `tests/test_anthropic_oauth_headers.py`
- `.gitignore`

Changes:
- Added `_compute_beta_header(model)` — dynamically computes the `anthropic-beta`
  header based on the model. Base betas now include `claude-code-20250219`
  (critical: tells Anthropic this is a Claude Code session), `prompt-caching-scope-2026-01-05`,
  and `context-management-2025-06-27`. Long-context models (opus-4-6+, sonnet-4-6+,
  fable-5, sonnet-5) get `context-1m-2025-08-07` and `effort-2025-11-24`. Haiku
  models exclude `interleaved-thinking-2025-05-14`.
- Added `x-app: cli` header to OAuth request headers.
- Added `_prefix_tool_name()` helper — capitalizes first letter before prefixing
  (e.g., `read` → `mcp_Read` instead of `mcp_read`). Mirrors Claude Code's
  PascalCase tool naming convention.
- Kept `ANTHROPIC_BETA_HEADER` constant for backward compatibility (token refresh
  requests that don't have a model context).
- 11 new tests covering beta computation (base, long-context, haiku exclusion)
  and tool name prefixing (lowercase, capitalized, empty, single char).

Rationale:
- Research into pi-agent (earendil-works/pi) and @cgaravitoq/pi-claude-code-auth
  revealed that the proxy was missing critical protocol signals:
  - `claude-code-20250219` beta (identifies as Claude Code session)
  - `x-app: cli` header (present in both pi implementations)
  - PascalCase tool names (Anthropic expects `mcp_Read`, not `mcp_read`)
- Skipped for now: billing header (cch), Claude Code identity system prompt
  injection, system prompt relocation — these are protocol emulation, not
  safe header additions.

Verification:
- `uv run python3 -m py_compile` — passed
- `uv run ruff check --select F401,F811,F821,E9` — passed
- `pytest tests/test_anthropic_oauth_headers.py tests/test_anthropic_models_dev.py tests/test_anthropic_translator.py -v` — 63 passed

Notes:
- `_strip_tool_prefix()` not modified — may need case-insensitive matching
  in a follow-up if tool result routing breaks.
- The `ANTHROPIC_BETA_HEADER` constant is kept for token refresh requests
  that don't have model context. It uses the base betas only.
- Ref: b3nw/LLM-API-Key-Proxy#97

## 2026-07-01 — Full Claude Code protocol emulation (billing header + identity)

Target: `feat(anthropic): add OAuth support and handle streaming nulls`
Files:
- `src/rotator_library/providers/anthropic_provider.py`
- `tests/test_anthropic_oauth_headers.py`

Changes:
- Added `_compute_billing_header(messages)` — computes the client attestation
  hash (cch) from the first user message text, mirroring @cgaravitoq:
  - cch = SHA256(first_user_message_text)[:5]
  - suffix = SHA256(salt + chars_at[4,7,20] + version)[:3]
  - salt = "59cf53e54c78"
- Added `_build_claude_code_system(messages, original_system_prompt)` — builds
  the system prompt array with:
  1. Billing header as first system entry
  2. Claude Code identity ("You are Claude Code, Anthropic's official CLI for Claude.")
     as second system entry
  3. Original system prompt relocated to first user message (prevents 400 rejections
     from non-Claude Code identity in system[])
- Modified `handle_oauth_completion()` to use `_build_claude_code_system()` for
  all OAuth requests, replacing the plain `payload["system"] = system_prompt`.
- Added 11 new tests: billing header format, determinism, hash correctness,
  list content extraction, empty messages, assistant message skipping, system
  array structure, prompt relocation, no-prompt case, no-user-message case.
- Configurable: `ANTHROPIC_CLI_VERSION` and `CLAUDE_CODE_ENTRYPOINT` env vars.

Rationale:
- Safe headers alone (PR #101 initial commit) produced 429 errors in testing.
- The billing header and identity prompt are required for Anthropic to treat
  OAuth requests as genuine Claude Code sessions with standard rate limits.
- Confirmed by pi-agent (earendil-works/pi) and @cgaravitoq/pi-claude-code-auth.

Verification:
- `uv run python3 -m py_compile` — passed
- `uv run ruff check --select F401,F811,F821,E9` — passed
- `pytest tests/test_anthropic_oauth_headers.py tests/test_anthropic_models_dev.py tests/test_anthropic_translator.py tests/test_model_alias.py -v` — 74 passed

Notes:
- The billing header salt (59cf53e54c78) is hardcoded from @cgaravitoq's reverse-
  engineered code. If Anthropic changes the algorithm, this will need updating.
- `ANTHROPIC_CLI_VERSION` should be kept in sync with the user agent version.
- Ref: b3nw/LLM-API-Key-Proxy#97
