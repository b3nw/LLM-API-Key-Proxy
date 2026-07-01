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
