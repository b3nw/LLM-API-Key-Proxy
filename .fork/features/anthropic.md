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
