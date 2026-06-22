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
