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
