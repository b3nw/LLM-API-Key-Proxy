# Umans feature ledger

## 2026-06-22 — Add Umans provider with request-based quota tracking

Target: `feat(umans): add Umans provider with request-based quota tracking`
Files:
- `src/rotator_library/providers/umans_provider.py`
- `src/rotator_library/providers/utilities/umans_quota_tracker.py`
- `tests/test_umans_quota_tracker.py`
- `.fork/stack.yml`
- `.fork/features/umans.md` (this file)

Working commits before autosquash:
- (new feature, no fixup)

Final stack commit:
- `feat(umans): add Umans provider with request-based quota tracking`

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/umans_provider.py` — passed
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/umans_quota_tracker.py` — passed
- `uv run ruff check src/rotator_library/providers/umans_provider.py --select F401,F811,F821,E9` — passed
- `uv run ruff check src/rotator_library/providers/utilities/umans_quota_tracker.py --select F401,F811,F821,E9` — passed
- `uv run --no-project python3 .fork/check-stack.py` — passed
- `uv run pytest tests/test_umans_quota_tracker.py -v` — passed
- Full test suite (`uv run pytest tests/ -q`) — passed

Notes:
- Authentication uses `Authorization: Bearer` against `https://api.code.umans.ai`.
- Two plans are detected from the `/v1/usage` response:
  - `code_pro`: 200 req / 5h soft limit, 400 hard cap, 3 concurrent sessions.
  - `max`: no request limit, 4 concurrent sessions.
- `UMANS_QUOTA_LIMIT` only overrides the soft request limit for `code_pro` keys.
- Request-quota tracking is **display-only** (`apply_exhaustion=False`) until the
  burst-ceiling enforcement behavior is observed. A 429 response will still put
  the credential on cooldown via the generic error handler.
- Concurrency tracking is display-only for all plans.
- The class-level `default_max_concurrent_per_key = 3` is the safe default;
  `get_credential_concurrency_limit()` returns 4 for detected max-plan keys.
- LiteLLM has no Umans pricing, so `skip_cost_calculation = True`.

Risks / follow-ups:
- Burst ceiling behavior is not yet confirmed. Once observed, consider switching
  `apply_exhaustion=True` at the soft limit or using `UMANS_QUOTA_LIMIT` to
  target the hard cap.
- The `/v1/messages` Anthropic-compatible endpoint is intentionally left to
  the standard OpenAI-compatible path in this change.

## 2026-06-22 — Address kilo-code-bot review (PR #62)

Target: `feat(umans): add Umans provider with request-based quota tracking`
Files:
- `src/rotator_library/providers/utilities/umans_quota_tracker.py`
- `src/rotator_library/providers/umans_provider.py`
- `tests/test_umans_quota_tracker.py`

Changes:
- Added `_safe_int()` defensive coercion helper and used it in
  `_detect_plan()` (lines 106-107) and `_resolve_request_limit()` (line 134)
  to prevent `ValueError` crashes on malformed env vars or API responses.
- Added class docstring note documenting the transient max-plan concurrency
  throttling at startup (before first `/v1/usage` fetch populates the
  per-credential override).
- Added 10 new tests covering `_safe_int`, malformed env var, malformed API
  limit strings, and end-to-end parse with string-typed limits.

Verification:
- `uv run python3 -m py_compile` — passed (all 3 files)
- `uv run ruff check --select F401,F811,F821,E9` — passed (all 3 files)
- `uv run pytest tests/test_umans_quota_tracker.py -v` — 33 passed
- `uv run pytest tests/ -q` — 412 passed
- `.fork/check-stack.py` fails on `dev` itself (pre-existing: `fix(ci): remove
  upstream community files` not in stack.yml manifest) — not introduced by
  this PR

Notes:
- Rebased onto current `origin/dev` (fe3ee86) to fix the stale base that
  caused PR #62 to show 16 commits instead of 1.

## 2026-06-22 — Fix Umans quota fetch 404 and Web UI visibility (post #62)

Target: `fix(umans): normalize API base and show quota without proxy requests`
Files:
- `src/rotator_library/providers/utilities/umans_quota_tracker.py`
- `src/rotator_library/providers/umans_provider.py`
- `src/rotator_library/client/quota.py`
- `tests/test_umans_quota_tracker.py`

Changes:
- `_normalize_umans_api_base()` strips trailing `/v1` so `UMANS_API_BASE=https://api.code.umans.ai/v1` does not call `/v1/v1/usage` (404).
- `_resolve_umans_api_key()` resolves `env://umans/N` to `UMANS_API_KEY_N` for Bearer auth.
- `QuotaService.get_quota_stats` keeps providers with quota baselines when `total_requests == 0`.
- Tests for URL normalization, env key resolution, and ISO timestamp assertion.

Verification:
- `uv run pytest tests/test_umans_quota_tracker.py -q` — 36 passed
- Live: `GET https://api.code.umans.ai/v1/usage` returns 200; double `/v1` returns 404.

Follow-up PR after merge of #62.
