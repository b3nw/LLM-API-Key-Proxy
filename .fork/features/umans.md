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

## 2026-06-22 — Static model definitions, cost tracking, and quota fixes

Target: `fix(umans): normalize API base and show quota without proxy requests`
Files:
- `src/rotator_library/providers/umans_provider.py`
- `src/rotator_library/providers/utilities/umans_quota_tracker.py`
- `src/rotator_library/client/quota.py`
- `.fork/stack.yml` (added umans to allowed_duplicate_features)

Changes:
- `UmansProvider.get_models()`: prioritize static definitions from `UMANS_MODELS`,
  dedup dynamic models against both suffix and explicit `id` fields.
- `UmansProvider.__init__`: cache upstream context lengths in `_upstream_context`,
  build reverse map (`_id_to_display`) from upstream IDs to display names.
- `UmansProvider.get_model_context_overrides()`: expose cached upstream context
  so `/v1/models` can apply authoritative context windows.
- `skip_cost_calculation` changed from `True` to `False` to enable cost lookup.
- `UmansProvider.calculate_cost()`: delegates to `ModelInfoService.compute_cost`
  using the display name via the reverse map.
- `UmansProvider.normalize_model_for_tracking()`: maps upstream model IDs to
  display names so usage/quota records under canonical names.
- `umans_quota_tracker._detect_plan()`: handle `plan` as dict (`{"slug": ...}`)
  or string; fallback to heuristic when `None`.
- `umans_quota_tracker._parse_usage_response()`: coalesce `"window": null` with
  `or {}` to prevent `NoneType.get()` crash.
- `quota.py.force_refresh_quota()`: access `status`/`error` as dataclass attrs
  on `UmansQuotaSnapshot` instead of dict `.get()`.

Verification:
- `uv run python3 -m py_compile` — passed (all 3 files)
- `uv run ruff check --select F401,F811,F821,E9` — passed (all 3 files)
- Hot-patched llm-proxy-dev: all 5 static models route and return responses
- `/v1/quota-stats`: usage tracked under display names, costs computed correctly
- Quota refresh endpoint functional, no dataclass attribute errors

Notes:
- Upstream context windows override models.dev values (e.g. glm-5.2: 405504 from
  upstream vs 1000000 from models.dev).
- Cost calculation relies on `PROVIDER_ALIASES` mapping sub-providers to canonical
  catalog providers (moonshot→moonshotai, z-ai→zai/zhipuai, qwen→alibaba).

## 2026-06-23 — Use hard cap (400) as effective quota limit

Target: `fix(umans): normalize API base and show quota without proxy requests`
Files:
- `src/rotator_library/providers/utilities/umans_quota_tracker.py`

Working commits before autosquash:
- `c487fc58 fixup! fix(umans): ...`

Verification:
- `uv run python3 -m py_compile` — passed
- `uv run ruff check --select F401,F811,F821,E9` — passed
- Live: `/v1/quota-stats` now shows `total_max: 400`, `remaining: 178`

Notes:
- The soft limit (200) has no observable enforcement — requests continue
  through to the hard cap (400). Changed `_store_baselines_to_usage_manager`
  to use `requests_hard_cap` as the effective limit, falling back to
  `requests_limit` if hard cap is 0.

## 2026-06-27 — Expose burst band and deprioritized state in quota stats and WebUI

Target: `fix(umans): normalize API base and show quota without proxy requests`
Branch: `fix/umans-quota-priority-display` (PR #86 into `dev`; PR contains **fixup!** commits only per AGENTS.md)
Files:
- `src/rotator_library/providers/utilities/umans_quota_tracker.py`
- `src/rotator_library/usage/manager.py`
- `src/rotator_library/client/usage_managers.py`
- `src/proxy_app/api/config.py`
- `tests/test_umans_quota_tracker.py`
- `tests/test_usage_manager_provider_instance.py`
- `webui/src/lib/umansQuota.ts`
- `webui/src/api/quota.ts`
- `webui/src/pages/Quota.tsx`
- `.fork/stack.yml`
- `.gitignore` (test allowlist entries)

Working commits before merge (fixup! only on PR branch):
- `fixup! fix(umans): normalize API base and show quota without proxy requests` (tracker, tests, ledger)
- `fixup! feat(usage): …` / `fixup! feat(core): …` / `fixup! feat(gemini-cli): …` / `fixup! feat(webui): …` / `fixup! feat(tests): …`

Verification:
- `uv run ruff check` (tracker, manager) — passed
- `uv run python3 -m py_compile` (tracker, manager) — passed
- `uv run python -m pytest tests/test_umans_quota_tracker.py -q` — 41 passed
- `uv run python -m pytest tests/test_usage_manager_provider_instance.py -q` — passed

Notes:
- Live `/v1/usage` uses `usage.priority.low`, `boxed_until`, `reason` for deprioritization; top-level `throttled` is absent (parser keeps legacy fallback).
- `upstream_quota` on each credential in `/v1/quota-stats` when provider implements `get_upstream_quota_for_accessor`.
- **PR #86 review:** `UsageManager` injects `get_provider_instance` from `RotatingClient._get_provider_instance` so `_quota_cache` matches `background_refresher`.
- WebUI: Deprioritized / Burst band badges; Umans upstream summary and per-credential panel.
- Distinct from proxy credential rotation `priority`.
