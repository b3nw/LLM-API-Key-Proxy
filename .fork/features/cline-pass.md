# ClinePass provider

Canonical feature ID: `cline-pass`
Stack subject: `feat(cline-pass): add ClinePass provider with 3-window quota tracking`
Manifest: `.fork/stack.yml`

This file is the shared, repo-tracked history for the ClinePass feature.
Local workspace state may contain run logs and scratch notes, but this
file is canonical across contributors and developer workspaces.

ClinePass is the subscription tier of the Cline API
(https://docs.cline.bot/getting-started/clinepass). It offers 10 curated
open coding models under the `cline-pass/<model>` identifier scheme with
a single monthly subscription and 3 percent-based usage windows (5h
rolling, weekly, monthly).

## 2026-07-11 — Initial provider implementation

Branch: `feat/cline-quota`
Files:
- `src/rotator_library/providers/cline_pass_provider.py`
- `src/rotator_library/providers/utilities/cline_pass_quota_tracker.py`
- `tests/test_cline_pass_quota_tracker.py`
- `.fork/stack.yml`
- `.fork/features/cline-pass.md` (this file)

Background:
- Prior research captured in `scratch/cline.md` (developer workspace)
  confirmed the dashboard quota endpoint accepts Bearer auth via
  `CLINE_PASS_API_KEY` or an account auth token; cookie-only exports
  fail with 401.
- Official docs (https://docs.cline.bot/getting-started/clinepass) name
  the catalog explicitly, so the first cut ships a hardcoded model
  catalog (`DEFAULT_CLINEPASS_MODELS`) with no live `/v1/models`
  dependency. Operators can override via the `CLINE_PASS_MODELS` env
  var (same shape as `UMANS_MODELS`).

Provider design:
- Provider class: `ClinePassProvider` (file: `cline_pass_provider.py`).
  Auto-registers as `cline_pass` in `PROVIDER_PLUGINS` via the
  `pkgutil.iter_modules` discovery in `providers/__init__.py`.
- Quota mixin: `ClinePassQuotaTracker` (file:
  `utilities/cline_pass_quota_tracker.py`).
  Polls `GET /api/v1/users/me/plan/usage-limits` every 15 min
  (configurable via `CLINE_PASS_QUOTA_REFRESH_INTERVAL`).
  Three separate quota groups: `5h` (rolling), `weekly`, `monthly`.
  Plan metadata from `GET /api/v1/users/me/plan` is fetched
  best-effort and surfaces `displayName`,
  `entitlements.cline_pass.inferenceCapThreshold` (USD cost cap).
- Exhaustion threshold: 95% (configurable via
  `CLINE_PASS_QUOTA_EXHAUSTION_PCT`). At or above this percent,
  `apply_exhaustion=True` on the initial fetch so the credential
  goes onto cooldown.
- Routing: chat completions go through
  `https://api.cline.bot/v1/chat/completions` using LiteLLM's
  `openai/` custom provider, with the Cline upstream `id` (e.g.
  `cline-pass/qwen3.7-plus`) substituted via the display-name →
  upstream-id map.
- Embeddings: not implemented (Cline API has no public embeddings
  endpoint for the subscription tier). The provider raises
  `NotImplementedError` so `MODEL_FALLBACK` can route around it.
- Model name handling (Umans-style): display names map to upstream
  IDs via the `CLINE_PASS_MODELS` env var (or the shipped defaults).
  `normalize_model_for_tracking()` rewrites upstream IDs back to
  display names for usage records and cost lookups.

Credentials:
- `CLINE_PASS_API_KEY_1`, `_2`, … — Bearer API keys
  (Settings > API Keys at app.cline.bot).
- `CLINE_PASS_API_KEY` — unnumbered shorthand for single-key setups.
- `CLINE_PASS_API_BASE` — base URL override (default
  `https://api.cline.bot/api/v1`); tolerated trailing `/v1` to avoid
  double-versioning in the path.
- `CLINE_PASS_QUOTA_REFRESH_INTERVAL` — seconds (default 900).
- `CLINE_PASS_QUOTA_EXHAUSTION_PCT` — float (default 95.0).

Scope: this PR ships the ClinePass subscription tier only. The full
Cline catalog (`anthropic/<model>`, `openai/<model>`, etc.) is left
for a follow-up — duplicating `anthropic/*` under `cline_pass/*`
adds no value while the existing first-party `anthropic` provider
already serves those models.

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/cline_pass_provider.py` — passed
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/cline_pass_quota_tracker.py` — passed
- `uv run ruff check src/rotator_library/providers/cline_pass_provider.py src/rotator_library/providers/utilities/cline_pass_quota_tracker.py --select F401,F811,F821,E9` — passed
- `uv run --with pytest python3 -m pytest tests/test_cline_pass_quota_tracker.py -v` — 26 passed
- `uv run --no-project python3 .fork/check-stack.py` — pending (stack.yml is updated; the script runs in CI on push)
- Manual smoke (TODO once a `CLINE_PASS_API_KEY` is wired in the dev
  container): confirm `claude-sonnet-4-6` equivalent is not exposed
  on ClinePass; the catalog here is the 10-model subscription list
  only.

Notes:
- The Cline API does not document a public models catalog endpoint
  for the subscription tier; the shipped defaults match the docs
  page as of 2026-07-11. Operators can override via
  `CLINE_PASS_MODELS` if Cline adds or retires models.
- Cost calculation is skipped (`skip_cost_calculation = True`)
  because ClinePass is a flat monthly subscription and the
  upstream's per-model reference pricing isn't billed.
- `get_model_quota_group()` defaults real models to the `5h` group
  (the most restrictive window) so a credential that hits the rolling
  limit short-circuits to the next one. The `weekly` and `monthly`
  groups still render as their own bars in the WebUI.

## 2026-07-11 — Fix double-prefix bug in `_build_reverse_map`

Target: `feat(cline-pass): add ClinePass provider with 3-window quota tracking`
Files:
- `src/rotator_library/providers/cline_pass_provider.py`
- `tests/test_cline_pass_quota_tracker.py`

Problem (caught by Kilo Code review on PR #122):
- `DEFAULT_CLINEPASS_MODELS` already stores upstream IDs with the
  `cline-pass/` prefix (e.g. `"cline-pass/glm-5.2"`), because the Cline
  docs document the upstream model ids in that form.
- The first cut of `_build_reverse_map()` wrapped those upstream ids
  with `f"cline_pass/{upstream_id}"` again, producing keys like
  `"cline_pass/cline-pass/glm-5.2"` that never matched any input to
  `normalize_model_for_tracking`. As a result, raw upstream IDs from
  Cline error messages and quota breakdowns were returned unchanged —
  breaking the very "WebUI/pricing maps upstream -> display" feature
  the PR description calls out.
- Umans does not have this bug because its `UMANS_MODELS` upstream ids
  are bare (`"umans-kimi-k2.6"`); wrapping with `umans/` produces a
  correct reverse-map key.

Fix:
- `_build_reverse_map()` now stores keys as the raw upstream id
  (`"cline-pass/<bare>"`) and values as the proxy display name
  (`"cline_pass/<bare>"`).
- `normalize_model_for_tracking()` rewritten to handle all three
  caller-input shapes — raw upstream id (`cline-pass/<bare>`), proxy
  display name (`cline_pass/<bare>`), and bare (`<bare>`) — and
  return the canonical proxy display name. Unknown inputs pass
  through unchanged.

Verification:
- `uv run python3 -m py_compile` — passed
- `uv run ruff check --select F401,F811,F821,E9` — passed
- `uv run --with pytest python3 -m pytest tests/test_cline_pass_quota_tracker.py -v` — 31 passed (added 5 round-trip regression tests:
  `test_build_reverse_map_keys_are_raw_upstream_ids`,
  `test_normalize_model_from_raw_upstream_id`,
  `test_normalize_model_from_proxy_display_name`,
  `test_normalize_model_from_bare_name`,
  `test_normalize_model_unknown_returns_input`)

Notes:
- Tests use `ClinePassProvider()` directly — the class is a
  `SingletonABCMeta` singleton so all test cases share one instance,
  and the `__init__`-built reverse map is reused across cases. No
  cleanup needed (and cleanup would actually break later tests).

## 2026-07-11 — Fix wrong API base path (deployment hotfix)

Target: `feat(cline-pass): add ClinePass provider with 3-window quota tracking`
Files:
- `src/rotator_library/providers/cline_pass_provider.py`
- `src/rotator_library/providers/utilities/cline_pass_quota_tracker.py`
- `tests/test_cline_pass_quota_tracker.py`

Problem (caught from production deployment on 2026-07-11, 02:34 UTC):
1. **Chat completions returned 404.** The provider class introduced
   a separate `CLINE_PASS_LITELLM_BASE = "https://api.cline.bot/v1"`
   for chat routing (assumption: "Cline is OpenAI-shaped, base is
   `/v1`"). The actual upstream path is
   `https://api.cline.bot/api/v1/chat/completions` — the prefix is
   `/api/v1`, not `/v1`. Every request landed at
   `https://api.cline.bot/v1/chat/completions` and 404'd.
2. **Quota card empty in WebUI.** The quota tracker's
   `_build_billing_url()` defensively stripped a trailing `/v1` from
   `_resolve_api_base()`, producing
   `https://api.cline.bot/api/users/me/plan/usage-limits` instead
   of the correct
   `https://api.cline.bot/api/v1/users/me/plan/usage-limits`. The
   upstream call 404'd, baseline writes never happened, and the
   `/v1/quota-stats` filter (`total_requests == 0 and no quota data`)
   then dropped the provider from the response.

Root cause (both bugs):
- Mistakenly assumed "OpenAI-compatible *body*" implied "OpenAI-compatible
  *path prefix*". Cline's API uses `/api/v1/`, not `/v1/`. The
  original writeup's `cline.md` documented the quota endpoint URL as
  `https://api.cline.bot/api/v1/users/me/plan/usage-limits` — if
  that had been more directly referenced during the first cut, the
  bug would have been caught.
- The first cut unhelpfully split into two bases (one for quota, one
  for chat) and then rewrote the quota URL. The chat URL was wrong
  by construction; the quota URL was wrong by post-processing.

Fix:
- Drop `CLINE_PASS_LITELLM_BASE` and the `self.litellm_base`
  attribute. Chat completions now use `self.api_base`, same as
  quota and model discovery. `self.api_base` defaults to
  `https://api.cline.bot/api/v1` and is overridable via
  `CLINE_PASS_API_BASE`.
- Remove the trailing-`/v1` strip in `_build_billing_url()`. The
  upstream is a flat `/api/v1` namespace — the helper just joins
  base + path.
- Update the docstring on the constant to call out the path-prefix
  gotcha so this doesn't regress.

Regression coverage (4 new tests, all pass):
- `test_build_billing_url_default_base_passes_path_through` — default
  base yields `https://api.cline.bot/api/v1/users/me/plan`.
- `test_build_billing_url_preserves_v1_in_api_v1_base` — pinning
  the no-strip behavior.
- `test_build_billing_url_trailing_slash_on_base_is_normalised` —
  trailing slash on the base doesn't double up.
- `test_provider_api_base_default_uses_documented_upstream` —
  provider default matches the Cline docs.
- `test_provider_uses_single_api_base_for_both_models_and_chat` —
  pinning the single-base invariant so a future split can't
  reintroduce Bug 1.

Verification:
- `uv run python3 -m py_compile` — passed (all 3 files)
- `uv run ruff check --select F401,F811,F821,E9` — passed (all 3 files)
- `uv run --with pytest python3 -m pytest tests/test_cline_pass_quota_tracker.py -q` — 34 passed
- `uv run --with pytest --with pytest-asyncio python3 -m pytest tests/ -q` — 493 passed (+3), same 15 pre-existing setup errors and same 2 pre-existing umans/xai test failures as `dev` (unrelated to this PR).
