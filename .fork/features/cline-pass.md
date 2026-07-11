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
