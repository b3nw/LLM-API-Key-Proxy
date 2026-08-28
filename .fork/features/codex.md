## 2026-08-28 — Resolve prompt cache busting across multi-turn sessions

Target: `feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports`
Files:
- `src/rotator_library/providers/codex_provider.py`
- `src/rotator_library/client/executor.py`
- `src/proxy_app/responses_compat.py`
- `tests/test_codex_prompt_caching.py`

Branch:
- `fix/codex-prompt-caching`

Verification:
- `uv run pytest tests/test_codex_prompt_caching.py -v` — passed (16 tests)
- `uv run python3 -m py_compile src/rotator_library/providers/codex_provider.py src/rotator_library/client/executor.py src/proxy_app/responses_compat.py tests/test_codex_prompt_caching.py` — passed
- `uv run ruff check src/rotator_library/providers/codex_provider.py src/rotator_library/client/executor.py src/proxy_app/responses_compat.py tests/test_codex_prompt_caching.py --select F401,F811,F821,E9` — passed

Notes:
- Bug: Near-zero prompt cache hit rates reported on Codex models during multi-turn chats.
- Root cause:
  1. `<think>` reasoning tags were injected into assistant responses and echoed back by clients in message history without being stripped in `_convert_messages_to_responses_input()`, causing prompt prefix divergence after turn 1.
  2. Missing `"type": "message"` on assistant input items in the Responses API payload.
  3. `SessionTracker` inferred `session_id`, but `executor._prepare_request_kwargs()` dropped `context.session_id`, leaving `headers["session_id"]` and `payload["prompt_cache_key"]` empty strings on every request.
  4. Missing tool call IDs in history generated random UUIDs on each request (`uuid.uuid4()`), mutating past turns.
  5. Responses API gateway (`responses_compat.py`) stripped `prompt_cache_key` and `session_id` when translating incoming `/v1/responses` requests to Chat Completions.
- Fix:
  1. Stripped leading `<think>` tags with prefix-anchored regex `^<(?:think|thought)(?:ing)?>.*?</(?:think|thought)(?:ing)?>\n?` while preserving exact whitespace and mid-sentence tags.
  2. Explicitly added `"type": "message"` to assistant input items.
  3. Propagated `context.session_id` through `_prepare_request_kwargs()` to populate `headers["session_id"]` and `payload["prompt_cache_key"]`.
  4. Replaced random UUID generation with request-scoped deterministic hash IDs (`call_gen_{global_tool_idx}_{content_hash}`).
  5. Passed through `prompt_cache_key`, `session_id`, and `conversation_id` in `responses_compat.py`.
  6. Generated unique `x-client-request-id` per request to avoid upstream deduplication conflicts.

---

## 2026-06-26 — Don't block routing when paid credits bypass included-window exhaustion

Target: `feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports`
Files:
- `src/rotator_library/providers/utilities/codex_quota_tracker.py`

Working commits before autosquash:
- `55ec727c fixup! feat(codex): ...`

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/codex_quota_tracker.py` — passed
- `uv run ruff check src/rotator_library/providers/utilities/codex_quota_tracker.py --select F401,F811,F821,E9` — passed

Notes:
- Bug: When a ChatGPT Business (team plan) credential has `credits.has_credits=true`
  but the included 5h primary window reaches 100%, the proxy incorrectly marks the
  credential exhausted with a `codex-global` cooldown. Native Codex continues by
  consuming workspace credits in this scenario.
- Root cause: `credits.has_credits` and `credits.unlimited` are parsed from the WHAM
  API response and stored on `CodexQuotaSnapshot.credits`, but never consulted in
  the three exhaustion-decision paths (`_push_quota_to_usage_manager`,
  `_store_baselines_to_usage_manager`, `run_background_job`).
- Fix: All three hierarchical exhaustion waterfall blocks now check
  `credits.has_credits or credits.unlimited` before setting `global_exhausted=True`.
  When paid credits are available, the included-window usage is still displayed but
  routing is not blocked.
- Added `CODEX_IGNORE_CREDITS` (global) and `CODEX_IGNORE_CREDITS_<SLUG>`
  (per-credential) env vars to disable the credits bypass if needed.
- Ref: https://github.com/b3nw/LLM-API-Key-Proxy/issues/85

---

## 2026-06-23 — Fix Codex CLI /v1/models "missing field models" warning

Target: `feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports`
Files:
- `src/rotator_library/providers/codex_provider.py`
- `src/proxy_app/main.py`

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/codex_provider.py` — passed
- `uv run python3 -m py_compile src/proxy_app/main.py` — passed
- `uv run ruff check src/rotator_library/providers/codex_provider.py --select F401,F811,F821,E9` — passed
- `uv run ruff check src/proxy_app/main.py --select F401,F811,F821,E9` — passed
- `uv run python .fork/check-stack.py` — passed

Notes:
- Issue: Codex CLI deserializes /v1/models as `{"models": [...]}` (ModelsResponse in
  codex-rs/protocol) while the proxy returns OpenAI-compatible `{"object": "list", "data": [...]}`.
  This causes a startup warning: "missing field `models`". Inference still works.
- Fix: Cache the raw upstream models.json catalog during GitHub fetch in codex_provider.py,
  and detect Codex CLI requests via the `client_version` query parameter on /v1/models.
  When detected, return `{"models": <raw_catalog>}` passthrough instead of the
  OpenAI-compatible format. Non-Codex clients are unaffected.
- Ref: https://github.com/b3nw/LLM-API-Key-Proxy/issues/59

---

## 2026-06-19 — Fix codex exhausted credential never cleared on quota recovery

Target: `feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports`
Files:
- `src/rotator_library/providers/codex_provider.py`
- `src/rotator_library/providers/utilities/codex_quota_tracker.py`

Working commits before autosquash:
- `cc599b6d fixup! feat(codex): ...`

Final stack commit after autosquash:
- `26977cec feat(codex): ...`

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/codex_provider.py` — passed
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/codex_quota_tracker.py` — passed
- `uv run ruff check src/rotator_library/providers/codex_provider.py --select F401,F811,F821,E9` — passed
- `uv run ruff check src/rotator_library/providers/utilities/codex_quota_tracker.py --select F401,F811,F821,E9` — passed
- Hotpatched to docker-test and verified live: 28/28 credentials active, 0 exhausted, 0 cooldown

Notes:
- Bug: Codex quota tracker had zero calls to `clear_cooldown_if_exists`, while `base_quota_tracker.py` used it on every recovery. Once exhausted, credentials stayed blocked until the original `cooldown.until` timestamp expired, even if the API reported quota recovery (used_percent < 100).
- Fix: Added `clear_cooldown_if_exists` to all three paths that push quota data to UsageManager:
  - `_push_quota_to_usage_manager` (header/response path) — clears per-tier cooldowns on every API response
  - `_store_baselines_to_usage_manager` (initial fetch) — clears stale cooldowns during startup
  - `run_background_job` (periodic 300s refresh) — fetches ALL credentials, evaluates exhaustion waterfall, and clears recovered cooldowns
- Also pre-registered tier quota groups (`5h-limit`, `weekly-limit`, `monthly-limit`) in deterministic ascending window-size order for consistent UI display regardless of credential fetch order.
