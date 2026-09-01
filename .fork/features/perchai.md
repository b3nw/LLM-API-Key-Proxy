# Perchai Provider Feature History

## 2026-08-23: Fix tool-call loop

Perchai was firing `tool_call_delta` SSE events as probes - wrong tool names (e.g. `ast_grep_replace`), empty args. The real call only shows up later in `done.toolCalls`. Proxy emitted both, client picked up the probe name from the first delta, executed the wrong tool, got an error, fed the error back, model tried the same wrong tool again. Loop.

Pulled from the transaction logs: requests showed `ast_grep_replace` firing over and over with `{"entity":"user"}` args and `invalid value 'undefined' for '--lang'` errors. Model actually wanted `bash`.

Two bugs working together:

1. `tool_call_delta` events got through to the client. They shouldn't - they're probes.
2. `wrap_stream` strips `finish_reason` from any chunk that doesn't carry usage tokens. Perchai never sends usage. So `finish_reason: "tool_calls"` was silently dropped, client never saw it.

What I changed:

- `perchai_provider.py` - `_parse_sse_line` returns `None` for `tool_call_delta`. Only `done.toolCalls` emits tool calls. Added `tool_call_finish_emitted` flag so we don't double-emit `finish_reason: "tool_calls"` and added a terminal chunk when `done` lands after tool deltas.
- `streaming.py` - track `finish_reason_emitted`. If the stream ends and nothing reached the client, synthesize a final chunk with the accumulated `finish_reason` before `[DONE]`.
- `transforms.py` - Perchai is exempt from `_guard_thinking_tool_calls`, and fixed how `extra_body` merges with model options.
- `model_definitions.py` - `get_model_definition` now also looks at the `id` field, supports multi-segment keys.
- `tests/test_perchai_provider.py` - updated the 6 delta tests to assert `None`, added tests for done-event real UUID, wrap_stream finish_reason, and the thinking guard exemption. 82 pass.

Confirmed with a curl test: only `done.toolCalls` shows up, `finish_reason: "tool_calls"` is there.

**Rebuild gotcha**: PyInstaller `--onefile` caches bytecode in `build/` and `__pycache__/`. Wipe both before rebuilding or stale code ends up in the binary. Check the PYZ archive for `finish_reason_emitted` in `rotator_library.client.streaming` co_varnames to confirm the new code is actually shipped.

**Debug tip**: `--enable-request-logging` writes per-request dirs to `/usr/local/bin/logs/transactions/`. The `request.json` `data` key has the full payload going to Perchai.

## 2026-08-23: Align tests with other provider patterns

**Branch**: `feat/provider-app.perchai`
**Files changed**:
- `tests/test_perchai_provider.py` - Added 7 tests aligned with test_provider_plugins.py and test_vertex_provider.py patterns, removed unused import

**Tests added**:
1. Singleton pattern - two instantiations return same instance
2. get_model_tier_requirement returns None (no tier restrictions)
3. get_credential_priority returns None (not yet discovered)
4. skip_cost_calculation is True
5. default_rotation_mode is 'sequential'
6. Plain request doesn't auto-enable thinking in payload
7. reasoning_effort passes through without thinking config

**Test categories now covered** (aligned with other providers):
- Plugin registration: PROVIDER_PLUGINS, PROVIDER_MAP, LITELLM_PROVIDERS, PROVIDER_URL_MAP, DEFAULT_OAUTH_DIRS, ENV_OAUTH_PROVIDERS
- Provider contract: has_custom_logic, skip_cost_calculation, default_rotation_mode, singleton, tier_requirement, credential_priority
- Error handling: 10 error codes, 429 status, malformed body
- SSE/streaming: answer_delta, reasoning_delta, tool_call_delta, tool_use_end, unknown events, finish_reason
- Tool calls: synthetic IDs/names, real name resolution, index mismatch, multi-delta consistency, full stream integration
- Thinking config: transform_request hook, thinking disabled suppression (stream + non-stream), thinking in payload, reasoning_effort pass-through, effort stripped when disabled, plain request no auto-thinking, thinking policy patterns
- Credential resolution: file path, env virtual path, empty identifier fallback
- 401 refresh: streaming + non-streaming
- Quota tracking: background job config, model quota groups, run_background_job with invalid token
- E2E routing: option IDs route to real upstream
- Envelope structure: thinking config in request field

56 unit tests pass. 5 live API tests deselected (expired token).

## 2026-08-23: Fix thinking config and reasoning_effort mapping

**Branch**: `feat/provider-app.perchai` (worktree: `feat-provider-app.perchai`)
**Files changed**:
- `src/rotator_library/providers/perchai_provider.py` - Added `thinking` and `reasoning_effort` to `SUPPORTED_PARAMS`, updated `_is_thinking_disabled` to check top-level `thinking` key (not just `extra_body.thinking`), strip `reasoning_effort` from payload when thinking disabled
- `tests/test_perchai_provider.py` - Added 6 new tests (RED-GREEN TDD) for thinking config in payload, reasoning_effort pass-through, stream stop chunk, and envelope structure

**Root cause**: `reasoning_effort` from model options (PERCHAI_MODELS env var) was set in `kwargs` by transforms.py step 3, but `_build_payload` only copied `SUPPORTED_PARAMS` keys. Since `reasoning_effort` was not in the set, it was silently dropped. The thinking config from `extra_body` was merged correctly, but `reasoning_effort` set directly in kwargs was lost.

**Fix**:
1. Added `thinking` and `reasoning_effort` to `SUPPORTED_PARAMS` so they pass through `_build_payload`
2. Updated `_is_thinking_disabled` to check top-level `thinking` key (after extra_body merge, thinking is at the top level, not nested in `extra_body`)
3. Strip `reasoning_effort` from payload when thinking is disabled (contradictory config)

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_provider.py
uv run ruff check src/rotator_library/providers/perchai_provider.py --select F401,F811,F821,E9
uv run python -m pytest tests/test_perchai_provider.py -v --tb=short -k "not expired_token and not option_id and not run_background"
```
49 unit tests pass (was 43 before this session, 48 after previous session's work).

**Pending**: Deploy to running container to verify gemma-4-31b responds. Token expired, needs `perch login` to re-authenticate for live API tests.

## 2026-08-22: Thinking disabled detection and normalization

**Branch**: `feat/provider-app.perchai`
**Files changed**:
- `src/rotator_library/providers/perchai_provider.py` - Added `transform_request` hook, `_is_thinking_disabled` helper, `_build_model_response` method, thinking suppression in stream and non-stream paths
- `tests/test_perchai_provider.py` - Added 6 tests for thinking normalization

**Changes**:
- Added `transform_request` hook to strip `reasoning_content` from assistant messages when thinking disabled
- Added `_parse_sse_line` `thinking_disabled` param to suppress `reasoning_delta` events
- Added `_build_model_response` for testable non-streaming response building
- Added `_extract_tool_names` for real tool name resolution from request payload

48 tests pass (43 unit + 5 live API).

## 2026-08-29: Live thinking-respect regression tests

**Branch**: `feat/provider-app.perchai` (worktree: `feat-provider-app.perchai`)

**Question answered**: does the Perch upstream actually honor the thinking values the proxy normalizes? Mocked tests only proved we SEND them. Live tests now prove upstream behavior.

**Files changed**:
- `tests/test_perchai_provider.py` - Added 2 live e2e thinking tests gated on `~/.perch/cli-auth-session.json`; removed module-level `pytestmark` skipif (mocked tests now run everywhere; live tests carry per-test `@live_only` gate; also applied to the 2 expired-token refresh tests and the option_id probe test).

**Tests added**:
1. `test_live_thinking_disabled_suppresses_reasoning` - `thinking={"type": "disabled"}` -> upstream returns 0 reasoning chars (streaming, real API.
2. `test_live_thinking_effort_modulates_reasoning_volume` - `reasoning_effort=high` produces MORE reasoning chars than `low` (soft monotonic, one retry guard on high). Model: `perchai/wandb-deepseek-ai-deepseek-v4-flash` (cheap Starter-tier option ID, overridable via `PERCHAI_THINKING_TEST_MODEL`).

**Result**: Perch upstream RESPECTS the normalized thinking values:
- `thinking: disabled` suppresses `reasoning_content` entirely (0 chars.
- `reasoning_effort` modulates reasoning volume (high > low.
- Expired access token auto-revived via `refresh_on_401` (refresh token still valid.

**Verification**:
```bash
uv run python3 -m py_compile tests/test_perchai_provider.py
uv run ruff check tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run python -m pytest tests/test_perchai_provider.py -v --tb=short
```
85 tests pass (79 mocked + 6 live: 2 expired-token refresh,  ​2 option_id probes,​ 2 thinking-respect.

## 2026-08-29: Re-read OAuth session file on every request

**Branch**: `feat/provider-app.perchai` (worktree: `feat-provider-app.perchai`)

**Problem**: `PerchaiAuthBase.load_session` cached the parsed session in `self._session` after the first call. When `perch login` rewrote `~/.perch/cli-auth-session.json` in a separate process, the running proxy still served the stale cached token. Fix required a reboot.

**Files changed**:
- `src/rotator_library/providers/perchai_auth_base.py` - Removed the `self._session` in-memory cache in `load_session` (dropped the early-return + two assignments). `_ensure_session` now always invokes `load_session`, which re-reads the file on every call. Supabase config cache (`_supabase_url`/`_supabase_anon_key`) kept (separate concern, expensive HTTP).
- `tests/test_perchai_provider.py` - Added `test_load_session_re_reads_file_on_every_call` (RED-first TDD): monkeypatches `_resolve_session_file` to a tmp path, writes a token, loads, rewrites the file with a new token (simulating `perch login`), loads again, asserts the new token is returned. Fails before the fix, passes after.

**Behaviour preserved**:
- `refresh_on_401` single-flight still works: it calls `_ensure_session` (now always reloads), compares the on-disk token to the expired one, and skips redundant refresh if another coroutine already wrote a new token via `_persist_session`.
- `refresh_token` reads the (rotated) refresh token from disk each time so single-use rotation still applies.
- `get_app_url` re-reads the file too; appUrl rarely changes, cost is one small JSON parse per call.

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_auth_base.py tests/test_perchai_provider.py
uv run ruff check src/rotator_library/providers/perchai_auth_base.py tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run python -m pytest tests/test_perchai_provider.py --tb=short
```
86 tests pass (80 mocked + 6 live: 2 expired-token refresh, 2 option_id probes, 2 thinking-respect).

## 2026-08-29: Fix sync transform_request hook await bug + truncation investigation

**Branch**: `feat/provider-app.perchai` (worktree: `feat-provider-app.perchai`)

**Bug**: `PerchaiProvider.transform_request` is a sync method returning `List[str]`, but `ProviderTransforms.apply` awaited it -> every Perchai request logged `Provider transform_request hook failed: object list can't be used in 'await' expression`. Side effects (reasoning_content stripping) ran before the await failure, but the error was logged per request and the modifications list was lost.

**Files changed**:
- `src/rotator_library/client/transforms.py` - `apply` now handles both sync and async `transform_request` hooks (`inspect.isawaitable` branch).
- `tests/test_perchai_provider.py` - Added `test_sync_transform_request_hook_runs_through_transforms` (asserts stripping happens AND no "hook failed" log via caplog; RED before fix).; hardened `test_live_thinking_effort_modulates_reasoning_volume` (max_tokens 512->2048, longer prompt, assertion changed from directional `high>low` to volume-difference `abs(high-low)>0.3*max` since observed effort direction is NOT monotonic for deepseek-v4-flash: low->~6.2K reasoning chars, high->~2.5K (stable, inverted direction).

**Truncation investigation (user report: thinking truncated at~15-20s)**:
- Proxy SSE output verified FULL via `curl -o` + python httpx (37.9KB/96 lines/[DONE] for a 15.2s long-thinking request; 333KB/815 lines/[DONE] for 17.9s request).. Earlier "truncated" captures were a bash-tool stdout-redirect artifact (120-char/line +3000-byte truncation with "..." suffix), NOT the proxy.

- No 15-20s timeout exists in the proxy chain: container `GLOBAL_TIMEOUT=600`, `TIMEOUT_READ_STREAMING=360`; code defaults 30/300 but env overrides. No premature-end warnings/errors/client-disconnects in proxy logs (15:43-16:53 window).
- Live effort test finding: Perchai upstream DOES respond to `reasoning_effort` (volume changes significantly;, but direction is inverted for this model/prompt (low effort -> MORE reasoning volume than high). May warrant config-side investigation (e.g. map effort values differently for perchai).

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/client/transforms.py tests/test_perchai_provider.py
uv run ruff check src/rotator_library/client/transforms.py tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run python -m pytest tests/test_perchai_provider.py tests/test_provider_transforms.py tests/test_request_sanitizer.py --tb=short
```
115 tests pass.

## 2026-08-31: Fix silent error dropping in streaming

**Branch**: `feat/provider-app.perchai`
**Files changed**:
- `src/rotator_library/providers/perchai_provider.py` - Changed error handling in `_parse_sse_line` when `done` event has `ok: false`
- `tests/test_perchai_provider.py` - Updated test to expect RuntimeError instead of None return

**Bug**: When Perch.ai sent `done` event with `ok: false` and error message, the error was logged at DEBUG level and silently dropped (`return None`). This caused streams to end prematurely without visible error, appearing as "thinks half second then interrupts" in long sessions.

**Root cause**: `lib_logger.debug()` + `return None` made the error invisible in normal logs and prevented error propagation to the streaming handler.

**Fix**:
- Changed `lib_logger.debug` to `lib_logger.warning` (visible in normal logs)
- Changed `return None` to `raise RuntimeError(f"Perchai upstream error: {error_text}")` (propagate error)
- Error now propagates to streaming handler for proper retry/rotation

**Test updated**: `test_parse_sse_done_event_with_error_returns_none` -> `test_parse_sse_done_event_with_error_raises_runtime_error`

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py
uv run ruff check src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_provider.py::test_parse_sse_done_event_with_error_raises_runtime_error -v
```
Test passes. All Perchai tests pass (except pre-existing credential-related failures).

**Impact**: Upstream errors now visible in logs and properly handled by retry/rotation logic instead of silent truncation.

## 2026-09-01: Fix whitespace mangling + DeepSeek thinking_budget cap

**Branch**: `feat/provider-app.perchai`

**Bug 1 - mangled markdown** (`##Title`, `needs200+`): `_parse_sse_line` called `.rstrip()` on every `answer_delta`/`text_delta`/`reasoning_delta` chunk. DeepSeek emits whitespace as separate chunks (`"needs"` + `" "` + `"200+"`); rstrip turned the `" "` chunk into `""`, losing the space. Qwen bundles spaces with words so the bug was latent there - DeepSeek-specific visibility, proxy-side cause.

**Bug 2 - reasoning truncation**: DeepSeek-v4-flash reasoning cut mid-sentence at ~13.3K chars (~3300 tokens) with `finish_reason=stop` + `[DONE]`. Proxy forwarded all chunks - NOT a proxy timeout. Confirmed upstream: Qwen 3.8 flash does not truncate. Reverse-engineered the `perchai-cli` bundle (`~/.asdf/installs/nodejs/24.8.0/lib/node_modules/perchai-cli/dist/perch.mjs`, binary - grep with `-a`) and found the wandb payload builder emits `chat_template_kwargs={enable_thinking: bool}` (vLLM/SGLang passthrough). Live probe proved the Perchai server ALSO honors `chat_template_kwargs.thinking_budget`.

**DeepSeek effort mapping** (official docs): `medium -> high`, `high -> high`. Only `low` is a real reduction. Prior cap `high->medium` was a no-op.

**Files changed**:
- `src/rotator_library/providers/perchai_provider.py`:
  - `_parse_sse_line`: removed `.rstrip()` from answer/text/reasoning delta text.
  - `transform_request`: cap `reasoning_effort` `high -> low`; inject `chat_template_kwargs={enable_thinking:true, thinking_budget:3000}` when thinking enabled and effort is medium/high.
- `tests/test_perchai_provider.py`:
  - `test_text_delta_preserves_whitespace_only_chunks` (RED before fix)
  - `test_reasoning_effort_capped_to_low` (parametrized, replaces broken `_capped_to_medium`)
  - `test_high_effort_injects_thinking_budget`

**Live verification** (thinking_budget probe): budget=1500 -> 4493 chars ends `.`; budget=3000 -> 3454 chars ends clean; no budget -> 13317/13366 chars truncated mid-word. 3000 is safe ceiling under the ~3300-token wall.

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py
uv run ruff check src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_provider.py tests/test_provider_transforms.py -k "not live"
```
102 pass; 4 live-gated failures pre-existing (`refresh_token_already_used` - stale session, unrelated).

**Deploy**: binary rebuilt at `dist/proxy_app`; copy into `llm-proxy` container and restart (docker ops manual).

## 2026-09-01: Configurable thinking budgets

**Branch**: `feat/provider-app.perchai`

**Problem**: Hardcoded `thinking_budget=3000` for all models. DeepSeek needs it (wall at ~3300 tokens), but other models (Gemma, Qwen, GLM) don't need the cap and may benefit from longer reasoning.

**Solution**: Env var configuration with fallback chain:
1. `PERCHAI_{MODEL}_THINKING_BUDGET_{LEVEL}` - model-specific override
2. `PERCHAI_THINKING_BUDGET_{LEVEL}_DEFAULT` - level-wide default
3. Hardcoded `3000` for DeepSeek models only (discovered wall)
4. No cap (upstream decides) for non-DeepSeek models

Model name normalization: strip `perchai/` prefix, replace `-` and `.` with `_`, uppercase. Example: `perchai/wandb-deepseek-ai-deepseek-v4-flash-0731` -> `WANDB_DEEPSEEK_AI_DEEPSEEK_V4_FLASH_0731`.

**Files changed**:
- `src/rotator_library/providers/perchai_provider.py`:
  - Added `_normalize_model_name_for_env(model_name)` - model name normalization
  - Added `_get_thinking_budget(model_name, reasoning_effort)` - env var lookup with fallback chain
  - Updated `transform_request` to use configurable budget instead of hardcoded 3000
- `tests/test_perchai_provider.py`:
  - 11 new tests (RED-GREEN TDD): model normalization, reasoning level normalization, env var lookup (model-specific, level default, override precedence, deepseek fallback, non-deepseek no cap), integration with transform_request
- `.env.example`: documented new env vars with examples
- `DOCUMENTATION.md`: added "Configurable Thinking Budgets" subsection under Perchai section

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py
uv run ruff check src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_provider.py -k "thinking_budget or normalize_model" -v
```
10 new tests pass. Existing `test_high_effort_injects_thinking_budget` still passes (deepseek fallback to 3000). 97 total tests pass (6 live-gated failures pre-existing - stale OAuth session).

**Startup logging**: Added `lib_logger.info()` in `__init__` that scans for `PERCHAI_*THINKING_BUDGET*` env vars and logs them at startup. Follows NanoGPT pattern - only log when non-default values exist. 2 additional tests verify logging behavior (singleton cache cleared for test isolation). 99 tests pass total.
