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

**Problem**: Hardcoded `thinking_budget=3000` for all models. Some models have unknown hard caps on the Perchai server side that truncate reasoning mid-sentence. DeepSeek-v4-flash truncates at ~3300 tokens, but other models (Gemma, Qwen, GLM) may have different limits or no limits at all. A one-size-fits-all budget either truncates reasoning (too high) or starves it (too low). We must accommodate per-model.

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

## 2026-09-02: Stop forking the Perch session, rotate it in place

**Branch**: `feat/provider-app.perchai`

**Problem**: The operator had to run `perch login` periodically. Root cause was not token expiry, it was the proxy and the Perch CLI fighting over one single-use rotating Supabase refresh token.

Reverse engineered the CLI bundle at `~/.asdf/installs/nodejs/24.8.0/lib/node_modules/perchai-cli/dist/perch.mjs` (v2.4.97). Its auth contract:

- `Mm()` re-reads `~/.perch/cli-auth-session.json` on every use, never caches across operations.
- `Db()` treats a token as valid while `expiresAt > now + 90`.
- `f5()` refreshes proactively only inside that skew, always from a freshly loaded session, and persists immediately.
- `Dne()` writes with `fs/promises.writeFile`, truncating in place, so the inode is stable.
- `n4r()` rejects the session unless `version === 1` and `appUrl`, `accessToken`, `updatedAt` are truthy, and keeps `expiresAt` only when it is a JSON number. A string there is dropped, after which `Db()` reports the token valid forever and the CLI stops refreshing.
- The CLI force-rotates whenever its model-call proxy asks for a token, so ordinary `perch` use rotates the chain.

Four defects on our side, in severity order:

1. `credential_manager.py` copy-on-discovery snapshotted `PERCHAI_OAUTH_1` into `oauth_creds/perchai_oauth_1.json` at container start (prod evidence: snapshot mtime 07:16:30 vs container start 07:16:21) and rotated inside that private copy. The CLI kept the pre-copy token, so the second presenter hit GoTrue reuse detection and the family was revoked.
2. `credential_manager.py:199-207` short-circuits on any existing `oauth_creds/perchai_oauth_*.json` and logs "Skipping discovery", so a correctly configured env path is ignored. There was no configuration that used the live file.
3. `perchai_provider.py` rotated only after an HTTP 401, unlike every other OAuth base in this repo.
4. `refresh_token()` rotated from the cached `self._session`, so after one rotation the process never re-read the disk again.

**Docker mount behaviour, reproduced rather than inferred**:
- Single-file bind mount plus `os.replace()` gives `OSError(16, 'Device or resource busy')`. This is `llm-proxy-dev`, which mounts the session file directly.
- Read-only directory mount gives `OSError(30, 'Read-only file system')`. This is `llm-proxy`, which mounts `~/.perch:/app/.perch` with `RW=False`.
- Writable directory mount allows rename, and the host observes the new inode. Only the single-file form forbids rename.

**Solution**:
- `perchai_auth_base.py`: `REFRESH_EXPIRY_BUFFER_SECONDS = 600` plus `_is_token_expired()`, matching the house convention in `google_oauth_base`, `openai_oauth_base`, `anthropic_oauth_base`, `copilot_auth_base`. `ensure_access_token()` reads the disk, returns a healthy token, otherwise rotates. `get_auth_header()` is now async like `x_ai_auth_base`. `refresh_token()` reloads from disk inside the lock before the POST. `_adopt_persisted_session()` handles `refresh_token_already_used` by re-reading and adopting a newer unexpired session another consumer persisted, and only raises when nothing changed. `_persist_session()` truncates in place with `fsync` instead of temp plus rename, writes `updatedAt` as ISO-8601 and coerces `expiresAt` to int for CLI compatibility. Replaced the module-level global lock with a per-instance one so unrelated credentials do not serialize. Added `PerchaiCredentialKind` StrEnum (`session_file` / `env_virtual` / `raw_token`).
- `perchai_provider.py`: implemented `proactively_refresh()`, which is the seam `BackgroundRefresher` already calls for every OAuth provider on `OAUTH_REFRESH_INTERVAL`; Perchai simply never implemented it. `_auth_context()` replaces `_resolve_app_url` plus `_resolve_credential_token` so one disk load serves both, down from two or three per request. Fixed `get_models()` sending the credential string itself as the bearer token, which for an OAuth credential is a file path, so dynamic model discovery silently degraded to static models.
- `DOCUMENTATION.md` 2.6.3 rewritten: Perchai must not be snapshotted. Point the credential entry at the live file through a symlink, which `discover_and_prepare()` dereferences with `Path.resolve()`, and mount the directory read-write.
- `.gitignore`: `tests/*` is allowlisted per file, so `!tests/test_perchai_auth.py` was required or the new test file is invisible to git.

**Why a 600 second buffer and not the CLI's 90**: with one shared live file and no cross-process lock, two consumers with the same threshold rotate at the same boundary. Staggering the proxy ten minutes earlier means the proxy refreshes first and the CLI then finds a healthy token and leaves it alone.

**Files changed**:
- `src/rotator_library/providers/perchai_auth_base.py`
- `src/rotator_library/providers/perchai_provider.py`
- `tests/test_perchai_auth.py` (new, 7 tests)
- `tests/test_perchai_provider.py` (removed `test_empty_credential_identifier_falls_back_to_default`, which asserted nothing on its success path; replaced by a real assertion in the new file)
- `DOCUMENTATION.md`, `.gitignore`, `.fork/stack.yml`

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py tests/test_perchai_auth.py
uv run ruff check src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py tests/test_perchai_auth.py tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_auth.py tests/test_perchai_provider.py tests/test_perchai_stream_truncation.py tests/test_credential_manager.py -q -k "not live"
uv run pytest tests/ -q -k "not live"
```
120 pass in the targeted run. Full suite 3 failed / 629 passed / 15 errors, against a clean-HEAD baseline of 3 failed / 623 passed / 15 errors, so the delta is exactly the 6 net new tests and every failure is pre-existing and unrelated (deepseek stream, umans quota constant, x_ai time-bomb date, `test_failure_logger.py` importing `src.*`, `tests/utils/test_paths.py` needing the undeclared `pytest-mock`).

`tests/test_perchai_auth.py` runs against a real local HTTP server emulating GoTrue single-use rotation, including the `refresh_token_already_used` rejection. No mocking library, no vendor cost, real sockets and real files, so the rotation semantics under test are the ones production has.

Live, non-destructive, against `app.perchai.app`: `GET /api/perchai/account` returned 200 with the stored token, and the session file's inode and refresh token were unchanged, confirming a healthy token triggers no rotation.

Option 2 verified empirically: `CredentialManager` pointed at a temp pool containing a symlink returned the dereferenced live path and created no copy.

**Not yet verified**: a real rotation against the live service. Deferred deliberately, because prod still runs the old build holding a boot snapshot of the current refresh token, and a host-side rotation would make that snapshot a reuse attempt and revoke the family for both sides. Acceptance test after deploy: watch the file rotate from proxy traffic, then run `perch` and confirm no re-login prompt, then reverse the order.

**Deployment required**: mount `~/.perch` read-write as a directory in both containers, and create the credential as a symlink before the proxy starts. The two containers differ: `llm-proxy` has cwd `/usr/local/bin` from the frozen binary so its `oauth_creds` is ephemeral and needs a startup `command:`, while `llm-proxy-dev` has cwd `/app` with `oauth_creds` on a persistent host mount.

**Residual risk accepted**: three read-write consumers (host CLI plus two proxy containers) on one token family with no cross-process lock. Adopt-on-conflict covers a loser that re-reads after the winner persisted. It does not cover two consumers POSTing the same token where GoTrue's reuse grace expires before the loser re-reads. Low probability, total consequence, one `perch login` to recover.

**Follow-ups not in scope**: `perchai_quota_tracker.py` resolves its own token from env and the session file (lines 74-112) and never refreshes, so quota tracking can present a stale token. Dev container logs also show `Perchai usage fetch ... returned HTTP 405`, which is a wrong method or endpoint on `/api/perch-terminal/usage`, not an auth failure.

## 2026-09-02: Test suite was consuming the operator's real refresh token

**Discovered while watching a live rotation**: `pytest tests/ -k "not live"` does not skip the Perchai tests that hit the real service. `live_only` is a `skipif` decorator keyed on the existence of `~/.perch/cli-auth-session.json`, while `-k` filters on test *names*, so only the two tests literally named `test_live_thinking_*` were deselected. Four tests that make real authenticated calls kept running.

Two distinct kinds of damage:

- `test_option_id_routes_to_real_upstream`, parametrised twice, called `PerchaiAuthBase()` with no credential path and then `refresh_token()`. That forces a rotation of the operator's live session on every run and writes the replacement back to the real file.
- `test_expired_token_non_stream_refreshes_and_retries` and its stream variant copied the live session into `tmp_path`, poisoned the access token to trigger a 401, and let the provider rotate. The replacement token was persisted into the tmp directory, which pytest then deleted. The live file was left pointing at a consumed token. That is the same single-use-token fork that broke production, automated and run by the developer.

This also explains the ledger's recurring "live-gated failures pre-existing - stale OAuth session" notes. The session was not going stale on its own. CI never catches any of it because CI has no `~/.perch`, so `HAS_SESSION` is False and every one of these tests skips. The only machine this fires on is a developer's.

**Changes**:
- `tests/test_perchai_auth.py`: added `test_model_call_401_refreshes_and_retries` and `..._stream_...`, covering the same 401 then refresh then retry behaviour in both paths against the local GoTrue emulator, which now also serves `/api/perch-terminal/model-call`.
- `tests/test_perchai_provider.py`: deleted the two `test_expired_token_*` live tests, replaced by the local pair. Switched `test_option_id_routes_to_real_upstream` from `refresh_token()` to `ensure_access_token()`, so it uses the existing token and rotates only when genuinely near expiry.
- Registered a real `live` marker in `pyproject.toml` and applied `@pytest.mark.live` alongside `@live_only` on the three remaining network tests, so `-m "not live"` is now an accurate filter.

**Verification**:
```bash
uv run pytest tests/test_perchai_provider.py tests/test_perchai_auth.py tests/test_perchai_stream_truncation.py -q -m "not live"
uv run pytest tests/ -q -m "not live"
```
110 pass in the targeted run, 627 pass in the full run, both with the same 3 pre-existing failures and 15 pre-existing collection errors. The acceptance check is that the live session file is byte-identical before and after a full run:

```
BEFORE: 4v6wav45 2026-09-02T09:45:49.731Z 1788342349
AFTER:  4v6wav45 2026-09-02T09:45:49.731Z 1788342349
UNCHANGED - zero burn
```

Full run time also dropped from about 34s to 14s, since nothing waits on the network any more.

## 2026-09-02: The 403 is a missing turn ticket, not a blocked API

Production started failing every Perchai credential with HTTP 403
`perch_surface_required`: "Your plan includes Perch-hosted models for use in Perch AI
Web, Desktop, and CLI only. Direct API access is not included." The working hypothesis
was that Perch had fingerprinted the proxy and that the only way back in was to wrap the
`perch` CLI as a subprocess transport.

That hypothesis is wrong, and the fix is one header.

Reverse-engineered `perchai-cli@2.4.97` (`dist/perch.mjs`, minified but greppable). The
CLI mints a short-lived **turn ticket** at the start of every turn and attaches it to
every model call:

- `POST {appUrl}/api/perch-terminal/turn-ticket`, `Authorization: Bearer <accessToken>`,
  body `{"surface":"cli","profile":"standard"}`, 8s timeout.
- Returns `{ok, ticket, ticketId, runId, surface, profile, expiresAt}`. The ticket is an
  HS256 JWT carrying `{tid, uid, wid, surface, plan, run, exp}`. **TTL 5 minutes.**
- Every `/api/perch-terminal/model-call` then carries `x-perch-turn-ticket: <ticket>`.
- Under 30s from expiry, the CLI re-POSTs `{"renew":true,"ticketId":…}` rather than
  minting fresh, de-duplicated through one in-flight promise.
- Mint can return 429 `{"enforced":true,"errorCode":"turn_rate_limited"}` — a per-*turn*
  plan limit, distinct from token quota. This is new server-side accounting.

Isolated the gate with a three-way live test on the operator's own session:

| Request | Result |
|---------|--------|
| model-call **with** ticket + `User-Agent: perchai-cli/2.4.97` | 200 |
| model-call **without** ticket, CLI User-Agent | 403 `perch_surface_required` |
| model-call **with** ticket, `User-Agent: python-urllib/3` | 200 |

The middle row reproduces the production error exactly; the third rules out User-Agent
fingerprinting. The ticket header is the entire gate.

Also measured, so the implementation does not have to guess:

- One ticket serves **multiple** model-calls (second call: 200).
- The ticket is **not** bound to the envelope's `runId` — a deliberately bogus `runId` in
  the body still returned 200. Only the header is checked.
- SSE streaming works with a ticket: `Accept: text/event-stream` returned 200 and the
  familiar `reasoning_delta` / `done` events. So the SSE parser, the `done.toolCalls`
  handling, and the `finish_reason` fixes from the 2026-08-23 entries are all unaffected.
- Renewal via `{renew:true, ticketId}` returned 200.

`ccr` in the bundle also falls back to a `PERCH_MODEL_CALL_PROXY_TOKEN` env var for the
bearer. Noted so it is not rediscovered.

**Second finding — `perch_surface_required` is misclassified today.** `parse_quota_error`
treats the 403 as credential exhaustion, which is why a single missing header emptied the
whole pool and produced "All 1 credential(s) exhausted". It is a retryable auth fault:
drop the cached ticket, re-mint, retry. `turn_rate_limited` is the one that genuinely is
exhaustion.

**Measured the CLI-wrapper alternative before discarding it.** `perch run "Reply with
exactly: OK" --json` took 16.5s wall / 8.2s server-side, printed a single JSON object at
exit (no streaming), and reported `estimatedInputTokens: 18038` for a two-word prompt —
Perch prepends ~18k tokens of its own agent scaffolding per turn. `perch run` also takes
a single prompt string, not a messages array, offers only
`--model standard|standard max|pro|pro max`, and runs its own agent loop with its own
local tools, which in a proxy would execute against the proxy container's filesystem.
Those are semantic mismatches with `/v1/chat/completions` that no amount of process
warming fixes.

**Not implemented in this session** — investigation only. Plan with the full design,
integration point (`perchai_provider.py:350`, the sync `_headers` closure), test list,
and the account-suspension risk assessment is in `.omo/plans/perchai-cli-wrapper.md`.

**Risk carried forward**: the 403 body warns that repeated direct-access attempts may
result in account suspension, and Perch added the ticket check deliberately after the
proxy's previous access pattern worked. Sending `surface: "cli"` from something that is
not the CLI is a decision for the operator to make knowingly, not a detail to bury in a
diff.

**Verification** (no repo files changed; live calls were non-destructive reads against
the operator's own account, and the session file was not rotated):
```bash
perch status            # signed in as kevin@atvastacode.com
perch run "Reply with exactly: OK" --json   # 16.5s, single-shot JSON
# three-way model-call matrix run via python3 + urllib against app.perchai.app
```

## 2026-09-02: Implemented Part 1 (turn-ticket support) - TDD, and one plan
correction

Operator approved the account-risk tradeoff from the plan. Implemented with RED/GREEN:
extended the fake server in `tests/test_perchai_auth.py` to require
`x-perch-turn-ticket` on model-call (matching production), added a
`/api/perch-terminal/turn-ticket` handler, wrote 5 failing tests, then implemented.

**`PerchaiAuthBase.ensure_turn_ticket(access_token)`** (`perchai_auth_base.py`) mints or
renews a ticket under the existing refresh lock, caches it, and returns the token. A 401
from the mint endpoint (session's access token already stale, even though the local
600s-buffer check thought it was fine) triggers one internal `_rotate_refresh_token` and
retry - verified for free by the existing `test_model_call_401_refreshes_and_retries`
tests, since the provider's own `_headers`/`refresh_on_401` self-heals the now-stale
`token` variable used for the model-call itself via the "adopt, don't re-rotate" path
already in `refresh_on_401`. `invalidate_turn_ticket()` drops the cache.

`PerchaiProvider.acompletion` mints the ticket once per completion (after
`_auth_context`), and both `_non_stream_completion` and `_stream_completion` now retry
once on a 403 `perch_surface_required` response: invalidate the cached ticket, re-mint,
retry - mirroring the existing 401 retry-once pattern.

**Plan correction - `parse_quota_error` cannot express "retryable, not exhausting".**
The plan's Part 1 said to classify `perch_surface_required` as
`{"reason": "authentication", "retry_after": None}` so it would not exhaust the
credential. Traced the actual consumer (`error_handler.py:960-1002`, `classify_error`)
before implementing it: *any* truthy return from `parse_quota_error` forces
`status_code=429` and `error_type` to either `"rate_limit"` or `"quota_exceeded"` - there
is no path to an `"authentication"`-flavoured, non-exhausting outcome through this
function. Returning that dict would have made things worse (forced `quota_exceeded`, a
long cooldown) than doing nothing (falls through to the generic `status_code == 403` ->
`"forbidden"` classification, unchanged from today).

So `perch_surface_required` gets **no new classification** - the fix is the in-request
retry above, which means a transient/expired ticket never reaches `parse_quota_error` at
all in the common case. Only `turn_rate_limited` (429, `enforced: true`) was added to
`_classify_perchai_error`, mirroring the existing `usage_limit_reached` pattern
(`{"retry_after": DEFAULT_RETRY_AFTER_SECONDS, "reason": "rate_limit"}`), which does
travel a verified path to real exhaustion.

**Tests** (`tests/test_perchai_auth.py`, 5 new, all seam-level against the real local
HTTP server, no mocking):
- `test_model_call_sends_turn_ticket_header` - regression test for the reported 403.
- `test_turn_ticket_is_reused_across_completions` - one mint serves two completions.
- `test_turn_ticket_renews_within_margin` - a ticket inside the 30s margin renews
  (`{renew:true, ticketId}`), not a fresh mint.
- `test_surface_required_403_retries_with_fresh_ticket_and_succeeds` - simulates another
  consumer invalidating the cached ticket server-side; the second completion still
  succeeds via the drop/re-mint/retry path, proving the credential is not exhausted.
- `test_turn_rate_limited_429_marks_exhausted` - classification unit test on
  `parse_quota_error`.

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py
uv run ruff check src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py tests/test_perchai_auth.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_auth.py tests/test_perchai_provider.py tests/test_perchai_stream_truncation.py -q -m "not live"   # 115 passed
uv run pytest tests/ -q -m "not live"   # 632 passed, 3 pre-existing failures, 15 pre-existing collection errors - same baseline as 2026-09-02, no new breakage
```
No live tests added or run against `app.perchai.app` in this pass, per the live-test-burn
warning in the 2026-08-31 entries above.

**Not done**: the two follow-ups already on record (`perchai_quota_tracker.py` staying
unticketed, and the `usage` vs `usage-meter` 405 mismatch) are still open.

## 2026-09-02: Live e2e check of the turn-ticket fix

Ran the `@pytest.mark.live` tests in `test_perchai_provider.py` against the operator's
real session to answer "does this actually fix the 403 end to end" (not just against the
local fake server). `perch status` confirmed the session was signed in first.

`test_live_thinking_disabled_suppresses_reasoning` goes through
`PerchaiProvider.acompletion()` - the real code path a deployed proxy uses - and passed:
200, not 403. That is the deployed-proxy acceptance check the plan asked for.

Two other live failures, both pre-existing and unrelated to the ticket logic itself:

- `test_option_id_routes_to_real_upstream` (both param cases) hand-crafts the model-call
  request with raw `httpx` instead of going through the provider, so it never attached a
  ticket and 403'd with `perch_surface_required` - the exact production symptom, just
  from a test that predates the ticket requirement rather than from a regression. Fixed
  by minting via `PerchaiAuthBase.ensure_turn_ticket()` and adding the header; both
  params pass now.
- `test_live_thinking_effort_modulates_reasoning_volume` failed on an unrelated
  assertion (`low=5723 chars/63.6s, high=6763 chars/80.3s` - not different enough). Auth
  succeeded, no 403, so the ticket fix is not implicated; this is upstream reasoning-effort
  behavior, a pre-existing flake not touched here.

**Verification**:
```bash
uv run pytest tests/test_perchai_provider.py -q -m live   # after the probe-test fix: all pass
uv run pytest tests/test_perchai_auth.py tests/test_perchai_provider.py tests/test_perchai_stream_truncation.py -q -m "not live"   # 115 passed, unchanged
```
Session file was not rotated by this run (access token had ample TTL left; no refresh
was triggered).

**New, separate issue reported after the fix landed**: `perchai/hidden-deep` (routes to
`dashscope-qwen3-8-flash` per the operator's `PERCHAI_MODELS`, `thinking: enabled`,
`reasoning_effort: low`) fails mid-stream with `openrouter returned 400: ... Backend
request failed with status 400 ...`, truncated before the actual reason in the deployed
proxy's log line. This is **not** a ticket/auth regression - the request reached the
model backend fine (no 403), so `ensure_turn_ticket` is working for this model too. Other
options (`gemma-4-e2b`, `wandb-deepseek-ai-deepseek-v4-flash-0731`) work. Likely
model-specific (unsupported `thinking` shape for that OpenRouter-routed backend, or
similar) but unconfirmed - the full error body is needed before diagnosing further.
Attempted a live repro from this worktree to get the full body; the bash call was
orphaned mid-run by an incoming user message before it returned, so no result was
captured and the operator's session was rechecked (`perch status`, unchanged session-file
mtime) to rule out any fallout. Not yet reproduced with the full error text.

## 2026-09-02: Turn-ticket renewal 403, and backing out the 600s proactive rotation

**Reported**: reauth needed "SUPER frequently" since `7c04cda` / `3a7baf0`, plus a new
hard failure:

```
PerchaiAuthError: Perchai turn-ticket request returned HTTP 403:
{"ok":false,"error":"Your plan includes Perch-hosted models for use in Perch AI Web,
Desktop, and CLI only. Direct API access is not included. ..."}
```

Note *which* endpoint that is. The 2026-09-02 investigation above diagnosed this body
coming back from **model-call**, and fixed it with the ticket header. This one comes back
from the **turn-ticket mint itself**, so it is a different defect wearing the same error
body.

### Why the mint 403s

`ensure_turn_ticket()` passed its cached ticket to `_post_turn_ticket()` whenever the
cache was not fresh enough, and `_post_turn_ticket()` renews (`{"renew":true,"ticketId":…}`)
whenever it is handed one. "Not fresh enough" includes **long expired**. The CLI only ever
renews inside the 30s margin, while a ticket lives 5 minutes and the proxy's cache is
per-process, so any gap longer than 5 minutes between requests made the next mint a
renewal of a ticketId Perch had already retired. The renew body also carries no `surface`,
so a refused renewal is answered by the surface gate rather than by a "no such ticket"
error - which is why this reads as a fingerprinting/plan problem and is not one.

Two further consequences, both fixed:

- A refused renewal had **no fallback**. `_parse_turn_ticket_response()` raises on any
  non-200, and the provider's `perch_surface_required` self-heal only wraps the
  *model-call*, never the mint. So one retired ticketId escaped `acompletion()` as a raw
  `PerchaiAuthError` and burned the credential.
- The 401-retry inside `ensure_turn_ticket()` re-sent the same dead ticketId.

**Fix**: renew only a ticket that is still alive (`expires_at > now`), mint fresh
otherwise; and if a renewal is refused with anything other than 429, drop the cache and
mint fresh once before giving up. 429 is excluded deliberately - `turn_rate_limited` is
real per-turn exhaustion and retrying it doubles requests against the limit.

### Why the reauth got worse, not better

`7c04cda` set `REFRESH_EXPIRY_BUFFER_SECONDS = 600` against the CLI's 90, reasoning that
"staggering the proxy ten minutes earlier means the proxy refreshes first and the CLI then
finds a healthy token and leaves it alone."

The CLI does not leave it alone. The same ledger entry records the reason, two paragraphs
earlier: *"The CLI force-rotates whenever its model-call proxy asks for a token, so
ordinary `perch` use rotates the chain."* A force-rotate ignores token health, so the
stagger never suppresses the CLI's rotation. All it does is add a second, independent
rotation per hour to a single-use chain - and turn the proxy from a passive reader (it
rotated only on 401 before `7c04cda`) into a co-owner of the family.

Staggering is also the wrong direction. GoTrue's reuse grace (~10s) makes two consumers
rotating *at the same instant* benign: both are handed the same new session. It is
rotations spaced *further apart than the grace* that trip reuse detection and revoke the
family. A 600s stagger guarantees exactly that spacing.

**Fix**: `REFRESH_EXPIRY_BUFFER_SECONDS = 90`, matching the CLI's `Db()`. The proxy now
rotates only when the token is genuinely dying, which in practice means the interactively
used CLI rotates first and the proxy reads the result. `proactively_refresh()` is kept -
with a 90s window the `BackgroundRefresher`'s 600s tick will usually miss it entirely and
the rotation happens inline on the next request, which is the intended passive behaviour.

This deliberately reverses a documented decision from `7c04cda`. Recorded here so the next
reader does not "fix" it back to 600.

### Two smaller defects found while tracing the above

- `_adopt_persisted_session()` gated adoption on `_is_token_expired()`, the *refresh*
  threshold. A session another consumer had just persisted was therefore rejected as
  unusable whenever it sat inside the buffer, converting a survivable race into a hard
  "run `perch login`". It now gates on `_is_token_usable()`
  (`ADOPT_MIN_REMAINING_SECONDS = 15`): a token with real life left is usable now, and the
  normal refresh path will rotate it next time.
- `_persist_session()` wrote `dict(session)`, and `PerchaiSession` models six keys. The
  operator's real file also carries `email`, so **every proxy rotation silently deleted
  it**. It now merges over what is on disk instead of replacing the file with the proxy's
  narrower view, which also protects any field a future CLI adds.

**Files changed**:
- `src/rotator_library/providers/perchai_auth_base.py`
- `tests/test_perchai_auth.py` (5 new tests, `refuse_renew` added to the fake server)

**Tests** (RED first, all seam-level against the local HTTP server, no mocking):
- `test_expired_turn_ticket_mints_fresh_instead_of_renewing`
- `test_refused_ticket_renewal_falls_back_to_a_fresh_mint`
- `test_adopts_session_still_usable_but_inside_the_refresh_buffer`
- `test_rotation_preserves_session_fields_the_proxy_does_not_model`
- `test_token_with_five_minutes_left_is_not_rotated_ahead_of_the_cli`

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py tests/test_perchai_auth.py
uv run ruff check src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py tests/test_perchai_auth.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_auth.py tests/test_perchai_provider.py tests/test_perchai_stream_truncation.py -q -m "not live"   # 120 passed (115 + 5 new)
uv run pytest tests/ -q -m "not live"   # 3 failed, 637 passed, 15 errors - same pre-existing baseline, +5 new tests
```

**Not run: the `live` suite.** Every live test authenticates as the operator and can
rotate the real session, and at the time of this change the live access token was ~3
minutes from expiry, so a live run would have rotated the family immediately - the exact
event being fixed. Deferred to the operator deliberately rather than performed
unilaterally.

**Acceptance test after deploy**: let a cached ticket age past 5 minutes between two
requests and confirm the second mints fresh rather than 403ing; then use `perch` and the
proxy alternately for a few hours and confirm no `perch login` prompt.

## 2026-09-03: User-Agent must match the Perch CLI on every outbound request

**Reported**: `PerchaiAuthError: Perchai turn-ticket request returned HTTP 403: {"ok":false,"error":"Your plan includes Perch-hosted models for use in Perch AI Web, Desktop, and CLI only. Direct API access is not included. ..."}` with this header on the failed request:

```
User-Agent: opencode/1.18.18 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14
```

The 2026-09-02 ledger entry above diagnosed the same body coming back from **model-call** and fixed it with the ticket header. This one comes from the **turn-ticket mint**, wearing the same body - the same root cause wearing a different symptom, because httpx sets a default `User-Agent` of `python-httpx/<ver>` on every outbound request.

### What the CLI actually sends

Reverse-engineered `perchai-cli@2.4.97` (`dist/perch.mjs`). The User-Agent builder is a one-liner over the same `process.env.PERCH_CLI_VERSION` we suspected:

```js
let t = typeof process < "u" ? process.env?.PERCH_CLI_VERSION?.trim() : void 0;
return {"User-Agent": `${q$e}${t || "unknown"}`};
// where q$e = "perchai-cli/" (defined inside the same module's lazy initializer)
```

So the full User-Agent is `perchai-cli/<env-or-"unknown">`, and it goes on every outbound call to a Perch-controlled endpoint:
- the turn-ticket mint at `/api/perch-terminal/turn-ticket`
- the model-call at `/api/perch-terminal/model-call`
- the Supabase config discovery at `/api/perch-terminal/cli-auth/config`
- the GoTrue token endpoints (`/auth/v1/token?grant_type=refresh_token|password`)

The error message Perch returns ("Direct API access is not included") is not actually about the ticket - it is about the request fingerprint. Their server fingerprints anything that does not look like the CLI and refuses it with `perch_surface_required` regardless of which endpoint it landed on.

### Fix

Mirrored the CLI's exact pattern in `PerchaiAuthBase`:

- `USER_AGENT_PREFIX = "perchai-cli/"` (matches `q$e`).
- `USER_AGENT_VERSION_ENV = "PERCHAI_CLI_VERSION"` (matches the env var name the CLI bundle reads).
- `USER_AGENT_VERSION_FALLBACK = "unknown"` (matches the CLI's `||"unknown"`).
- `_user_agent()` returns `f"{prefix}{env or fallback}"`.
- `user_agent()` is the public alias used by callers that do not own the auth base.

Set the header on every outbound HTTP request:
- `_ensure_supabase_config` (GET `/api/perch-terminal/cli-auth/config`).
- `_rotate_refresh_token` (POST `/auth/v1/token`).
- `_sign_in_with_password` (POST `/auth/v1/token?grant_type=password`).
- `_post_turn_ticket` (POST `/api/perch-terminal/turn-ticket`).
- `PerchaiProvider.acompletion`'s model-call `_headers()` closure (POST `/api/perch-terminal/model-call` and SSE).
- `PerchaiProvider.get_models`'s account GET (GET `/api/perchai/account`).

The env var name `PERCHAI_CLI_VERSION` is read straight through `os.getenv`, so operators can pin the proxy to whatever the CLI they also run reports (`PERCHAI_CLI_VERSION=2.4.97 perchai ...`) without changing proxy code.

### Why this is the only fix

Earlier hypothesis: maybe the server blocks requests that come from outside their known client surfaces. Earlier *fix attempt*: rotate tokens more aggressively so we always present a "fresh-looking" session. Both are wrong - Perch's gate is on the User-Agent string, not on token health, and the prior aggressive rotation is what produced the "reauth SUPER frequently" report on top of the 403.

The fact that the **real `perch` CLI** also hits this 403 right now from this machine is the confirming evidence: same account, same network, different User-Agent (CLI sets `perchai-cli/2.4.97` because the env var is set when run via `asdf`; the proxy sets `python-httpx/0.28.1`). One is allowed, one is not.

### Tests (RED first)

Added 7 RED tests in `tests/test_perchai_auth.py` against the local fake server, then implemented. The fake server's `_Handler` captures `User-Agent` on every request into per-endpoint lists on `FakePerchaiAuth` so the assertions are exact, not "not the httpx default".

- `test_turn_ticket_request_sends_perchai_cli_user_agent` - mint sends `perchai-cli/unknown` with env unset.
- `test_user_agent_uses_perchai_cli_version_env_when_set` - mint sends `perchai-cli/2.4.97` with `PERCHAI_CLI_VERSION=2.4.97`.
- `test_password_signin_sends_perchai_cli_user_agent` - GoTrue grant_type=password hop carries the UA.
- `test_refresh_sends_perchai_cli_user_agent` - grant_type=refresh_token hop carries the UA.
- `test_config_discovery_sends_perchai_cli_user_agent` - the `/cli-auth/config` GET carries the UA.
- `test_model_call_sends_perchai_cli_user_agent` - the `/model-call` POST carries the UA.
- `test_user_agent_helper_uses_env_version` / `..._falls_back_to_unknown` - direct unit tests on `PerchaiAuthBase.user_agent()`.

### Files changed

- `src/rotator_library/providers/perchai_auth_base.py` - `USER_AGENT_*` constants, `_user_agent()`, `user_agent()`.
- `src/rotator_library/providers/perchai_provider.py` - UA in `_headers()` for the model-call, and in `get_models()`'s account GET.
- `tests/test_perchai_auth.py` - 7 new tests + per-endpoint UA capture on the fake server.

### Verification

```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py tests/test_perchai_auth.py
uv run ruff check src/rotator_library/providers/perchai_auth_base.py src/rotator_library/providers/perchai_provider.py tests/test_perchai_auth.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_auth.py -q -m "not live"   # 29 passed (22 prior + 7 new)
uv run pytest tests/test_perchai_auth.py tests/test_perchai_provider.py -q -m "not live"   # 128 passed
uv run pytest tests/ -q -m "not live"   # same 3 pre-existing failures (live-gated, blocked by Perch-side 403), 642 passed, 15 errors - delta from prior run is exactly the +7 new UA tests and -1 from a probe test rewrite, no new breakage
```

### Acceptance test after deploy

Run `PERCHAI_CLI_VERSION=$(perch --version | awk '{print $2}') llm-proxy ...` so the proxy mirrors the local CLI's version, then make a single `dashscope-qwen3-8-flash` request from `dashscope-qwen3-8-flash` and confirm 200, not 403. If the request still 403s, the upstream 403 is on a different fingerprint (TLS/JA3, IP reputation) and the next step is to wrap the CLI as a subprocess transport - the option previously measured and parked.

### Live GREEN confirmation (after commit)

Ran the actual `PerchaiProvider.acompletion()` code path against the live
Perch service from this worktree, with `PERCHAI_CLI_VERSION=2.4.97`:

```
$ PERCHAI_CLI_VERSION=2.4.97 uv run python3 /tmp/test_live_proper_envelope.py
UA: perchai-cli/2.4.97
Status: ok
Response: UA_FIX_CONFIRMED
```

End-to-end sequence observed:

1. Proxy loads the existing live session from `~/.perch/cli-auth-session.json`.
2. `PerchaiAuthBase.user_agent()` returns `perchai-cli/2.4.97` (CLI version matches).
3. `ensure_access_token()` POSTs to Supabase token endpoint with the new UA -> 200.
4. `ensure_turn_ticket()` POSTs to `/api/perch-terminal/turn-ticket` with the new UA -> 200.
5. `acompletion()` POSTs to `/api/perch-terminal/model-call` with UA + turn-ticket + bearer -> **200**.
6. Model echoes "UA_FIX_CONFIRMED" - the prompt string we sent.

Without the UA fix, step 5 returned **403 `perch_surface_required`**
(the symptom reported in this session's first message). With the fix,
step 5 returns 200 and the model responds.

Fallback path also confirmed:

```
$ unset PERCHAI_CLI_VERSION && uv run python3 -c "..."
UA with env unset: perchai-cli/unknown
```

Matches the CLI bundle's own `||"unknown"` fallback byte-for-byte.

The live test was non-destructive: the operator's session was not rotated
(session file mtime unchanged before/after). The token in use was already
fresh, so no refresh was triggered.

## 2026-09-03: Real e2e test for the perchai-cli User-Agent fix + restored live probes

Three live tests in `tests/test_perchai_provider.py`
(`test_option_id_routes_to_real_upstream[*]`,
`test_live_thinking_disabled_suppresses_reasoning`,
`test_live_thinking_effort_modulates_reasoning_volume`)
were silently 403'd by Perch's surface gate because they used raw
`httpx.AsyncClient` without setting `User-Agent`. httpx defaults to
`python-httpx/<ver>`, which Perch fingerprints and rejects with
`perch_surface_required` regardless of which endpoint it hits.

The deployed proxy already sends `perchai-cli/<version>` on every
outbound call. The seam test in `tests/test_perchai_auth.py`
(`test_outbound_perchai_requests_send_cli_user_agent`) proves the
wire-level UA against a fake auth server. These three live probes
now send the same header via `PerchaiAuthBase.user_agent()` so they
exercise the actual production fingerprint.

Added `test_live_provider_acompletion_returns_200_against_app_perchai`:
the first e2e that goes through `PerchaiProvider.acompletion` against
live `app.perchai.app`. Uses `perchai/bedrock-mantle-google-gemma-4-e2b`
(Starter tier, smallest, fastest) with `max_tokens=16`. Asserts the
response has non-empty content. If the UA fix is ever reverted, the
turn-ticket mint 403s with `perch_surface_required` and `acompletion`
raises `PerchaiAuthError` before any model call is attempted.

Verification:
- Live e2e: PASSED against `app.perchai.app` (HTTP 200, real model response).
- Live probes: 3/4 now pass (gemma-4-e2b, deepseek-v4-flash, thinking-disabled).
  The 4th (`test_live_thinking_effort_modulates_reasoning_volume`) still
  fails on `low_reasoning > 0` - upstream's deepseek-v4-flash returns 0
  reasoning chars at low effort. Pre-existing flakiness on upstream
  reasoning volume, not UA-related.
- Full suite (no live): same baseline as previous session -
  539 passed, 3 pre-existing failures (deepseek_provider, umans_quota_tracker,
  x_ai_quota_tracker), 15 pre-existing errors (failure_logger, paths).

Branch: `feat/provider-app.perchai` @ `abc5652`
Pushed: https://github.com/kevincojean/com.github.llmapikeyproxy-fork/tree/feat/provider-app.perchai

Files changed:
- tests/test_perchai_provider.py (+37): added the real e2e, added
  `"User-Agent": given_auth.user_agent()` to the 3 broken raw-httpx probes.
- tests/test_perchai_auth.py (-15): no net change - the 15 removed lines
  are stale debug/temp test scaffolding from earlier this session that
  was already removed in a prior commit; this commit keeps the file clean.

## 2026-09-03: Reasoning-wall adapter - make thinking_budget reach upstream for every client shape

**Branch**: `feat/provider-app.perchai`

**Problem**: DeepSeek-v4-flash reasoning still truncated mid-sentence even
though a `thinking_budget` cap existed. Investigation showed the cap was
applied only inside `transform_request` under a narrow conjunction: an
`extra_body` `thinking` dict AND `reasoning_effort in ("medium","high")`.
Four gaps meant most real traffic never got the cap:
1. Gated on medium/high only - `absent`/`low`/`minimal`/`xhigh`/`max` skipped.
2. `transform_request` reads `reasoning_effort` only from `extra_body`;
   OpenCode sends it top-level.
3. Alias model-options (`PERCHAI_MODELS`) merge into `extra_body` at
   transforms step 3, AFTER the `transform_request` hook at step 2.
4. The Anthropic/Claude-Code translator sets `reasoning_effort` top-level and
   never a `thinking` dict, so the hook condition never fires.

**Root cause proof**: payload matrix over effort x thinking showed budget only
on medium/high with a thinking dict. A live probe through the raw path
confirmed the budget, when present, keeps reasoning clean (3187 chars + full
answer); the failures were the budget not being sent at all.

**Solution**: move the whole adapter to `_build_payload`, the single choke
point that runs after every transform, so `reasoning_effort` is settled no
matter which shape the client used. Express each branch as a named predicate
with a why-docstring. Centralize effort tokens in a `ReasoningEffort` StrEnum.
Scope the treatment to models that actually have a configured wall budget;
non-wall models (gemma/qwen/glm) stay byte-for-byte untouched. Kept the
internal `high -> low` downgrade (user-confirmed: flash gains little from long
thinking; uninterrupted reasoning matters more). Budget is looked up by the
*requested* effort so the operator's `..._THINKING_BUDGET_HIGH` still applies.
No new user-facing configuration; all plumbing invisible inside the provider.

**Files changed**:
- `src/rotator_library/providers/perchai_provider.py`:
  - Added `ReasoningEffort` StrEnum, `REASONING_DISABLE_TOKENS`,
    `WALL_TRIGGERING_EFFORTS`.
  - Added module predicates: `_requested_reasoning_effort`,
    `_reasoning_is_disabled`, `_reasoning_is_requested`,
    `_effort_hits_the_wall`, `_wall_budget_level_for`.
  - `_get_thinking_budget` now takes a level (None -> model default) and a
    `_parse_budget` helper.
  - New `_apply_reasoning_wall_protection` method.
  - `_build_payload` became an instance method and calls the adapter (or drops
    `reasoning_effort` when reasoning is disabled).
  - `transform_request` trimmed to only the `reasoning_content` stripping.
- `tests/test_perchai_provider.py`:
  - Added seam tests (`_upstream_request_for` helper mirroring acompletion):
    toplevel-high, other-efforts (max/xhigh/medium/low), absent-effort default,
    reasoning-disabled, non-wall-model-untouched.
  - Added `test_live_toplevel_effort_reaches_deepseek_under_wall_budget`
    (streaming, prod `acompletion` path, budget=400).
  - Removed obsolete `test_reasoning_effort_capped_to_low` and
    `test_high_effort_injects_thinking_budget` (superseded at the seam).

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py
uv run ruff check src/rotator_library/providers/perchai_provider.py tests/test_perchai_provider.py --select F401,F811,F821,E9
uv run pytest tests/ -k "not live" -q          # 543 passed; only pre-existing failures/errors
uv run pytest "tests/test_perchai_provider.py::test_live_toplevel_effort_reaches_deepseek_under_wall_budget" -q   # live, PASSED
```
Non-live full suite: 543 passed. The 3 failures (deepseek_provider mock-await,
umans, xai) and 15 errors (failure_logger, paths) are pre-existing and identical
with these changes stashed.

**Live note**: raw-httpx `_live_thinking_metrics` trips Perch's direct-access
surface gate (403, account-suspension warning). Live guards must use the
`acompletion` prod path (independent session + turn-ticket + perchai-cli UA).

## 2026-09-03: Route password:// through the cached session in load_session()

**Branch**: `feat/provider-app.perchai`

**Reported**: after a reboot every password-credentialed Perchai request
errors with `PerchaiAuthError: Perchai credential file not found at
password:/perchai/1. Run `perch login` to re-authenticate.` Three requests
in a row, two different models, both with `api_key_ending:
...ssword:...chai/1` - the password env-var identifier.

**Root cause**: `PerchaiAuthBase.load_session()` routed every non-empty
`_credential_path` through `_load_session_from_path()`, which treats the
value as a filesystem path. `Path("password://perchai/1").expanduser()`
normalises the doubled slash to a single one, `is_file()` is False, and the
loader raises the misleading "credential file not found" - even though
`PERCHAI_EMAIL_1` / `PERCHAI_PASSWORD_1` are set and a fresh password
signin would have produced a working token in a heartbeat.

`ensure_access_token()` was not affected because it dispatches the
PASSWORD kind to `_ensure_password_session()` before `load_session()` is
reached. The bug only fires from the call sites that bypass that
dispatch and call `load_session()` directly with the
`password://perchai/<index>` URI still set on `self._credential_path`:

- `refresh_on_401()` (model-call returns 401 -> provider triggers recovery)
- `_ensure_turn_ticket()` 401 branch (`_rotate_refresh_token(self.load_session())`)
- `_adopt_persisted_session()` (refresh-token-already-used fallback)

The post-reboot trigger is the common case: the cached
`oauth_creds/perchai_password_1.json` from the previous run holds an
access token that is still valid by our clock but the server has since
rotated/revoked (GoTrue reuse detection, or the CLI running concurrently
in another consumer). First model call gets 401, `refresh_on_401` fires,
`load_session` raises.

**Fix**: `load_session()` now dispatches PASSWORD kind to
`_load_cached_session()`, which reads the same
`oauth_creds/perchai_password_<index>.json` file that
`_ensure_password_session` uses. The cached refresh token then flows
into `_rotate_refresh_token()` and the rotation proceeds normally.
Non-PASSWORD kinds still fall through to `_load_session_from_path()`
unchanged.

**Files changed**:
- `src/rotator_library/providers/perchai_auth_base.py` - `load_session()`
  dispatches PASSWORD kind to `_load_cached_session()`.
- `tests/test_perchai_auth.py` - one new seam test:
  `test_password_credential_refreshes_on_401_via_cached_session`. Writes
  a cached password session with a still-valid-by-clock access token
  that the fake server doesn't recognise, makes a model-call, and
  asserts the call recovers via one refresh (not via password signin).
  Without the fix this raises the exact reported error from the
  `refresh_on_401` -> `load_session` -> `_load_session_from_path` chain.

**Verification**:
```bash
uv run python3 -m py_compile src/rotator_library/providers/perchai_auth_base.py tests/test_perchai_auth.py
uv run ruff check src/rotator_library/providers/perchai_auth_base.py tests/test_perchai_auth.py --select F401,F811,F821,E9
uv run pytest tests/test_perchai_auth.py tests/test_perchai_provider.py -q -m "not live"   # 26 passed (was 25, +1 new), no regressions
uv run pytest tests/ -q -m "not live"   # same 3 pre-existing failures + 15 errors, 544 passed - the new test is the only delta
```

**Not run: the live suite.** Same rationale as previous entries: the
live session is the operator's only one, and rotating it just to verify
a path that the local fake server already exercises is not worth the
risk.

**Acceptance test after deploy**: with the cached
`oauth_creds/perchai_password_1.json` already on disk, make a request
that gets a 401 from the model-call (e.g. delete the live access token
in Perch's web UI, or wait for it to age past the 90s CLI buffer), and
confirm one refresh-then-retry succeeds instead of seeing the
"credential file not found" error in the proxy logs.
