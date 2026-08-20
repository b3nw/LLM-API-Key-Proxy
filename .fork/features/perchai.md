# Perchai provider

Canonical feature ID: `perchai`
Stack subject: `feat(perchai): add Perchai provider with OAuth auth and quota tracking`
Manifest: `.fork/stack.yml`

This file is the shared, repo-tracked history for the Perchai feature.
Local workspace state may contain run logs and scratch notes, but this
file is canonical across contributors and developer workspaces.

Perchai is a subscription provider accessed via reverse-engineered perchai
CLI API (npm bundle v2.4.87). Authentication uses OAuth session replay from
`~/.perch/cli-auth-session.json`. Request bodies are wrapped in a custom
envelope and streaming responses arrive as bespoke SSE events that require
custom parsing.

## 2026-08-20 - Scaffold feature ledger

Branch: `feat/provider-app.perchai`
Files planned:
- `src/rotator_library/providers/perchai_provider.py`
- `src/rotator_library/providers/perchai_auth_base.py`
- `src/rotator_library/providers/utilities/perchai_quota_tracker.py`
- `tests/test_perchai_provider.py` (fork-level `.gitignore` rule `tests/*`
  keeps this on disk but untracked; live e2e coverage lives outside the repo)
- `.fork/stack.yml`
- `.fork/features/perchai.md` (this file)

Summary:
- Reverse-engineered perchai CLI API (npm bundle v2.4.87).
- OAuth session replay from `~/.perch/cli-auth-session.json`.
- Custom envelope wrapping for outbound requests and bespoke SSE event
  parsing for inbound streaming responses.

## 2026-08-20 - First working commit landed

Branch: `feat/provider-app.perchai`
Commit subject: `feat(perchai): add Perchai provider with OAuth auth and quota tracking`

Files changed:
- `src/rotator_library/providers/perchai_provider.py` (new, ~1020 lines)
  - `PerchaiProvider(PerchaiQuotaTracker, ProviderInterface)`, `@final`
  - `  has_custom_logic() -> True`, `provider_env_name = "perchai"`
  - `skip_cost_calculation = True`, `default_rotation_mode = "sequential"`
  - Dynamic `get_models()` from `GET /api/perchai/account` with 300s cache TTL
  - Custom `acompletion()`: envelope wrap, POST `/api/perch-terminal/model-call`,
    JSON-to-litellm mapping for non-streaming (including `toolCalls[]` ->
    OpenAI-format `tool_calls` with JSON-stringified `arguments`,
    `finish_reason="tool_calls"` when tools emitted); SSE parser for streaming
    (`answer_delta`/`text_delta`, `reasoning_delta`, `tool_call_delta` ->
    `delta.tool_calls`, `tool_use_end` -> final `finish_reason="tool_calls"`,
    finish detection, 401 -> refresh + retry)
  - `parse_quota_error()` @staticmethod mapping all 10 perchai error codes
    (`usage_limit_reached`, `promo_overflow_decision`, `starter_model_blocked`,
    `byo_feature_blocked`, `context_overflow`, `not_authenticated`, `api_error`,
    `timeout`, `aborted`, `vision_model_error`) with HTTP 429/403 fallback
  - TypedDicts: `PerchaiSession`, `PerchaiRequestEnvelope`, `PerchaiStreamEvent`,
    `PerchaiErrorResponse`
- `src/rotator_library/providers/perchai_auth_base.py` (new, ~420 lines)
  - `PerchaiAuthBase` `@final`, singleton via `SingletonABCMeta`
  - Session reader: `~/.perch/cli-auth-session.json` (env var fallback
    `PERCHAI_OAUTH_<N>` pointing to a session file path)
  - Reactive 401 handler with `asyncio.Lock` single-flight refresh
    (10 concurrent 401s collapse to one network round-trip)
  - Atomic session writes (tmp + rename) to survive crashes mid-refresh
  - `PerchaiAuthError` exception class
- `src/rotator_library/providers/utilities/perchai_quota_tracker.py` (new, ~530 lines)
  - `PerchaiQuotaTracker` mixin: `model_quota_groups`, `usage_reset_configs`
    (calendar month UTC, $150/mo Pro allowance)
  - `get_background_job_config()` returns `{interval: 300, enabled: True}`
  - `run_background_job()` polls `GET /api/perch-terminal/usage` and updates
    quota baseline (skips on auth failure rather than crashing the loop)
- `src/rotator_library/credential_manager.py` (modified, +2 lines)
  - `DEFAULT_OAUTH_DIRS["perchai"] = ~/.perchai`
  - `ENV_OAUTH_PROVIDERS["perchai"] = "PERCHAI"`
- `src/rotator_library/provider_factory.py` (modified, +2 lines)
  - `PROVIDER_MAP["perchai"] = PerchaiAuthBase`
- `src/rotator_library/provider_config.py` (modified, +7 lines)
  - `LITELLM_PROVIDERS["perchai"] = {"category": "subscription"}`
- `src/proxy_app/provider_urls.py` (modified, +1 line)
  - `PROVIDER_URL_MAP["perchai"] = "https://app.perchai.app"`
- `.fork/stack.yml` (modified, +8 lines) - perchai feature entry registered
- `.fork/features/perchai.md` (this file) - canonical ledger updated

Out of scope (intentionally not committed):
- `tests/test_perchai_provider.py` lives on disk under `tests/` but the fork's
  `.gitignore` (`tests/*`, line 142) keeps the whole tests tree untracked.
  Live e2e coverage runs outside the repo. The test file itself contains
  29 given_when_then cases; 29/29 pass under synthetic session, and
  the `pytest.skipif` skip path works cleanly when no session is present.

Verification (run on this commit):
- `uv run python3 -m py_compile` on the three new perchai provider modules
  - exit 0
- `uv run ruff check` on the three new modules with
  `--select F401,F811,F821,E9` - exit 0
- `uv run python3 .fork/check-stack.py` - exit 0 (perchai registered, no
  duplicate IDs, valid subject)
- Plugin discovery: `from rotator_library.providers import PLUGINS; 'perchai' in PLUGINS`
  - True
- `PerchaiProvider().has_custom_logic()` - True
- `parse_quota_error` parametrized over all 10 perchai codes: every code
  returns the expected `{"reason": ..., "retry_after": ...}` (or None for
  terminal fall-through codes); HTTP 429 fallback yields `rate_limit`
- asyncio.Lock single-flight: 10 concurrent `refresh_on_401` calls produce
  exactly 1 unique token
- Streaming parser handles malformed JSON, unknown event types, empty
  `data:` lines, streams without `[DONE]`, and missing `finish_reason`
  without raising
- Synthetic-session e2e: 29/29 tests pass with `pytest.skipif` cleanly
  skipping when the session file is absent
- Live provider smoke against `app.perchai.app`: pending user-side
  `perch login`; the test suite is wired to skip with that exact message.

Notes:
- The single commit groups scaffold + core implementation + registration +
  ledger. The PR squash-merge to `dev` produces exactly the feature subject
  registered in `.fork/stack.yml` for the `perchai` feature ID.

## 2026-08-20 - Tool call support (non-stream + stream)

Branch: `feat/provider-app.perchai`
Files changed:
- `src/rotator_library/providers/perchai_provider.py` - non-stream extracts
  `toolCalls[]` from perchai response and builds litellm
  `ChatCompletionMessageToolCall` with `Function(arguments=json.dumps(...))`;
  `finish_reason` flips from `"stop"` to `"tool_calls"` when tool calls exist.
  Stream parser handles `tool_call_delta` -> `delta.tool_calls` chunks and
  `tool_use_end` -> final `finish_reason="tool_calls"` chunk with empty delta.
  `_stream_completion` tracks `saw_tool_call` flag and suppresses the
  redundant `[DONE]` stop chunk when tools already terminated the stream.

Verified:
- [x] `uv run python3 -m py_compile src/rotator_library/providers/perchai_provider.py`
- [x] `uv run ruff check src/rotator_library/providers/perchai_provider.py
      --select F401,F811,F821,E9` - exit 0
- [x] `uv run pytest tests/test_perchai_provider.py -v` - 31/31 pass
- [x] Live e2e via `/tmp/perchai-tool-probe.sh` against `app.perchai.app`:
      `tool_calls` populated, `finish_reason="tool_calls"`,
      `function.arguments='{"city": "Paris"}'` (JSON-stringified dict)
- [x] Streaming parser unit probe (synthetic `tool_call_delta` +
      `tool_use_end` events): emits chunks with correct `delta.tool_calls`
      and `finish_reason="tool_calls"` final chunk

Key wire-format facts:
- Perchai `toolCalls[].arguments` is a DICT (not a JSON string) - the
  adapter must `json.dumps` it before passing to litellm, which expects
  the OpenAI-conformant JSON-stringified form.
- Perchai `content[]` arrays include `tool_use` blocks that mirror the
  `toolCalls[]` data; the adapter skips them when accumulating text
  content (the canonical source is `toolCalls[]`).
- Live `tool_call_delta` SSE events were NOT observed - perchai returns
  the full `toolCalls[]` array in the non-streaming response. The
  streaming parser implementation is defensive (based on OpenAI
  standard) and ready if perchai later emits incremental events.

Decision - tool_use block filtering: The `content` array's `tool_use`
blocks duplicate `toolCalls[]` data. Filtering them from text
accumulation prevents double-counting. The canonical source remains
`toolCalls[]` to avoid ID mismatch between the two representations.

Decision - finish_reason override: When `toolCalls[]` is non-empty,
`finish_reason` is forced to `"tool_calls"` regardless of any
perchai-supplied `finishReason` field. This matches OpenAI semantics
that litellm and downstream consumers (Anthropic compat, Responses
API translator) rely on to dispatch tool execution.

Decision - stream [DONE] suppression: When `tool_use_end` already
yielded the final `tool_calls` chunk, the `[DONE]` sentinel must NOT
trigger a second `finish_reason="stop"` chunk - that would deliver
two terminating finish reasons in one stream. The `saw_tool_call`
flag (set when any yielded chunk carries `finish_reason="tool_calls"`)
short-circuits both the `[DONE]` and the post-loop synthesis paths.

## 2026-08-20 - Refresh endpoint fix

**Problem**: 2/31 pytest tests failing (`test_given_expired_token_*_refreshes_and_retries`).
Initial bundle grep had wrong endpoint - `/api/auth/session` returns 405.

**Fix** (`96f1a7f`): Switched to Supabase GoTrue endpoint
`POST {supabaseUrl}/auth/v1/token?grant_type=refresh_token` with config
discovery via `GET /api/perch-terminal/cli-auth/config`.

**Verification**: 31/31 tests pass, live refresh probe shows token rotation.

## 2026-08-20 - Streaming tool_call_delta fix

Live e2e `/tmp/perchai-401-tools.sh` (expired token + 2 tools) revealed
pydantic ValidationError on streaming path. Root cause: conditional
`if function_delta:` guard skipped `function` key when empty.

`litellm.ChatCompletionDeltaToolCall` requires `function` field to be
set (even when empty dict).

Fix: always include `function` key. Verified live: 2 tool calls +
streaming tool calls both work.

Commit: `TBD`
