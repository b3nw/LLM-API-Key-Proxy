# Perchai Provider Feature History

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
