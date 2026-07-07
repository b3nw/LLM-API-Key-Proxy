# model-routing — MODEL_ALIASES and Cross-Provider Rotation

## 2026-06-23 — Fix streaming coroutine not awaited in CrossProviderExecutor

Target: `feat(model-routing): MODEL_ALIASES and cross-provider rotation`
Files:
- `src/rotator_library/client/cross_provider_executor.py`

Working commits before autosquash:
- (see fixup commit hash after commit)

Verification:
- `uv run python3 -m py_compile src/rotator_library/client/cross_provider_executor.py` — passed
- `uv run ruff check src/rotator_library/client/cross_provider_executor.py --select F401,F811,F821,E9` — passed

Notes:
- `CrossProviderExecutor.execute()` returned `self._execute_streaming(...)` without
  `await`. Since `_execute_streaming` is `async def` (returns a coroutine wrapping the
  inner `_stream_with_failover()` async generator), callers received a coroutine object
  instead of the async generator. The Anthropic `/v1/messages` streaming wrapper then
  failed with `'async for' requires an object with __aiter__ method, got coroutine`.
- Fix: `return await self._execute_streaming(...)` so the coroutine resolves to the
  inner async generator before being returned to the caller.
- Ref: https://github.com/b3nw/LLM-API-Key-Proxy/issues/58

## 2026-07-01 — Add default Claude model aliases for bare-ID routing

Target: `feat(model-routing): MODEL_ALIASES and cross-provider rotation`
Files:
- `src/rotator_library/model_alias_registry.py`
- `tests/test_model_alias.py`

Changes:
- Added `DEFAULT_MODEL_ALIASES` dict mapping bare Claude model IDs to
  `anthropic:<model_id>` targets. This enables clients like Claude Code
  (which send unprefixed model IDs such as `claude-opus-4-8`) to route
  without requiring MODEL_ALIAS_* environment variable configuration.
- Modified `_load_from_env()` to load built-in defaults first, then load
  MODEL_ALIAS_* env vars which override defaults for the same canonical name.
- Default aliases for 4-5 family target date-suffixed IDs (matching
  `OAUTH_MODELS`): `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`,
  `claude-haiku-4-5-20251001`.
- Default aliases for newer models target bare IDs: `claude-fable-5`,
  `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`.
- Added 4 tests in `TestDefaultClaudeAliases` class: defaults loaded without
  env vars, correct target model IDs, env var override behavior, and
  canonical models listing.

Verification:
- `uv run python3 -m py_compile src/rotator_library/model_alias_registry.py` — passed
- `uv run ruff check src/rotator_library/model_alias_registry.py --select F401,F811,F821,E9` — passed
- `pytest tests/test_model_alias.py -v` — passed (all tests including new ones)

Notes:
- `_register_alias()` replaces (not appends) when the same canonical name is
  registered twice, so env vars cleanly override defaults — no duplication.
- Operators who want cross-provider failover (e.g. anthropic + copilot) can
  still set `MODEL_ALIAS_CLAUDE_OPUS_4_8="anthropic:...,copilot:..."` to
  override the single-provider default.
- `/v1/models` endpoint automatically includes default aliases via
  `get_canonical_models()`.
- Ref: b3nw/LLM-API-Key-Proxy#97

## 2026-07-03 — Skip alias models already listed by their provider

Target: `fix(models): skip alias models already listed by their provider`
Files:
- `src/proxy_app/main.py`

Working commits before autosquash:
- `8f4c5078 fix(models): skip alias models already listed by their provider`

Verification:
- `uv run python3 -m py_compile src/proxy_app/main.py` — passed
- `uv run ruff check src/proxy_app/main.py --select F401,F811,F821,E9` — passed

Notes:
- `list_models` was appending all canonical alias model names unconditionally,
  causing duplicates when the provider already returned the same model under its
  prefixed name (e.g. `anthropic/claude-opus-4-6` from the provider plus bare
  `claude-opus-4-6` from the alias registry).
- Fix: before appending an alias, resolve its targets and skip if any target's
  `full_model` is already present in the discovered `model_ids` set.
- This prevents `/v1/models` from returning duplicate entries without breaking
  aliases that route to providers not yet discovered.

## 2026-07-07 — Add default Codex desktop model aliases

Target: `feat(model-routing): MODEL_ALIASES and cross-provider rotation`
Files:
- `src/rotator_library/model_alias_registry.py`
- `tests/test_model_alias.py`

Working commits before autosquash:
- (see fixup commit hash after commit)

Verification:
- `uv run --with pytest pytest tests/test_model_alias.py tests/test_proxy_endpoints.py` — passed
- `uv run ruff check README.md src/proxy_app/main.py src/rotator_library/client/rotating_client.py src/rotator_library/model_alias_registry.py src/rotator_library/providers/codex_provider.py tests/test_model_alias.py tests/test_proxy_endpoints.py --select F401,F811,F821,E9` — passed
- `uv run python3 -m py_compile src/proxy_app/main.py src/rotator_library/client/rotating_client.py src/rotator_library/model_alias_registry.py src/rotator_library/providers/codex_provider.py tests/test_model_alias.py tests/test_proxy_endpoints.py` — passed

Notes:
- Added built-in default aliases for Codex desktop/OpenAI Codex clients that send
  bare model IDs: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`,
  `gpt-5.2`, and `codex-auto-review`.
- Added OpenAI-prefixed compatibility aliases for the same models, e.g.
  `openai/gpt-5.5` -> `codex/gpt-5.5`, so downstream apps that special-case
  provider prefixes can use an OpenAI-shaped public model ID.
- Each alias routes to the existing Codex provider target with the same model ID,
  e.g. `gpt-5.5` -> `codex/gpt-5.5` via `AliasTarget.full_model`.
- Codex provider model discovery now exposes OpenAI-prefixed public IDs while
  keeping the internal `codex` provider key for OAuth credentials, quota routing,
  and provider implementation code.
- This mirrors the existing built-in Claude aliases for clients that cannot or do
  not prefix model IDs with a provider name.
- Env vars still override defaults when operators need custom routing or
  multi-provider failover.
- Practical use case: Codex desktop/macOS apps and HumanLayer's OpenAI Codex
  provider expose/send model IDs like `gpt-5.5` or `openai/gpt-5.5` rather than
  `codex/gpt-5.5`; without default aliases the proxy rejects those requests or
  routes them to the wrong provider before they can reach Codex credentials.
