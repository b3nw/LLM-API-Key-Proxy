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
