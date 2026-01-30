# Bug Report: LiteLLM Returns None for Streaming Requests

**Date**: 2026-01-30  
**Severity**: High  
**Component**: LiteLLM `acompletion()` with `stream=True`  
**LiteLLM Version**: 1.81.5  
**Status**: Workaround Applied

---

## Summary

`litellm.acompletion()` intermittently returns `None` instead of an async iterator when called with `stream=True` for custom OpenAI-compatible providers. This causes an `AttributeError` when the caller attempts to iterate over the response.

## Error Signature

```
AttributeError: 'NoneType' object has no attribute '__aiter__'
```

### Stack Trace (Simplified)

```python
# src/rotator_library/client/streaming.py:82
stream_iterator = stream.__aiter__()  # stream is None here
```

## Environment

| Component | Version |
|-----------|---------|
| LiteLLM | 1.81.5 |
| Python | 3.12 |
| Platform | Linux (Docker) |
| Provider | Custom OpenAI-compatible (Dedalus Labs) |

## Reproduction

### Conditions That Trigger the Bug

1. **Provider**: Custom OpenAI-compatible endpoint via `api_base` override
2. **Payload Size**: Large requests (~50KB+)
3. **Streaming**: `stream=True`
4. **Model Pattern**: `openai/<provider>/<model>` with `custom_llm_provider="openai"`

### Failing Request Pattern

```python
stream = await litellm.acompletion(
    model="openai/anthropic/claude-opus-4-5",
    api_base="https://api.dedaluslabs.ai/v1",
    custom_llm_provider="openai",
    api_key="dsk-live-xxxxx",
    messages=[...],  # Large payload ~56KB
    max_tokens=64000,
    stream=True,
    stream_options={"include_usage": True},
)
# stream is None instead of CustomStreamWrapper
```

### Working Request Pattern (Same Model, Small Payload)

```python
stream = await litellm.acompletion(
    model="openai/anthropic/claude-opus-4-5",
    api_base="https://api.dedaluslabs.ai/v1",
    custom_llm_provider="openai",
    api_key="dsk-live-xxxxx",
    messages=[{"role": "user", "content": "hi"}],  # Small payload
    max_tokens=5,
    stream=True,
)
# stream is CustomStreamWrapper ✓
```

## Evidence That Upstream Provider Is NOT the Cause

### 1. Direct API Calls Work

```bash
curl -X POST "https://api.dedaluslabs.ai/v1/chat/completions" \
  -H "Authorization: Bearer dsk-live-xxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-opus-4-5", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
# ✅ Returns valid SSE stream
```

### 2. Standalone LiteLLM Tests Work

```python
# Tested inside the same Docker container
import asyncio
import litellm

async def test():
    stream = await litellm.acompletion(
        model="openai/anthropic/claude-opus-4-5",
        api_base="https://api.dedaluslabs.ai/v1",
        api_key="dsk-live-xxxxx",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        stream=True
    )
    print(f"Stream type: {type(stream)}")  # CustomStreamWrapper
    print(f"Stream is None: {stream is None}")  # False
    async for chunk in stream:
        print("Chunk received")
        break

asyncio.run(test())
# ✅ Works correctly
```

### 3. Small Requests Through Proxy Work

```bash
curl -X POST "https://llm-proxy.ext.ben.io/v1/chat/completions" \
  -H "Authorization: Bearer sk-proxy-xxxxx" \
  -d '{"model": "dedaluslabs/anthropic/claude-opus-4-5", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
# ✅ Returns valid SSE stream
```

### 4. Other Models on Same Provider Work

```
✅ dedaluslabs/gpt-4o (large payload) - Works
✅ dedaluslabs/anthropic/claude-sonnet-4-5 (large payload) - Works
❌ dedaluslabs/anthropic/claude-opus-4-5 (large payload) - Fails intermittently
```

## Failure Log Entries

From `failures.log`:

```json
{
  "timestamp": "2026-01-30T21:49:03.373483",
  "api_key_ending": "...bfd2e6",
  "model": "dedaluslabs/anthropic/claude-opus-4-5",
  "attempt_number": 1,
  "error_type": "AttributeError",
  "error_message": "'NoneType' object has no attribute '__aiter__'",
  "raw_response": null,
  "request_headers": {
    "content-length": "56887",
    "user-agent": "opencode/local ai-sdk/provider-utils/3.0.20 runtime/bun/1.3.5"
  },
  "error_chain": null
}
```

## Root Cause Analysis

### Identified Code Path

The bug is in **LiteLLM's `acompletion` function** (`litellm/main.py`, lines 593-625):

```python
# litellm/main.py:593-625
try:
    func = partial(completion, **completion_kwargs, **kwargs)
    ctx = contextvars.copy_context()
    func_with_context = partial(ctx.run, func)

    init_response = await loop.run_in_executor(None, func_with_context)
    if isinstance(init_response, dict) or isinstance(init_response, ModelResponse):
        # CACHING SCENARIO
        if isinstance(init_response, dict):
            response = ModelResponse(**init_response)
        response = init_response
    elif asyncio.iscoroutine(init_response):
        response = await init_response  # <-- Could return None if coroutine fails silently
    else:
        response = init_response  # type: ignore  # <-- Could be None

    if isinstance(response, CustomStreamWrapper):
        response.set_logging_event_loop(loop=loop)
    return response  # <-- NO VALIDATION THAT response IS NOT None!
```

### Missing Validation

LiteLLM lacks a critical check before returning:

```python
# What SHOULD exist:
if response is None:
    raise OpenAIError(
        status_code=500,
        message="acompletion returned None - empty response from provider"
    )
return response
```

### Flow for OpenAI-Compatible Streaming

1. `litellm.acompletion()` calls `completion()` in a thread executor
2. `completion()` calls `openai_chat_completions.completion()` with `acompletion=True`, `stream=True`
3. This returns a **coroutine** to `async_streaming()`
4. Back in `acompletion()`, `init_response` is the coroutine
5. `response = await init_response` awaits the coroutine
6. If something in `async_streaming()` fails silently (no exception raised), `response` is `None`
7. **Return `None` without any validation**

### Potential Triggers

1. **Silent HTTP errors** from custom OpenAI-compatible providers not properly surfaced by httpx/openai SDK
2. **Memory pressure** during large payload handling causing object creation to fail
3. **Race condition** in the sync-to-async boundary via `run_in_executor`
4. **OpenAI SDK edge case** where `with_raw_response.create()` returns empty on specific error conditions

## Workaround Applied

Added defensive check in `src/rotator_library/client/streaming.py`:

```python
# Line 82+
if stream is None:
    lib_logger.error(
        f"Received None stream for model {model} - provider returned empty response"
    )
    if cred_context:
        from ..error_handler import ClassifiedError
        cred_context.mark_failure(
            ClassifiedError(
                error_type="empty_response",
                message="Provider returned empty stream",
                retry_after=None,
            )
        )
    raise StreamedAPIError("Provider returned empty stream", data=None)

stream_iterator = stream.__aiter__()
```

This prevents the `AttributeError` crash and enables the credential rotation/retry logic to handle the failure gracefully.

## Recommended Actions

### Completed ✅
1. **✅ Confirmed on latest LiteLLM** (v1.81.5) - Issue persists
2. **✅ Applied workaround** - Defensive check in `streaming.py`
3. **✅ Root cause analysis** - Identified missing validation in `acompletion()`

### Next Steps
1. **File upstream bug** with LiteLLM GitHub including:
   - This root cause analysis
   - Reproduction steps
   - Proposed fix (add None validation before return)
2. **Monitor logs** for `"Received None stream"` entries to track occurrence rate
3. **Consider adding retry** at executor level for empty responses (optional, current workaround allows credential rotation which may suffice)

### Proposed Upstream Fix

Add validation in `litellm/main.py` before returning from `acompletion()`:

```python
# After line 624 (before return response)
if response is None:
    raise OpenAIError(
        status_code=500,
        message=f"acompletion returned None for model={model}, stream={stream}. "
                "This typically indicates a silent failure in the response handling."
    )
return response
```

## Related Files

| File | Purpose |
|------|---------|
| `src/rotator_library/client/streaming.py:82-95` | Workaround - None stream check |
| `src/rotator_library/client/executor.py:796` | LiteLLM acompletion call |
| `venv/.../litellm/main.py:593-625` | Root cause - acompletion function |
| `venv/.../litellm/llms/openai/openai.py:953-1023` | async_streaming function |

## Commit Reference

```
fix/anthropic-nonstreaming-null-response @ e45d7f3
```

---

**Prepared by**: Antigravity (AI Assistant)  
**Updated**: 2026-01-30  
**Reviewed by**: Pending

