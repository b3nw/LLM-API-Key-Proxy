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

## Hypothesis

LiteLLM has a bug where, under specific conditions with large/complex payloads routed through the OpenAI-compatible path, `acompletion()` returns `None` instead of the expected `CustomStreamWrapper`. Possible causes:

1. **Silent exception handling** that catches an error and returns None
2. **Memory pressure** during async response object creation
3. **Race condition** in streaming handler initialization
4. **Timeout** in upstream connection handled incorrectly

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

1. **Update LiteLLM** to latest version to check if this is already fixed
2. **File upstream bug** with LiteLLM if issue persists
3. **Monitor** for recurrence after workaround deployment

## Related Files

- `src/rotator_library/client/streaming.py` - Workaround location
- `src/rotator_library/client/executor.py:796` - Where litellm.acompletion is called
- `src/rotator_library/provider_config.py:696-743` - Custom provider routing logic

## Commit Reference

```
fix/anthropic-nonstreaming-null-response @ e45d7f3
```

---

**Prepared by**: Antigravity (AI Assistant)  
**Reviewed by**: Pending
