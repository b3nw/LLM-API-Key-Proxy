# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from src.proxy_app.responses_compat import (
    ResponsesStreamConverter,
    convert_chat_response_to_responses,
    convert_responses_input_to_messages,
)
from rotator_library.core.errors import StreamedAPIError
from rotator_library.error_handler import classify_error, should_retry_same_key, should_rotate_on_error
from rotator_library.providers.codex_provider import _convert_messages_to_responses_input


def events(raw):
    return [json.loads(line[5:].strip()) for line in raw.splitlines() if line.startswith("data:")]


def chunk(data):
    return "data: " + json.dumps(data) + "\n\n"


@pytest.mark.parametrize("cached,reasoning", [(800, 12), (0, 0)])
def test_usage_details_survive_both_response_modes(cached, reasoning):
    usage = {
        "prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }
    response = convert_chat_response_to_responses(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": usage},
        "resp_test", {},
    )
    converter = ResponsesStreamConverter("resp_test", "test")
    converter.convert_chunk(chunk({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))
    converter.convert_chunk(chunk({"choices": [], "usage": usage}))
    final = events(converter.convert_chunk("data: [DONE]"))[-1]["response"]
    assert final["usage"] == response["usage"] == {
        "input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


def test_null_usage_is_safe():
    result = convert_chat_response_to_responses({"usage": None}, "r", {})
    assert result["usage"]["input_tokens"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_error_never_becomes_completed_or_repeats(partial):
    converter = ResponsesStreamConverter("resp_test", "test")
    if partial:
        converter.convert_chunk(chunk({"choices": [{"delta": {"content": "partial"}}]}))
    output = converter.convert_chunk(chunk({"error": {
        "code": "context_length_exceeded", "message": "Too much input",
    }}))
    terminal = events(output)[-1]
    assert terminal["type"] == "response.failed"
    assert terminal["response"]["status"] == "failed"
    assert terminal["response"]["error"]["code"] == "context_length_exceeded"
    assert "response.completed" not in output
    assert converter.convert_chunk("data: [DONE]") == ""
    assert converter.finish() == ""
    assert converter.fail("another error") == ""


@pytest.mark.parametrize("end", ["done", "eof"])
def test_truncated_stream_is_failed(end):
    converter = ResponsesStreamConverter("resp_test", "test")
    converter.convert_chunk(chunk({"choices": [{"delta": {"content": "partial"}}]}))
    output = converter.finish() if end == "eof" else converter.convert_chunk("data: [DONE]")
    assert events(output)[-1]["response"]["error"]["code"] == "incomplete_stream"
    assert "response.completed" not in output


@pytest.mark.parametrize("reason,status", [("stop", "completed"), ("tool_calls", "completed"), ("length", "incomplete")])
def test_terminal_success_is_idempotent(reason, status):
    converter = ResponsesStreamConverter("resp_test", "test")
    converter.convert_chunk(chunk({"choices": [{"delta": {}, "finish_reason": reason}]}))
    assert events(converter.convert_chunk("data: [DONE]"))[-1]["response"]["status"] == status
    assert converter.convert_chunk("data: [DONE]") == ""
    assert converter.finish() == ""


def test_image_tool_result_stays_multimodal_through_codex():
    url = "data:image/png;base64," + "QUJD" * 10000
    messages = convert_responses_input_to_messages([
        {"type": "function_call", "call_id": "call_1", "name": "screenshot", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": [
            {"type": "input_text", "text": "Screenshot captured"},
            {"type": "input_image", "image_url": url, "detail": "high"},
        ]},
    ])
    assert messages[1] == {"role": "tool", "tool_call_id": "call_1", "content": "Screenshot captured"}
    assert messages[2]["content"][0]["image_url"] == {"url": url, "detail": "high"}
    converted, _ = _convert_messages_to_responses_input(messages)
    tool = next(item for item in converted if item["type"] == "function_call_output")
    assert tool["output"] == "Screenshot captured"
    image = next(part for item in converted for part in item.get("content", []) if part["type"] == "input_image")
    assert image["image_url"] == url
    assert image["detail"] == "high"
    assert all(url not in item.get("output", "") for item in converted)


@pytest.mark.parametrize("output,expected", [
    ([{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}], "a\nb"),
    ({"result": 42}, '{"result": 42}'),
    ("raw", "raw"),
])
def test_text_tool_results_preserved(output, expected):
    messages = convert_responses_input_to_messages([{"type": "function_call_output", "call_id": "c", "output": output}])
    assert messages[0]["content"] == expected


@pytest.mark.parametrize("kind", ["stream", "dict", "wrapped_dict", "http"])
def test_context_overflow_never_rotates(kind):
    payload = {"error": {"code": "context_length_exceeded", "message": "Input exceeds the context window"}}
    if kind == "stream":
        error = StreamedAPIError(payload["error"]["message"])
    elif kind == "dict":
        error = payload
    elif kind == "wrapped_dict":
        error = StreamedAPIError("failed", data=payload)
    else:
        error = httpx.HTTPStatusError("Bad request", request=httpx.Request("POST", "https://example.com"), response=httpx.Response(400, json=payload))
    classified = classify_error(error, "codex")
    assert classified.error_type == "context_window_exceeded"
    assert not should_rotate_on_error(classified)
    assert not should_retry_same_key(classified)


@pytest.mark.parametrize("kind", ["stream", "dict", "http"])
def test_codex_account_eligibility_rotates_without_global_invalidation(kind):
    message = "The 'test-model' model is not supported when using Codex with a ChatGPT account."
    if kind == "stream":
        error = StreamedAPIError(message)
    elif kind == "dict":
        error = {"error": {"message": message}}
    else:
        error = httpx.HTTPStatusError("Bad request", request=httpx.Request("POST", "https://example.com"), response=httpx.Response(400, json={"error": {"message": message}}))
    classified = classify_error(error, "codex")
    assert classified.error_type == "model_not_supported"
    assert should_rotate_on_error(classified)
    assert not should_retry_same_key(classified)
    assert classified.retry_after == 900


def test_other_invalid_requests_and_auth_keep_their_policies():
    invalid = classify_error(StreamedAPIError("Malformed tool arguments"), "codex")
    assert not should_rotate_on_error(invalid)
    auth = classify_error({"error": {"code": "token_invalidated", "message": "Invalid token"}}, "codex")
    assert auth.error_type == "authentication"
    assert should_rotate_on_error(auth)


@pytest.mark.asyncio
async def test_eligibility_cooldown_is_model_only(tmp_path):
    from rotator_library.usage.manager import UsageManager
    from rotator_library.usage.types import CredentialState
    from rotator_library.usage.limits.cooldowns import CooldownChecker

    manager = UsageManager(provider="codex", file_path=tmp_path / "usage.json")
    state = CredentialState(stable_id="test", provider="codex", accessor="test", created_at=time.time())
    manager._states["test"] = state
    manager._save_if_needed = AsyncMock()
    error = classify_error(StreamedAPIError("The model is not supported for this account"), "codex")
    await manager._record_failure("test", "test-model", quota_group="codex-global", error=error)
    assert set(state.cooldowns) == {"test-model"}
    checker = CooldownChecker()
    assert not checker.check(state, "test-model", "codex-global").allowed
    assert checker.check(state, "other-model", "codex-global").allowed
    assert not any(cycle.exhausted for cycle in state.fair_cycle.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [0, 768])
async def test_codex_parser_retains_cache_and_reasoning(cached):
    from rotator_library.providers.codex_provider import _parse_response_events

    async def upstream():
        yield {"type": "response.completed", "response": {"usage": {
            "input_tokens": 1000, "output_tokens": 30, "total_tokens": 1030,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens_details": {"reasoning_tokens": 20},
        }}}

    chunks = [item async for item in _parse_response_events(upstream(), "test")]
    usage = chunks[-1].model_dump()["usage"]
    assert usage["prompt_tokens_details"]["cached_tokens"] == cached
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 20


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["error", "exception", "eof", "success"])
async def test_responses_http_stream_terminal_state(monkeypatch, mode):
    from types import SimpleNamespace
    monkeypatch.setenv("PROXY_API_KEY", "test-key")
    from src.proxy_app import main

    async def upstream():
        yield chunk({"choices": [{"delta": {"content": "partial"}}]})
        if mode == "exception":
            raise RuntimeError("Upstream transport failed")
        if mode == "error":
            yield chunk({"error": {"code": "context_length_exceeded", "message": "Input too large"}})
            yield "data: [DONE]\n\n"
        if mode == "success":
            yield chunk({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {
                "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
                "prompt_tokens_details": {"cached_tokens": 768},
            }})
            yield "data: [DONE]\n\n"

    monkeypatch.setattr(main.app.state, "rotating_client", SimpleNamespace(
        acompletion=AsyncMock(return_value=upstream()),
        resolve_latest_async=AsyncMock(return_value=None),
    ), raising=False)
    monkeypatch.setattr(main, "_normalize_model_id", lambda model, client: model)
    monkeypatch.setattr(main, "PROXY_API_KEY", "test-key")
    monkeypatch.setattr(main, "ENABLE_RAW_LOGGING", False)
    monkeypatch.setattr(main, "log_request_to_console", lambda **kwargs: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        result = await client.post("/v1/responses", headers={"Authorization": "Bearer test-key"}, json={
            "model": "codex/test", "input": "hi", "stream": True,
        })
    assert result.status_code == 200
    terminal = [event for event in events(result.text) if event["type"] in ("response.failed", "response.completed")]
    assert len(terminal) == 1
    assert terminal[0]["type"] == ("response.completed" if mode == "success" else "response.failed")
    if mode == "success":
        assert terminal[0]["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 768


def test_stream_overload_retries_same_credential():
    classified = classify_error(StreamedAPIError("server_is_overloaded"), "codex")
    assert classified.error_type == "server_error"
    assert should_retry_same_key(classified)
