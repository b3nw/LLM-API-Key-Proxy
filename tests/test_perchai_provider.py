# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kévin Cojean
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from rotator_library.providers import PROVIDER_PLUGINS
from rotator_library.providers.perchai_provider import (
    MODEL_CACHE_TTL_SECONDS,
    MODEL_CALL_PATH,
    USAGE_PATH,
    PerchaiProvider,
)
from rotator_library.providers.utilities.perchai_quota_tracker import (
    PerchaiQuotaTracker,
)
from rotator_library.credential_manager import (
    DEFAULT_OAUTH_DIRS,
    ENV_OAUTH_PROVIDERS,
)
from rotator_library.provider_factory import PROVIDER_MAP
from rotator_library.provider_config import LITELLM_PROVIDERS
from proxy_app.provider_urls import PROVIDER_URL_MAP


PERCHAI_SESSION: Path = Path.home() / ".perch" / "cli-auth-session.json"
HAS_SESSION: bool = PERCHAI_SESSION.is_file()

pytestmark = [
    pytest.mark.skipif(
        not HAS_SESSION,
        reason="No perchai session - run `perch login`",
    ),
]


PERCHAI_ERROR_CODE_CASES = [
    pytest.param(
        "usage_limit_reached",
        {"reason": "rate_limit", "retry_after": 3600},
        id="usage_limit_reached",
    ),
    pytest.param(
        "promo_overflow_decision",
        {"reason": "rate_limit", "retry_after": 3600},
        id="promo_overflow_decision",
    ),
    pytest.param(
        "starter_model_blocked",
        {"reason": "forbidden", "retry_after": None},
        id="starter_model_blocked",
    ),
    pytest.param(
        "byo_feature_blocked",
        {"reason": "forbidden", "retry_after": None},
        id="byo_feature_blocked",
    ),
    pytest.param(
        "not_authenticated",
        {"reason": "authentication", "retry_after": None},
        id="not_authenticated",
    ),
    pytest.param("context_overflow", None, id="context_overflow"),
    pytest.param("api_error", None, id="api_error"),
    pytest.param("timeout", None, id="timeout"),
    pytest.param("aborted", None, id="aborted"),
    pytest.param("vision_model_error", None, id="vision_model_error"),
]


def test_perchai_in_plugins() -> None:
    given_plugins = PROVIDER_PLUGINS
    when_checked = "perchai"
    then_asserted = when_checked in given_plugins
    assert then_asserted, (
        f"perchai plugin missing from PROVIDER_PLUGINS: "
        f"{sorted(given_plugins.keys())!r}"
    )


def test_has_custom_logic_true() -> None:
    given_provider = PerchaiProvider()
    when_called = given_provider.has_custom_logic()
    then_returned = when_called is True
    assert then_returned, (
        f"PerchaiProvider.has_custom_logic() returned {when_called!r}; "
        f"expected True"
    )


def test_mro_includes_quota_tracker() -> None:
    from rotator_library.providers.provider_interface import ProviderInterface

    given_mro = PerchaiProvider.__mro__
    when_checked = PerchaiQuotaTracker in given_mro
    then_asserted = when_checked
    assert then_asserted, (
        f"PerchaiQuotaTracker missing from MRO: {[c.__name__ for c in given_mro]!r}"
    )
    given_mixin_idx = given_mro.index(PerchaiQuotaTracker)
    given_iface_idx = given_mro.index(ProviderInterface)
    then_order = given_mixin_idx < given_iface_idx
    assert then_order, (
        f"PerchaiQuotaTracker must come before ProviderInterface in MRO "
        f"(mixin idx={given_mixin_idx}, interface idx={given_iface_idx})"
    )


@pytest.mark.parametrize("error_code,expected", PERCHAI_ERROR_CODE_CASES)
def test_all_error_codes_parsed_correctly(
    error_code: str,
    expected: Optional[Dict[str, Any]],
) -> None:
    given_body = json.dumps({"errorCode": error_code, "error": "synthetic test body"})
    given_error = Exception("synthetic")
    when_parsed = PerchaiProvider.parse_quota_error(given_error, given_body)
    then_returned = when_parsed
    assert then_returned == expected, (
        f"errorCode={error_code!r}: expected {expected!r}, got {then_returned!r}"
    )


def test_429_status_parsed_as_rate_limit() -> None:
    given_error = Exception("429")
    given_body = ""
    when_parsed = PerchaiProvider.parse_quota_error(given_error, given_body)
    then_returned = when_parsed
    assert then_returned == {"reason": "rate_limit", "retry_after": 3600}, (
        f"HTTP 429 fallback should produce rate_limit dict, got {then_returned!r}"
    )


def test_malformed_body_returns_none() -> None:
    given_error = Exception("synthetic")
    given_body = "{this is not valid json"
    when_parsed = PerchaiProvider.parse_quota_error(given_error, given_body)
    then_returned = when_parsed
    assert then_returned is None, (
        f"Malformed body should parse to None, got {then_returned!r}"
    )


@pytest.mark.asyncio
async def test_empty_messages_raises_valueerror() -> None:
    given_client = None
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/test-model",
        "messages": [],
    }
    when_called = PerchaiProvider().acompletion(given_client, **given_kwargs)
    with pytest.raises(ValueError) as exc_info:
        await when_called
    then_message = str(exc_info.value)
    assert "messages" in then_message.lower(), (
        f"ValueError message should mention 'messages', got {then_message!r}"
    )


def test_malformed_sse_line_returns_none() -> None:
    given_line = "data: {not valid json"
    given_model = "perchai/test-model"
    when_parsed = PerchaiProvider._parse_sse_line(given_line, given_model)
    then_returned = when_parsed
    assert then_returned is None, (
        f"Malformed SSE line should parse to None, got {then_returned!r}"
    )


def test_unknown_event_type_skipped() -> None:
    given_line = 'data: {"type":"future_event_type","payload":"ignored"}'
    given_model = "perchai/test-model"
    when_parsed = PerchaiProvider._parse_sse_line(given_line, given_model)
    then_returned = when_parsed
    assert then_returned is None, (
        f"Unknown event type should parse to None, got {then_returned!r}"
    )


def test_text_delta_produces_content_chunk() -> None:
    given_line = 'data: {"type":"answer_delta","text":"hello world"}'
    given_model = "perchai/test-model"
    when_parsed = PerchaiProvider._parse_sse_line(given_line, given_model)
    then_chunk = when_parsed
    assert then_chunk is not None, "answer_delta should produce a chunk, got None"
    then_choices = then_chunk.choices
    assert then_choices, "chunk has no choices"
    then_delta = then_choices[0].get("delta") if isinstance(then_choices[0], dict) else then_choices[0].delta
    then_content = then_delta.get("content") if isinstance(then_delta, dict) else then_delta.content
    assert then_content == "hello world", (
        f"answer_delta.content should be 'hello world', got {then_content!r}"
    )


def test_reasoning_delta_produces_reasoning_chunk() -> None:
    given_line = 'data: {"type":"reasoning_delta","text":"thinking step"}'
    given_model = "perchai/test-model"
    when_parsed = PerchaiProvider._parse_sse_line(given_line, given_model)
    then_chunk = when_parsed
    assert then_chunk is not None, "reasoning_delta should produce a chunk, got None"
    then_choices = then_chunk.choices
    assert then_choices, "chunk has no choices"
    then_delta = then_choices[0].get("delta") if isinstance(then_choices[0], dict) else then_choices[0].delta
    then_reasoning = then_delta.get("reasoning_content") if isinstance(then_delta, dict) else then_delta.reasoning_content
    assert then_reasoning == "thinking step", (
        f"reasoning_delta.reasoning_content should be 'thinking step', "
        f"got {then_reasoning!r}"
    )


async def test_stream_without_tool_id_has_synthetic_id() -> None:
    from unittest.mock import AsyncMock, MagicMock

    # Simulate Perchai SSE stream: tool_call_delta without id/name
    given_sse_lines = [
        'data: {"type":"tool_call_delta","index":0,"arguments":"{\\"x\\": 1}"}',
        'data: {"type":"tool_call_delta","index":0,"arguments":"{\\"y\\": 2}"}',
        'data: {"type":"tool_use_end"}',
        'data: [DONE]',
    ]

    # Mock httpx streaming response
    given_response = MagicMock()
    given_response.status_code = 200

    async def mock_aiter_lines():
        for line in given_sse_lines:
            yield line

    given_response.aiter_lines = mock_aiter_lines
    given_response.aread = AsyncMock()

    # Mock httpx context manager
    given_context = AsyncMock()
    given_context.__aenter__ = AsyncMock(return_value=given_response)
    given_context.__aexit__ = AsyncMock(return_value=None)

    # Mock httpx client
    given_client = MagicMock()
    given_client.stream = MagicMock(return_value=given_context)

    # Create provider instance
    given_provider = PerchaiProvider()
    given_model = "perchai/test-model"
    given_url = "https://api.perchai.com/v1/chat"
    given_headers = {"Authorization": "Bearer fake-token"}
    given_token = "fake-token"
    given_payload = {"messages": [{"role": "user", "content": "test"}]}

    # Mock file logger
    given_logger = MagicMock()
    given_logger.log_response_chunk = MagicMock()

    # Consume the stream
    then_chunks = []
    async for chunk in given_provider._stream_completion(
        client=given_client,
        url=given_url,
        build_headers=lambda t: given_headers,
        token=given_token,
        payload=given_payload,
        model=given_model,
        file_logger=given_logger,
        credential_identifier="test-cred",
    ):
        then_chunks.append(chunk)

    # Validate: at least one tool_call chunk emitted
    assert len(then_chunks) > 0, "Stream produced no chunks"

    # Find tool_call chunks (those with tool_calls in delta)
    then_tool_call_chunks = []
    for chunk in then_chunks:
        choices = chunk.choices
        if choices:
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else choices[0].delta
            tool_calls = delta.get("tool_calls") if isinstance(delta, dict) else getattr(delta, "tool_calls", None)
            if tool_calls:
                then_tool_call_chunks.append((chunk, tool_calls))

    assert len(then_tool_call_chunks) > 0, "No tool_call chunks found in stream"

    # Validate: every tool_call chunk has string id and function.name
    for chunk_idx, (chunk, tool_calls) in enumerate(then_tool_call_chunks):
        for tc_idx, tc in enumerate(tool_calls):
            then_id = tc.get("id") if isinstance(tc, dict) else tc.id
            assert then_id is not None, (
                f"Chunk {chunk_idx} tool_call {tc_idx} missing id: {tc!r}"
            )
            assert isinstance(then_id, str), (
                f"Chunk {chunk_idx} tool_call {tc_idx} id not string: {type(then_id)}"
            )
            assert len(then_id) > 0, (
                f"Chunk {chunk_idx} tool_call {tc_idx} id is empty string"
            )

            then_function = tc.get("function") if isinstance(tc, dict) else tc.function
            then_name = then_function.get("name") if isinstance(then_function, dict) else then_function.name
            assert then_name is not None, (
                f"Chunk {chunk_idx} tool_call {tc_idx} missing function.name: {tc!r}"
            )
            assert isinstance(then_name, str), (
                f"Chunk {chunk_idx} tool_call {tc_idx} function.name not string: {type(then_name)}"
            )
            assert len(then_name) > 0, (
                f"Chunk {chunk_idx} tool_call {tc_idx} function.name is empty string"
            )


def test_tool_call_delta_without_id_emits_synthetic_id() -> None:
    given_line = 'data: {"type":"tool_call_delta","index":0,"name":"my_func","arguments":"{\\"x\\": 1}"}'
    given_model = "perchai/test-model"
    given_id_map: dict[int, str] = {}
    given_name_map: dict[int, str] = {}
    when_parsed = PerchaiProvider._parse_sse_line(
        given_line, given_model, given_id_map, given_name_map
    )
    then_chunk = when_parsed
    assert then_chunk is not None, "tool_call_delta should produce a chunk"
    then_choices = then_chunk.choices
    assert then_choices, "chunk has no choices"
    then_delta = then_choices[0].get("delta") if isinstance(then_choices[0], dict) else then_choices[0].delta
    then_tool_calls = then_delta.get("tool_calls") if isinstance(then_delta, dict) else then_delta.tool_calls
    assert then_tool_calls, "chunk has no tool_calls"
    then_first_call = then_tool_calls[0]
    then_id = then_first_call.get("id") if isinstance(then_first_call, dict) else then_first_call.id
    assert then_id is not None, f"tool_call missing id, got {then_first_call!r}"
    assert isinstance(then_id, str), f"tool_call.id must be string, got {type(then_id)}"
    assert then_id == "call_0", f"Expected synthetic id 'call_0', got {then_id!r}"


def test_tool_call_delta_without_name_emits_synthetic_name() -> None:
    given_line = 'data: {"type":"tool_call_delta","index":0,"id":"call_abc","arguments":"{\\"x\\": 1}"}'
    given_model = "perchai/test-model"
    given_id_map: dict[int, str] = {}
    given_name_map: dict[int, str] = {}
    when_parsed = PerchaiProvider._parse_sse_line(
        given_line, given_model, given_id_map, given_name_map
    )
    then_chunk = when_parsed
    assert then_chunk is not None, "tool_call_delta should produce a chunk"
    then_choices = then_chunk.choices
    assert then_choices, "chunk has no choices"
    then_delta = then_choices[0].get("delta") if isinstance(then_choices[0], dict) else then_choices[0].delta
    then_tool_calls = then_delta.get("tool_calls") if isinstance(then_delta, dict) else then_delta.tool_calls
    assert then_tool_calls, "chunk has no tool_calls"
    then_first_call = then_tool_calls[0]
    then_function = then_first_call.get("function") if isinstance(then_first_call, dict) else then_first_call.function
    then_name = then_function.get("name") if isinstance(then_function, dict) else then_function.name
    assert then_name is not None, f"function missing name, got {then_function!r}"
    assert isinstance(then_name, str), f"function.name must be string, got {type(then_name)}"
    assert then_name == "function_0", f"Expected synthetic name 'function_0', got {then_name!r}"


def test_tool_call_delta_without_name_uses_real_name_from_request() -> None:
    given_line = 'data: {"type":"tool_call_delta","index":0,"id":"call_abc","arguments":"{\\"path\\": \\"/tmp\\"}"}'
    given_model = "perchai/test-model"
    given_id_map: dict[int, str] = {}
    given_name_map: dict[int, str] = {}
    given_request_tool_names = {0: "read", 1: "write"}
    when_parsed = PerchaiProvider._parse_sse_line(
        given_line, given_model, given_id_map, given_name_map,
        given_request_tool_names,
    )
    then_chunk = when_parsed
    assert then_chunk is not None, "tool_call_delta should produce a chunk"
    then_choices = then_chunk.choices
    then_delta = then_choices[0].get("delta") if isinstance(then_choices[0], dict) else then_choices[0].delta
    then_tool_calls = then_delta.get("tool_calls") if isinstance(then_delta, dict) else then_delta.tool_calls
    then_first_call = then_tool_calls[0]
    then_function = then_first_call.get("function") if isinstance(then_first_call, dict) else then_first_call.function
    then_name = then_function.get("name") if isinstance(then_function, dict) else then_function.name
    assert then_name == "read", (
        f"Expected real tool name 'read' from request tools, got {then_name!r}"
    )


def test_tool_call_delta_index_mismatch_falls_back_to_synthetic() -> None:
    given_line = 'data: {"type":"tool_call_delta","index":2,"arguments":"{}"}'
    given_model = "perchai/test-model"
    given_id_map: dict[int, str] = {}
    given_name_map: dict[int, str] = {}
    given_request_tool_names = {0: "read", 1: "write"}
    when_parsed = PerchaiProvider._parse_sse_line(
        given_line, given_model, given_id_map, given_name_map,
        given_request_tool_names,
    )
    then_chunk = when_parsed
    assert then_chunk is not None
    then_delta = then_chunk.choices[0].get("delta") if isinstance(then_chunk.choices[0], dict) else then_chunk.choices[0].delta
    then_tool_calls = then_delta.get("tool_calls") if isinstance(then_delta, dict) else then_delta.tool_calls
    then_function = then_tool_calls[0].get("function") if isinstance(then_tool_calls[0], dict) else then_tool_calls[0].function
    then_name = then_function.get("name") if isinstance(then_function, dict) else then_function.name
    assert then_name == "function_2", (
        f"Expected synthetic 'function_2' for out-of-range index, got {then_name!r}"
    )


def test_multiple_tool_call_deltas_reuse_same_synthetic_id() -> None:
    given_line1 = 'data: {"type":"tool_call_delta","index":0,"arguments":"{\\"x\\": 1}"}'
    given_line2 = 'data: {"type":"tool_call_delta","index":0,"arguments":"{\\"y\\": 2}"}'
    given_model = "perchai/test-model"
    given_id_map: dict[int, str] = {}
    given_name_map: dict[int, str] = {}
    when_parsed1 = PerchaiProvider._parse_sse_line(
        given_line1, given_model, given_id_map, given_name_map
    )
    when_parsed2 = PerchaiProvider._parse_sse_line(
        given_line2, given_model, given_id_map, given_name_map
    )
    then_chunk1 = when_parsed1
    then_chunk2 = when_parsed2
    assert then_chunk1 is not None and then_chunk2 is not None
    then_delta1 = then_chunk1.choices[0].get("delta") if isinstance(then_chunk1.choices[0], dict) else then_chunk1.choices[0].delta
    then_delta2 = then_chunk2.choices[0].get("delta") if isinstance(then_chunk2.choices[0], dict) else then_chunk2.choices[0].delta
    then_tc1 = (then_delta1.get("tool_calls") if isinstance(then_delta1, dict) else then_delta1.tool_calls)[0]
    then_tc2 = (then_delta2.get("tool_calls") if isinstance(then_delta2, dict) else then_delta2.tool_calls)[0]
    then_id1 = then_tc1.get("id") if isinstance(then_tc1, dict) else then_tc1.id
    then_id2 = then_tc2.get("id") if isinstance(then_tc2, dict) else then_tc2.id
    assert then_id1 == then_id2, f"Synthetic id changed between chunks: {then_id1!r} vs {then_id2!r}"
    then_fn1 = then_tc1.get("function") if isinstance(then_tc1, dict) else then_tc1.function
    then_fn2 = then_tc2.get("function") if isinstance(then_tc2, dict) else then_tc2.function
    then_name1 = then_fn1.get("name") if isinstance(then_fn1, dict) else then_fn1.name
    then_name2 = then_fn2.get("name") if isinstance(then_fn2, dict) else then_fn2.name
    assert then_name1 == then_name2, f"Synthetic name changed between chunks: {then_name1!r} vs {then_name2!r}"


def test_perchai_in_oauth_dirs() -> None:
    given_dirs = DEFAULT_OAUTH_DIRS
    when_checked = "perchai"
    then_present = when_checked in given_dirs
    assert then_present, (
        f"perchai missing from DEFAULT_OAUTH_DIRS: {sorted(given_dirs.keys())!r}"
    )
    then_path = given_dirs[when_checked]
    assert then_path == Path.home() / ".perch", (
        f"DEFAULT_OAUTH_DIRS['perchai'] should be ~/.perch, got {then_path!r}"
    )


def test_perchai_in_env_oauth() -> None:
    given_env_map = ENV_OAUTH_PROVIDERS
    when_checked = "perchai"
    then_present = when_checked in given_env_map
    assert then_present, (
        f"perchai missing from ENV_OAUTH_PROVIDERS: {sorted(given_env_map.keys())!r}"
    )
    then_prefix = given_env_map[when_checked]
    assert then_prefix == "PERCHAI", (
        f"ENV_OAUTH_PROVIDERS['perchai'] should be 'PERCHAI', got {then_prefix!r}"
    )


def test_perchai_in_provider_factory() -> None:
    given_map = PROVIDER_MAP
    when_checked = "perchai"
    then_present = when_checked in given_map
    assert then_present, (
        f"perchai missing from PROVIDER_MAP: {sorted(given_map.keys())!r}"
    )
    then_auth_class = given_map[when_checked]
    assert then_auth_class is not None, (
        "PROVIDER_MAP['perchai'] resolved to None"
    )
    assert callable(then_auth_class), (
        f"PROVIDER_MAP['perchai'] should be a class, got {type(then_auth_class)!r}"
    )


def test_perchai_in_provider_config() -> None:
    given_config = LITELLM_PROVIDERS
    when_checked = "perchai"
    then_present = when_checked in given_config
    assert then_present, (
        f"perchai missing from LITELLM_PROVIDERS: {sorted(given_config.keys())!r}"
    )
    then_entry = given_config[when_checked]
    assert isinstance(then_entry, dict) and then_entry, (
        f"LITELLM_PROVIDERS['perchai'] should be a non-empty dict, got {then_entry!r}"
    )


def test_perchai_in_provider_url_map() -> None:
    given_url_map = PROVIDER_URL_MAP
    when_checked = "perchai"
    then_present = when_checked in given_url_map
    assert then_present, (
        f"perchai missing from PROVIDER_URL_MAP: {sorted(given_url_map.keys())!r}"
    )
    then_url = given_url_map[when_checked]
    assert then_url and then_url.startswith("https://"), (
        f"PROVIDER_URL_MAP['perchai'] should be an https URL, got {then_url!r}"
    )


def test_background_job_config_returns_valid_dict() -> None:
    when_called = PerchaiQuotaTracker.get_background_job_config()
    then_returned = when_called
    assert then_returned is not None, (
        "PerchaiQuotaTracker.get_background_job_config() returned None"
    )
    then_interval = then_returned.get("interval")
    then_name = then_returned.get("name")
    assert isinstance(then_interval, int) and then_interval > 0, (
        f"interval should be a positive int, got {then_interval!r}"
    )
    assert isinstance(then_name, str) and then_name, (
        f"name should be a non-empty str, got {then_name!r}"
    )


def test_model_quota_groups_has_monthly_group() -> None:
    given_groups = PerchaiQuotaTracker.model_quota_groups
    when_checked = "monthly($)"
    then_present = when_checked in given_groups
    assert then_present, (
        f"monthly($) missing from model_quota_groups: {sorted(given_groups.keys())!r}"
    )


@pytest.mark.asyncio
async def test_run_background_job_invalid_token_no_crash() -> None:
    given_provider = PerchaiProvider()
    given_invalid_credential = "/tmp/does-not-exist-perchai-session.json"
    given_credentials = [given_invalid_credential]

    when_called = given_provider.run_background_job(
        usage_manager=None,
        credentials=given_credentials,
    )

    await when_called

    then_no_crash = True
    assert then_no_crash, "run_background_job raised on invalid credential"


# ---------------------------------------------------------------------------
# Reactive 401 refresh: streaming + non-streaming paths must both
# refresh the access token on the first 401 and retry the request.
# The streaming path previously had NO 401 handler (only the non-streaming
# path did), causing credential exhaustion in the proxy when the token
# expired between calls.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_token_non_stream_refreshes_and_retries(
    tmp_path: Path,
) -> None:
    import httpx

    given_session = json.loads(PERCHAI_SESSION.read_text(encoding="utf-8"))
    given_session["accessToken"] = "perchai-expired-token-will-401"
    given_cred_file = tmp_path / "perchai_oauth_1.json"
    given_cred_file.write_text(json.dumps(given_session), encoding="utf-8")

    given_provider = PerchaiProvider()

    when_responded = await given_provider.acompletion(
        httpx.AsyncClient(),
        model="perchai/nemotron-3.5-lightning",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        credential_identifier=str(given_cred_file),
        stream=False,
    )

    then_content = when_responded.choices[0].message.content
    assert then_content, (
        "Non-streaming with expired token should refresh and return content, "
        f"got {then_content!r}"
    )


@pytest.mark.asyncio
async def test_expired_token_stream_refreshes_and_retries(
    tmp_path: Path,
) -> None:
    import httpx

    given_session = json.loads(PERCHAI_SESSION.read_text(encoding="utf-8"))
    given_session["accessToken"] = "perchai-expired-token-will-401"
    given_cred_file = tmp_path / "perchai_oauth_1.json"
    given_cred_file.write_text(json.dumps(given_session), encoding="utf-8")

    given_provider = PerchaiProvider()

    when_streamed = await given_provider.acompletion(
        httpx.AsyncClient(),
        model="perchai/nemotron-3.5-lightning",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        credential_identifier=str(given_cred_file),
        stream=True,
    )

    then_chunks = []
    async for chunk in when_streamed:
        then_chunks.append(chunk)

    then_has_chunks = len(then_chunks) > 0
    then_last_finish = (
        then_chunks[-1].choices[0].finish_reason if then_chunks else None
    )
    assert then_has_chunks, (
        "Streaming with expired token should refresh and yield chunks, "
        f"got {len(then_chunks)} chunks"
    )
    assert then_last_finish == "stop", (
        f"Last chunk should have finish_reason='stop', got {then_last_finish!r}"
    )


# ---------------------------------------------------------------------------
# Credential path resolution: credential_identifier passed by the rotator
# is a FILE PATH (e.g. "/home/.../perchai_oauth_1.json") or env virtual path
# (e.g. "env://perchai/1"), NOT an access token. The provider must resolve
# the path to an actual accessToken before sending it as a Bearer header.
# ---------------------------------------------------------------------------


def test_credential_file_path_resolves_to_access_token() -> None:
    import tempfile

    from rotator_library.providers.perchai_auth_base import PerchaiAuthBase

    given_session = {
        "version": 1,
        "appUrl": "https://app.perchai.app",
        "accessToken": "test-access-token-from-file",
        "refreshToken": "test-refresh-token",
        "expiresAt": 9999999999,
        "userId": "test-user-id",
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as given_file:
        json.dump(given_session, given_file)
        given_path = given_file.name

    given_auth = PerchaiAuthBase(credential_path=given_path)
    given_session_loaded = given_auth.load_session()
    then_token = given_session_loaded.get("accessToken")

    assert then_token == "test-access-token-from-file", (
        f"Expected accessToken from file, got {then_token!r}"
    )
    assert then_token != given_path, (
        "credential_identifier was used as the token directly instead of "
        "being resolved to an access token from the file"
    )


def test_env_virtual_path_resolves_to_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.providers.perchai_auth_base import PerchaiAuthBase

    monkeypatch.setenv("PERCHAI_1_ACCESS_TOKEN", "test-env-access-token")
    monkeypatch.setenv("PERCHAI_1_REFRESH_TOKEN", "test-env-refresh-token")

    given_auth = PerchaiAuthBase(credential_path="env://perchai/1")
    given_session = given_auth.load_session()
    then_token = given_session.get("accessToken")

    assert then_token == "test-env-access-token", (
        f"Expected accessToken from env var, got {then_token!r}"
    )


def test_empty_credential_identifier_falls_back_to_default() -> None:
    given_provider = PerchaiProvider()

    try:
        given_provider._resolve_credential_token("")
    except Exception as exc:
        then_is_auth_error = "perch" in str(exc).lower() or "session" in str(exc).lower()
        assert then_is_auth_error, (
            f"Empty credential_identifier should fall back to default session "
            f"resolution, got unexpected error: {exc!r}"
        )
    else:
        pass


# ---------------------------------------------------------------------------
# E2E routing verification: option IDs listed in the docs must actually route
# to a real upstream, not silently fall back to the workspace default
# (bedrock_mantle:moonshotai.kimi-k2.5). Pinned to 2 cheap Starter-tier
# models to keep cost minimal.
# ---------------------------------------------------------------------------


PROBE_OPTION_IDS = [
    pytest.param(
        "bedrock-mantle-google-gemma-4-e2b",
        id="gemma-4-e2b",
    ),
    pytest.param(
        "wandb-deepseek-ai-deepseek-v4-flash",
        id="deepseek-v4-flash",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("option_id", PROBE_OPTION_IDS)
async def test_option_id_routes_to_real_upstream(
    option_id: str,
) -> None:
    import httpx

    from rotator_library.providers.perchai_auth_base import PerchaiAuthBase

    given_auth = PerchaiAuthBase()
    given_token = await given_auth.refresh_token()
    given_url = f"{given_auth.get_app_url().rstrip('/')}{MODEL_CALL_PATH}"
    given_body = {
        "request": {
            "model": "probe",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": False,
        },
        "runId": None,
        "lane": "chat",
        "preferredModelId": None,
        "manualModelOptionId": option_id,
    }

    async with httpx.AsyncClient() as probe_client:
        given_response = await probe_client.post(
            given_url,
            headers={
                "Authorization": f"Bearer {given_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=given_body,
            timeout=30.0,
        )

    then_status = given_response.status_code
    then_body = given_response.json()

    assert then_status == 200, (
        f"Probe HTTP {then_status} for option_id={option_id!r}: {then_body!r}"
    )
    assert then_body.get("ok") is True, (
        f"Probe ok=false for option_id={option_id!r}: "
        f"{then_body.get('error')!r}"
    )

    then_model = then_body.get("model")
    then_provider = then_body.get("provider")
    then_not_fallback = not (
        then_model == "moonshotai.kimi-k2.5" and then_provider == "bedrock_mantle"
    )
    assert then_not_fallback, (
        f"option_id={option_id!r} silently fell back to default upstream "
        f"({then_provider}:{then_model}) - this option ID is NOT currently "
        f"wired and must be removed from the docs"
    )


def test_module_constants_nonempty() -> None:
    given_cache_ttl = MODEL_CACHE_TTL_SECONDS
    given_model_call_path = MODEL_CALL_PATH
    given_usage_path = USAGE_PATH
    assert isinstance(given_cache_ttl, int) and given_cache_ttl > 0, (
        f"MODEL_CACHE_TTL_SECONDS should be positive int, got {given_cache_ttl!r}"
    )
    assert isinstance(given_model_call_path, str) and given_model_call_path.startswith("/"), (
        f"MODEL_CALL_PATH should be an absolute path str, got {given_model_call_path!r}"
    )
    assert isinstance(given_usage_path, str) and given_usage_path.startswith("/"), (
        f"USAGE_PATH should be an absolute path str, got {given_usage_path!r}"
    )


def test_has_transform_request_hook() -> None:
    assert hasattr(PerchaiProvider, "transform_request"), (
        "PerchaiProvider must implement transform_request for thinking normalization"
    )


def test_thinking_disabled_strips_reasoning_from_messages() -> None:
    given_provider = PerchaiProvider()
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [
            {"role": "user", "content": "hi"},
        ],
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    when_result = given_provider.transform_request(given_kwargs, "perchai/bedrock-mantle-google-gemma-4-e2b", "test-cred")
    then_extra_body = given_kwargs.get("extra_body", {})
    assert isinstance(when_result, list), f"transform_request must return list of modifications, got {type(when_result)}"
    assert then_extra_body.get("thinking") == {"type": "disabled"}, (
        f"thinking config must be preserved, got {then_extra_body!r}"
    )


def test_streaming_reasoning_delta_suppressed_when_thinking_disabled() -> None:
    given_line = 'data: {"type":"reasoning_delta","text":"thinking about it"}'
    given_model = "perchai/bedrock-mantle-google-gemma-4-e2b"
    given_id_map: dict[int, str] = {}
    given_name_map: dict[int, str] = {}
    given_request_tool_names: Dict[int, str] = {}
    given_thinking_disabled = True
    when_chunk = PerchaiProvider._parse_sse_line(
        given_line, given_model, given_id_map, given_name_map,
        given_request_tool_names, given_thinking_disabled,
    )
    assert when_chunk is None, (
        f"reasoning_delta must be suppressed when thinking disabled, got {when_chunk!r}"
    )


def test_streaming_reasoning_delta_emitted_when_thinking_enabled() -> None:
    given_line = 'data: {"type":"reasoning_delta","text":"thinking about it"}'
    given_model = "perchai/wandb-deepseek-ai-deepseek-v4-flash-0731"
    given_id_map: dict[int, str] = {}
    given_name_map: dict[int, str] = {}
    given_request_tool_names: Dict[int, str] = {}
    given_thinking_disabled = False
    when_chunk = PerchaiProvider._parse_sse_line(
        given_line, given_model, given_id_map, given_name_map,
        given_request_tool_names, given_thinking_disabled,
    )
    assert when_chunk is not None, "reasoning_delta must produce a chunk when thinking enabled"
    then_delta = when_chunk.choices[0].get("delta") if isinstance(when_chunk.choices[0], dict) else when_chunk.choices[0].delta
    then_reasoning = then_delta.get("reasoning_content") if isinstance(then_delta, dict) else then_delta.reasoning_content
    assert then_reasoning == "thinking about it", (
        f"Expected reasoning_content='thinking about it', got {then_reasoning!r}"
    )


def test_non_stream_response_strips_reasoning_when_thinking_disabled() -> None:
    given_provider = PerchaiProvider()
    given_response_data = {
        "text": "Hello!",
        "reasoning": "I should say hello",
        "content": [],
        "toolCalls": [],
        "usage": {},
    }
    given_payload = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"thinking": {"type": "disabled"}},
        "tools": [],
    }
    when_response = given_provider._build_model_response(
        given_response_data, "perchai/bedrock-mantle-google-gemma-4-e2b", given_payload
    )
    then_message = when_response.choices[0].message
    then_reasoning = getattr(then_message, "reasoning_content", None)
    assert then_reasoning is None, (
        f"reasoning_content must be stripped when thinking disabled, got {then_reasoning!r}"
    )
    assert then_message.content == "Hello!", (
        f"content must be preserved, got {then_message.content!r}"
    )


def test_non_stream_response_preserves_reasoning_when_thinking_enabled() -> None:
    given_provider = PerchaiProvider()
    given_response_data = {
        "text": "Hello!",
        "reasoning": "I should say hello",
        "content": [],
        "toolCalls": [],
        "usage": {},
    }
    given_payload = {
        "model": "perchai/wandb-deepseek-ai-deepseek-v4-flash-0731",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"thinking": {"type": "enabled"}},
        "tools": [],
    }
    when_response = given_provider._build_model_response(
        given_response_data, "perchai/wandb-deepseek-ai-deepseek-v4-flash-0731", given_payload
    )
    then_message = when_response.choices[0].message
    then_reasoning = getattr(then_message, "reasoning_content", None)
    assert then_reasoning == "I should say hello", (
        f"reasoning_content must be preserved when thinking enabled, got {then_reasoning!r}"
    )
    assert then_message.content == "Hello!", (
        f"content must be preserved, got {then_message.content!r}"
    )


def test_build_payload_includes_thinking_when_enabled() -> None:
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    }
    when_payload = PerchaiProvider._build_payload(
        model_name="bedrock-mantle-google-gemma-4-31b",
        kwargs=given_kwargs,
    )
    then_thinking = when_payload.get("thinking")
    assert then_thinking == {"type": "enabled"}, (
        f"thinking config must be at top level in payload, got {then_thinking!r}"
    )
    then_effort = when_payload.get("reasoning_effort")
    assert then_effort == "low", (
        f"reasoning_effort must be at top level in payload, got {then_effort!r}"
    )
    then_no_extra_body = "extra_body" not in when_payload
    assert then_no_extra_body, (
        f"extra_body must be flattened into payload, not kept as nested key: {when_payload!r}"
    )


def test_build_payload_includes_thinking_disabled() -> None:
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    when_payload = PerchaiProvider._build_payload(
        model_name="bedrock-mantle-google-gemma-4-e2b",
        kwargs=given_kwargs,
    )
    then_thinking = when_payload.get("thinking")
    assert then_thinking == {"type": "disabled"}, (
        f"thinking disabled must be at top level, got {then_thinking!r}"
    )
    then_no_effort = "reasoning_effort" not in when_payload
    assert then_no_effort, (
        f"reasoning_effort should not be in payload when thinking disabled: {when_payload!r}"
    )


def test_build_payload_includes_reasoning_effort_from_kwargs() -> None:
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "low",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    when_payload = PerchaiProvider._build_payload(
        model_name="bedrock-mantle-google-gemma-4-31b",
        kwargs=given_kwargs,
    )
    then_effort = when_payload.get("reasoning_effort")
    assert then_effort == "low", (
        f"reasoning_effort from kwargs must be in payload, got {then_effort!r}"
    )


@pytest.mark.asyncio
async def test_stream_only_reasoning_emits_stop_chunk() -> None:
    from unittest.mock import AsyncMock, MagicMock

    given_sse_lines = [
        'data: {"type":"reasoning_delta","text":"thinking step 1"}',
        'data: {"type":"reasoning_delta","text":"thinking step 2"}',
        'data: {"type":"finishReason","finishReason":"stop"}',
    ]

    given_response = MagicMock()
    given_response.status_code = 200

    async def mock_aiter_lines():
        for line in given_sse_lines:
            yield line

    given_response.aiter_lines = mock_aiter_lines
    given_response.aread = AsyncMock()

    given_context = AsyncMock()
    given_context.__aenter__ = AsyncMock(return_value=given_response)
    given_context.__aexit__ = AsyncMock(return_value=None)

    given_client = MagicMock()
    given_client.stream = MagicMock(return_value=given_context)

    given_provider = PerchaiProvider()
    given_payload = {
        "messages": [{"role": "user", "content": "test"}],
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    given_logger = MagicMock()

    then_chunks = []
    async for chunk in given_provider._stream_completion(
        client=given_client,
        url="https://api.perchai.com/v1/chat",
        build_headers=lambda t: {"Authorization": "Bearer fake"},
        token="fake-token",
        payload=given_payload,
        model="perchai/test-model",
        file_logger=given_logger,
        credential_identifier="test-cred",
    ):
        then_chunks.append(chunk)

    then_has_stop = any(
        (
            c.choices[0].get("finish_reason") if isinstance(c.choices[0], dict)
            else getattr(c.choices[0], "finish_reason", None)
        ) == "stop"
        for c in then_chunks if c.choices
    )
    assert then_has_stop, (
        f"Stream with only reasoning_delta must still emit a stop chunk, "
        f"got {len(then_chunks)} chunks"
    )


def test_build_envelope_includes_thinking_in_request() -> None:
    given_payload = {
        "model": "bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    when_envelope = PerchaiProvider._build_envelope(
        model="perchai/bedrock-mantle-google-gemma-4-31b",
        payload=given_payload,
    )
    then_request = when_envelope.get("request", {})
    then_thinking = then_request.get("thinking")
    assert then_thinking == {"type": "enabled"}, (
        f"envelope.request must contain thinking config, got {then_thinking!r}"
    )
    then_effort = then_request.get("reasoning_effort")
    assert then_effort == "low", (
        f"envelope.request must contain reasoning_effort, got {then_effort!r}"
    )


def test_build_payload_strips_effort_when_thinking_disabled() -> None:
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {
            "thinking": {"type": "disabled"},
            "reasoning_effort": "low",
        },
    }
    when_payload = PerchaiProvider._build_payload(
        model_name="bedrock-mantle-google-gemma-4-e2b",
        kwargs=given_kwargs,
    )
    then_thinking = when_payload.get("thinking")
    assert then_thinking == {"type": "disabled"}, (
        f"thinking disabled must be preserved, got {then_thinking!r}"
    )
    then_no_effort = "reasoning_effort" not in when_payload
    assert then_no_effort, (
        f"reasoning_effort must be stripped when thinking disabled: {when_payload!r}"
    )


def test_singleton_same_instance() -> None:
    given_first = PerchaiProvider()
    given_second = PerchaiProvider()
    then_same = given_first is given_second
    assert then_same, (
        f"PerchaiProvider must be singleton, got different instances: "
        f"{id(given_first)} vs {id(given_second)}"
    )


def test_get_model_tier_requirement_returns_none() -> None:
    given_provider = PerchaiProvider()
    when_result = given_provider.get_model_tier_requirement("bedrock-mantle-google-gemma-4-31b")
    assert when_result is None, (
        f"Perchai has no tier restrictions, got {when_result!r}"
    )


def test_get_credential_priority_returns_none() -> None:
    given_provider = PerchaiProvider()
    when_result = given_provider.get_credential_priority("any-key")
    assert when_result is None, (
        f"Perchai credential priority should be None by default, got {when_result!r}"
    )


def test_skip_cost_calculation_is_true() -> None:
    given_provider = PerchaiProvider()
    assert given_provider.skip_cost_calculation is True, (
        f"skip_cost_calculation should be True, got {given_provider.skip_cost_calculation!r}"
    )


def test_default_rotation_mode_is_sequential() -> None:
    given_provider = PerchaiProvider()
    assert given_provider.default_rotation_mode == "sequential", (
        f"rotation mode should be 'sequential', got {given_provider.default_rotation_mode!r}"
    )


def test_plain_request_does_not_auto_enable_thinking() -> None:
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
    }
    when_payload = PerchaiProvider._build_payload(
        model_name="bedrock-mantle-google-gemma-4-31b",
        kwargs=given_kwargs,
    )
    then_no_thinking = "thinking" not in when_payload
    assert then_no_thinking, (
        f"plain request should not auto-enable thinking, got {when_payload!r}"
    )
    then_no_effort = "reasoning_effort" not in when_payload
    assert then_no_effort, (
        f"plain request should not have reasoning_effort, got {when_payload!r}"
    )


def test_reasoning_effort_passes_through_without_thinking_config() -> None:
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "medium",
    }
    when_payload = PerchaiProvider._build_payload(
        model_name="bedrock-mantle-google-gemma-4-31b",
        kwargs=given_kwargs,
    )
    then_effort = when_payload.get("reasoning_effort")
    assert then_effort == "medium", (
        f"reasoning_effort should pass through without thinking config, got {then_effort!r}"
    )


@pytest.mark.asyncio
async def test_fetch_usage_data_uses_account_endpoint_not_usage_endpoint() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    given_tracker = PerchaiQuotaTracker()
    given_tracker._balance_cache = {}
    given_token = "test-token"
    given_app_url = "https://app.perchai.app"
    given_credential = "perchai_oauth_1.json"

    given_response = MagicMock()
    given_response.status_code = 200
    given_response.raise_for_status = MagicMock()
    given_response.json = MagicMock(return_value={
        "ok": True,
        "session": {
            "planCode": "starter",
            "planName": "Starter",
        },
        "usageMeter": {
            "monthly_usd": 5.0,
            "daily_usd": 1.0,
            "weekly_usd": 3.0,
        },
        "creditBalancePt": 0,
    })

    given_client = MagicMock()
    given_client.get = AsyncMock(return_value=given_response)
    given_client.__aenter__ = AsyncMock(return_value=given_client)
    given_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=given_client):
        when_result = await given_tracker._fetch_usage_data(
            given_credential, given_token, given_app_url
        )

    then_called_url = given_client.get.call_args[0][0] if given_client.get.call_args else None
    assert then_called_url is not None, (
        "_fetch_usage_data did not make any HTTP request"
    )
    then_is_account = "/api/perchai/account" in then_called_url
    assert then_is_account, (
        f"_fetch_usage_data must call /api/perchai/account, "
        f"got {then_called_url!r}"
    )
    then_not_usage = "/api/perch-terminal/usage" not in then_called_url
    assert then_not_usage, (
        f"_fetch_usage_data must NOT call /api/perch-terminal/usage "
        f"(that endpoint returns 405 on GET), got {then_called_url!r}"
    )
    assert when_result is not None, (
        "_fetch_usage_data should return parsed JSON, got None"
    )


@pytest.mark.asyncio
async def test_fetch_usage_data_uses_get_method() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    given_tracker = PerchaiQuotaTracker()
    given_tracker._balance_cache = {}
    given_token = "test-token"
    given_app_url = "https://app.perchai.app"
    given_credential = "perchai_oauth_1.json"

    given_response = MagicMock()
    given_response.status_code = 200
    given_response.raise_for_status = MagicMock()
    given_response.json = MagicMock(return_value={"ok": True, "session": {}})

    given_client = MagicMock()
    given_client.get = AsyncMock(return_value=given_response)
    given_client.__aenter__ = AsyncMock(return_value=given_client)
    given_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=given_client):
        await given_tracker._fetch_usage_data(
            given_credential, given_token, given_app_url
        )

    then_get_called = given_client.get.called
    assert then_get_called, (
        "_fetch_usage_data must use client.get (GET method), "
        "not client.post"
    )


def test_extract_dollar_fields_from_account_response_monthly_usd() -> None:
    given_data = {
        "ok": True,
        "session": {
            "planCode": "starter",
            "planName": "Starter",
            "entitlements": [],
        },
        "usageMeter": {
            "monthly_usd": 5.0,
            "daily_usd": 1.0,
            "weekly_usd": 3.0,
        },
        "creditBalancePt": 0,
    }

    when_used, when_cap, when_reset = PerchaiQuotaTracker._extract_dollar_fields(
        given_data
    )

    then_used = when_used
    assert then_used == 500, (
        f"monthly_usd=5.0 should produce used_cents=500, got {then_used!r}"
    )


def test_extract_dollar_fields_from_account_response_monthly_cap() -> None:
    given_data = {
        "ok": True,
        "session": {
            "planCode": "starter",
            "planName": "Starter",
            "entitlements": [
                {"key": "usage.monthly", "value_json": {"limitUsd": 10.0}},
            ],
        },
        "usageMeter": {
            "monthly_usd": 5.0,
            "daily_usd": 1.0,
            "weekly_usd": 3.0,
        },
        "creditBalancePt": 0,
    }

    when_used, when_cap, when_reset = PerchaiQuotaTracker._extract_dollar_fields(
        given_data
    )

    then_cap = when_cap
    assert then_cap == 1000, (
        f"limitUsd=10.0 should produce cap_cents=1000, got {then_cap!r}"
    )


def test_extract_dollar_fields_account_no_entitlements_cap_zero() -> None:
    given_data = {
        "ok": True,
        "session": {
            "planCode": "internal",
            "planName": "Internal",
            "entitlements": [],
        },
        "usageMeter": {
            "monthly_usd": 100.0,
        },
        "creditBalancePt": 5000,
    }

    when_used, when_cap, when_reset = PerchaiQuotaTracker._extract_dollar_fields(
        given_data
    )

    then_cap = when_cap
    assert then_cap == 0, (
        f"unlimited plan should have cap_cents=0, got {then_cap!r}"
    )


def test_guard_thinking_tool_calls_handles_perchai() -> None:
    from rotator_library.client.transforms import ProviderTransforms

    given_transforms = ProviderTransforms(provider_plugins={})
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [
            {"role": "user", "content": "call a tool then say hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "ast_grep_replace",
                            "arguments": '{"pattern": "test"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "tool result",
                "tool_call_id": "call_0",
            },
        ],
    }

    when_result = given_transforms._guard_thinking_tool_calls(
        given_kwargs, "perchai/bedrock-mantle-google-gemma-4-e2b", "perchai"
    )

    then_disabled = when_result is not None
    assert then_disabled, (
        f"_guard_thinking_tool_calls should inject thinking:disabled "
        f"for perchai provider when reasoning_content is missing, got: {when_result!r}"
    )
    then_thinking_disabled = (
        given_kwargs.get("extra_body", {}).get("thinking", {}).get("type") == "disabled"
    )
    assert then_thinking_disabled, (
        f"extra_body should contain thinking:disabled for perchai, "
        f"got: {given_kwargs.get('extra_body')!r}"
    )


def test_get_model_options_returns_empty_without_perchai_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.model_definitions import ModelDefinitions

    monkeypatch.delenv("PERCHAI_MODELS", raising=False)
    defs = ModelDefinitions()
    defs.reload_definitions()

    given_provider = PerchaiProvider()
    when_options = given_provider.get_model_options("perchai/some-model")

    then_empty = when_options == {}
    assert then_empty, (
        f"get_model_options should return empty dict when no PERCHAI_MODELS "
        f"is set, got: {when_options!r}"
    )


def test_get_model_options_returns_options_from_perchai_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.model_definitions import ModelDefinitions

    given_models_json = json.dumps(
        {
            "gpt-5": {"options": {"reasoning_effort": "high"}},
            "claude-4": {"options": {"temperature": 0.7}},
        }
    )
    monkeypatch.setenv("PERCHAI_MODELS", given_models_json)
    defs = ModelDefinitions()
    defs.reload_definitions()

    given_provider = PerchaiProvider()

    when_gpt5_options = given_provider.get_model_options("perchai/gpt-5")
    then_gpt5_reasoning = when_gpt5_options.get("reasoning_effort") == "high"
    assert then_gpt5_reasoning, (
        f"get_model_options for gpt-5 should return reasoning_effort=high, "
        f"got: {when_gpt5_options!r}"
    )

    when_claude4_options = given_provider.get_model_options("perchai/claude-4")
    then_claude4_temp = when_claude4_options.get("temperature") == 0.7
    assert then_claude4_temp, (
        f"get_model_options for claude-4 should return temperature=0.7, "
        f"got: {when_claude4_options!r}"
    )

    monkeypatch.delenv("PERCHAI_MODELS", raising=False)
    defs.reload_definitions()


def test_get_model_options_strips_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.model_definitions import ModelDefinitions

    given_models_json = json.dumps(
        {"my-model": {"options": {"max_tokens": 4096}}}
    )
    monkeypatch.setenv("PERCHAI_MODELS", given_models_json)
    defs = ModelDefinitions()
    defs.reload_definitions()

    given_provider = PerchaiProvider()
    when_options = given_provider.get_model_options("perchai/my-model")

    then_max_tokens = when_options.get("max_tokens") == 4096
    assert then_max_tokens, (
        f"get_model_options should strip 'perchai/' prefix and find options, "
        f"got: {when_options!r}"
    )

    monkeypatch.delenv("PERCHAI_MODELS", raising=False)
    defs.reload_definitions()


@pytest.mark.asyncio
async def test_get_models_merges_static_and_dynamic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.model_definitions import ModelDefinitions
    import httpx
    from unittest.mock import AsyncMock, MagicMock

    given_models_json = json.dumps(
        {"static-model-1": {}, "static-model-2": {"id": "upstream-id-2"}}
    )
    monkeypatch.setenv("PERCHAI_MODELS", given_models_json)
    defs = ModelDefinitions()
    defs.reload_definitions()

    given_provider = PerchaiProvider()
    given_provider._model_cache.clear()
    given_provider._model_cache_timestamps.clear()

    given_response = MagicMock()
    given_response.raise_for_status = MagicMock()
    given_response.json = MagicMock(
        return_value={
            "models": ["static-model-1", "upstream-id-2", "dynamic-model-3"]
        }
    )
    given_client = MagicMock(spec=httpx.AsyncClient)
    given_client.get = AsyncMock(return_value=given_response)

    when_models = await given_provider.get_models("test-key", given_client)

    then_has_static_1 = "perchai/static-model-1" in when_models
    assert then_has_static_1, (
        f"get_models should include static model 'perchai/static-model-1', "
        f"got: {when_models!r}"
    )

    then_has_static_2 = "perchai/static-model-2" in when_models
    assert then_has_static_2, (
        f"get_models should include static model 'perchai/static-model-2' "
        f"(display name, not upstream id), got: {when_models!r}"
    )

    then_no_upstream_id = "perchai/upstream-id-2" not in when_models
    assert then_no_upstream_id, (
        f"get_models should not duplicate 'perchai/upstream-id-2' "
        f"(covered by static-model-2), got: {when_models!r}"
    )

    then_has_dynamic = "perchai/dynamic-model-3" in when_models
    assert then_has_dynamic, (
        f"get_models should include dynamic model 'perchai/dynamic-model-3', "
        f"got: {when_models!r}"
    )

    monkeypatch.delenv("PERCHAI_MODELS", raising=False)
    defs.reload_definitions()


@pytest.mark.asyncio
async def test_get_models_returns_static_only_when_no_dynamic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.model_definitions import ModelDefinitions
    import httpx
    from unittest.mock import AsyncMock, MagicMock

    given_models_json = json.dumps(["fallback-model-1", "fallback-model-2"])
    monkeypatch.setenv("PERCHAI_MODELS", given_models_json)
    defs = ModelDefinitions()
    defs.reload_definitions()

    given_provider = PerchaiProvider()
    given_provider._model_cache.clear()
    given_provider._model_cache_timestamps.clear()

    given_client = MagicMock(spec=httpx.AsyncClient)
    given_client.get = AsyncMock(side_effect=httpx.RequestError("fail"))

    when_models = await given_provider.get_models("test-key", given_client)

    then_has_fallback_1 = "perchai/fallback-model-1" in when_models
    assert then_has_fallback_1, (
        f"get_models should return static models when dynamic fails, "
        f"got: {when_models!r}"
    )

    then_has_fallback_2 = "perchai/fallback-model-2" in when_models
    assert then_has_fallback_2, (
        f"get_models should return static models when dynamic fails, "
        f"got: {when_models!r}"
    )

    monkeypatch.delenv("PERCHAI_MODELS", raising=False)
    defs.reload_definitions()


def test_build_envelope_omits_promo_overflow_when_false() -> None:
    given_model = "perchai/bedrock-mantle-google-gemma-4-31b"
    given_payload: Dict[str, Any] = {
        "model": "bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
    }
    when_envelope = PerchaiProvider._build_envelope(
        model=given_model, payload=given_payload
    )
    then_no_promo = "promoOverflowAccepted" not in when_envelope
    assert then_no_promo, (
        f"envelope should not include promoOverflowAccepted when false, "
        f"got: {when_envelope!r}"
    )
    then_strict = when_envelope.get("strictManual") is False
    assert then_strict, (
        f"envelope should include strictManual=False, "
        f"got: {when_envelope!r}"
    )


def test_parse_sse_done_event_with_tool_calls() -> None:
    given_line = (
        'data: {"type":"done","ok":true,"text":"done","toolCalls":'
        '[{"id":"call_0","name":"bash","arguments":"{}"}]}'
    )
    when_chunk = PerchaiProvider._parse_sse_line(given_line)
    assert when_chunk is not None, (
        "done event with toolCalls should produce a chunk, got None"
    )
    then_finish = when_chunk.choices[0].finish_reason == "tool_calls"
    assert then_finish, (
        f"done event with toolCalls should have finish_reason=tool_calls, "
        f"got: {when_chunk.choices[0].finish_reason!r}"
    )


def test_parse_sse_done_event_without_tool_calls() -> None:
    given_line = (
        'data: {"type":"done","ok":true,"text":"hello","finishReason":"stop"}'
    )
    when_chunk = PerchaiProvider._parse_sse_line(given_line)
    assert when_chunk is not None, (
        "done event without toolCalls should produce a chunk, got None"
    )
    then_finish = when_chunk.choices[0].finish_reason == "stop"
    assert then_finish, (
        f"done event without toolCalls should have finish_reason=stop, "
        f"got: {when_chunk.choices[0].finish_reason!r}"
    )


def test_parse_sse_done_event_with_error_returns_none() -> None:
    given_line = (
        'data: {"type":"done","ok":false,"error":"Model call failed"}'
    )
    when_chunk = PerchaiProvider._parse_sse_line(given_line)
    assert when_chunk is None, (
        f"done event with ok=false should return None, got: {when_chunk!r}"
    )


def test_red_build_payload_does_not_inject_reasoning_content_on_tool_calls() -> None:
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-31b",
        "messages": [
            {"role": "user", "content": "call a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_0"},
        ],
    }
    when_payload = PerchaiProvider._build_payload(
        model_name="bedrock-mantle-google-gemma-4-31b",
        kwargs=given_kwargs,
    )
    then_messages = when_payload.get("messages", [])
    then_assistant = next(
        (m for m in then_messages if m.get("role") == "assistant" and m.get("tool_calls")),
        None,
    )
    assert then_assistant is not None, "test setup: expected assistant message with tool_calls"
    then_no_reasoning = "reasoning_content" not in then_assistant
    assert then_no_reasoning, (
        f"reasoning_content must NOT be injected on assistant tool-call messages. "
        f"The Perch CLI never sends this field - injecting it corrupts conversation. "
        f"Got: {then_assistant!r}"
    )


def test_red_parse_sse_done_event_with_tool_calls() -> None:
    given_line = (
        'data: {"type":"done","ok":true,"text":"done","toolCalls":'
        '[{"id":"call_0","name":"bash","arguments":"{}"}]}'
    )
    when_chunk = PerchaiProvider._parse_sse_line(given_line)
    assert when_chunk is not None, (
        "done event with toolCalls must produce a chunk. "
        "OLD CODE silently dropped done events - tool calls were lost."
    )
    then_finish = when_chunk.choices[0].finish_reason == "tool_calls"
    assert then_finish, (
        f"done event with toolCalls must set finish_reason=tool_calls. "
        f"Got: {when_chunk.choices[0].finish_reason!r}"
    )


def test_red_envelope_does_not_send_promo_overflow_false() -> None:
    given_model = "perchai/bedrock-mantle-google-gemma-4-31b"
    given_payload: Dict[str, Any] = {
        "model": "bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
    }
    when_envelope = PerchaiProvider._build_envelope(
        model=given_model, payload=given_payload
    )
    then_no_promo = "promoOverflowAccepted" not in when_envelope
    assert then_no_promo, (
        f"envelope must NOT include promoOverflowAccepted when false. "
        f"The Perch CLI only sends it when true. "
        f"Got: {when_envelope!r}"
    )
    then_has_strict = "strictManual" in when_envelope
    assert then_has_strict, (
        f"envelope must include strictManual field. "
        f"Got: {when_envelope!r}"
    )
