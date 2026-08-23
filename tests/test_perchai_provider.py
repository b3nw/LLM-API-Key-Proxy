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


# =========================================================================
# TEST CASES
# =========================================================================


def test_given_perchai_provider_when_loaded_then_perchai_in_plugins() -> None:
    """Given the providers package is imported, when the plugin auto-discovery
    runs, then the ``perchai`` plugin must be registered in PROVIDER_PLUGINS.
    """
    given_plugins = PROVIDER_PLUGINS
    when_checked = "perchai"
    then_asserted = when_checked in given_plugins
    assert then_asserted, (
        f"perchai plugin missing from PROVIDER_PLUGINS: "
        f"{sorted(given_plugins.keys())!r}"
    )


def test_given_provider_class_when_instantiated_then_has_custom_logic_true() -> None:
    """Given the PerchaiProvider class, when instantiated (singleton), then
    ``has_custom_logic()`` must return ``True`` so the rotator knows to
    route through our custom acompletion() implementation.
    """
    given_provider = PerchaiProvider()
    when_called = given_provider.has_custom_logic()
    then_returned = when_called is True
    assert then_returned, (
        f"PerchaiProvider.has_custom_logic() returned {when_called!r}; "
        f"expected True"
    )


def test_given_class_when_inspected_then_mro_includes_quota_tracker() -> None:
    """Given the PerchaiProvider class, when its MRO is inspected, then
    PerchaiQuotaTracker must appear before ProviderInterface so the
    mixin's __init__ side effects run first.
    """
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
def test_given_all_error_codes_when_parsed_then_correct_category(
    error_code: str,
    expected: Optional[Dict[str, Any]],
) -> None:
    """Given any of the 10 upstream perchai error codes, when
    ``parse_quota_error`` is called with a body containing that
    ``errorCode``, then the returned dict (or ``None``) must match the
    documented classification.
    """
    given_body = json.dumps({"errorCode": error_code, "error": "synthetic test body"})
    given_error = Exception("synthetic")
    when_parsed = PerchaiProvider.parse_quota_error(given_error, given_body)
    then_returned = when_parsed
    assert then_returned == expected, (
        f"errorCode={error_code!r}: expected {expected!r}, got {then_returned!r}"
    )


def test_given_429_status_when_parsed_then_rate_limit() -> None:
    """Given an Exception whose ``args[0]`` is the string ``"429"`` (no
    upstream body), when ``parse_quota_error`` is called, then the
    HTTP-status fallback must yield ``{"reason": "rate_limit",
    "retry_after": 3600}``.
    """
    given_error = Exception("429")
    given_body = ""
    when_parsed = PerchaiProvider.parse_quota_error(given_error, given_body)
    then_returned = when_parsed
    assert then_returned == {"reason": "rate_limit", "retry_after": 3600}, (
        f"HTTP 429 fallback should produce rate_limit dict, got {then_returned!r}"
    )


def test_given_malformed_body_when_parsed_then_returns_none() -> None:
    """Given a syntactically-broken JSON body, when ``parse_quota_error``
    is called, then the defensive parser must return ``None`` rather
    than raise.
    """
    given_error = Exception("synthetic")
    given_body = "{this is not valid json"
    when_parsed = PerchaiProvider.parse_quota_error(given_error, given_body)
    then_returned = when_parsed
    assert then_returned is None, (
        f"Malformed body should parse to None, got {then_returned!r}"
    )


@pytest.mark.asyncio
async def test_given_empty_messages_when_acompletion_then_raises_valueerror() -> None:
    """Given acompletion() is called with empty messages, when the provider
    validates input, then it must raise ``ValueError`` BEFORE attempting
    any HTTP I/O.
    """
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


def test_given_malformed_sse_line_when_parsed_then_returns_none() -> None:
    """Given a malformed JSON SSE ``data:`` line, when ``_parse_sse_line``
    runs, then the defensive parser must return ``None`` (not raise).
    """
    given_line = "data: {not valid json"
    given_model = "perchai/test-model"
    when_parsed = PerchaiProvider._parse_sse_line(given_line, given_model)
    then_returned = when_parsed
    assert then_returned is None, (
        f"Malformed SSE line should parse to None, got {then_returned!r}"
    )


def test_given_unknown_event_type_when_parsed_then_skipped() -> None:
    """Given an SSE line whose ``type`` field is not in the recognized set,
    when ``_parse_sse_line`` runs, then it must return ``None`` so the
    stream loop continues.
    """
    given_line = 'data: {"type":"future_event_type","payload":"ignored"}'
    given_model = "perchai/test-model"
    when_parsed = PerchaiProvider._parse_sse_line(given_line, given_model)
    then_returned = when_parsed
    assert then_returned is None, (
        f"Unknown event type should parse to None, got {then_returned!r}"
    )


def test_given_text_delta_when_parsed_then_content_chunk() -> None:
    """Given an ``answer_delta`` SSE event (the raw wire format from
    ``/api/perch-terminal/model-call``), when ``_parse_sse_line`` runs,
    then the returned ``ModelResponseStream`` must carry the text in
    ``choices[0].delta.content``.
    """
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


def test_given_reasoning_delta_when_parsed_then_reasoning_chunk() -> None:
    """Given a ``reasoning_delta`` SSE event, when ``_parse_sse_line``
    runs, then the returned ``ModelResponseStream`` must carry the text
    in ``choices[0].delta.reasoning_content``.
    """
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


async def test_given_perchai_stream_without_tool_id_when_consumed_then_synthetic_id_present() -> None:
    """Given a Perchai streaming response that emits ``tool_call_delta`` events
    without ``id`` or ``name`` fields (the actual wire format Perchai sends),
    when the full streaming pipeline consumes the stream, then every tool_call
    chunk must have a string ``id`` and ``function.name`` so @ai-sdk clients
    don't crash with "Expected 'id' to be a string" or
    "Expected 'function.name' to be a string".

    This is an integration test that exercises the full streaming path:
    httpx response -> aiter_lines -> _parse_sse_line -> chunk emission.
    """
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


def test_given_tool_call_delta_without_id_when_parsed_then_synthetic_id_emitted() -> None:
    """Given a ``tool_call_delta`` SSE event without ``id`` field (as Perchai
    currently sends), when ``_parse_sse_line`` runs with tracking maps, then
    the emitted chunk must contain a synthetic ``id`` so @ai-sdk clients
    don't crash with "Expected 'id' to be a string".
    """
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


def test_given_tool_call_delta_without_name_when_parsed_then_synthetic_name_emitted() -> None:
    """Given a ``tool_call_delta`` SSE event without ``name`` field and no
    request tool names, when ``_parse_sse_line`` runs with tracking maps,
    then the emitted chunk must contain a synthetic ``function.name`` so
    @ai-sdk clients don't crash with "Expected 'function.name' to be a string".
    """
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


def test_given_tool_call_delta_without_name_when_request_tools_provided_then_real_name_used() -> None:
    """Given a ``tool_call_delta`` SSE event without ``name`` field, when
    ``_parse_sse_line`` runs with ``request_tool_names`` from the original
    request's tools definition, then the emitted chunk must use the real
    tool name (e.g. ``read``) instead of a synthetic ``function_0`` so
    downstream clients call the correct tool.
    """
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


def test_given_tool_call_delta_without_name_when_index_mismatch_then_falls_back_to_synthetic() -> None:
    """Given a ``tool_call_delta`` SSE event with index 2 but only 2 tools
    in the request (indices 0, 1), when ``_parse_sse_line`` runs, then it
    must fall back to the synthetic ``function_2`` name since the index
    is out of range of the request tools.
    """
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


def test_given_multiple_tool_call_deltas_when_parsed_then_same_synthetic_id_reused() -> None:
    """Given multiple ``tool_call_delta`` events for same index without id/name,
    when ``_parse_sse_line`` runs with tracking maps, then all chunks must
    use the same synthetic id/name (not generate new ones per chunk).
    """
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


def test_given_registration_when_checked_then_perchai_in_oauth_dirs() -> None:
    """Given the credential manager module is loaded, when DEFAULT_OAUTH_DIRS
    is inspected, then ``perchai`` must be present pointing at
    ``~/.perch`` (the actual CLI directory, not ``~/.perchai``).
    """
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


def test_given_registration_when_checked_then_perchai_in_env_oauth() -> None:
    """Given ENV_OAUTH_PROVIDERS is populated, when checked, then
    ``perchai`` must map to the ``PERCHAI`` env-var prefix for stateless
    deployments (PERCHAI_ACCESS_TOKEN / PERCHAI_N_ACCESS_TOKEN).
    """
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


def test_given_registration_when_checked_then_perchai_in_provider_factory() -> None:
    """Given the provider factory module is loaded, when PROVIDER_MAP is
    inspected, then ``perchai`` must be present and resolve to a real
    auth class (PerchaiAuthBase) - not ``None`` or a placeholder.
    """
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


def test_given_provider_config_when_checked_then_perchai_listed() -> None:
    """Given the provider_config module is loaded, when LITELLM_PROVIDERS
    is inspected, then ``perchai`` must be present with a real category
    dict (not missing, not ``None``).
    """
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


def test_given_provider_urls_when_checked_then_perchai_url() -> None:
    """Given the proxy_app provider_urls module is loaded, when
    PROVIDER_URL_MAP is inspected, then ``perchai`` must be present
    pointing at the upstream app URL.
    """
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


def test_given_get_background_job_config_when_called_then_returns_valid_dict() -> None:
    """Given the background-job config static method is called, when
    invoked on either PerchaiProvider or PerchaiQuotaTracker, then the
    returned dict must contain ``interval`` and ``name`` keys (the
    fields the executor scheduler reads).
    """
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


def test_given_model_quota_groups_when_checked_then_monthly_group() -> None:
    """Given PerchaiQuotaTracker.model_quota_groups, when inspected, then
    the ``monthly($)`` quota group must be present (TUI surfaces this
    under the dollar-balance view).
    """
    given_groups = PerchaiQuotaTracker.model_quota_groups
    when_checked = "monthly($)"
    then_present = when_checked in given_groups
    assert then_present, (
        f"monthly($) missing from model_quota_groups: {sorted(given_groups.keys())!r}"
    )


@pytest.mark.asyncio
async def test_given_run_background_job_with_invalid_token_when_called_then_no_crash() -> None:
    """Given ``run_background_job`` is called with an invalid credential
    path and no usable session token, when it executes, then it must
    swallow the resolution failure and return without raising.
    """
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
async def test_given_expired_token_when_non_stream_chat_then_refreshes_and_retries(
    tmp_path: Path,
) -> None:
    """Given a credential file containing an expired access token, when
    non-streaming ``acompletion`` is called with that file path as
    ``credential_identifier``, then the provider must resolve the file to
    the token, receive a 401, refresh the token via
    ``PerchaiAuthBase.refresh_on_401`` and retry, returning a
    ``ModelResponse`` with non-empty content.
    """
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
async def test_given_expired_token_when_stream_chat_then_refreshes_and_retries(
    tmp_path: Path,
) -> None:
    """Given a credential file containing an expired access token, when
    streaming ``acompletion`` is called with that file path as
    ``credential_identifier``, then the provider must resolve the file to
    the token, receive a 401, refresh the token via
    ``PerchaiAuthBase.refresh_on_401`` and retry the stream, yielding at
    least one content chunk and a final chunk with
    ``finish_reason="stop"``.
    """
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


def test_given_credential_file_path_when_resolved_then_returns_access_token() -> None:
    """Given a credential_identifier that is a file path to a valid session
    JSON, when ``_resolve_credential_token`` is called, then it must read the
    file and return the ``accessToken`` value, NOT the file path itself.
    """
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


def test_given_env_virtual_path_when_resolved_then_returns_env_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a credential_identifier that is an ``env://perchai/1`` virtual
    path, when ``load_session`` is called, then it must read the
    ``PERCHAI_1_ACCESS_TOKEN`` env var and return it as the accessToken.
    """
    from rotator_library.providers.perchai_auth_base import PerchaiAuthBase

    monkeypatch.setenv("PERCHAI_1_ACCESS_TOKEN", "test-env-access-token")
    monkeypatch.setenv("PERCHAI_1_REFRESH_TOKEN", "test-env-refresh-token")

    given_auth = PerchaiAuthBase(credential_path="env://perchai/1")
    given_session = given_auth.load_session()
    then_token = given_session.get("accessToken")

    assert then_token == "test-env-access-token", (
        f"Expected accessToken from env var, got {then_token!r}"
    )


def test_given_empty_credential_identifier_when_resolved_then_falls_back_to_default() -> None:
    """Given an empty credential_identifier, when ``_resolve_credential_token``
    is called, then it must fall back to the default session resolution
    (``_resolve_session_token``), NOT use an empty string as the token.
    """
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
async def test_given_option_id_when_probed_then_routes_to_real_upstream(
    option_id: str,
) -> None:
    """Given a perchai option ID from the docs, when probed against the real
    API with ``manualModelOptionId``, then the response's ``model``/``provider``
    must NOT be the default-fallback pair (``moonshotai.kimi-k2.5`` /
    ``bedrock_mantle``) - otherwise the proxy would silently serve the
    default model while the caller thinks they got the requested one.
    """
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


# =========================================================================
# MODULE-LEVEL SANITY CHECKS (cheap, always run when module is collected)
# =========================================================================


def test_given_module_constants_when_imported_then_paths_nonempty() -> None:
    """Given the module-level constants are imported, when their values
    are checked, then the path constants must be non-empty strings so
    callers can build URLs without needing to re-discover them.
    """
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


# =========================================================================
# Reasoning / thinking normalization tests (RED phase)
# =========================================================================


def test_given_perchai_provider_when_checked_then_has_transform_request_hook() -> None:
    """Given the PerchaiProvider class, when checked for the transform_request
    hook, then it must exist so the proxy can apply thinking normalization."""
    assert hasattr(PerchaiProvider, "transform_request"), (
        "PerchaiProvider must implement transform_request for thinking normalization"
    )


def test_given_thinking_disabled_when_transform_request_then_thinking_stripped() -> None:
    """Given a request with extra_body.thinking set to disabled, when
    transform_request runs, then reasoning_content must be stripped from
    assistant messages in the conversation (perchai does not require it
    when thinking is off) and the thinking config must be preserved."""
    from unittest.mock import MagicMock
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


def test_given_streaming_reasoning_delta_when_thinking_disabled_then_reasoning_stripped() -> None:
    """Given a perchai stream that emits reasoning_delta events even when
    thinking is disabled (gemma models do this), when _parse_sse_line
    processes a reasoning_delta, then it must return None to suppress
    the reasoning_content chunk - downstream clients like Opencode
    should not see reasoning_content when thinking is off."""
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


def test_given_streaming_reasoning_delta_when_thinking_enabled_then_reasoning_emitted() -> None:
    """Given a perchai stream that emits reasoning_delta events when
    thinking is enabled, when _parse_sse_line processes a reasoning_delta,
    then it must return a chunk with reasoning_content so downstream
    clients can display the reasoning."""
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


def test_given_non_stream_response_when_thinking_disabled_then_reasoning_stripped() -> None:
    """Given a non-streaming perchai response that includes reasoning text
    when thinking was disabled in the request, when the response is parsed,
    then reasoning_content must NOT be included in the message - clients
    should not see reasoning when thinking is off."""
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


def test_given_non_stream_response_when_thinking_enabled_then_reasoning_preserved() -> None:
    """Given a non-streaming perchai response that includes reasoning text
    when thinking was enabled in the request, when the response is parsed,
    then reasoning_content must be included in the message."""
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


# =========================================================================
# Thinking config in payload tests (RED phase)
# =========================================================================


def test_given_thinking_enabled_when_build_payload_then_thinking_in_payload() -> None:
    """Given a perchai request with extra_body.thinking set to enabled and
    reasoning_effort=low (as sent by the proxy model options), when
    _build_payload runs, then the payload must include thinking and
    reasoning_effort at the top level - not nested inside extra_body -
    so the perchai upstream API receives them in the request body."""
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


def test_given_thinking_disabled_when_build_payload_then_thinking_disabled_in_payload() -> None:
    """Given a perchai request with extra_body.thinking set to disabled,
    when _build_payload runs, then the payload must include
    thinking: {type: disabled} at the top level and must NOT include
    reasoning_effort (it's meaningless when thinking is off)."""
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


def test_given_reasoning_effort_in_kwargs_when_build_payload_then_in_payload() -> None:
    """Given a perchai request where reasoning_effort is passed directly in
    kwargs (from the transforms apply step 3 model_options), when
    _build_payload runs, then reasoning_effort must be included in the
    payload so the upstream perchai API receives it."""
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
async def test_given_stream_only_reasoning_when_consumed_then_stop_chunk_emitted() -> None:
    """Given a perchai stream that emits only reasoning_delta events (no
    answer_delta) followed by a finishReason, when the stream is consumed,
    then at least one chunk with finish_reason='stop' must be emitted so
    downstream clients like Opencode see the turn as complete - not as a
    hung stream with no terminal event."""
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


def test_given_envelope_when_built_then_thinking_in_request() -> None:
    """Given a payload with thinking config at top level, when
    _build_envelope wraps it, then the envelope's request field must
    contain the thinking config so the perchai upstream API receives it."""
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


def test_given_thinking_disabled_with_effort_when_build_payload_then_effort_stripped() -> None:
    """Given a perchai request with extra_body.thinking set to disabled AND
    reasoning_effort present (which can happen when model options set both),
    when _build_payload runs, then reasoning_effort must be stripped from
    the payload - sending reasoning_effort with thinking disabled is
    contradictory and may confuse the upstream API."""
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
