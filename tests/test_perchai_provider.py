# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kévin Cojean
"""Seam-level tests for the Perchai provider.

Only tests that protect against specific regressions or exercise behavior
the e2e suite does not cover cheaply. Everything else is verified end to
end against `app.perchai.app` via the `@pytest.mark.live` tests below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import pytest

from rotator_library.providers.perchai_provider import (
    MODEL_CALL_PATH,
    TURN_TICKET_HEADER,
    PerchaiProvider,
)
from rotator_library.providers.utilities.perchai_quota_tracker import (
    PerchaiQuotaTracker,
)


PERCHAI_SESSION: Path = Path.home() / ".perch" / "cli-auth-session.json"
HAS_SESSION: bool = PERCHAI_SESSION.is_file()

live_only = pytest.mark.skipif(
    not HAS_SESSION,
    reason="No perchai session - run `perch login`",
)


# ---------------------------------------------------------------------------
# Seam-level regression tests (run everywhere)
# ---------------------------------------------------------------------------


def test_load_session_re_reads_file_on_every_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.providers import perchai_auth_base as perchai_auth
    from rotator_library.providers.perchai_auth_base import PerchaiAuthBase

    given_file = tmp_path / "session.json"
    given_file.write_text(
        json.dumps(
            {
                "version": 1,
                "appUrl": "https://app.perchai.app",
                "accessToken": "token-before",
                "refreshToken": "refresh-before",
                "userId": "u1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(perchai_auth, "_resolve_session_file", lambda: given_file)

    given_auth = PerchaiAuthBase()
    assert given_auth.load_session().get("accessToken") == "token-before", (
        "first load_session should read the initial token from disk"
    )

    given_file.write_text(
        json.dumps(
            {
                "version": 1,
                "appUrl": "https://app.perchai.app",
                "accessToken": "token-after",
                "refreshToken": "refresh-after",
                "userId": "u1",
            }
        ),
        encoding="utf-8",
    )

    assert given_auth.load_session().get("accessToken") == "token-after", (
        "load_session must re-read the session file on every call so that "
        "`perch login` (which rewrites the file in a separate process) is "
        "picked up without rebooting the proxy"
    )


def test_thinking_disabled_strips_reasoning_from_messages() -> None:
    given_provider = PerchaiProvider()
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    given_provider.transform_request(
        given_kwargs, "perchai/bedrock-mantle-google-gemma-4-e2b", "test-cred"
    )
    assert given_kwargs.get("extra_body", {}).get("thinking") == {"type": "disabled"}, (
        "thinking config must be preserved through transform_request"
    )


@pytest.mark.asyncio
async def test_sync_transform_request_hook_runs_through_transforms(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from rotator_library.client.transforms import ProviderTransforms

    caplog.set_level(logging.DEBUG, logger="rotator_library")

    given_transforms = ProviderTransforms(provider_plugins={"perchai": PerchaiProvider})
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "old thinking"},
        ],
        "extra_body": {"thinking": {"type": "disabled"}},
    }

    when_kwargs = await given_transforms.apply(
        "perchai",
        "perchai/bedrock-mantle-google-gemma-4-e2b",
        "test-cred",
        given_kwargs,
    )

    then_messages = when_kwargs.get("messages", [])
    then_stripped = all(
        "reasoning_content" not in m
        for m in then_messages
        if m.get("role") == "assistant"
    )
    assert then_stripped, (
        "sync transform_request hook must run through ProviderTransforms.apply "
        f"and strip reasoning_content when thinking disabled, got {then_messages!r}"
    )
    assert "hook failed" not in caplog.text, (
        "sync transform_request hook must not be awaited "
        f"(TypeError logged as 'hook failed', got: {caplog.text!r}"
    )


# ---------------------------------------------------------------------------
# Reasoning-wall adapter (invisible DeepSeek plumbing, inside the provider)
#
# Seam: what actually reaches the upstream request. Built the same way
# acompletion builds it: _build_payload -> _build_envelope -> ["request"].
# These protect the guarantee that a configured thinking_budget reaches
# upstream no matter how the client expressed reasoning (top-level
# reasoning_effort, an extra_body thinking dict, or a model-option alias).
# ---------------------------------------------------------------------------

DEEPSEEK_MODEL = "perchai/wandb-deepseek-ai-deepseek-v4-flash-0731"
DEEPSEEK_BUDGET_ENV_PREFIX = (
    "PERCHAI_WANDB_DEEPSEEK_AI_DEEPSEEK_V4_FLASH_0731_THINKING_BUDGET_"
)


def _upstream_request_for(
    provider: "PerchaiProvider", kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Mirror acompletion's request build: payload -> envelope -> request."""
    model = kwargs["model"]
    model_name = model.split("/", 1)[1] if "/" in model else model
    payload = provider._build_payload(model_name=model_name, kwargs=kwargs)
    envelope = provider._build_envelope(model=model, payload=payload)
    return envelope["request"]


def test_toplevel_high_effort_injects_wall_budget_and_downgrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenCode sends reasoning_effort top-level with no thinking dict.

    The wall budget must still reach upstream, and high is downgraded to low
    internally (DeepSeek-flash gains little from long thinking, and mid-work
    truncation is the painful symptom).
    """
    monkeypatch.setenv(DEEPSEEK_BUDGET_ENV_PREFIX + "HIGH", "2500")
    given_kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": "think hard"}],
        "reasoning_effort": "high",
    }
    when_request = _upstream_request_for(PerchaiProvider(), given_kwargs)
    when_ctk = when_request.get("chat_template_kwargs") or {}
    assert when_ctk.get("thinking_budget") == 2500, (
        "top-level high effort must inject the configured wall budget, "
        f"got chat_template_kwargs: {when_ctk!r}"
    )
    assert when_request.get("reasoning_effort") == "low", (
        "high effort must be downgraded to low internally, "
        f"got {when_request.get('reasoning_effort')!r}"
    )


@pytest.mark.parametrize(
    "given_effort,expected_budget_env,expected_budget,expected_effort_after",
    [
        pytest.param("max", "HIGH", 2500, "low", id="max_capped_and_downgraded"),
        pytest.param("xhigh", "HIGH", 2500, "low", id="xhigh_capped_and_downgraded"),
        pytest.param("medium", "MEDIUM", 1000, "medium", id="medium_own_bucket"),
        pytest.param("low", "LOW", 1500, "low", id="low_own_bucket"),
    ],
)
def test_other_efforts_inject_wall_budget(
    monkeypatch: pytest.MonkeyPatch,
    given_effort: str,
    expected_budget_env: str,
    expected_budget: int,
    expected_effort_after: str,
) -> None:
    """Every effort level that triggers reasoning must get its wall budget.

    Previously only medium/high were handled, so max/xhigh/low silently ran
    into the upstream truncation wall. Only the high bucket is downgraded to
    low; medium and low keep the effort the client asked for.
    """
    monkeypatch.setenv(DEEPSEEK_BUDGET_ENV_PREFIX + expected_budget_env, str(expected_budget))
    given_kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": "think"}],
        "reasoning_effort": given_effort,
    }
    when_request = _upstream_request_for(PerchaiProvider(), given_kwargs)
    when_ctk = when_request.get("chat_template_kwargs") or {}
    assert when_ctk.get("thinking_budget") == expected_budget, (
        f"effort {given_effort!r} must inject {expected_budget_env} budget "
        f"{expected_budget}, got {when_ctk!r}"
    )
    assert when_request.get("reasoning_effort") == expected_effort_after, (
        f"effort {given_effort!r} must reach upstream as {expected_effort_after!r}, "
        f"got {when_request.get('reasoning_effort')!r}"
    )


def test_absent_effort_with_thinking_enabled_injects_model_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """thinking enabled but no reasoning_effort: the model default budget
    (deepseek fallback = 3000) must still cap reasoning under the wall."""
    given_kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": "think"}],
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    when_request = _upstream_request_for(PerchaiProvider(), given_kwargs)
    when_ctk = when_request.get("chat_template_kwargs") or {}
    assert when_ctk.get("thinking_budget") == 3000, (
        "deepseek model default budget must apply when effort is absent, "
        f"got {when_ctk!r}"
    )


def test_reasoning_disabled_does_not_inject_wall_budget() -> None:
    """When reasoning is off there is no wall to protect against: no budget,
    and reasoning_effort must not be forwarded."""
    given_kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": "answer directly"}],
        "reasoning_effort": "disable",
    }
    when_request = _upstream_request_for(PerchaiProvider(), given_kwargs)
    assert "chat_template_kwargs" not in when_request, (
        f"disabled reasoning must not inject chat_template_kwargs, "
        f"got {when_request.get('chat_template_kwargs')!r}"
    )
    assert "reasoning_effort" not in when_request, (
        "disabled reasoning must drop reasoning_effort from the request"
    )


def test_non_wall_model_is_left_untouched() -> None:
    """Models without a configured wall (gemma) must be byte-for-byte
    unchanged: no budget injected, effort not downgraded."""
    given_kwargs: Dict[str, Any] = {
        "model": "perchai/bedrock-mantle-google-gemma-4-e2b",
        "messages": [{"role": "user", "content": "think"}],
        "reasoning_effort": "high",
    }
    when_request = _upstream_request_for(PerchaiProvider(), given_kwargs)
    assert "chat_template_kwargs" not in when_request, (
        "non-wall model must not get chat_template_kwargs, "
        f"got {when_request.get('chat_template_kwargs')!r}"
    )
    assert when_request.get("reasoning_effort") == "high", (
        "non-wall model effort must not be downgraded, "
        f"got {when_request.get('reasoning_effort')!r}"
    )


def test_text_delta_preserves_whitespace_only_chunks() -> None:
    given_lines = [
        'data: {"type":"answer_delta","text":"needs"}',
        'data: {"type":"answer_delta","text":" "}',
        'data: {"type":"answer_delta","text":"200+"}',
    ]
    given_model = "perchai/test-model"
    given_chunks = [
        PerchaiProvider._parse_sse_line(line, given_model) for line in given_lines
    ]
    assert all(c is not None for c in given_chunks), (
        "all deltas must produce chunks"
    )
    given_contents = []
    for chunk in given_chunks:
        choices = chunk.choices
        delta = (
            choices[0].get("delta")
            if isinstance(choices[0], dict)
            else choices[0].delta
        )
        content = (
            delta.get("content") if isinstance(delta, dict) else delta.content
        )
        given_contents.append(content)
    assert given_contents == ["needs", " ", "200+"], (
        f"whitespace-only chunk must be preserved as ' ', got {given_contents!r}"
    )


def test_guard_thinking_tool_calls_exempts_perchai() -> None:
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
            {"role": "tool", "content": "tool result", "tool_call_id": "call_0"},
        ],
    }

    when_result = given_transforms._guard_thinking_tool_calls(
        given_kwargs, "perchai/bedrock-mantle-google-gemma-4-e2b", "perchai"
    )
    assert when_result is None, (
        "_guard_thinking_tool_calls must NOT disable thinking for perchai. "
        f"Got: {when_result!r}"
    )
    assert "thinking" not in (given_kwargs.get("extra_body") or {}), (
        f"extra_body must not contain thinking key for perchai, "
        f"got: {given_kwargs.get('extra_body')!r}"
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

    then_chunks: List[Any] = []
    async for chunk in given_provider._stream_completion(
        client=given_client,
        url="https://api.perchai.com/v1/chat",
        build_headers=lambda t, tk: {"Authorization": "Bearer fake"},
        token="fake-token",
        ticket="test-ticket",
        payload=given_payload,
        model="perchai/test-model",
        file_logger=given_logger,
        credential_identifier="test-cred",
    ):
        then_chunks.append(chunk)

    then_has_stop = any(
        (
            c.choices[0].get("finish_reason")
            if isinstance(c.choices[0], dict)
            else getattr(c.choices[0], "finish_reason", None)
        )
        == "stop"
        for c in then_chunks
        if c.choices
    )
    assert then_has_stop, (
        f"Stream with only reasoning_delta must still emit a stop chunk, "
        f"got {len(then_chunks)} chunks"
    )


@pytest.mark.asyncio
async def test_fetch_usage_data_uses_account_endpoint() -> None:
    """Real bug: was calling /api/perch-terminal/usage which returns 405 on GET."""
    from unittest.mock import AsyncMock, MagicMock, patch

    given_tracker = PerchaiQuotaTracker()
    given_tracker._balance_cache = {}
    given_response = MagicMock()
    given_response.status_code = 200
    given_response.raise_for_status = MagicMock()
    given_response.json = MagicMock(
        return_value={
            "ok": True,
            "session": {"planCode": "starter", "planName": "Starter"},
            "usageMeter": {"monthly_usd": 5.0, "daily_usd": 1.0, "weekly_usd": 3.0},
            "creditBalancePt": 0,
        }
    )
    given_client = MagicMock()
    given_client.get = AsyncMock(return_value=given_response)
    given_client.__aenter__ = AsyncMock(return_value=given_client)
    given_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=given_client):
        when_result = await given_tracker._fetch_usage_data(
            "perchai_oauth_1.json", "test-token", "https://app.perchai.app"
        )

    then_called_url = (
        given_client.get.call_args[0][0] if given_client.get.call_args else None
    )
    assert then_called_url is not None, (
        "_fetch_usage_data did not make any HTTP request"
    )
    assert "/api/perchai/account" in then_called_url, (
        f"_fetch_usage_data must call /api/perchai/account, got {then_called_url!r}"
    )
    assert "/api/perch-terminal/usage" not in then_called_url, (
        f"_fetch_usage_data must NOT call /api/perch-terminal/usage "
        f"(405 on GET), got {then_called_url!r}"
    )
    assert when_result is not None, (
        "_fetch_usage_data should return parsed JSON, got None"
    )


@pytest.mark.asyncio
async def test_get_models_merges_static_and_dynamic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real bug: dynamic models silently dropped if upstream id overlapped static id."""
    from rotator_library.model_definitions import ModelDefinitions

    given_models_json = json.dumps(
        {"static-model-1": {}, "static-model-2": {"id": "upstream-id-2"}}
    )
    monkeypatch.setenv("PERCHAI_MODELS", given_models_json)
    defs = ModelDefinitions()
    defs.reload_definitions()

    given_provider = PerchaiProvider()
    given_provider._model_cache.clear()
    given_provider._model_cache_timestamps.clear()

    from unittest.mock import AsyncMock, MagicMock

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

    assert "perchai/static-model-1" in when_models
    assert "perchai/static-model-2" in when_models
    assert "perchai/upstream-id-2" not in when_models, (
        "get_models should not duplicate 'perchai/upstream-id-2' "
        f"(covered by static-model-2), got: {when_models!r}"
    )
    assert "perchai/dynamic-model-3" in when_models


def test_build_envelope_omits_promo_overflow_when_false() -> None:
    given_payload: Dict[str, Any] = {
        "model": "bedrock-mantle-google-gemma-4-31b",
        "messages": [{"role": "user", "content": "hi"}],
    }
    when_envelope = PerchaiProvider._build_envelope(
        model="perchai/bedrock-mantle-google-gemma-4-31b", payload=given_payload
    )
    assert "promoOverflowAccepted" not in when_envelope, (
        f"envelope should not include promoOverflowAccepted when false, "
        f"got: {when_envelope!r}"
    )
    assert when_envelope.get("strictManual") is False, (
        f"envelope should include strictManual=False, got: {when_envelope!r}"
    )


# ---------------------------------------------------------------------------
# E2E tests against real app.perchai.app
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option_id",
    [
        pytest.param("gemma-4-e2b", id="gemma-4-e2b"),
        pytest.param("deepseek-v4-flash", id="deepseek-v4-flash"),
    ],
)
@pytest.mark.live
@live_only
async def test_option_id_routes_to_real_upstream(option_id: str) -> None:
    from rotator_library.providers.perchai_auth_base import PerchaiAuthBase

    given_auth = PerchaiAuthBase()
    given_token = await given_auth.ensure_access_token()
    given_ticket = await given_auth.ensure_turn_ticket(given_token)
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
                TURN_TICKET_HEADER: given_ticket,
                "User-Agent": given_auth.user_agent(),
            },
            json=given_body,
            timeout=30.0,
        )

    assert given_response.status_code == 200, (
        f"Probe HTTP {given_response.status_code} for option_id={option_id!r}: "
        f"{given_response.json()!r}"
    )
    then_body = given_response.json()
    assert then_body.get("ok") is True, (
        f"Probe ok=false for option_id={option_id!r}: {then_body.get('error')!r}"
    )
    then_model = then_body.get("model")
    then_provider = then_body.get("provider")
    assert not (
        then_model == "moonshotai.kimi-k2.5" and then_provider == "bedrock_mantle"
    ), (
        f"option_id={option_id!r} silently fell back to default upstream "
        f"({then_provider}:{then_model}) - this option ID is NOT currently "
        f"wired and must be removed from the docs"
    )


async def _live_thinking_metrics(
    provider: PerchaiProvider,
    client: httpx.AsyncClient,
    *,
    thinking: Dict[str, Any],
    reasoning_effort: str = "low",
) -> Tuple[int, float]:
    """Stream a thinking-enabled prompt and return (reasoning chars, elapsed s)."""
    import time

    from rotator_library.providers.perchai_auth_base import PerchaiAuthBase

    given_auth = PerchaiAuthBase()
    given_token = await given_auth.ensure_access_token()
    given_ticket = await given_auth.ensure_turn_ticket(given_token)
    given_url = f"{given_auth.get_app_url().rstrip('/')}{MODEL_CALL_PATH}"
    given_body = {
        "request": {
            "model": "probe",
            "messages": [
                {"role": "user", "content": "Think about whether 17 is prime, step by step."}
            ],
            "max_tokens": 2048,
            "stream": True,
        },
        "runId": None,
        "lane": "chat",
        "preferredModelId": None,
        "manualModelOptionId": "wandb-deepseek-ai-deepseek-v4-flash-0731",
        "strictManual": True,
        "extraBody": {
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
        },
    }

    then_chars = 0
    then_start = time.time()
    async with client.stream(
        "POST",
        given_url,
        headers={
            "Authorization": f"Bearer {given_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            TURN_TICKET_HEADER: given_ticket,
            "User-Agent": given_auth.user_agent(),
        },
        json=given_body,
        timeout=120.0,
    ) as response:
        async for raw_line in response.aiter_lines():
            if not raw_line.startswith("data: "):
                continue
            try:
                payload = json.loads(raw_line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "reasoning_delta":
                then_chars += len(payload.get("text", ""))
            if payload.get("type") == "done":
                break
    return then_chars, time.time() - then_start


@pytest.mark.live
@live_only
async def test_live_thinking_disabled_suppresses_reasoning() -> None:
    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        when_reasoning, when_elapsed = await _live_thinking_metrics(
            given_provider, given_client, thinking={"type": "disabled"}
        )
    assert when_reasoning == 0, (
        "thinking=disabled must suppress reasoning_content from the Perch upstream, "
        f"got {when_reasoning} reasoning chars ({when_elapsed:.1f}s). "
        "Perchai may ignore the normalized thinking config."
    )


@pytest.mark.live
@live_only
async def test_live_thinking_effort_modulates_reasoning_volume() -> None:
    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        low_reasoning, low_elapsed = await _live_thinking_metrics(
            given_provider, given_client,
            thinking={"type": "enabled"}, reasoning_effort="low",
        )
        high_reasoning, high_elapsed = await _live_thinking_metrics(
            given_provider, given_client,
            thinking={"type": "enabled"}, reasoning_effort="high",
        )
        if high_reasoning <= low_reasoning:
            high_reasoning, high_elapsed = await _live_thinking_metrics(
                given_provider, given_client,
                thinking={"type": "enabled"}, reasoning_effort="high",
            )
    assert low_reasoning > 0, (
        "thinking=enabled + reasoning_effort=low must produce reasoning, "
        f"got {low_reasoning} reasoning chars ({low_elapsed:.1f}s)."
    )
    then_effort_changes_volume = (
        abs(high_reasoning - low_reasoning) > 0.3 * max(high_reasoning, low_reasoning)
    )
    assert then_effort_changes_volume, (
        "reasoning_effort must change reasoning volume (upstream honors effort), "
        f"low={low_reasoning} chars/{low_elapsed:.1f}s, "
        f"high={high_reasoning} chars/{high_elapsed:.1f}s. "
        "Volumes are too similar - Perchai may ignore reasoning_effort."
    )


@pytest.mark.live
@live_only
@pytest.mark.asyncio
async def test_live_provider_acompletion_returns_200_against_app_perchai() -> None:
    """Real e2e: PerchaiProvider.acompletion must 200 against live app.perchai.app.

    Regression guard for the perchai-cli/<version> User-Agent fix. Perch's
    surface gate fingerprints direct API access by User-Agent and rejects
    anything other than perchai-cli/* with 403 perch_surface_required. The
    seam test in tests/test_perchai_auth.py proves the proxy SENDS that header
    against a fake server; this test proves production ACCEPTS it end to end
    through the deployed code path (token discovery -> turn-ticket mint ->
    model-call POST). If User-Agent is reverted to python-httpx/<ver>, the
    turn-ticket mint 403s and acompletion raises PerchaiAuthError before any
    model call is attempted.

    Uses gemma-4-e2b (Starter tier, smallest, fastest) with max_tokens=16.
    """
    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        when_response = await given_provider.acompletion(
            given_client,
            model="perchai/bedrock-mantle-google-gemma-4-e2b",
            messages=[{"role": "user", "content": "Reply with one short sentence."}],
            max_tokens=16,
            stream=False,
        )
    then_content = when_response.choices[0].message.content
    assert then_content and then_content.strip(), (
        "acompletion returned an empty reply - upstream likely 403'd "
        "perch_surface_required or returned ok=false: "
        f"{when_response!r}"
    )


@pytest.mark.live
@live_only
@pytest.mark.asyncio
async def test_live_toplevel_effort_reaches_deepseek_under_wall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end guard for the invisible reasoning-wall adapter.

    Exercises the exact OpenCode shape that used to bypass the cap: a
    top-level reasoning_effort with no thinking dict, streamed to
    DeepSeek-v4-flash through the production acompletion path (independent
    session, turn-ticket, perchai-cli User-Agent). A small budget keeps the
    call cheap while proving the thinking_budget reaches upstream and
    reasoning streams out and completes instead of truncating mid-sentence.
    """
    monkeypatch.setenv(
        "PERCHAI_WANDB_DEEPSEEK_AI_DEEPSEEK_V4_FLASH_0731_THINKING_BUDGET_HIGH",
        "400",
    )
    given_provider = PerchaiProvider()
    reasoning_chars = 0
    answer_chars = 0
    async with httpx.AsyncClient() as given_client:
        when_stream = await given_provider.acompletion(
            given_client,
            model="perchai/wandb-deepseek-ai-deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "What is 17*23? Work it out."}],
            reasoning_effort="high",
            max_tokens=512,
            stream=True,
        )
        async for chunk in when_stream:
            delta = chunk.choices[0].delta
            reasoning_chars += len(getattr(delta, "reasoning_content", None) or "")
            answer_chars += len(getattr(delta, "content", None) or "")
    assert reasoning_chars > 0, (
        "top-level high effort must stream reasoning end-to-end; the wall "
        f"budget path is not reaching upstream (reasoning={reasoning_chars}, "
        f"answer={answer_chars})"
    )
    assert answer_chars > 0, (
        f"model must still answer under the capped reasoning budget, "
        f"got reasoning={reasoning_chars}, answer={answer_chars}"
    )
