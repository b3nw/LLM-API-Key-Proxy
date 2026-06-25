# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for Lightning AI provider thinking/reasoning parameter handling.

Verifies that:
- Anthropic-style `thinking` param is converted to `reasoning_effort`
- `reasoning` dict (Responses API format) is converted to `reasoning_effort`
- `extra_body.thinking` from _guard_thinking_tool_calls is cleaned up
- Guard's `type: disabled` takes precedence over client's `type: enabled`

NO network calls, NO API keys needed — OpenAI SDK is mocked.
"""

import asyncio
from unittest.mock import MagicMock, patch

import httpx

from rotator_library.providers.lightning_ai_provider import LightningAiProvider


def _make_provider():
    """Create a LightningAiProvider instance."""
    return LightningAiProvider()


def _capture_call_kwargs(provider, kwargs):
    """
    Run provider.acompletion() with a mocked OpenAI client.

    Returns the kwargs passed to openai_client.chat.completions.create().
    """
    captured = {}

    async def fake_create(**call_kwargs):
        captured.update(call_kwargs)
        # Return a minimal mock response
        response = MagicMock()
        response.model_dump.return_value = {
            "id": "test",
            "choices": [{"message": {"role": "assistant", "content": "test"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return response

    mock_client = MagicMock()
    mock_client.chat.completions.create = fake_create

    with patch("rotator_library.providers.lightning_ai_provider.openai") as mock_openai:
        mock_openai.AsyncOpenAI.return_value = mock_client
        mock_http_client = MagicMock(spec=httpx.AsyncClient)

        asyncio.run(
            provider.acompletion(
                client=mock_http_client,
                credential_identifier="test-key",
                **kwargs,
            )
        )

    return captured


class TestThinkingConversion:
    """Test that Anthropic-style thinking param is converted to reasoning_effort."""

    def test_thinking_enabled_converts_to_reasoning_effort(self):
        """thinking: {type: enabled} → reasoning_effort: high"""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        captured = _capture_call_kwargs(provider, kwargs)
        assert captured.get("reasoning_effort") == "high"
        assert "thinking" not in captured

    def test_thinking_disabled_does_not_set_reasoning_effort(self):
        """thinking: {type: disabled} → no reasoning_effort set"""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
        }
        captured = _capture_call_kwargs(provider, kwargs)
        assert "reasoning_effort" not in captured
        assert "thinking" not in captured

    def test_no_thinking_no_reasoning_effort(self):
        """No thinking param → no reasoning_effort set"""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        captured = _capture_call_kwargs(provider, kwargs)
        assert "reasoning_effort" not in captured

    def test_reasoning_dict_converts_to_reasoning_effort(self):
        """reasoning: {effort: medium} → reasoning_effort: medium"""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning": {"effort": "medium"},
        }
        captured = _capture_call_kwargs(provider, kwargs)
        assert captured.get("reasoning_effort") == "medium"
        assert "reasoning" not in captured

    def test_reasoning_string_converts_to_reasoning_effort(self):
        """reasoning: 'low' → reasoning_effort: low"""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning": "low",
        }
        captured = _capture_call_kwargs(provider, kwargs)
        assert captured.get("reasoning_effort") == "low"

    def test_explicit_reasoning_effort_not_overridden_by_thinking(self):
        """If reasoning_effort is already set, thinking doesn't override it."""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "medium",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        captured = _capture_call_kwargs(provider, kwargs)
        # setdefault means existing value wins
        assert captured.get("reasoning_effort") == "medium"


class TestGuardInteraction:
    """Test interaction with _guard_thinking_tool_calls extra_body injection."""

    def test_guard_disabled_overrides_client_enabled(self):
        """Guard's extra_body.thinking: {type: disabled} takes precedence."""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        captured = _capture_call_kwargs(provider, kwargs)
        # Guard disabled → no reasoning_effort
        assert "reasoning_effort" not in captured
        # extra_body.thinking should be cleaned up
        if "extra_body" in captured:
            assert "thinking" not in captured["extra_body"]

    def test_guard_disabled_without_client_thinking(self):
        """Guard injects disabled, no client thinking → no reasoning_effort."""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        captured = _capture_call_kwargs(provider, kwargs)
        assert "reasoning_effort" not in captured
        # extra_body.thinking should be removed
        if "extra_body" in captured:
            assert "thinking" not in captured["extra_body"]

    def test_guard_disabled_with_extra_body_other_keys(self):
        """Guard's thinking is removed but other extra_body keys are kept."""
        provider = _make_provider()
        kwargs = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {"thinking": {"type": "disabled"}, "other_key": "value"},
        }
        captured = _capture_call_kwargs(provider, kwargs)
        # other_key should still be there
        assert captured.get("extra_body", {}).get("other_key") == "value"
        # thinking should be removed from extra_body
        assert "thinking" not in captured.get("extra_body", {})
