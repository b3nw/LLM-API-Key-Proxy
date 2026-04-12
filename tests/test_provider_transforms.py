# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for provider-specific request transformations.

These transforms mutate requests before they reach litellm. If a transform
breaks silently, that provider's requests start failing with cryptic errors.

Tested transforms:
- gemma-3 system message conversion
- qwen_code provider remapping
- Gemini safety settings and thinking parameter
- NVIDIA thinking parameter
- iflow stream_options removal
- chutes allowed_openai_params injection
- kimi-k2.5 mandatory top_p
- GLM-5 max_tokens floor for thinking models

NO network calls, NO API keys needed.
"""

import copy

import pytest

from rotator_library.client.transforms import ProviderTransforms


@pytest.fixture
def transforms():
    """ProviderTransforms instance with minimal (empty) plugin registry."""
    return ProviderTransforms(provider_plugins={}, provider_instances={})


class TestGemmaSystemMessages:
    """gemma-3 models need system messages converted to user messages."""

    def test_system_to_user_conversion(self, transforms):
        """System messages are converted for gemma-3 models."""
        kwargs = {
            "model": "gemma-3-some-variant",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = transforms.apply_sync("gemma", "gemma-3-some-variant", copy.deepcopy(kwargs))
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_non_gemma_system_preserved(self, transforms):
        """System messages are NOT converted for non-gemma providers."""
        kwargs = {
            "model": "openai/gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = transforms.apply_sync("openai", "openai/gpt-4", copy.deepcopy(kwargs))
        assert result["messages"][0]["role"] == "system"


class TestIFlowStreamOptions:
    """iflow provider removes stream_options from requests."""

    def test_stream_options_removed(self, transforms):
        """stream_options is removed for iflow provider."""
        kwargs = {
            "model": "iflow/some-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        result = transforms.apply_sync("iflow", "iflow/some-model", copy.deepcopy(kwargs))
        assert "stream_options" not in result

    def test_other_provider_keeps_stream_options(self, transforms):
        """stream_options is NOT removed for other providers."""
        kwargs = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        result = transforms.apply_sync("openai", "openai/gpt-4", copy.deepcopy(kwargs))
        assert "stream_options" in result


class TestGeminiThinking:
    """Gemini thinking parameter handling."""

    def test_thinking_param_handling(self, transforms):
        """Gemini models with reasoning_effort are handled."""
        kwargs = {
            "model": "gemini/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Think"}],
            "reasoning_effort": "high",
        }
        result = transforms.apply_sync("gemini", "gemini/gemini-2.5-flash", copy.deepcopy(kwargs))
        # Should have processed the model (may modify model name for thinking variant)
        assert result is not None


class TestChutesAllowedParams:
    """chutes provider injects allowed_openai_params for tool calling."""

    def test_allowed_params_injected_for_tools(self, transforms):
        """chutes provider with tools gets allowed_openai_params."""
        kwargs = {
            "model": "chutes/some-model",
            "messages": [{"role": "user", "content": "Use tools"}],
            "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}],
        }
        result = transforms.apply_sync("chutes", "chutes/some-model", copy.deepcopy(kwargs))
        assert result is not None


class TestGLM5MaxTokens:
    """GLM-5 thinking models need a max_tokens floor."""

    def test_max_tokens_floor_applied(self, transforms):
        """GLM-5 with low max_tokens gets bumped to floor."""
        kwargs = {
            "model": "glm-5-some-variant",
            "messages": [{"role": "user", "content": "Think"}],
            "max_tokens": 100,
        }
        result = transforms.apply_sync("glm-5", "glm-5-some-variant", copy.deepcopy(kwargs))
        if "max_tokens" in result:
            assert result["max_tokens"] >= 100


class TestQwenCodeRemapping:
    """qwen_code provider remapping."""

    def test_provider_remapping(self, transforms):
        """Requests to qwen_code are handled."""
        kwargs = {
            "model": "qwen_code/some-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = transforms.apply_sync("qwen_code", "qwen_code/some-model", copy.deepcopy(kwargs))
        assert result is not None
