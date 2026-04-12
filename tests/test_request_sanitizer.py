# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for request sanitization.

Sanitization removes unsupported parameters from requests before
they reach providers. If this breaks:
- `dimensions` on non-OpenAI models → 400 Bad Request
- `thinking` on non-Gemini models → 400 Bad Request

NO network calls, NO API keys needed.
"""

import copy

import pytest

from rotator_library.request_sanitizer import sanitize_request_payload


class TestSanitizeDimensions:
    """Test removal of `dimensions` parameter for non-OpenAI embedding models."""

    def test_dimensions_removed_for_non_openai(self):
        """dimensions is removed for any model that isn't OpenAI text-embedding-3-*."""
        payload = {"model": "some-other-model", "input": "test", "dimensions": 512}
        result = sanitize_request_payload(copy.deepcopy(payload), "some-other-model")
        assert "dimensions" not in result

    def test_dimensions_kept_for_openai_embedding(self):
        """dimensions is preserved for OpenAI text-embedding-3 models."""
        for model in ["openai/text-embedding-3-small", "openai/text-embedding-3-large"]:
            payload = {"model": model, "input": "test", "dimensions": 512}
            result = sanitize_request_payload(copy.deepcopy(payload), model)
            assert result["dimensions"] == 512

    def test_no_dimensions_key(self):
        """Payload without dimensions is unchanged."""
        payload = {"model": "test-model", "input": "test"}
        result = sanitize_request_payload(copy.deepcopy(payload), "test-model")
        assert result == payload


class TestSanitizeThinking:
    """Test removal of `thinking` parameter for non-Gemini models."""

    def test_thinking_removed_for_non_gemini(self):
        """thinking is removed for models that aren't gemini/gemini-2.5-pro/flash."""
        payload = {
            "model": "claude-sonnet-4-5",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), "claude-sonnet-4-5")
        assert "thinking" not in result

    def test_thinking_kept_for_gemini_25_pro(self):
        """thinking is preserved for gemini/gemini-2.5-pro."""
        payload = {
            "model": "gemini/gemini-2.5-pro",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), "gemini/gemini-2.5-pro")
        assert "thinking" in result

    def test_thinking_kept_for_gemini_25_flash(self):
        """thinking is preserved for gemini/gemini-2.5-flash."""
        payload = {
            "model": "gemini/gemini-2.5-flash",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), "gemini/gemini-2.5-flash")
        assert "thinking" in result

    def test_thinking_not_removed_if_different_value(self):
        """Only the exact thinking={type:enabled, budget:-1} is affected."""
        payload = {
            "model": "some-model",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), "some-model")
        # Different budget_tokens value should NOT be removed by current logic
        # (the sanitizer only targets the exact -1 pattern)
        assert "thinking" in result

    def test_empty_payload(self):
        """Empty payload doesn't crash."""
        result = sanitize_request_payload({}, "any-model")
        assert result == {}
