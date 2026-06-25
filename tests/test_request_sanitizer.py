# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for request sanitization.

Sanitization removes unsupported parameters from requests before
they reach providers. If this breaks:
- `dimensions` on non-embedding models → 400 Bad Request

Note: `thinking` parameter filtering is no longer handled by the sanitizer.
Each provider handles thinking/reasoning params in its own acompletion()
method (e.g. Lightning AI converts `thinking` to `reasoning_effort`).

NO network calls, NO API keys needed.
"""

import copy

from rotator_library.request_sanitizer import sanitize_request_payload


class TestSanitizeDimensions:
    """Test removal of `dimensions` parameter for non-embedding models."""

    def test_dimensions_removed_for_non_embedding_model(self):
        """dimensions is removed for any model without 'embedding' in its name."""
        payload = {"model": "openai/gpt-4o", "input": "test", "dimensions": 512}
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "dimensions" not in result

    def test_dimensions_kept_for_openai_embedding(self):
        """dimensions is preserved for OpenAI text-embedding-3 models."""
        for model in ["openai/text-embedding-3-small", "openai/text-embedding-3-large"]:
            payload = {"model": model, "input": "test", "dimensions": 512}
            result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
            assert result["dimensions"] == 512

    def test_dimensions_kept_for_gemini_embedding(self):
        """dimensions is preserved for Gemini embedding models."""
        payload = {"model": "google/gemini-embedding-2", "input": "test", "dimensions": 768}
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert result["dimensions"] == 768

    def test_no_dimensions_key(self):
        """Payload without dimensions is unchanged."""
        payload = {"model": "test-model", "input": "test"}
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert result == payload


class TestSanitizeThinking:
    """Test that `thinking` parameter is passed through (not stripped).

    The sanitizer no longer strips `thinking` — each provider handles
    thinking/reasoning params in its own acompletion() method.
    """

    def test_thinking_kept_for_non_whitelisted(self):
        """thinking is preserved for all models — providers handle filtering."""
        payload = {
            "model": "openai/gpt-4o",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" in result

    def test_thinking_kept_for_anthropic(self):
        """thinking is preserved for anthropic models."""
        payload = {
            "model": "anthropic/claude-3-7-sonnet",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" in result

    def test_thinking_kept_for_gemini(self):
        """thinking is preserved for gemini models."""
        payload = {
            "model": "gemini/gemini-2.0-flash",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" in result

    def test_thinking_kept_regardless_of_value(self):
        """thinking is preserved for all models regardless of param values."""
        payload = {
            "model": "some-model",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" in result

    def test_empty_payload(self):
        """Empty payload doesn't crash."""
        result = sanitize_request_payload({}, "any-model")
        assert result == {}

    def test_thinking_with_invalid_type_doesnt_crash(self):
        """thinking parameter with non-dict type doesn't crash and is passed through."""
        payload = {
            "model": "some-model",
            "messages": [],
            "thinking": "enabled",  # String instead of dict
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" in result

    def test_extra_body_thinking_kept(self):
        """extra_body.thinking is preserved — providers handle it."""
        payload = {
            "model": "lightning_ai/gpt-5.2",
            "messages": [],
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "extra_body" in result
        assert result["extra_body"]["thinking"] == {"type": "disabled"}


class TestSanitizeCombined:
    """Test payloads containing multiple parameters that need sanitization."""

    def test_dimensions_removed_thinking_kept(self):
        """dimensions is removed but thinking is kept for unsupported model."""
        payload = {
            "model": "some/unsupported-model",
            "input": "test",
            "dimensions": 1024,
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "dimensions" not in result
        assert "thinking" in result

    def test_dimensions_removed_for_openai_non_embedding(self):
        """Dimensions removed for OpenAI chat models."""
        payload = {
            "model": "openai/gpt-4o",
            "messages": [],
            "dimensions": 512,
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "dimensions" not in result
