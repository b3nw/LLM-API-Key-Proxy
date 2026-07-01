# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for dynamic Anthropic model discovery via models.dev.

Verifies:
- Fetching and parsing the models.dev catalog
- Filtering of deprecated/non-tool-call models
- Module-level cache behavior (fresh → stale → fallback)
- Fallback to hardcoded OAUTH_MODELS when fetch fails
- get_model_quota_group() override

NO network calls — all HTTP is mocked.
NO API keys needed.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

import rotator_library.providers.anthropic_provider as ap
from rotator_library.providers.anthropic_provider import (
    OAUTH_MODELS,
    _fetch_anthropic_models_from_models_dev,
    _get_dynamic_models,
)


class TestModelsDevFetch:
    """Test the _fetch_anthropic_models_from_models_dev function."""

    @staticmethod
    def _mock_urlopen(data: dict):
        """Create a mock for urllib.request.urlopen returning JSON data."""
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        return mock_resp

    def test_fetch_returns_models_and_max_tokens(self):
        """Fetch parses model IDs and max output tokens from models.dev response."""
        fake_data = {
            "anthropic": {
                "models": {
                    "claude-opus-4-8": {
                        "name": "Claude Opus 4.8",
                        "tool_call": True,
                        "limit": {"context": 1000000, "output": 128000},
                    },
                    "claude-sonnet-4-6": {
                        "name": "Claude Sonnet 4.6",
                        "tool_call": True,
                        "limit": {"context": 1000000, "output": 64000},
                    },
                }
            }
        }
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(fake_data)):
            result = _fetch_anthropic_models_from_models_dev()

        assert result is not None
        assert "claude-opus-4-8" in result["models"]
        assert "claude-sonnet-4-6" in result["models"]
        assert result["max_tokens"]["claude-opus-4-8"] == 128000
        assert result["max_tokens"]["claude-sonnet-4-6"] == 64000

    def test_fetch_filters_3x_models(self):
        """Retired 3.x models are excluded from the result."""
        fake_data = {
            "anthropic": {
                "models": {
                    "claude-3-5-sonnet-20241022": {
                        "name": "Claude 3.5 Sonnet",
                        "tool_call": True,
                        "limit": {"output": 8192},
                    },
                    "claude-opus-4-8": {
                        "name": "Claude Opus 4.8",
                        "tool_call": True,
                        "limit": {"output": 128000},
                    },
                }
            }
        }
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(fake_data)):
            result = _fetch_anthropic_models_from_models_dev()

        assert result is not None
        assert "claude-opus-4-8" in result["models"]
        assert "claude-3-5-sonnet-20241022" not in result["models"]

    def test_fetch_filters_mythos_models(self):
        """Restricted mythos models are excluded."""
        fake_data = {
            "anthropic": {
                "models": {
                    "claude-mythos-5": {
                        "name": "Claude Mythos 5",
                        "tool_call": True,
                        "limit": {"output": 128000},
                    },
                    "claude-fable-5": {
                        "name": "Claude Fable 5",
                        "tool_call": True,
                        "limit": {"output": 128000},
                    },
                }
            }
        }
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(fake_data)):
            result = _fetch_anthropic_models_from_models_dev()

        assert result is not None
        assert "claude-fable-5" in result["models"]
        assert "claude-mythos-5" not in result["models"]

    def test_fetch_filters_no_tool_call(self):
        """Models without tool_call support are excluded."""
        fake_data = {
            "anthropic": {
                "models": {
                    "claude-opus-4-8": {
                        "name": "Claude Opus 4.8",
                        "tool_call": True,
                        "limit": {"output": 128000},
                    },
                    "some-embedding-model": {
                        "name": "Embedding Model",
                        "tool_call": False,
                        "limit": {"output": 4096},
                    },
                }
            }
        }
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(fake_data)):
            result = _fetch_anthropic_models_from_models_dev()

        assert result is not None
        assert "claude-opus-4-8" in result["models"]
        assert "some-embedding-model" not in result["models"]

    def test_fetch_returns_none_on_network_error(self):
        """Returns None when the network request fails."""
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = _fetch_anthropic_models_from_models_dev()
        assert result is None

    def test_fetch_returns_none_on_invalid_json(self):
        """Returns None when the response is not valid JSON."""
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"not valid json"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _fetch_anthropic_models_from_models_dev()
        assert result is None

    def test_fetch_returns_none_on_empty_models(self):
        """Returns None when models.dev has no Anthropic models."""
        fake_data = {"anthropic": {"models": {}}}
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(fake_data)):
            result = _fetch_anthropic_models_from_models_dev()
        assert result is None

    def test_fetch_returns_none_when_all_filtered(self):
        """Returns None when all models are filtered out (not a truthy empty dict)."""
        fake_data = {
            "anthropic": {
                "models": {
                    "claude-3-5-sonnet-20241022": {  # Will be filtered (3.x prefix)
                        "tool_call": True,
                        "limit": {"output": 8192},
                    },
                    "claude-mythos-5": {  # Will be filtered (mythos prefix)
                        "tool_call": True,
                        "limit": {"output": 128000},
                    },
                }
            }
        }
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(fake_data)):
            result = _fetch_anthropic_models_from_models_dev()
        assert result is None


class TestDynamicModelsCache:
    """Test the _get_dynamic_models cache and fallback logic."""

    def setup_method(self):
        """Reset module-level cache before each test."""
        ap._models_dev_cache = None
        ap._models_dev_cache_time = 0.0

    @staticmethod
    def _mock_urlopen(data: dict):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        return mock_resp

    def test_caches_successful_fetch(self):
        """Second call within TTL returns cached data without fetching again."""
        fake_data = {
            "anthropic": {
                "models": {
                    "claude-opus-4-8": {
                        "tool_call": True,
                        "limit": {"output": 128000},
                    },
                }
            }
        }
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(fake_data)) as mock_urlopen:
            result1 = _get_dynamic_models()
            assert result1 is not None
            assert "claude-opus-4-8" in result1["models"]

            result2 = _get_dynamic_models()
            assert result2 is not None

            # urlopen should only be called once (second call uses cache)
            assert mock_urlopen.call_count == 1

    def test_falls_back_to_stale_cache_on_failure(self):
        """When fetch fails but stale cache exists, returns stale data."""
        # Prime the cache with stale data
        ap._models_dev_cache = {
            "models": ["claude-opus-4-8"],
            "max_tokens": {"claude-opus-4-8": 128000},
        }
        ap._models_dev_cache_time = 0.0  # Epoch → definitely stale

        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = _get_dynamic_models()

        assert result is not None
        assert "claude-opus-4-8" in result["models"]

    def test_returns_none_when_no_data_available(self):
        """Returns None when fetch fails and no cache exists."""
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = _get_dynamic_models()
        assert result is None


class TestQuotaGroupOverride:
    """Test the get_model_quota_group override."""

    def test_quota_group_always_anthropic_global(self):
        """get_model_quota_group returns 'anthropic-global' for any model."""
        from rotator_library.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        assert provider.get_model_quota_group("claude-opus-4-8") == "anthropic-global"
        assert provider.get_model_quota_group("claude-sonnet-4-6") == "anthropic-global"
        assert provider.get_model_quota_group("anthropic/claude-fable-5") == "anthropic-global"
        assert provider.get_model_quota_group("any-unknown-model") == "anthropic-global"
