# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for provider configuration accessors.
"""

import pytest
from unittest.mock import patch

from rotator_library.provider_config import get_full_provider_config, get_provider_ui_config

def test_get_provider_ui_config_known_provider():
    """Test get_provider_ui_config with a known provider."""
    mock_litellm_providers = {
        "test_provider": {
            "category": "test_category",
            "note": "Test note",
        }
    }
    with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
        config = get_provider_ui_config("test_provider")
        assert config == {"category": "test_category", "note": "Test note"}

def test_get_provider_ui_config_unknown_provider():
    """Test get_provider_ui_config with an unknown provider."""
    mock_litellm_providers = {}
    with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
        config = get_provider_ui_config("unknown_provider")
        assert config == {"category": "other"}

def test_get_full_provider_config_known_provider():
    """Test get_full_provider_config with a known provider."""
    mock_scraped_providers = {
        "test_provider": {
            "api_base": "https://api.test.com",
            "models": ["model-a"],
        }
    }
    mock_litellm_providers = {
        "test_provider": {
            "category": "test_category",
            "note": "Test note",
        }
    }
    with patch("rotator_library.provider_config.SCRAPED_PROVIDERS", mock_scraped_providers):
        with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
            config = get_full_provider_config("test_provider")

            # Should have properties from both
            assert config["api_base"] == "https://api.test.com"
            assert config["models"] == ["model-a"]
            assert config["category"] == "test_category"
            assert config["note"] == "Test note"

def test_get_full_provider_config_unknown_provider():
    """Test get_full_provider_config with an unknown provider."""
    mock_scraped_providers = {}
    mock_litellm_providers = {}

    with patch("rotator_library.provider_config.SCRAPED_PROVIDERS", mock_scraped_providers):
        with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
            config = get_full_provider_config("unknown_provider")

            # Should fallback to default category and have no scraped properties
            assert config == {"category": "other"}

def test_get_full_provider_config_partial_data():
    """Test get_full_provider_config when provider is only in scraped or only in UI config."""
    # Only in UI config
    mock_litellm_providers = {"test_ui_only": {"category": "ui_only_category"}}
    mock_scraped_providers = {}

    with patch("rotator_library.provider_config.SCRAPED_PROVIDERS", mock_scraped_providers):
        with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
            config = get_full_provider_config("test_ui_only")
            assert config == {"category": "ui_only_category"}

    # Only in scraped config
    mock_litellm_providers = {}
    mock_scraped_providers = {"test_scraped_only": {"api_base": "https://test.com"}}

    with patch("rotator_library.provider_config.SCRAPED_PROVIDERS", mock_scraped_providers):
        with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
            config = get_full_provider_config("test_scraped_only")
            assert config == {"api_base": "https://test.com", "category": "other"}
