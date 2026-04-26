import pytest
from rotator_library.provider_config import get_provider_ui_config

@pytest.mark.parametrize("provider,expected_category", [
    ("openai", "popular"),
    ("anthropic", "popular"),
    ("gemini", "popular"),
])
def test_get_provider_ui_config_existing(provider, expected_category):
    # Test getting an existing provider's UI config using hardcoded expectations
    result = get_provider_ui_config(provider)
    assert result["category"] == expected_category

def test_get_provider_ui_config_missing():
    # Test getting a missing provider's UI config
    result = get_provider_ui_config("unknown_provider_xyz")
    assert result == {"category": "other"}
