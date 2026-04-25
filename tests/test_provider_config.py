import pytest
from rotator_library.provider_config import get_provider_ui_config, LITELLM_PROVIDERS

def test_get_provider_ui_config_existing():
    # Test getting an existing provider's UI config
    # We use the first key in the dictionary to avoid hardcoding a specific provider name,
    # ensuring the test remains robust if the configuration changes.
    if not LITELLM_PROVIDERS:
        pytest.skip("LITELLM_PROVIDERS is empty, cannot test existing provider")

    first_provider = next(iter(LITELLM_PROVIDERS.keys()))
    expected_config = LITELLM_PROVIDERS[first_provider]

    result = get_provider_ui_config(first_provider)
    assert result == expected_config
    assert "category" in result

def test_get_provider_ui_config_missing():
    # Test getting a missing provider's UI config
    result = get_provider_ui_config("unknown_provider_xyz")
    assert result == {"category": "other"}
