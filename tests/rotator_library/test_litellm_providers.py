import pytest
from unittest.mock import patch

from rotator_library.litellm_providers import get_provider_route

MOCK_PROVIDERS = {
    "provider_with_slash": {
        "route": "myroute/",
    },
    "provider_without_slash": {
        "route": "myroute",
    },
    "provider_empty_route": {
        "route": "",
    },
    "provider_no_route": {
        "other_key": "value",
    },
}

@patch("rotator_library.litellm_providers.SCRAPED_PROVIDERS", MOCK_PROVIDERS)
def test_get_provider_route_with_trailing_slash():
    assert get_provider_route("provider_with_slash") == "myroute"

@patch("rotator_library.litellm_providers.SCRAPED_PROVIDERS", MOCK_PROVIDERS)
def test_get_provider_route_without_trailing_slash():
    assert get_provider_route("provider_without_slash") == "myroute"

@patch("rotator_library.litellm_providers.SCRAPED_PROVIDERS", MOCK_PROVIDERS)
def test_get_provider_route_empty():
    assert get_provider_route("provider_empty_route") is None

@patch("rotator_library.litellm_providers.SCRAPED_PROVIDERS", MOCK_PROVIDERS)
def test_get_provider_route_missing_key():
    assert get_provider_route("provider_no_route") is None

@patch("rotator_library.litellm_providers.SCRAPED_PROVIDERS", MOCK_PROVIDERS)
def test_get_provider_route_unknown_provider():
    assert get_provider_route("unknown_provider") is None
