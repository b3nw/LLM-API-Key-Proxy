"""Regression coverage for Gemini CLI CCPA model registration."""

from rotator_library.providers.gemini_cli_provider import (
    AVAILABLE_MODELS,
    CCPA_AI_MODEL_MAPPINGS,
    GeminiCliProvider,
)


def test_new_flash_models_use_ccpa_supported_identifiers() -> None:
    """User-facing Gemini 3.6/3.5-lite IDs must be remapped for CCPA."""
    assert CCPA_AI_MODEL_MAPPINGS["gemini-3.6-flash"] == "gemini-3-flash"
    assert CCPA_AI_MODEL_MAPPINGS["gemini-3.5-flash-lite"] == "gemini-3.1-flash-lite"


def test_new_flash_models_are_discoverable_and_share_their_quota_pools() -> None:
    """Registered models are exposed and tracked with the matching family."""
    assert "gemini-3.6-flash" in AVAILABLE_MODELS
    assert "gemini-3.5-flash-lite" in AVAILABLE_MODELS
    assert "gemini-3.6-flash" in GeminiCliProvider.model_quota_groups["flash"]
    assert "gemini-3.5-flash-lite" in GeminiCliProvider.model_quota_groups["flash-lite"]
