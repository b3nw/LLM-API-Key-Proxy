# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import importlib
import logging
import pkgutil
import os
from typing import Dict, Type

lib_logger = logging.getLogger("rotator_library")
from .provider_interface import ProviderInterface

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {}


class DynamicOpenAICompatibleProvider:
    """
    Dynamic provider class for custom OpenAI-compatible providers.
    Created at runtime for providers with _API_BASE environment variables
    that are NOT known LiteLLM providers.

    Environment variable pattern:
        <NAME>_API_BASE - The API base URL (required)
        <NAME>_API_KEY  - The API key

    Example:
        MYSERVER_API_BASE=http://localhost:8000/v1
        MYSERVER_API_KEY=sk-xxx

    Note: For known providers (openai, anthropic, etc.), setting _API_BASE
    will override their default endpoint without creating a custom provider.
    """

    # Cost calculation is handled by this plugin using pricing from the
    # provider's /v1/models response, not by LiteLLM's internal database.
    skip_cost_calculation: bool = False

    # Class-level pricing cache shared across all instances.
    # Key: full model name (e.g. "lightning_ai/anthropic/claude-opus-4-6")
    # Value: (input_cost_per_token, output_cost_per_token)
    _all_provider_pricing: Dict[str, Dict[str, tuple]] = {}

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        # Get API base URL from environment (using _API_BASE pattern)
        self.api_base = os.getenv(f"{provider_name.upper()}_API_BASE")
        if not self.api_base:
            raise ValueError(
                f"Environment variable {provider_name.upper()}_API_BASE is required for custom OpenAI-compatible provider"
            )

        # Import model definitions
        from ..model_definitions import ModelDefinitions

        self.model_definitions = ModelDefinitions()

        # Eagerly fetch pricing on first instantiation for this provider
        if provider_name not in DynamicOpenAICompatibleProvider._all_provider_pricing:
            DynamicOpenAICompatibleProvider._all_provider_pricing[provider_name] = {}
            self._fetch_pricing_sync()

    @property
    def _model_pricing(self) -> Dict[str, tuple]:
        """Access the shared pricing dict for this provider."""
        return DynamicOpenAICompatibleProvider._all_provider_pricing.get(
            self.provider_name, {}
        )

    def _fetch_pricing_sync(self):
        """Eagerly fetch model pricing from the provider's /v1/models endpoint."""
        import httpx as _httpx

        provider_upper = self.provider_name.upper()
        # Find the first API key for this provider
        api_key = None
        for i in range(1, 20):
            key = os.getenv(f"{provider_upper}_API_KEY_{i}")
            if key:
                api_key = key
                break
        if not api_key:
            return

        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            resp = _httpx.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            resp.raise_for_status()
            api_models = resp.json().get("data", [])
            pricing_dict = DynamicOpenAICompatibleProvider._all_provider_pricing[
                self.provider_name
            ]
            captured = 0
            for model_data in api_models:
                model_id = model_data.get("id", "")
                full_model_name = f"{self.provider_name}/{model_id}"
                pricing = model_data.get("pricing")
                if pricing:
                    input_cost = pricing.get("input_cost_per_million_tokens")
                    output_cost = pricing.get("output_cost_per_million_tokens")
                    if input_cost is not None or output_cost is not None:
                        pricing_dict[full_model_name] = (
                            float(input_cost or 0),
                            float(output_cost or 0),
                        )
                        captured += 1
            if captured:
                lib_logger.info(
                    f"Captured pricing for {captured} models from {self.provider_name}"
                )
        except Exception as exc:
            lib_logger.debug(
                f"Failed to fetch pricing for {self.provider_name}: {exc}"
            )

    async def get_models(self, api_key: str, client):
        """
        Fetch models from the OpenAI-compatible API.
        Combines static definitions with dynamic discovery.
        Also captures per-model pricing if provided by the API.

        Note: We can't delegate to OpenAICompatibleProvider because it's a singleton,
        and concurrent calls from multiple dynamic providers would share the same instance.
        """
        models = []

        # Get static model definitions from PROVIDER_MODELS env var
        static_models = self.model_definitions.get_all_provider_models(
            self.provider_name
        )
        if static_models:
            models.extend(static_models)

        # Try dynamic discovery to get additional models
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url, headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()

            static_model_names = {m.split("/")[-1] for m in static_models}
            api_models = response.json().get("data", [])

            pricing_dict = DynamicOpenAICompatibleProvider._all_provider_pricing.setdefault(
                self.provider_name, {}
            )

            for model_data in api_models:
                model_id = model_data.get("id", "")
                full_model_name = f"{self.provider_name}/{model_id}"

                if model_id not in static_model_names:
                    models.append(full_model_name)

                # Capture pricing if the API provides it
                pricing = model_data.get("pricing")
                if pricing:
                    input_cost = pricing.get("input_cost_per_million_tokens")
                    output_cost = pricing.get("output_cost_per_million_tokens")
                    if input_cost is not None or output_cost is not None:
                        # Despite the field name, these are per-token costs
                        # (verified: $15/M tokens = $0.000015 per token)
                        pricing_dict[full_model_name] = (
                            float(input_cost or 0),
                            float(output_cost or 0),
                        )


            if self._model_pricing:
                lib_logger.info(
                    f"Captured pricing for {len(self._model_pricing)} models "
                    f"from {self.provider_name}"
                )

        except Exception:
            pass  # Static models are sufficient if dynamic discovery fails

        return models

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate cost using pricing captured from the provider's API.

        Args:
            model: Full model name (e.g. "lightning_ai/anthropic/claude-opus-4-6")
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Approximate cost in dollars, or 0.0 if no pricing available
        """
        pricing = self._model_pricing.get(model)
        if not pricing:
            return 0.0

        input_cost_per_token, output_cost_per_token = pricing
        return (prompt_tokens * input_cost_per_token) + (
            completion_tokens * output_cost_per_token
        )

    def get_model_options(self, model_name: str) -> Dict[str, any]:
        """Get model options from static definitions."""
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """Returns False since we want to use the standard litellm flow."""
        return False

    def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Returns the standard Bearer token header."""
        return {"Authorization": f"Bearer {credential_identifier}"}


def _register_providers():
    """
    Dynamically discovers and imports provider plugins from this directory.
    Also creates dynamic plugins for custom OpenAI-compatible providers.
    """
    package_path = __path__
    package_name = __name__

    # First, register file-based providers
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        # Construct the full module path
        full_module_path = f"{package_name}.{module_name}"

        # Import the module
        module = importlib.import_module(full_module_path)

        # Look for a class that inherits from ProviderInterface
        # and is defined in this module (not just imported)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, ProviderInterface)
                and attribute is not ProviderInterface
                and getattr(attribute, "__module__", None) == full_module_path
            ):
                # Derives 'gemini_cli' from 'gemini_cli_provider.py'
                # Remap 'nvidia' to 'nvidia_nim' to align with litellm's provider name
                provider_name = module_name.replace("_provider", "")
                if provider_name == "nvidia":
                    provider_name = "nvidia_nim"
                PROVIDER_PLUGINS[provider_name] = attribute
                lib_logger.debug(f"Registered provider: {provider_name}")

    # Then, create dynamic plugins for custom OpenAI-compatible providers
    # These use the pattern: <NAME>_API_BASE where NAME is not a known LiteLLM provider
    # Known providers just get their api_base overridden via ProviderConfig

    # Import KNOWN_PROVIDERS to check against
    from ..provider_config import KNOWN_PROVIDERS

    for env_var in os.environ:
        if env_var.endswith("_API_BASE"):
            provider_name = env_var[:-9].lower()  # Remove '_API_BASE' suffix

            # Skip if this is a known LiteLLM provider (not a custom provider)
            if provider_name in KNOWN_PROVIDERS:
                continue

            # Skip if this provider name already exists (file-based plugin)
            if provider_name in PROVIDER_PLUGINS:
                continue

            # Create a dynamic plugin class
            def create_plugin_class(name):
                class DynamicPlugin(DynamicOpenAICompatibleProvider):
                    def __init__(self):
                        super().__init__(name)

                return DynamicPlugin

            # Create and register the plugin class
            plugin_class = create_plugin_class(provider_name)
            PROVIDER_PLUGINS[provider_name] = plugin_class
            lib_logger.debug(f"Registered dynamic provider: {provider_name}")


# Discover and register providers when the package is imported
_register_providers()
