# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Provider-specific request transformations.

This module isolates all provider-specific request mutations that were
scattered throughout client.py, including:
- gemma-3 system message conversion
- qwen_code provider remapping
- Gemini safety settings and thinking parameter
- NVIDIA thinking parameter
- iflow stream_options removal
- dedaluslabs tool_choice=auto removal
- kimi-k2.5 mandatory top_p
- GLM-5 max_tokens floor for thinking models

Transforms are applied in a defined order with logging of modifications.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


class ProviderTransforms:
    """
    Centralized provider-specific request transformations.

    Transforms are applied in order:
    1. Built-in transforms (gemma-3, qwen_code, etc.)
    2. Provider hook transforms (from provider plugins)
    3. Safety settings conversions
    """

    def __init__(
        self,
        provider_plugins: Dict[str, Any],
        provider_config: Optional[Any] = None,
        provider_instances: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ProviderTransforms.

        Args:
            provider_plugins: Dict mapping provider names to plugin classes
            provider_config: ProviderConfig instance for LiteLLM conversions
            provider_instances: Shared dict for caching provider instances.
                If None, creates a new dict (not recommended - leads to duplicate instances).
        """
        self._plugins = provider_plugins
        self._plugin_instances: Dict[str, Any] = (
            provider_instances if provider_instances is not None else {}
        )
        self._config = provider_config

        # Registry of built-in transforms
        # Each provider can have multiple transform functions
        self._transforms: Dict[str, List[Callable]] = {
            "gemma": [self._transform_gemma_system_messages],
            "qwen_code": [self._transform_qwen_code_provider],
            "gemini": [self._transform_gemini_safety, self._transform_gemini_thinking],
            "nvidia_nim": [self._transform_nvidia_thinking],
            "iflow": [self._transform_iflow_stream_options],
            "dedaluslabs": [self._transform_dedaluslabs_tool_choice],
            "chutes": [self._transform_chutes_allowed_params],
            "kimi-k2.5": [self._transform_kimi_parameters],
            "glm-5": [self._transform_glm5_max_tokens],
            "glm-4": [self._transform_glm5_max_tokens],
        }

    def _get_plugin_instance(self, provider: str) -> Optional[Any]:
        """Get or create a plugin instance for a provider."""
        if provider not in self._plugin_instances:
            plugin_class = self._plugins.get(provider)
            if plugin_class:
                if isinstance(plugin_class, type):
                    self._plugin_instances[provider] = plugin_class()
                else:
                    self._plugin_instances[provider] = plugin_class
            else:
                return None
        return self._plugin_instances[provider]

    async def apply(
        self,
        provider: str,
        model: str,
        credential: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply all applicable transforms to request kwargs.

        Args:
            provider: Provider name
            model: Model being requested
            credential: Selected credential
            kwargs: Request kwargs (will be mutated)

        Returns:
            Modified kwargs
        """
        modifications: List[str] = []

        # 1. Apply built-in transforms
        for transform_provider, transforms in self._transforms.items():
            # Check if transform applies (provider match or model contains pattern)
            if transform_provider == provider or transform_provider in model.lower():
                for transform in transforms:
                    result = transform(kwargs, model, provider)
                    if result:
                        modifications.append(result)

        # 2. Apply provider hook transforms (async)
        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "transform_request"):
            try:
                hook_result = await plugin.transform_request(kwargs, model, credential)
                if hook_result:
                    modifications.extend(hook_result)
            except Exception as e:
                lib_logger.debug(f"Provider transform_request hook failed: {e}")

        # 3. Apply model-specific options from provider
        if plugin and hasattr(plugin, "get_model_options"):
            model_options = plugin.get_model_options(model)
            if model_options:
                for key, value in model_options.items():
                    if key == "reasoning_effort":
                        kwargs["reasoning_effort"] = value
                    elif key not in kwargs:
                        kwargs[key] = value
                modifications.append(f"applied model options for {model}")

        # 4. Apply LiteLLM conversion if config available
        if self._config and hasattr(self._config, "convert_for_litellm"):
            kwargs = self._config.convert_for_litellm(**kwargs)

        if modifications:
            lib_logger.debug(
                f"Applied transforms for {provider}/{model}: {modifications}"
            )

        return kwargs

    def apply_sync(
        self,
        provider: str,
        model: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply built-in transforms synchronously (no provider hooks).

        Useful when async is not available.

        Args:
            provider: Provider name
            model: Model being requested
            kwargs: Request kwargs

        Returns:
            Modified kwargs
        """
        modifications: List[str] = []

        for transform_provider, transforms in self._transforms.items():
            if transform_provider == provider or transform_provider in model.lower():
                for transform in transforms:
                    result = transform(kwargs, model, provider)
                    if result:
                        modifications.append(result)

        if modifications:
            lib_logger.debug(
                f"Applied sync transforms for {provider}/{model}: {modifications}"
            )

        return kwargs

    # =========================================================================
    # BUILT-IN TRANSFORMS
    # =========================================================================

    def _transform_gemma_system_messages(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Convert system messages to user messages for Gemma-3.

        Gemma-3 models don't support system messages, so we convert them
        to user messages to maintain functionality.
        """
        if "gemma-3" not in model.lower():
            return None

        messages = kwargs.get("messages", [])
        if not messages:
            return None

        converted = False
        new_messages = []
        for m in messages:
            if m.get("role") == "system":
                new_messages.append({"role": "user", "content": m["content"]})
                converted = True
            else:
                new_messages.append(m)

        if converted:
            kwargs["messages"] = new_messages
            return "gemma-3: converted system->user messages"
        return None

    def _transform_qwen_code_provider(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Remap qwen_code to qwen provider for LiteLLM.

        The qwen_code provider is a custom wrapper that needs to be
        translated to the qwen provider for LiteLLM compatibility.
        """
        if provider != "qwen_code":
            return None

        kwargs["custom_llm_provider"] = "qwen"
        if "/" in model:
            kwargs["model"] = model.split("/", 1)[1]
        return "qwen_code: remapped to qwen provider"

    def _transform_gemini_safety(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        # Safety settings are passed through unchanged. No defaults are injected
        # because some Gemini-family models (e.g. Gemma) reject unknown safety
        # categories with a 400 error.
        return None

    def _transform_gemini_thinking(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Handle thinking parameter for Gemini.

        Delegates to provider plugin's handle_thinking_parameter method.
        """
        if provider != "gemini":
            return None

        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "handle_thinking_parameter"):
            plugin.handle_thinking_parameter(kwargs, model)
            return "gemini: handled thinking parameter"
        return None

    def _transform_nvidia_thinking(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Handle thinking parameter for NVIDIA NIM.

        Delegates to provider plugin's handle_thinking_parameter method.
        """
        if provider != "nvidia_nim":
            return None

        plugin = self._get_plugin_instance(provider)
        if plugin and hasattr(plugin, "handle_thinking_parameter"):
            plugin.handle_thinking_parameter(kwargs, model)
            return "nvidia_nim: handled thinking parameter"
        return None

    def _transform_iflow_stream_options(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Remove stream_options for iflow provider.

        The iflow provider returns HTTP 406 if stream_options is present.
        """
        if provider != "iflow":
            return None

        if "stream_options" in kwargs:
            del kwargs["stream_options"]
            return "iflow: removed stream_options"
        return None

    def _transform_dedaluslabs_tool_choice(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Remove tool_choice=auto for dedaluslabs provider.

        Dedaluslabs API returns HTTP 422 if tool_choice is passed as a string
        ("auto") instead of an object. Since "auto" is the default behavior,
        removing it fixes the issue without changing functionality.
        """
        if provider != "dedaluslabs":
            return None

        if kwargs.get("tool_choice") == "auto":
            del kwargs["tool_choice"]
            return "dedaluslabs: removed tool_choice=auto"
        return None

    # OpenAI-compatible params that LiteLLM's Chutes provider config
    # doesn't declare support for.  Without this list, drop_params=True
    # causes LiteLLM to silently strip tools / tool_choice / etc.
    _CHUTES_ALLOWED_OPENAI_PARAMS = [
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    ]

    def _transform_chutes_allowed_params(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Inject allowed_openai_params for Chutes provider.

        LiteLLM's built-in Chutes provider config doesn't advertise support
        for tool calling parameters (tools, tool_choice, etc.), so with
        litellm.drop_params=True they get silently removed.  This transform
        tells LiteLLM these standard OpenAI params are safe to pass through
        to the Chutes API, which is fully OpenAI-compatible.
        """
        if provider != "chutes":
            return None

        # Only inject if the request actually uses any of these params
        has_tool_params = any(k in kwargs for k in self._CHUTES_ALLOWED_OPENAI_PARAMS)
        if not has_tool_params:
            return None

        existing = kwargs.get("allowed_openai_params", [])
        merged = list(set(existing) | set(self._CHUTES_ALLOWED_OPENAI_PARAMS))
        kwargs["allowed_openai_params"] = merged
        return "chutes: injected allowed_openai_params for tool calling"

    def _transform_kimi_parameters(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Set top_p=0.95 for Kimi K2.5 models.

        The Kimi K2.5 API (via various providers) strictly requires top_p to be 0.95.
        Other values or missing top_p results in a 400 error.
        """
        if "kimi-k2.5" not in model.lower():
            return None

        if kwargs.get("top_p") != 0.95:
            kwargs["top_p"] = 0.95
            return "kimi-k2.5: set top_p=0.95 (mandatory)"
        return None

    # GLM-5 / GLM-4 thinking model minimum token floor
    GLM_MIN_MAX_TOKENS = 4096

    def _transform_glm5_max_tokens(
        self,
        kwargs: Dict[str, Any],
        model: str,
        provider: str,
    ) -> Optional[str]:
        """
        Enforce a minimum max_tokens floor for GLM-5/GLM-4 thinking models.

        GLM-5 (and GLM-4.x) thinking variants share a single max_tokens budget
        between reasoning tokens and content tokens. When max_tokens is too low,
        the model exhausts the entire budget on chain-of-thought reasoning and
        returns content: null/"". This affects all providers hosting these models
        (Modal, NanoGPT, Kilo, Zenmux, etc.).

        This transform enforces a minimum floor so the model always has enough
        headroom to produce actual response content after reasoning.
        """
        model_lower = model.lower()
        # Only apply to GLM thinking/reasoning model variants
        if not any(prefix in model_lower for prefix in ("glm-5", "glm-4")):
            return None

        current = kwargs.get("max_tokens")
        if current is None or current < self.GLM_MIN_MAX_TOKENS:
            kwargs["max_tokens"] = self.GLM_MIN_MAX_TOKENS
            if current is not None:
                return (
                    f"glm: raised max_tokens from {current} to "
                    f"{self.GLM_MIN_MAX_TOKENS} (thinking budget floor)"
                )
            return (
                f"glm: set max_tokens to {self.GLM_MIN_MAX_TOKENS} "
                f"(thinking budget floor)"
            )
        return None

    # =========================================================================
    # SAFETY SETTINGS CONVERSION (REMOVED)
    # =========================================================================
    # Previously had convert_safety_settings() wrapper that delegated to
    # provider plugins. Removed because auto-injecting/merging safety defaults
    # caused 400 errors on models that don't support those categories (e.g. Gemma).
    # See gemini_provider.py for the full removal comment with previous defaults.
    # =========================================================================
