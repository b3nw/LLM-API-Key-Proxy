# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Slim RotatingClient facade.

This is a lightweight facade that delegates to extracted components:
- RequestExecutor: Unified retry/rotation logic
- CredentialFilter: Tier compatibility filtering
- ModelResolver: Model name resolution and filtering
- ProviderTransforms: Provider-specific request mutations
- StreamingHandler: Streaming response processing

The original client.py was ~3000 lines. This facade is ~300 lines,
with all complexity moved to specialized modules.
"""

import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, TYPE_CHECKING

import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter

from ..core.types import RequestContext
from ..core.errors import mask_credential
from ..core.config import ConfigLoader
from ..core.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_ROTATION_TOLERANCE,
)

from .filters import CredentialFilter
from .models import ModelResolver
from .transforms import ProviderTransforms
from .executor import RequestExecutor
from .anthropic import AnthropicHandler
from .cross_provider_executor import CrossProviderExecutor


# Import providers and other dependencies
from ..providers import PROVIDER_PLUGINS
from ..cooldown_manager import CooldownManager
from ..credential_manager import CredentialManager
from ..background_refresher import BackgroundRefresher
from ..model_definitions import ModelDefinitions
from ..transaction_logger import TransactionLogger
from ..provider_config import ProviderConfig as LiteLLMProviderConfig
from ..utils.paths import get_default_root, get_logs_dir, get_oauth_dir
from ..utils.suppress_litellm_warnings import suppress_litellm_serialization_warnings

from ..model_latest_registry import ModelLatestRegistry
from ..model_alias_registry import ModelAliasRegistry
from ..failure_logger import configure_failure_logger

# Import new usage package
from ..usage import UsageManager as NewUsageManager
from ..usage.config import load_provider_usage_config, WindowDefinition

if TYPE_CHECKING:
    from ..anthropic_compat import AnthropicMessagesRequest, AnthropicCountTokensRequest

lib_logger = logging.getLogger("rotator_library")


class RotatingClient:
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.

    This is a slim facade that delegates to specialized components:
    - RequestExecutor: Handles retry/rotation logic
    - CredentialFilter: Filters credentials by tier
    - ModelResolver: Resolves model names
    - ProviderTransforms: Applies provider-specific transforms
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, List[str]]] = None,
        oauth_credentials: Optional[Dict[str, List[str]]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        usage_file_path: Optional[Union[str, Path]] = None,
        configure_logging: bool = True,
        global_timeout: int = DEFAULT_GLOBAL_TIMEOUT,
        abort_on_callback_error: bool = True,
        litellm_provider_params: Optional[Dict[str, Any]] = None,
        ignore_models: Optional[Dict[str, List[str]]] = None,
        whitelist_models: Optional[Dict[str, List[str]]] = None,
        enable_request_logging: bool = False,
        max_concurrent_requests_per_key: Optional[Dict[str, int]] = None,
        rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the RotatingClient.

        See original client.py for full parameter documentation.
        """
        # Resolve data directory
        self.data_dir = Path(data_dir).resolve() if data_dir else get_default_root()

        # Configure logging
        configure_failure_logger(get_logs_dir(self.data_dir))
        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.set_verbose = False
        litellm.drop_params = True
        suppress_litellm_serialization_warnings()

        if configure_logging:
            lib_logger.propagate = True
            if lib_logger.hasHandlers():
                lib_logger.handlers.clear()
                lib_logger.addHandler(logging.NullHandler())
        else:
            lib_logger.propagate = False

        # Process credentials
        api_keys = api_keys or {}
        oauth_credentials = oauth_credentials or {}
        api_keys = {p: k for p, k in api_keys.items() if k}
        oauth_credentials = {p: c for p, c in oauth_credentials.items() if c}

        if not api_keys and not oauth_credentials:
            lib_logger.warning(
                "No provider credentials configured. Client will be unable to make requests."
            )

        # Discover OAuth credentials if not provided
        if oauth_credentials:
            self.oauth_credentials = oauth_credentials
        else:
            cred_manager = CredentialManager(
                os.environ, oauth_dir=get_oauth_dir(self.data_dir)
            )
            self.oauth_credentials = cred_manager.discover_and_prepare()

        # Build combined credentials
        self.all_credentials: Dict[str, List[str]] = {}
        for provider, keys in api_keys.items():
            self.all_credentials.setdefault(provider, []).extend(keys)
        for provider, paths in self.oauth_credentials.items():
            self.all_credentials.setdefault(provider, []).extend(paths)

        self.api_keys = api_keys
        self.oauth_providers = set(self.oauth_credentials.keys())

        # Store configuration
        self.max_retries = max_retries
        self.global_timeout = global_timeout
        self.abort_on_callback_error = abort_on_callback_error
        self.litellm_provider_params = litellm_provider_params or {}
        self._litellm_logger_fn = self._litellm_logger_callback
        self.enable_request_logging = enable_request_logging
        self.max_concurrent_requests_per_key = max_concurrent_requests_per_key or {}

        # Validate concurrent requests config
        for provider, max_val in self.max_concurrent_requests_per_key.items():
            if max_val < 1:
                lib_logger.warning(
                    f"Invalid max_concurrent for '{provider}': {max_val}. Setting to 1."
                )
                self.max_concurrent_requests_per_key[provider] = 1

        # Initialize configuration loader
        self._config_loader = ConfigLoader(PROVIDER_PLUGINS)

        # Initialize components
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances: Dict[str, Any] = {}

        # Initialize managers
        self.cooldown_manager = CooldownManager()
        self.background_refresher = BackgroundRefresher(self)
        self.model_definitions = ModelDefinitions()
        self.provider_config = LiteLLMProviderConfig()
        self.http_client = httpx.AsyncClient()

        # Initialize extracted components
        self._credential_filter = CredentialFilter(
            PROVIDER_PLUGINS,
            provider_instances=self._provider_instances,
        )
        self._model_resolver = ModelResolver(
            PROVIDER_PLUGINS,
            self.model_definitions,
            ignore_models or {},
            whitelist_models or {},
            provider_instances=self._provider_instances,
        )
        self._provider_transforms = ProviderTransforms(
            PROVIDER_PLUGINS,
            self.provider_config,
            provider_instances=self._provider_instances,
        )

        # Initialize UsageManagers (one per provider) using new usage package
        self._usage_managers: Dict[str, NewUsageManager] = {}

        # Resolve usage file path base
        if usage_file_path:
            base_path = Path(usage_file_path)
            if base_path.suffix:
                base_path = base_path.parent
            self._usage_base_path = base_path / "usage"
        else:
            self._usage_base_path = self.data_dir / "usage"
        self._usage_base_path.mkdir(parents=True, exist_ok=True)

        # Build provider configs using ConfigLoader
        provider_configs = {}
        for provider in self.all_credentials.keys():
            provider_configs[provider] = self._config_loader.load_provider_config(
                provider
            )

        # Create UsageManager for each provider
        for provider, credentials in self.all_credentials.items():
            config = load_provider_usage_config(provider, PROVIDER_PLUGINS)
            # Override tolerance from constructor param
            config.rotation_tolerance = rotation_tolerance

            self._apply_usage_reset_config(provider, credentials, config)

            usage_file = self._usage_base_path / f"usage_{provider}.json"

            # Get max concurrent for this provider
            # Per-provider env var MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER> takes priority
            # Falls back to global MAX_CONCURRENT_REQUESTS_PER_KEY (default 1)
            default_max = int(os.getenv("MAX_CONCURRENT_REQUESTS_PER_KEY", "1"))
            max_concurrent = self.max_concurrent_requests_per_key.get(provider, default_max)

            manager = NewUsageManager(
                provider=provider,
                file_path=usage_file,
                provider_plugins=PROVIDER_PLUGINS,
                config=config,
                max_concurrent_per_key=max_concurrent,
            )
            self._usage_managers[provider] = manager

        # Initialize executor with new usage managers
        self._executor = RequestExecutor(
            usage_managers=self._usage_managers,
            cooldown_manager=self.cooldown_manager,
            credential_filter=self._credential_filter,
            provider_transforms=self._provider_transforms,
            provider_plugins=PROVIDER_PLUGINS,
            http_client=self.http_client,
            max_retries=max_retries,
            global_timeout=global_timeout,
            abort_on_callback_error=abort_on_callback_error,
            litellm_provider_params=self.litellm_provider_params,
            litellm_logger_fn=self._litellm_logger_fn,
            provider_instances=self._provider_instances,
        )

        self._model_list_cache: Dict[str, List[str]] = {}
        self._model_list_cache_time: Dict[str, float] = {}
        self.MODEL_LIST_CACHE_TTL = 6 * 60 * 60  # 6 hours
        self._usage_initialized = False
        self._usage_init_lock = asyncio.Lock()



        # Initialize smart "latest" model alias registry
        self._latest_registry = ModelLatestRegistry()
        if self._latest_registry.has_rules():
            self._latest_registry.set_pricing_resolver(self._pricing_resolver_callback)

        # Initialize cross-provider model alias registry
        self._alias_registry = ModelAliasRegistry()
        self._cross_provider_executor = CrossProviderExecutor(
            client=self,
            alias_registry=self._alias_registry,
        )

        # Initialize Anthropic compatibility handler
        self._anthropic_handler = AnthropicHandler(self)

    async def __aenter__(self):
        await self.initialize_usage_managers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize_usage_managers(self) -> None:
        """Initialize usage managers once before background jobs run."""
        if self._usage_initialized:
            return
        async with self._usage_init_lock:
            if self._usage_initialized:
                return
            for provider, manager in self._usage_managers.items():
                credentials = self.all_credentials.get(provider, [])
                priorities, tiers = self._get_credential_metadata(provider, credentials)
                await manager.initialize(
                    credentials, priorities=priorities, tiers=tiers
                )
            summaries = []
            for provider, manager in self._usage_managers.items():
                credentials = self.all_credentials.get(provider, [])
                status = (
                    f"loaded {manager.loaded_credentials}"
                    if manager.loaded_from_storage
                    else "fresh"
                )
                summaries.append(f"{provider}:{len(credentials)} ({status})")
            if summaries:
                lib_logger.info(
                    f"Usage managers initialized: {', '.join(sorted(summaries))}"
                )

            # Inject usage manager references into providers that support it
            # (e.g., CodexProvider via CodexQuotaTracker for header-based quota updates)
            for provider, manager in self._usage_managers.items():
                instance = self._get_provider_instance(provider)
                if instance and hasattr(instance, "set_usage_manager"):
                    instance.set_usage_manager(manager)

            self._usage_initialized = True

    async def close(self):
        """Close the HTTP client and save usage data."""
        # Save and shutdown new usage managers
        for manager in self._usage_managers.values():
            await manager.shutdown()

        if hasattr(self, "http_client") and self.http_client:
            await self.http_client.aclose()

    async def acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        """
        Dispatcher for completion requests.

        Routes to the provider specified in the provider/model format.

        Supports two routing modes:
        1. provider/model format: routed directly to the specified provider
        2. unprefixed model name: if it matches a registered alias, routed
           across multiple providers via CrossProviderExecutor

        Returns:
            Response object or async generator for streaming
        """
        model = kwargs.get("model", "")
        provider = model.split("/")[0] if "/" in model else ""

        # Check if this is an unprefixed alias model
        if not provider or provider not in self.all_credentials:
            # Try cross-provider alias resolution
            alias_targets = self._alias_registry.resolve(model)
            if alias_targets:
                lib_logger.info(
                    f"Model '{model}' matched alias → routing across "
                    f"{len(alias_targets)} providers"
                )
                return await self._cross_provider_executor.execute(
                    canonical_model=model,
                    targets=alias_targets,
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **kwargs,
                )

            raise ValueError(
                f"Invalid model format or no credentials for provider: {model}"
            )

        # Standard single-provider path (unchanged)
        # Extract internal logging parameters (not passed to API)
        parent_log_dir = kwargs.pop("_parent_log_dir", None)

        # Resolve model ID
        resolved_model = self._model_resolver.resolve_model_id(model, provider)
        kwargs["model"] = resolved_model

        # Create transaction logger if enabled
        transaction_logger = None
        if self.enable_request_logging:
            transaction_logger = TransactionLogger(
                provider=provider,
                model=resolved_model,
                enabled=True,
                parent_dir=parent_log_dir,
            )
            transaction_logger.log_request(kwargs)

        # Build request context
        context = RequestContext(
            model=resolved_model,
            provider=provider,
            kwargs=kwargs,
            streaming=kwargs.get("stream", False),
            credentials=self.all_credentials.get(provider, []),
            deadline=time.time() + self.global_timeout,
            request=request,
            pre_request_callback=pre_request_callback,
            transaction_logger=transaction_logger,
        )

        return await self._executor.execute(context)

    def aembedding(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """
        Execute an embedding request with retry logic.
        """
        model = kwargs.get("model", "")
        provider = model.split("/")[0] if "/" in model else ""

        if not provider or provider not in self.all_credentials:
            raise ValueError(
                f"Invalid model format or no credentials for provider: {model}"
            )

        # Build request context (embeddings are never streaming)
        context = RequestContext(
            model=model,
            provider=provider,
            kwargs=kwargs,
            streaming=False,
            request_type="embedding",
            credentials=self.all_credentials.get(provider, []),
            deadline=time.time() + self.global_timeout,
            request=request,
            pre_request_callback=pre_request_callback,
        )

        return self._executor.execute(context)

    def token_count(self, **kwargs) -> int:
        """Calculate token count for text or messages.

        For Antigravity provider models, this also includes the preprompt tokens
        that get injected during actual API calls (agent instruction + identity override).
        This ensures token counts match actual usage.
        """
        model = kwargs.get("model")
        text = kwargs.get("text")
        messages = kwargs.get("messages")

        if not model:
            raise ValueError("'model' is required")

        # Calculate base token count
        if messages:
            base_count = token_counter(model=model, messages=messages)
        elif text:
            base_count = token_counter(model=model, text=text)
        else:
            raise ValueError("Either 'text' or 'messages' must be provided")

        # Add preprompt tokens for Antigravity provider
        # The Antigravity provider injects system instructions during actual API calls,
        # so we need to account for those tokens in the count
        provider = model.split("/")[0] if "/" in model else ""
        if provider == "antigravity":
            try:
                from ..providers.antigravity_provider import (
                    get_antigravity_preprompt_text,
                )

                preprompt_text = get_antigravity_preprompt_text()
                if preprompt_text:
                    preprompt_tokens = token_counter(model=model, text=preprompt_text)
                    base_count += preprompt_tokens
            except ImportError:
                # Provider not available, skip preprompt token counting
                pass

        return base_count

    async def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a provider with caching."""
        now = time.time()
        if provider in self._model_list_cache:
            age = now - self._model_list_cache_time.get(provider, 0)
            if age < self.MODEL_LIST_CACHE_TTL:
                return self._model_list_cache[provider]
            # Stale cache — evict so we re-fetch
            del self._model_list_cache[provider]
            self._model_list_cache_time.pop(provider, None)

        credentials = self.all_credentials.get(provider, [])
        if not credentials:
            return []

        # Shuffle and try each credential
        shuffled = list(credentials)
        random.shuffle(shuffled)

        plugin = self._get_provider_instance(provider)
        if not plugin:
            return []

        for cred in shuffled:
            try:
                models = await plugin.get_models(cred, self.http_client)

                # Apply whitelist/blacklist
                final = [
                    m
                    for m in models
                    if self._model_resolver.is_model_allowed(m, provider)
                ]

                self._model_list_cache[provider] = final
                self._model_list_cache_time[provider] = now
                # Underlying model catalog changed — invalidate alias resolutions
                self._latest_registry.clear_resolution_cache()
                return final

            except Exception as e:
                lib_logger.debug(
                    f"Failed to get models for {provider} with {mask_credential(cred)}: {e}"
                )
                continue

        return []

    async def get_all_available_models(
        self,
        grouped: bool = True,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get all available models across all providers."""
        providers = list(self.all_credentials.keys())
        tasks = [self.get_available_models(p) for p in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_models: Dict[str, List[str]] = {}
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                lib_logger.error(f"Failed to get models for {provider}: {result}")
                all_models[provider] = []
            else:
                all_models[provider] = result

        if grouped:
            return all_models
        else:
            flat = []
            for models in all_models.values():
                flat.extend(models)
            return flat

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get quota and usage stats for all credentials.

        Args:
            provider_filter: Optional provider name to filter results

        Returns:
            Dict with stats per provider
        """
        providers = {}

        for provider, manager in self._usage_managers.items():
            if provider_filter and provider != provider_filter:
                continue

            stats = await manager.get_stats_for_endpoint()


            # Filter out stale quota groups that no longer exist in the provider's
            # current model_quota_groups (e.g. after a group rename like
            # firmware_global → credits($))
            plugin = self._get_provider_instance(provider)
            if plugin and hasattr(plugin, "model_quota_groups"):
                valid_groups = set(plugin.model_quota_groups.keys())
                if valid_groups:
                    stale_groups = [
                        g
                        for g in stats.get("quota_groups", {})
                        if g not in valid_groups
                    ]
                    for g in stale_groups:
                        del stats["quota_groups"][g]
                    # Also clean stale groups from per-credential group_usage
                    for cred_data in stats.get("credentials", {}).values():
                        for g in stale_groups:
                            cred_data.get("group_usage", {}).pop(g, None)


            # Skip providers with no activity AND no quota data
            # (filters out invalid/unused providers, but keeps quota-tracked providers visible)
            has_requests = stats.get("total_requests", 0) > 0
            has_quota_data = any(
                any(
                    ws.get("total_max", 0) > 0
                    for ws in group_stats.get("windows", {}).values()
                )
                for group_stats in stats.get("quota_groups", {}).values()
            )
            if not has_requests and not has_quota_data:
                continue

            providers[provider] = stats

        def _build_summary(providers_data, source_key=None):
            """Build a summary dict from provider data.

            Args:
                providers_data: Dict of provider stats
                source_key: If set, read stats from this sub-key of each
                    provider (e.g. "current_period"). If None, read from
                    the provider-level fields directly (lifetime/totals).
            """
            s = {
                "total_providers": len(providers_data),
                "total_credentials": 0,
                "active_credentials": 0,
                "exhausted_credentials": 0,
                "total_requests": 0,
                "tokens": {
                    "input_cached": 0,
                    "input_uncached": 0,
                    "input_cache_pct": 0,
                    "output": 0,
                },
                "approx_total_cost": None,
            }

            approx_total_cost = 0.0
            has_cost = False

            for prov in providers_data.values():
                s["total_credentials"] += prov.get("credential_count", 0)
                s["active_credentials"] += prov.get("active_count", 0)
                s["exhausted_credentials"] += prov.get("exhausted_count", 0)

                if source_key:
                    src = prov.get(source_key, {})
                    s["total_requests"] += src.get("total_requests", 0)
                    tokens = src.get("tokens", {})
                    cost = src.get("approx_cost")
                else:
                    s["total_requests"] += prov.get("total_requests", 0)
                    tokens = prov.get("tokens", {})
                    cost = prov.get("approx_cost")

                s["tokens"]["input_cached"] += tokens.get("input_cached", 0)
                s["tokens"]["input_uncached"] += tokens.get("input_uncached", 0)
                s["tokens"]["output"] += tokens.get("output", 0)

                if cost:
                    approx_total_cost += cost
                    has_cost = True

            total_input = (
                s["tokens"]["input_cached"] + s["tokens"]["input_uncached"]
            )
            s["tokens"]["input_cache_pct"] = (
                round(s["tokens"]["input_cached"] / total_input * 100, 1)
                if total_input > 0
                else 0
            )
            s["approx_total_cost"] = approx_total_cost if has_cost else None
            return s

        # summary = current period stats (from primary window)
        summary = _build_summary(providers, source_key="current_period")

        # global_summary = lifetime/all-time stats (from totals)
        global_summary = _build_summary(providers)

        return {
            "providers": providers,
            "summary": summary,
            "global_summary": global_summary,
            "data_source": "cache",
            "timestamp": time.time(),
        }

    def get_oauth_credentials(self) -> Dict[str, List[str]]:
        """Get discovered OAuth credentials."""
        return self.oauth_credentials

    def _get_provider_instance(self, provider: str) -> Optional[Any]:
        """Get or create a provider plugin instance."""
        if provider not in self.all_credentials:
            return None

        if provider not in self._provider_instances:
            plugin_class = self._provider_plugins.get(provider)
            if plugin_class:
                self._provider_instances[provider] = plugin_class()
            else:
                return None

        return self._provider_instances[provider]

    def _get_credential_metadata(
        self,
        provider: str,
        credentials: List[str],
    ) -> tuple[Dict[str, int], Dict[str, str]]:
        """Resolve priority and tier metadata for credentials."""
        plugin = self._get_provider_instance(provider)
        priorities: Dict[str, int] = {}
        tiers: Dict[str, str] = {}

        if not plugin:
            return priorities, tiers

        for credential in credentials:
            if hasattr(plugin, "get_credential_priority"):
                priority = plugin.get_credential_priority(credential)
                if priority is not None:
                    priorities[credential] = priority
            if hasattr(plugin, "get_credential_tier_name"):
                tier_name = plugin.get_credential_tier_name(credential)
                if tier_name:
                    tiers[credential] = tier_name

        return priorities, tiers

    def get_usage_manager(self, provider: str) -> Optional[NewUsageManager]:
        """
        Get the new UsageManager for a specific provider.

        Args:
            provider: Provider name

        Returns:
            UsageManager for the provider, or None if not found
        """
        return self._usage_managers.get(provider)

    @property
    def usage_managers(self) -> Dict[str, NewUsageManager]:
        """Get all new usage managers."""
        return self._usage_managers

    @property
    def latest_registry(self) -> "ModelLatestRegistry":
        """Get the smart 'latest' model alias registry."""
        return self._latest_registry

    def resolve_latest(self, model: str) -> Optional[str]:
        """Try to resolve a 'latest' model alias using cached model lists."""
        if not self._latest_registry.has_rules():
            return None
        return self._latest_registry.resolve(model, self._model_list_cache)

    async def resolve_latest_async(self, model: str) -> Optional[str]:
        """Resolve a 'latest' alias, warming the model cache if needed.

        Unlike resolve_latest(), this will fetch the provider's model list
        on-demand if the cache is cold (e.g. right after a container restart).
        """
        if not self._latest_registry.has_rules():
            return None

        # Try with current cache first
        resolved = self._latest_registry.resolve(model, self._model_list_cache)
        if resolved:
            return resolved

        # If this is a known alias but cache is empty, warm it
        if self._latest_registry.is_latest_alias(model):
            rule = self._latest_registry.get_all_rules().get(model.lower())
            if rule and rule.provider not in self._model_list_cache:
                lib_logger.info(
                    f"Latest alias '{model}': warming model cache for "
                    f"provider '{rule.provider}'"
                )
                await self.get_available_models(rule.provider)
                return self._latest_registry.resolve(
                    model, self._model_list_cache
                )

        return None

    def _pricing_resolver_callback(
        self, provider: str, model_id: str
    ) -> Optional[float]:
        """
        Pricing resolver callback for cost-based tiebreaking in latest aliases.

        Checks provider-specific pricing cache first, then falls back to the
        global ModelRegistry (ModelInfoService).
        """
        # 1. Try provider-specific pricing cache (e.g., ChutesProvider._pricing_cache)
        plugin = self._provider_instances.get(provider)
        if plugin and hasattr(plugin, "_pricing_cache"):
            # Strip org prefix for cache lookup
            bare_name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
            pricing = plugin._pricing_cache.get(bare_name) or plugin._pricing_cache.get(
                model_id
            )
            if pricing:
                return pricing.get("input", 0.0)

        # 2. Fall back to global ModelRegistry (ModelInfoService)
        try:
            from ..model_info_service import get_model_info_service

            registry = get_model_info_service()
            if registry.is_ready:
                pricing = registry.get_pricing(f"{provider}/{model_id}")
                if pricing:
                    return pricing.get("input_cost_per_token")
        except Exception:
            pass

        return None

    @property
    def alias_registry(self) -> "ModelAliasRegistry":
        """Get the model alias registry for cross-provider routing."""
        return self._alias_registry

    def _apply_usage_reset_config(
        self,
        provider: str,
        credentials: List[str],
        config: Any,
    ) -> None:
        """Apply provider-specific usage reset config to window definitions."""
        if not credentials:
            return

        plugin = self._get_provider_instance(provider)
        if not plugin or not hasattr(plugin, "get_usage_reset_config"):
            return

        try:
            reset_config = plugin.get_usage_reset_config(credentials[0])
        except Exception as exc:
            lib_logger.debug(f"Failed to load usage reset config for {provider}: {exc}")
            return

        if not reset_config:
            return

        window_seconds = reset_config.get("window_seconds")
        if not window_seconds:
            return

        mode = reset_config.get("mode", "credential")
        applies_to = "credential" if mode == "credential" else "model"

        if window_seconds == 86400:
            window_name = "daily"
        elif window_seconds % 3600 == 0:
            window_name = f"{window_seconds // 3600}h"
        else:
            window_name = "window"

        config.windows = [
            WindowDefinition.rolling(
                name=window_name,
                duration_seconds=int(window_seconds),
                is_primary=True,
                applies_to=applies_to,
            ),
        ]

    def _sanitize_litellm_log(self, log_data: dict) -> dict:
        """Remove large/sensitive fields from LiteLLM logs."""
        if not isinstance(log_data, dict):
            return log_data

        keys_to_pop = [
            "messages",
            "input",
            "response",
            "data",
            "api_key",
            "api_base",
            "original_response",
            "additional_args",
        ]
        nested_keys = ["kwargs", "litellm_params", "model_info", "proxy_server_request"]

        clean_data = json.loads(json.dumps(log_data, default=str))

        def clean_recursively(data_dict: dict) -> None:
            for key in keys_to_pop:
                data_dict.pop(key, None)
            for key in nested_keys:
                if key in data_dict and isinstance(data_dict[key], dict):
                    clean_recursively(data_dict[key])
            for value in list(data_dict.values()):
                if isinstance(value, dict):
                    clean_recursively(value)

        clean_recursively(clean_data)
        return clean_data

    def _litellm_logger_callback(self, log_data: dict) -> None:
        """Redirect LiteLLM logs into rotator library logger."""
        log_event_type = log_data.get("log_event_type")
        if log_event_type in ["pre_api_call", "post_api_call"]:
            return

        if not log_data.get("exception"):
            sanitized_log = self._sanitize_litellm_log(log_data)
            lib_logger.debug(f"LiteLLM Log: {sanitized_log}")
            return

        model = log_data.get("model", "N/A")
        error_info = log_data.get("standard_logging_object", {}).get(
            "error_information", {}
        )
        error_class = error_info.get("error_class", "UnknownError")
        error_message = error_info.get(
            "error_message", str(log_data.get("exception", ""))
        )
        error_message = " ".join(error_message.split())

        lib_logger.debug(
            f"LiteLLM Callback Handled Error: Model={model} | "
            f"Type={error_class} | Message='{error_message}'"
        )

    # =========================================================================
    # USAGE MANAGEMENT METHODS
    # =========================================================================

    async def reload_usage_from_disk(self) -> None:
        """
        Force reload usage data from disk.

        Useful when wanting fresh stats without making external API calls.
        """
        for manager in self._usage_managers.values():
            await manager.reload_from_disk()

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Force refresh quota from external API.

        For Antigravity, this fetches live quota data from the API.
        For other providers, this is a no-op (just reloads from disk).

        Args:
            provider: If specified, only refresh this provider
            credential: If specified, only refresh this specific credential

        Returns:
            Refresh result dict with success/failure info
        """
        result = {
            "action": "force_refresh",
            "scope": "credential"
            if credential
            else ("provider" if provider else "all"),
            "provider": provider,
            "credential": credential,
            "credentials_refreshed": 0,
            "success_count": 0,
            "failed_count": 0,
            "duration_ms": 0,
            "errors": [],
        }

        start_time = time.time()

        # Determine which providers to refresh
        if provider:
            providers_to_refresh = (
                [provider] if provider in self.all_credentials else []
            )
        else:
            providers_to_refresh = list(self.all_credentials.keys())

        for prov in providers_to_refresh:
            provider_class = self._provider_plugins.get(prov)
            if not provider_class:
                continue

            # Get or create provider instance
            provider_instance = self._get_provider_instance(prov)
            if not provider_instance:
                continue

            # Check if provider supports quota refresh (like Antigravity)
            if hasattr(provider_instance, "fetch_initial_baselines"):
                # Get credentials to refresh
                if credential:
                    # Find full path for this credential
                    creds_to_refresh = []
                    for cred_path in self.all_credentials.get(prov, []):
                        if cred_path.endswith(credential) or cred_path == credential:
                            creds_to_refresh.append(cred_path)
                            break
                else:
                    creds_to_refresh = self.all_credentials.get(prov, [])

                if not creds_to_refresh:
                    continue

                try:
                    # Fetch live quota from API for ALL specified credentials
                    quota_results = await provider_instance.fetch_initial_baselines(
                        creds_to_refresh
                    )

                    # Store baselines in usage manager
                    usage_manager = self._usage_managers.get(prov)
                    if usage_manager and hasattr(
                        provider_instance, "_store_baselines_to_usage_manager"
                    ):
                        stored = await provider_instance._store_baselines_to_usage_manager(
                            quota_results,
                            usage_manager,
                            force=True,
                            is_initial_fetch=True,  # Manual refresh checks exhaustion
                        )
                        result["success_count"] += stored

                    result["credentials_refreshed"] += len(creds_to_refresh)

                    # Count failures
                    for cred_path, data in quota_results.items():
                        if data.get("status") != "success":
                            result["failed_count"] += 1
                            result["errors"].append(
                                f"{Path(cred_path).name}: {data.get('error', 'Unknown error')}"
                            )

                except Exception as e:
                    lib_logger.error(f"Failed to refresh quota for {prov}: {e}")
                    result["errors"].append(f"{prov}: {str(e)}")
                    result["failed_count"] += len(creds_to_refresh)

        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result

    # =========================================================================
    # ANTHROPIC API COMPATIBILITY METHODS
    # =========================================================================

    async def anthropic_messages(
        self,
        request: "AnthropicMessagesRequest",
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
    ) -> Any:
        """
        Handle Anthropic Messages API requests.

        This method accepts requests in Anthropic's format, translates them to
        OpenAI format internally, processes them through the existing acompletion
        method, and returns responses in Anthropic's format.

        Args:
            request: An AnthropicMessagesRequest object
            raw_request: Optional raw request object for disconnect checks
            pre_request_callback: Optional async callback before each API request

        Returns:
            For non-streaming: dict in Anthropic Messages format
            For streaming: AsyncGenerator yielding Anthropic SSE format strings
        """
        return await self._anthropic_handler.messages(
            request=request,
            raw_request=raw_request,
            pre_request_callback=pre_request_callback,
        )

    async def anthropic_count_tokens(
        self,
        request: "AnthropicCountTokensRequest",
    ) -> dict:
        """
        Handle Anthropic count_tokens API requests.

        Counts the number of tokens that would be used by a Messages API request.
        This is useful for estimating costs and managing context windows.

        Args:
            request: An AnthropicCountTokensRequest object

        Returns:
            Dict with input_tokens count in Anthropic format
        """
        return await self._anthropic_handler.count_tokens(request=request)
