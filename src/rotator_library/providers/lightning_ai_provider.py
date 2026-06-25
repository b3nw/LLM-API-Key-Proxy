# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Lightning AI Provider

Provider for Lightning AI (https://lightning.ai).
OpenAI-compatible API with dollar-based monthly credit quota tracking.

Features:
- Dynamic model discovery from /api/v1/models endpoint
- Environment variable model override (LIGHTNING_AI_MODELS)
- Monthly credit balance monitoring via /v1/memberships
- Balance tracked in cents (dollars × 100) for integer quota compatibility

Quota system:
Lightning AI uses a dollar-based credit balance that reloads monthly.
The balance is fetched from the memberships API and converted to integer
cents so the UsageManager can track it with its standard quota machinery.
All models share the same credential-level balance pool.

Environment variables:
    LIGHTNING_AI_API_BASE=https://lightning.ai/api/v1
    LIGHTNING_AI_API_KEY_1=<uuid>
    LIGHTNING_AI_API_KEY_2=<uuid>
    LIGHTNING_AI_QUOTA_REFRESH_INTERVAL=300   # optional, seconds
"""

import asyncio
import httpx
import os
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, TYPE_CHECKING

import litellm
import openai

if TYPE_CHECKING:
    from ..usage import UsageManager

from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.lightning_ai_quota_tracker import LightningAiQuotaTracker
from ..error_handler import mask_credential
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

# Lightning AI API base URL (OpenAI-compatible endpoint)
LIGHTNING_AI_API_BASE = "https://lightning.ai/api/v1"

# Concurrency limit for parallel balance fetches
BALANCE_FETCH_CONCURRENCY = 5

# Default monthly free credit grant per account (dollars).
# Lightning AI plan tiers:
#   free:  $15/month
#   pro:   $20/month
#   teams: $50/month
# Override with LIGHTNING_AI_MONTHLY_GRANT (in whole dollars) if on a paid plan.
DEFAULT_MONTHLY_GRANT_DOLLARS = 15

# Parameters accepted by the OpenAI SDK's chat.completions.create().
# Lightning AI is an OpenAI-compatible endpoint, so we bypass litellm
# entirely to avoid the responses_api_bridge_check() that routes GPT-5
# models with tools + reasoning to the /responses endpoint (405).
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "reasoning_effort",
    "extra_headers",
    "extra_body",
    "user",
}


class LightningAiProvider(LightningAiQuotaTracker, ProviderInterface):
    """
    Provider for Lightning AI API.

    Supports dollar-based monthly credit quota tracking.
    All models share the same credential-level balance pool.
    """

    # Skip LiteLLM cost calculation — Lightning AI provides per-model pricing
    # in the /v1/models response; cost is tracked via the balance API instead.
    skip_cost_calculation = True

    # =========================================================================
    # PROVIDER CONFIGURATION
    # =========================================================================

    provider_env_name = "lightning_ai"

    # Single quota group: all models share the same monthly credit balance.
    # Named 'credits($)' so the TUI shows a human-readable dollar label.
    model_quota_groups = {
        "credits($)": ["_balance"],
    }

    # Monthly rolling window — credits reload on the nextFreeCreditsGrant date
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=2592000,  # ~30 days
            mode="per_model",
            description="Lightning AI monthly credit balance",
            field_name="monthly",
        )
    }

    def __init__(self):
        self.model_definitions = ModelDefinitions()

        # Balance cache: credential_identifier → balance data dict
        self._balance_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = int(
            os.getenv("LIGHTNING_AI_QUOTA_REFRESH_INTERVAL", "300")
        )
        # Monthly grant in cents (×100 scale) for accurate TUI display.
        # The TUI detects the ($) suffix in 'credits($)' and formats as dollars:
        #   e.g. 1485/1500 → displays as $14.85/$15.00
        # Lightning AI tiers: free=$15, pro=$20, teams=$50
        # Set LIGHTNING_AI_MONTHLY_GRANT (whole dollars) to match your plan.
        grant_dollars = int(
            os.getenv("LIGHTNING_AI_MONTHLY_GRANT") or DEFAULT_MONTHLY_GRANT_DOLLARS
        )
        self._monthly_grant_cents: int = grant_dollars * 100  # e.g. 15 → 1500

    # =========================================================================
    # USAGE TRACKING CONFIGURATION
    # =========================================================================

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Lightning AI credentials.

        Uses per_model mode with a monthly window to match the credit reload cycle.
        """
        return {
            "mode": "per_model",
            "window_seconds": 2592000,  # ~30 days
        }

    # =========================================================================
    # QUOTA GROUPING
    # =========================================================================

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        All Lightning AI models share the same monthly credit balance pool.

        Args:
            model: Model name (ignored — all models share one balance)

        Returns:
            Quota group name
        """
        return "credits($)"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Return all models belonging to the given quota group.

        Args:
            group: Quota group identifier

        Returns:
            List of model names in the group
        """
        if group == "credits($)":
            return ["_balance"]
        return []

    def get_quota_groups(self) -> List[str]:
        """Return the list of quota groups for this provider."""
        return ["credits($)"]

    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return Lightning AI models from:
        1. Environment variable (LIGHTNING_AI_MODELS) — priority
        2. Dynamic discovery from API
        3. Empty list if both fail
        """
        models: List[str] = []
        seen_ids: set = set()

        # Source 1: Static model definitions via LIGHTNING_AI_MODELS env var
        static_models = self.model_definitions.get_all_provider_models("lightning_ai")
        if static_models:
            for model in static_models:
                model_id = model.split("/")[-1] if "/" in model else model
                models.append(model)
                seen_ids.add(model_id)
            lib_logger.debug(
                f"Loaded {len(static_models)} static models for lightning_ai"
            )

        # Source 2: Dynamic discovery from the OpenAI-compatible /models endpoint
        api_base = os.getenv("LIGHTNING_AI_API_BASE", LIGHTNING_AI_API_BASE)
        try:
            response = await client.get(
                f"{api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            dynamic_count = 0
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                if model_id and model_id not in seen_ids:
                    models.append(f"lightning_ai/{model_id}")
                    seen_ids.add(model_id)
                    dynamic_count += 1

            if dynamic_count > 0:
                lib_logger.debug(
                    f"Discovered {dynamic_count} models for lightning_ai from API"
                )

        except Exception as e:
            lib_logger.debug(f"Dynamic model discovery failed for lightning_ai: {e}")

        return models

    # =========================================================================
    # CUSTOM COMPLETION LOGIC
    # =========================================================================

    def has_custom_logic(self) -> bool:
        """
        Lightning AI bypasses litellm's standard completion path.

        litellm 1.85+ has ``responses_api_bridge_check()`` that automatically
        routes GPT-5.4+ models to the Responses API (``/responses`` endpoint)
        when ``reasoning_effort`` and ``tools`` are present.  Lightning AI only
        supports ``/chat/completions``, so this bridge produces a 405 error.

        By returning ``True``, the executor calls our ``acompletion()``
        directly, which uses the OpenAI SDK to call ``/chat/completions``
        without litellm's internal routing.
        """
        return True

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[
        litellm.ModelResponse,
        AsyncGenerator[litellm.ModelResponse, None],
    ]:
        """
        Make a chat completion request directly to Lightning AI's API.

        Uses the OpenAI SDK instead of litellm to avoid the
        ``responses_api_bridge_check()`` that would route GPT-5 models with
        tools + reasoning to the ``/responses`` endpoint (unsupported → 405).

        The OpenAI SDK calls ``/chat/completions`` directly, and the response
        objects are compatible with the executor's duck-typed usage extraction
        (``hasattr(response, "usage")`` / ``getattr(response.usage, ...)``).
        """
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        # Strip lightning_ai/ prefix to get the bare model name
        model_bare = model.split("/", 1)[1] if "/" in model else model

        api_base = os.getenv("LIGHTNING_AI_API_BASE", LIGHTNING_AI_API_BASE)

        # Normalize reasoning param: the proxy may pass ``reasoning`` as a dict
        # (e.g. {"effort": "medium"}) from the /v1/responses format.
        reasoning = kwargs.pop("reasoning", None)
        if reasoning and isinstance(reasoning, dict) and "effort" in reasoning:
            kwargs.setdefault("reasoning_effort", reasoning["effort"])
        elif reasoning and isinstance(reasoning, str):
            kwargs.setdefault("reasoning_effort", reasoning)

        # Normalize thinking param (Anthropic-style) to OpenAI reasoning_effort.
        # Lightning AI uses the OpenAI-compatible reasoning_effort parameter,
        # not Anthropic's thinking field.  Map enabled → high as a reasonable
        # default.  The _guard_thinking_tool_calls transform may inject
        # extra_body: {"thinking": {"type": "disabled"}} for tool-call turns
        # missing reasoning_content — honor that by not enabling reasoning.
        thinking_enabled = False

        # Check top-level thinking param (from client)
        thinking = kwargs.pop("thinking", None)
        if thinking and isinstance(thinking, dict) and thinking.get("type") == "enabled":
            thinking_enabled = True

        # Check extra_body.thinking (may be injected by _guard_thinking_tool_calls)
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict) and "thinking" in extra_body:
            thinking_extra = extra_body.pop("thinking")
            if not extra_body:
                del kwargs["extra_body"]
            if isinstance(thinking_extra, dict) and thinking_extra.get("type") == "disabled":
                thinking_enabled = False  # Guard takes precedence

        if thinking_enabled:
            kwargs.setdefault("reasoning_effort", "high")

        # Create OpenAI client pointing at Lightning AI's endpoint.
        # max_retries=0 matches the executor's litellm path (executor.py:774,
        # 1056) — the outer retry loop owns retry policy, not the SDK.
        openai_client = openai.AsyncOpenAI(
            api_key=credential,
            base_url=api_base,
            http_client=client,
            max_retries=0,
        )

        # Filter to supported params only — drop internal/litellm-specific keys
        unsupported = set(kwargs.keys()) - SUPPORTED_PARAMS
        if unsupported:
            lib_logger.debug(
                f"lightning_ai: stripping unsupported params for "
                f"{mask_credential(model)}: {unsupported}"
            )
        call_kwargs = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}
        call_kwargs["model"] = model_bare

        return await openai_client.chat.completions.create(**call_kwargs)

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Configure periodic credit balance refresh."""
        return {
            "interval": self._quota_refresh_interval,
            "name": "lightning_ai_balance_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh credit balance for all credentials in parallel.

        Converts the dollar balance to integer cents and pushes it to the
        UsageManager as a quota baseline for the virtual "_balance" model.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys
        """
        semaphore = asyncio.Semaphore(BALANCE_FETCH_CONCURRENCY)

        async def refresh_single(api_key: str, client: httpx.AsyncClient) -> None:
            async with semaphore:
                try:
                    balance_data = await self.refresh_balance(
                        api_key,
                        credential_identifier=api_key,
                        client=client,
                    )

                    if balance_data.get("status") == "success":
                        balance_dollars = balance_data["balance_dollars"]
                        # Use cents (×100 scale) for full precision.
                        # The TUI detects 'credits($)' and formats as dollars:
                        #   1485/1500 → $14.85/$15.00
                        balance_cents = balance_data["balance_cents"]
                        max_cents = self._monthly_grant_cents
                        next_grant_ts = balance_data.get("next_grant_ts")

                        # Compute cents used relative to grant
                        used_cents = max(0, max_cents - balance_cents)

                        await usage_manager.update_quota_baseline(
                            api_key,
                            "lightning_ai/_balance",
                            quota_max_requests=max_cents,
                            quota_reset_ts=next_grant_ts,
                            quota_used=used_cents,
                        )

                        lib_logger.debug(
                            f"Updated Lightning AI balance baseline: "
                            f"${balance_dollars:.2f} remaining "
                            f"({balance_cents}¢ / {max_cents}¢)"
                        )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh Lightning AI balance: {e}"
                    )

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [refresh_single(api_key, client) for api_key in credentials]
            await asyncio.gather(*tasks, return_exceptions=True)
