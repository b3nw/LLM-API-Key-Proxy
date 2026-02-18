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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..usage import UsageManager

from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.lightning_ai_quota_tracker import LightningAiQuotaTracker
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

# Lightning AI API base URL (OpenAI-compatible endpoint)
LIGHTNING_AI_API_BASE = "https://lightning.ai/api/v1"

# Concurrency limit for parallel balance fetches
BALANCE_FETCH_CONCURRENCY = 5

# Default monthly free credit grant per account, in tenths of a dollar (×10).
# Lightning AI plan tiers:
#   free:  $15/month  → 150
#   pro:   $20/month  → 200
#   teams: $50/month  → 500
# Override with LIGHTNING_AI_MONTHLY_GRANT (in whole dollars) if on a paid plan.
DEFAULT_MONTHLY_GRANT_DOLLARS = 15  # whole dollars; converted to tenths internally


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
        # Monthly grant in tenths of a dollar (×10 scale) for meaningful TUI display.
        # e.g. $15 grant → 150, so $14.85 shows as 148/150 (not 14/15 which loses precision)
        # Lightning AI tiers: free=$15, pro=$20, teams=$50
        # Set LIGHTNING_AI_MONTHLY_GRANT (whole dollars) to match your plan.
        grant_dollars = int(
            os.getenv("LIGHTNING_AI_MONTHLY_GRANT", str(DEFAULT_MONTHLY_GRANT_DOLLARS))
        )
        self._monthly_grant_tenths: int = grant_dollars * 10  # e.g. 15 → 150

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
                        # Use tenths-of-a-dollar scale for meaningful TUI display.
                        # $14.85 → 148 (floor), $14.98 → 149, $15.00 → 150
                        # This lets keys with tiny spending show differently from full ones.
                        balance_tenths = int(balance_dollars * 10)  # floor in tenths
                        max_tenths = self._monthly_grant_tenths
                        next_grant_ts = balance_data.get("next_grant_ts")

                        # Compute tenths used relative to grant
                        used_tenths = max(0, max_tenths - balance_tenths)

                        await usage_manager.update_quota_baseline(
                            api_key,
                            "lightning_ai/_balance",
                            quota_max_requests=max_tenths,
                            quota_reset_ts=next_grant_ts,
                            quota_used=used_tenths,
                        )

                        lib_logger.debug(
                            f"Updated Lightning AI balance baseline: "
                            f"${balance_dollars:.2f} remaining "
                            f"({balance_tenths} / {max_tenths} tenths)"
                        )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh Lightning AI balance: {e}"
                    )

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [refresh_single(api_key, client) for api_key in credentials]
            await asyncio.gather(*tasks, return_exceptions=True)
