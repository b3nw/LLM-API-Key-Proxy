"""
Firmware.ai Provider with Quota Tracking

Provider implementation for the Firmware.ai API with credit-based quota tracking.
Uses the FirmwareQuotaTracker mixin to fetch credit balance from their API.

Environment variables:
    FIRMWARE_API_BASE: API base URL (default: https://app.firmware.ai/api/v1)
    FIRMWARE_API_KEY: API key for authentication
    FIRMWARE_QUOTA_REFRESH_INTERVAL: Quota refresh interval in seconds (default: 300)
"""

import asyncio
import httpx
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .provider_interface import ProviderInterface
from .utilities.firmware_quota_tracker import FirmwareQuotaTracker

if TYPE_CHECKING:
    from ..usage import UsageManager

import logging

lib_logger = logging.getLogger("rotator_library")

# Concurrency limit for parallel quota fetches
QUOTA_FETCH_CONCURRENCY = 5


class FirmwareProvider(FirmwareQuotaTracker, ProviderInterface):
    """
    Provider implementation for the Firmware.ai API with quota tracking.
    """

    # Quota groups for tracking credit-based limits
    # Uses a virtual model "firmware/_quota" for credential-level quota tracking
    model_quota_groups = {
        "firmware_global": ["firmware/_quota"],
    }

    def __init__(self, *args, **kwargs):
        """Initialize FirmwareProvider with quota tracking."""
        super().__init__(*args, **kwargs)

        # Quota tracking cache and refresh interval
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        try:
            self._quota_refresh_interval = int(
                os.environ.get("FIRMWARE_QUOTA_REFRESH_INTERVAL", "300")
            )
        except ValueError:
            lib_logger.warning(
                "Invalid FIRMWARE_QUOTA_REFRESH_INTERVAL value, using default 300"
            )
            self._quota_refresh_interval = 300

        # API base URL (default to Firmware.ai)
        self.api_base = os.environ.get(
            "FIRMWARE_API_BASE", "https://app.firmware.ai/api/v1"
        )

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All Firmware.ai models share the same credential-level quota pool,
        so they all belong to the same quota group.

        Args:
            model: Model name (ignored - all models share quota)

        Returns:
            Quota group identifier for shared credential-level tracking
        """
        return "firmware_global"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get all models in a quota group.

        For Firmware.ai, we use a virtual model "firmware/_quota" to track the
        credential-level 5-hour rolling window quota.

        Args:
            group: Quota group name

        Returns:
            List of model names in the group
        """
        if group == "firmware_global":
            return ["firmware/_quota"]
        return []

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Firmware.ai credentials.

        Firmware.ai uses per_model mode to track usage at the model level,
        with credit-based quota managed via the background job.

        Args:
            credential: The API key (unused, same config for all)

        Returns:
            Configuration with per_model mode
        """
        return {
            "mode": "per_model",
            "field_name": "models",
        }

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from the Firmware.ai API.

        Args:
            api_key: Firmware.ai API key
            client: HTTP client

        Returns:
            List of model names prefixed with 'firmware/'
        """
        try:
            response = await client.get(
                f"{self.api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"firmware/{model['id']}" for model in response.json().get("data", [])
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch Firmware.ai models: {e}")
            return []

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic quota usage refresh.

        Returns:
            Background job configuration for quota refresh
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "firmware_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh quota usage for all credentials in parallel.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys
        """
        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def refresh_single_credential(
            api_key: str, client: httpx.AsyncClient
        ) -> None:
            async with semaphore:
                try:
                    usage_data = await self.fetch_quota_usage(api_key, client)

                    if usage_data.get("status") == "success":
                        # Update quota cache
                        self._quota_cache[api_key] = usage_data

                        # Calculate values for usage manager
                        credits_balance = usage_data.get("credits", 0.0)
                        remaining_fraction = usage_data.get("remaining_fraction", 0.0)

                        # Store baseline in usage manager
                        # Since Firmware.ai uses credential-level quota, we use a virtual model name
                        # Express credits as synthetic request-count values (in cents)
                        # so the quota group window gets a non-zero limit for visibility
                        credits_as_cents = int(credits_balance * 100)

                        if remaining_fraction <= 0.0:
                            stable_id = usage_manager.registry.get_stable_id(
                                api_key, usage_manager.provider
                            )
                            state = usage_manager.states.get(stable_id)
                            if state:
                                await usage_manager.tracking.apply_cooldown(
                                    state=state,
                                    reason="quota_exhausted",
                                    duration=3600,  # 1 hour cooldown
                                    model_or_group="firmware/_quota",
                                    source="api_quota",
                                )
                        await usage_manager.update_quota_baseline(
                            api_key,
                            "firmware/_quota",  # Virtual model for credential-level tracking
                            quota_max_requests=credits_as_cents,
                            quota_used=0,
                            quota_reset_ts=None,
                            force=True,
                        )

                        lib_logger.debug(
                            f"Updated Firmware.ai quota baseline: "
                            f"{credits_balance:.2f} credits remaining"
                        )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh Firmware.ai quota usage: {e}"
                    )

        # Fetch all credentials in parallel with shared HTTP client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(api_key, client) for api_key in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
