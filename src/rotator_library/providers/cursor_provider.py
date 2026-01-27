"""
Cursor Provider with Quota Tracking

Provider implementation for Cursor AI via the cursor-sidecar with quota monitoring.
Uses the CursorQuotaTracker mixin to fetch quota usage from cursor.com's web API.

The Cursor provider works in two parts:
1. Chat completions: Via cursor-sidecar (OpenAI-compatible API at CURSOR_API_BASE)
2. Quota monitoring: Via cursor.com web API using CURSOR_SESSION_TOKEN

Environment variables:
    CURSOR_API_BASE: Sidecar API base URL (e.g., http://cursor-sidecar:18741/v1)
    CURSOR_API_KEY: API key for sidecar (can be "not-needed")
    CURSOR_SESSION_TOKEN: WorkosCursorSessionToken cookie for quota API
    CURSOR_QUOTA_REFRESH_INTERVAL: Quota refresh interval in seconds (default: 300)
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

from .openai_compatible_provider import OpenAICompatibleProvider
from .utilities.cursor_quota_tracker import CursorQuotaTracker

if TYPE_CHECKING:
    from ..usage import UsageManager

import logging

lib_logger = logging.getLogger("rotator_library")


class CursorProvider(CursorQuotaTracker, OpenAICompatibleProvider):
    """
    Provider implementation for Cursor AI with quota tracking.

    Cursor uses a sidecar container that provides an OpenAI-compatible API.
    This class adds quota tracking via the cursor.com web API.
    """

    # Skip LiteLLM cost calculation - Cursor models use custom naming
    skip_cost_calculation: bool = True

    # Quota groups for tracking monthly limits
    # Cursor tracks usage per-model but we use a virtual model for overall tracking
    model_quota_groups = {
        "cursor_premium": ["cursor/_quota"],
    }

    def __init__(self):
        """Initialize CursorProvider with quota tracking."""
        # Initialize OpenAICompatibleProvider with provider name
        super().__init__("cursor")

        # Quota tracking cache and refresh interval
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        try:
            self._quota_refresh_interval = int(
                os.environ.get("CURSOR_QUOTA_REFRESH_INTERVAL", "300")
            )
        except ValueError:
            lib_logger.warning(
                "Invalid CURSOR_QUOTA_REFRESH_INTERVAL value, using default 300"
            )
            self._quota_refresh_interval = 300

        # Track whether session token is configured
        self._session_token_configured = bool(os.environ.get("CURSOR_SESSION_TOKEN"))
        if not self._session_token_configured:
            lib_logger.info(
                "CURSOR_SESSION_TOKEN not set - Cursor quota tracking disabled. "
                "To enable, extract WorkosCursorSessionToken cookie from cursor.com"
            )

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        Cursor tracks usage per-model (gpt-4, etc.) but we map all models
        to a single quota group since they share the same monthly limit.

        Args:
            model: Model name (ignored - all premium models share quota)

        Returns:
            Quota group identifier for shared tracking
        """
        return "cursor_premium"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get all models in a quota group.

        For Cursor, we use a virtual model "cursor/_quota" to track the
        overall monthly quota.

        Args:
            group: Quota group name

        Returns:
            List of model names in the group
        """
        if group == "cursor_premium":
            return ["cursor/_quota"]
        return []

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Cursor credentials.

        Cursor uses monthly quotas that reset at the start of each billing period.

        Args:
            credential: The API key (unused, same config for all)

        Returns:
            Configuration with per_model mode and monthly window
        """
        return {
            "mode": "per_model",
            "window_seconds": 2592000,  # ~30 days (monthly billing)
            "field_name": "models",
        }

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic quota usage refresh.

        Only returns config if CURSOR_SESSION_TOKEN is set.

        Returns:
            Background job configuration for quota refresh, or None if disabled
        """
        if not self._session_token_configured:
            return None

        return {
            "interval": self._quota_refresh_interval,
            "name": "cursor_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh quota usage from cursor.com web API.

        Note: Cursor quota is account-level, not per-credential, so we only
        need to fetch once regardless of how many API keys are configured.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys (not used for quota fetch)
        """
        session_token = os.environ.get("CURSOR_SESSION_TOKEN")
        if not session_token:
            return

        try:
            usage_data = await self.fetch_cursor_quota_usage(session_token)

            if usage_data.get("status") == "success":
                # Update quota cache
                self._quota_cache["cursor_session"] = usage_data

                # Extract model quotas
                model_quotas = self.extract_cursor_model_quotas(usage_data)

                # Find the primary model (usually gpt-4) for overall tracking
                # Use the model with the most restrictive quota
                min_remaining = 1.0
                primary_max_requests = None
                primary_used = 0
                for model_name, remaining, max_requests in model_quotas:
                    if remaining < min_remaining:
                        min_remaining = remaining
                        primary_max_requests = max_requests
                        # Calculate used from the models dict
                        model_data = usage_data.get("models", {}).get(model_name, {})
                        primary_used = model_data.get("numRequests", 0)

                # Calculate reset timestamp from start_of_month + ~30 days
                # Note: Cursor billing resets monthly; using 30 days as approximation
                start_of_month = usage_data.get("start_of_month")
                reset_ts = None
                if start_of_month:
                    try:
                        # Parse ISO format
                        start_str = start_of_month
                        if start_str.endswith("Z"):
                            start_str = start_str.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(start_str)
                        # Add ~30 days for next reset (Cursor billing cycle)
                        reset_dt = dt + timedelta(days=30)
                        reset_ts = reset_dt.timestamp()
                    except Exception:
                        pass

                # Apply to all credentials using virtual model
                for api_key in credentials:
                    # Check if exhausted
                    if min_remaining <= 0.0 and reset_ts:
                        stable_id = usage_manager.registry.get_stable_id(
                            api_key, usage_manager.provider
                        )
                        state = usage_manager.states.get(stable_id)
                        if state:
                            await usage_manager.tracking.apply_cooldown(
                                state=state,
                                reason="quota_exhausted",
                                until=reset_ts,
                                model_or_group="cursor/_quota",
                                source="api_quota",
                            )

                    await usage_manager.update_quota_baseline(
                        api_key,
                        "cursor/_quota",  # Virtual model for tracking
                        quota_max_requests=primary_max_requests,
                        quota_used=primary_used,
                        quota_reset_ts=reset_ts,
                    )

                lib_logger.debug(
                    f"Updated Cursor quota baseline: "
                    f"{primary_used}/{primary_max_requests} used ({min_remaining * 100:.1f}% remaining)"
                )

        except Exception as e:
            lib_logger.warning(f"Failed to refresh Cursor quota usage: {e}")

    # =========================================================================
    # QUOTA BASELINE FETCHING (for force_refresh_quota)
    # =========================================================================

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch quota baselines for credentials.

        For Cursor, quota is account-level so we fetch once and apply to all.

        Args:
            credential_paths: All credential paths (API keys)

        Returns:
            Dict mapping credential_path -> fetched quota data
        """
        session_token = os.environ.get("CURSOR_SESSION_TOKEN")
        if not session_token:
            lib_logger.debug("CURSOR_SESSION_TOKEN not set - skipping quota fetch")
            return {}

        if not credential_paths:
            return {}

        lib_logger.debug(f"Fetching Cursor quota baseline...")

        # Fetch once - account level quota
        usage_data = await self.fetch_cursor_quota_usage(session_token)

        # Apply to all credentials
        results = {}
        for cred_path in credential_paths:
            results[cred_path] = usage_data

        if usage_data.get("status") == "success":
            model_quotas = self.extract_cursor_model_quotas(usage_data)
            if model_quotas:
                summary = ", ".join(
                    f"{m}: {r * 100:.1f}% ({mx or 'unlimited'})"
                    for m, r, mx in model_quotas
                )
                lib_logger.debug(f"Cursor quota: {summary}")

        return results
