# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Lightning AI Quota Tracking Mixin

Provides quota tracking for the Lightning AI provider using their memberships API.
Lightning AI tracks credit balances (in dollars) that reload monthly.

API Details:
- Endpoint: GET https://lightning.ai/v1/memberships
- Auth: Authorization: Bearer <api_key>
- Response: { memberships: [{ displayName, balance, nextFreeCreditsGrant, freeCreditsEnabled, ... }] }

The balance is a float (dollars). We convert to integer cents (× 100) so the
UsageManager's integer-based quota system can track it accurately.

Required from provider:
    - self._balance_cache: Dict[str, Dict[str, Any]] = {}
    - self._quota_refresh_interval: int = 300
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

lib_logger = logging.getLogger("rotator_library")

LIGHTNING_AI_BASE_URL = "https://lightning.ai"
LIGHTNING_AI_MEMBERSHIPS_URL = f"{LIGHTNING_AI_BASE_URL}/v1/memberships"

# Scale factor: dollars → integer cents for UsageManager compatibility
CENTS_PER_DOLLAR = 100


class LightningAiQuotaTracker:
    """
    Mixin class providing quota tracking functionality for the Lightning AI provider.

    Lightning AI uses a dollar-based credit system that reloads monthly.
    The balance is fetched from the /v1/memberships endpoint and converted
    to integer cents for compatibility with the UsageManager's quota system.

    Usage:
        class LightningAiProvider(LightningAiQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._balance_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes from provider
    _balance_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # =========================================================================
    # MEMBERSHIPS / BALANCE API
    # =========================================================================

    async def fetch_balance(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch credit balance from the Lightning AI memberships API.

        Aggregates balance across all memberships for the given API key.

        Args:
            api_key: Lightning AI API key (UUID format)
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "balance_dollars": float,       # Total balance across memberships
                "balance_cents": int,           # balance_dollars × 100 (integer)
                "next_grant_ts": float | None,  # Unix timestamp of next credit reload
                "memberships": [                # Raw membership data
                    {
                        "name": str,
                        "balance": float,
                        "next_grant": str | None,
                        "free_credits_enabled": bool,
                    }
                ],
                "fetched_at": float,
            }
        """
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }

            if client is not None:
                response = await client.get(
                    LIGHTNING_AI_MEMBERSHIPS_URL, headers=headers, timeout=30
                )
            else:
                async with httpx.AsyncClient() as new_client:
                    response = await new_client.get(
                        LIGHTNING_AI_MEMBERSHIPS_URL, headers=headers, timeout=30
                    )
            response.raise_for_status()
            data = response.json()

            memberships_raw = data.get("memberships", [])
            total_balance = 0.0
            next_grant_ts: Optional[float] = None
            memberships_parsed: List[Dict[str, Any]] = []

            for m in memberships_raw:
                name = m.get("displayName") or m.get("name", "unknown")
                balance = float(m.get("balance", 0.0))
                total_balance += balance

                free_credits_enabled = bool(m.get("freeCreditsEnabled", False))
                next_grant_raw = (
                    m.get("nextFreeCreditsGrant") if free_credits_enabled else None
                )

                # Parse ISO-8601 grant date → Unix timestamp
                grant_ts: Optional[float] = None
                if next_grant_raw:
                    try:
                        dt = datetime.fromisoformat(
                            next_grant_raw.replace("Z", "+00:00")
                        )
                        grant_ts = dt.timestamp()
                        # Keep the earliest upcoming grant date as the reset time
                        if next_grant_ts is None or grant_ts < next_grant_ts:
                            next_grant_ts = grant_ts
                    except (ValueError, AttributeError):
                        pass

                memberships_parsed.append(
                    {
                        "name": name,
                        "balance": balance,
                        "next_grant": next_grant_raw,
                        "free_credits_enabled": free_credits_enabled,
                    }
                )

            balance_cents = int(round(total_balance * CENTS_PER_DOLLAR))

            return {
                "status": "success",
                "error": None,
                "balance_dollars": total_balance,
                "balance_cents": balance_cents,
                "next_grant_ts": next_grant_ts,
                "memberships": memberships_parsed,
                "fetched_at": time.time(),
            }

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                body = e.response.text
                if body:
                    error_msg = f"{error_msg}: {body[:200]}"
            except Exception:
                pass
            lib_logger.warning(f"Failed to fetch Lightning AI balance: {error_msg}")
            return self._error_response(error_msg)

        except Exception as e:
            lib_logger.warning(f"Failed to fetch Lightning AI balance: {e}")
            return self._error_response(str(e))

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return a standardised error response."""
        return {
            "status": "error",
            "error": error_msg,
            "balance_dollars": 0.0,
            "balance_cents": 0,
            "next_grant_ts": None,
            "memberships": [],
            "fetched_at": time.time(),
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def get_remaining_fraction(self, balance_data: Dict[str, Any]) -> float:
        """
        Calculate remaining quota fraction from balance data.

        Because Lightning AI doesn't expose a "max" balance, we use the cached
        maximum observed balance as the denominator.  On first fetch the fraction
        is 1.0 (full).  As balance decreases the fraction drops.

        Args:
            balance_data: Response from fetch_balance()

        Returns:
            Remaining fraction (0.0 to 1.0)
        """
        # We store max_balance_cents in the cache; see refresh_balance below.
        max_cents = balance_data.get("max_balance_cents")
        current_cents = balance_data.get("balance_cents", 0)
        if not max_cents or max_cents <= 0:
            return 1.0
        return min(1.0, max(0.0, current_cents / max_cents))

    def get_reset_timestamp(self, balance_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the next credit-reload timestamp from balance data.

        Args:
            balance_data: Response from fetch_balance()

        Returns:
            Unix timestamp of next monthly credit reload, or None
        """
        ts = balance_data.get("next_grant_ts")
        return ts if ts and ts > 0 else None

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    async def refresh_balance(
        self,
        api_key: str,
        credential_identifier: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Refresh and cache credit balance for a credential.

        Tracks the maximum observed balance so we can compute a meaningful
        remaining fraction even though Lightning AI doesn't expose a quota cap.

        Args:
            api_key: Lightning AI API key
            credential_identifier: Identifier for caching (e.g. "env://lightning_ai/1")
            client: Optional HTTP client for connection reuse

        Returns:
            Enriched balance data (includes max_balance_cents)
        """
        balance_data = await self.fetch_balance(api_key, client)

        if balance_data.get("status") == "success":
            # Retrieve previously cached max to preserve the high-water mark
            cached = self._balance_cache.get(credential_identifier, {})
            prev_max = cached.get("max_balance_cents", 0)
            current_cents = balance_data["balance_cents"]
            balance_data["max_balance_cents"] = max(prev_max, current_cents)

            self._balance_cache[credential_identifier] = balance_data

            lib_logger.debug(
                f"Lightning AI balance for {credential_identifier}: "
                f"${balance_data['balance_dollars']:.2f} "
                f"({balance_data['balance_cents']} ¢ / "
                f"{balance_data['max_balance_cents']} ¢ max)"
            )

        return balance_data

    def get_cached_balance(
        self, credential_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached balance data for a credential.

        Args:
            credential_identifier: Identifier used in caching

        Returns:
            Cached balance data or None
        """
        return self._balance_cache.get(credential_identifier)

    async def get_all_balance_info(
        self,
        api_keys: List[Tuple[str, str]],  # List of (identifier, api_key) tuples
    ) -> Dict[str, Any]:
        """
        Get balance info for all credentials.

        Args:
            api_keys: List of (identifier, api_key) tuples

        Returns:
            {
                "credentials": { identifier: { ... } },
                "summary": {
                    "total_credentials": int,
                    "total_balance_dollars": float,
                },
                "timestamp": float,
            }
        """
        import asyncio

        results: Dict[str, Any] = {}
        total_balance = 0.0

        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(
            identifier: str, api_key: str, client: httpx.AsyncClient
        ) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                data = await self.refresh_balance(api_key, identifier, client)
                return identifier, data

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                fetch_with_semaphore(ident, key, client) for ident, key in api_keys
            ]
            import asyncio as _asyncio
            fetch_results = await _asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Lightning AI balance fetch failed: {result}")
                continue

            identifier, data = result
            if data.get("status") == "success":
                total_balance += data.get("balance_dollars", 0.0)

            results[identifier] = {
                "identifier": identifier,
                "status": data.get("status", "error"),
                "error": data.get("error"),
                "balance_dollars": data.get("balance_dollars"),
                "balance_cents": data.get("balance_cents"),
                "max_balance_cents": data.get("max_balance_cents"),
                "next_grant_ts": data.get("next_grant_ts"),
                "memberships": data.get("memberships", []),
                "fetched_at": data.get("fetched_at"),
            }

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(api_keys),
                "total_balance_dollars": total_balance,
            },
            "timestamp": time.time(),
        }
