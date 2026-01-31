"""
Cursor Quota Tracking Mixin

Provides quota tracking for the Cursor provider using their web API.
Cursor uses a monthly quota system where requests are tracked per model.

API Details:
- Endpoint: GET https://cursor.com/api/usage?user={user_id}
- Auth: Cookie header with WorkosCursorSessionToken
- Response: { "gpt-4": {"numRequests": int, "maxRequestUsage": int, ...}, "startOfMonth": str }

The user_id is extracted from the session token (format: user_XXXX::jwt)

Required from provider:
    - self._quota_cache: Dict[str, Dict[str, Any]] = {}
    - self._quota_refresh_interval: int = 300
"""

import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# Cursor API configuration
CURSOR_API_BASE = "https://cursor.com/api"
CURSOR_USAGE_ENDPOINT = "/usage"


class CursorQuotaTracker:
    """
    Mixin class providing quota tracking functionality for the Cursor provider.

    This mixin adds the following capabilities:
    - Fetch quota usage from the Cursor web API
    - Track monthly request limits per model
    - Parse user ID from session token

    Usage:
        class CursorProvider(CursorQuotaTracker, OpenAICompatibleProvider):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes from provider
    _quota_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # =========================================================================
    # TOKEN PARSING
    # =========================================================================

    def _extract_user_id_from_token(self, session_token: str) -> Optional[str]:
        """
        Extract user ID from the session token.

        Token format: user_XXXX%3A%3Ajwt... (URL-encoded :: separator)
        or: user_XXXX::jwt... (decoded format)

        Args:
            session_token: The WorkosCursorSessionToken value

        Returns:
            User ID (e.g., "user_01JWV7FARDJPMQ5QZSANMJDS9A") or None
        """
        try:
            # URL-decode first in case it's encoded
            decoded = urllib.parse.unquote(session_token)

            # Split on :: separator
            if "::" in decoded:
                user_id = decoded.split("::")[0]
                if user_id.startswith("user_"):
                    return user_id

            # Try extracting from the token prefix directly
            if session_token.startswith("user_"):
                # Find the separator (either %3A%3A or ::)
                if "%3A%3A" in session_token:
                    return session_token.split("%3A%3A")[0]
                elif "::" in session_token:
                    return session_token.split("::")[0]

            return None
        except Exception as e:
            lib_logger.warning(f"Failed to extract user ID from session token: {e}")
            return None

    # =========================================================================
    # QUOTA USAGE API
    # =========================================================================

    async def fetch_cursor_quota_usage(
        self,
        session_token: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch quota usage from the Cursor web API.

        Args:
            session_token: The WorkosCursorSessionToken cookie value
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "models": {
                    "gpt-4": {"numRequests": int, "maxRequestUsage": int, "remaining_fraction": float},
                    ...
                },
                "start_of_month": str | None,
                "fetched_at": float,
            }
        """
        try:
            # Extract user ID from token
            user_id = self._extract_user_id_from_token(session_token)
            if not user_id:
                return {
                    "status": "error",
                    "error": "Could not extract user ID from session token",
                    "models": {},
                    "start_of_month": None,
                    "fetched_at": time.time(),
                }

            headers = {
                "Accept": "application/json",
                "Cookie": f"WorkosCursorSessionToken={session_token}",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            }

            # URL-encode user_id for safety
            encoded_user_id = urllib.parse.quote(user_id, safe="")
            url = f"{CURSOR_API_BASE}{CURSOR_USAGE_ENDPOINT}?user={encoded_user_id}"

            if client is not None:
                response = await client.get(url, headers=headers, timeout=30, follow_redirects=True)
            else:
                async with httpx.AsyncClient() as new_client:
                    response = await new_client.get(
                        url, headers=headers, timeout=30, follow_redirects=True
                    )

            response.raise_for_status()
            data = response.json()

            # Check for auth errors
            if "error" in data:
                error_msg = data.get("description", data.get("error", "Unknown error"))
                if data.get("error") == "not_authenticated":
                    lib_logger.warning(
                        "Cursor session token expired or invalid. "
                        "Please update CURSOR_SESSION_TOKEN with a fresh cookie from cursor.com"
                    )
                return {
                    "status": "error",
                    "error": error_msg,
                    "models": {},
                    "start_of_month": None,
                    "fetched_at": time.time(),
                }

            # Parse the response
            # Format: {"gpt-4": {...}, "startOfMonth": "2026-01-23T22:27:08.000Z"}
            start_of_month = data.pop("startOfMonth", None)

            models = {}
            for model_name, usage_data in data.items():
                if isinstance(usage_data, dict):
                    num_requests = usage_data.get("numRequests", 0)
                    max_requests = usage_data.get("maxRequestUsage")

                    # Calculate remaining fraction
                    if max_requests and max_requests > 0:
                        remaining = max(0, max_requests - num_requests)
                        remaining_fraction = remaining / max_requests
                    else:
                        # No limit or unknown limit
                        remaining_fraction = 1.0

                    models[model_name] = {
                        "numRequests": num_requests,
                        "numRequestsTotal": usage_data.get("numRequestsTotal", num_requests),
                        "numTokens": usage_data.get("numTokens", 0),
                        "maxRequestUsage": max_requests,
                        "maxTokenUsage": usage_data.get("maxTokenUsage"),
                        "remaining_fraction": remaining_fraction,
                    }

            return {
                "status": "success",
                "error": None,
                "models": models,
                "start_of_month": start_of_month,
                "fetched_at": time.time(),
            }

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code in (401, 403):
                lib_logger.warning(
                    f"Cursor API authentication failed ({error_msg}). "
                    "Please update CURSOR_SESSION_TOKEN with a fresh cookie from cursor.com"
                )
            else:
                lib_logger.warning(f"Failed to fetch Cursor quota: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "models": {},
                "start_of_month": None,
                "fetched_at": time.time(),
            }
        except Exception as e:
            lib_logger.warning(f"Failed to fetch Cursor quota: {type(e).__name__}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "models": {},
                "start_of_month": None,
                "fetched_at": time.time(),
            }

    def get_cursor_remaining_fraction(
        self, usage_data: Dict[str, Any], model: str
    ) -> Optional[float]:
        """
        Get remaining quota fraction for a specific model.

        Args:
            usage_data: Response from fetch_cursor_quota_usage()
            model: Model name (e.g., "gpt-4")

        Returns:
            Remaining fraction (0.0 to 1.0) or None if not found
        """
        models = usage_data.get("models", {})
        model_data = models.get(model)
        if model_data:
            return model_data.get("remaining_fraction", 1.0)
        return None

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    async def refresh_cursor_quota_usage(
        self,
        credential_identifier: str,
    ) -> Dict[str, Any]:
        """
        Refresh and cache quota usage for a credential.

        The credential_identifier for Cursor is the session token from
        CURSOR_SESSION_TOKEN environment variable.

        Args:
            credential_identifier: Identifier for caching (typically "cursor_session")

        Returns:
            Usage data from fetch_cursor_quota_usage()
        """
        session_token = os.environ.get("CURSOR_SESSION_TOKEN")
        if not session_token:
            lib_logger.warning(
                "CURSOR_SESSION_TOKEN not set - cannot fetch quota"
            )
            return {
                "status": "error",
                "error": "CURSOR_SESSION_TOKEN not configured",
                "models": {},
                "start_of_month": None,
                "fetched_at": time.time(),
            }

        usage_data = await self.fetch_cursor_quota_usage(session_token)

        if usage_data.get("status") == "success":
            self._quota_cache[credential_identifier] = usage_data

            models = usage_data.get("models", {})
            if models:
                model_summary = ", ".join(
                    f"{m}: {d.get('remaining_fraction', 0) * 100:.1f}%"
                    for m, d in models.items()
                )
                lib_logger.debug(f"Cursor quota: {model_summary}")

        return usage_data

    def get_cached_cursor_usage(
        self, credential_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached quota usage for a credential.

        Args:
            credential_identifier: Identifier used in caching

        Returns:
            Copy of cached usage data or None
        """
        cached = self._quota_cache.get(credential_identifier)
        return dict(cached) if cached else None

    # =========================================================================
    # MODEL QUOTA EXTRACTION
    # =========================================================================

    def extract_cursor_model_quotas(
        self, usage_data: Dict[str, Any]
    ) -> List[Tuple[str, float, Optional[int]]]:
        """
        Extract model quota information from usage data.

        Args:
            usage_data: Response from fetch_cursor_quota_usage()

        Returns:
            List of tuples: (model_name, remaining_fraction, max_requests)
            - model_name: Model name from Cursor API (e.g., "gpt-4")
            - remaining_fraction: 0.0 to 1.0
            - max_requests: Maximum requests for this model, or None if unlimited
        """
        result = []
        models = usage_data.get("models", {})

        for model_name, model_data in models.items():
            remaining_fraction = model_data.get("remaining_fraction", 1.0)
            max_requests = model_data.get("maxRequestUsage")
            result.append((model_name, remaining_fraction, max_requests))

        return result
