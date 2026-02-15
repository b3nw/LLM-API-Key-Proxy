# src/rotator_library/providers/utilities/anthropic_quota_tracker.py
"""
Anthropic Quota Tracking Mixin

Provides quota tracking functionality for the Anthropic provider by:
1. Parsing rate limit headers from API responses
2. Caching rate limit snapshots per credential
3. Pushing quota data to UsageManager for TUI and /quota-stats display

Anthropic Rate Limit Headers:
- anthropic-ratelimit-requests-limit: Max requests per minute
- anthropic-ratelimit-requests-remaining: Requests remaining
- anthropic-ratelimit-requests-reset: When request limit resets (ISO 8601)
- anthropic-ratelimit-tokens-limit: Max tokens per minute
- anthropic-ratelimit-tokens-remaining: Tokens remaining
- anthropic-ratelimit-tokens-reset: When token limit resets (ISO 8601)

Required from provider:
    - self._credentials_cache: Dict[str, Dict[str, Any]]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...usage import UsageManager

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# HEADER NAMES
# =============================================================================

HEADER_REQUESTS_LIMIT = "anthropic-ratelimit-requests-limit"
HEADER_REQUESTS_REMAINING = "anthropic-ratelimit-requests-remaining"
HEADER_REQUESTS_RESET = "anthropic-ratelimit-requests-reset"
HEADER_TOKENS_LIMIT = "anthropic-ratelimit-tokens-limit"
HEADER_TOKENS_REMAINING = "anthropic-ratelimit-tokens-remaining"
HEADER_TOKENS_RESET = "anthropic-ratelimit-tokens-reset"

# Stale threshold - snapshots older than this are considered stale (10 minutes)
QUOTA_STALE_THRESHOLD_SECONDS = 600


# =============================================================================
# DATA CLASSES
# =============================================================================


def _get_credential_identifier(credential_path: str) -> str:
    """Extract a short identifier from a credential path."""
    if credential_path.startswith("env://"):
        return credential_path
    return Path(credential_path).name


def _parse_iso_timestamp(iso_string: str) -> Optional[float]:
    """Parse an ISO 8601 timestamp to Unix timestamp in seconds."""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


@dataclass
class AnthropicRateLimitSnapshot:
    """Snapshot of Anthropic rate limit state for a credential."""

    credential_path: str
    identifier: str

    # Request limits (per minute)
    requests_limit: Optional[int] = None
    requests_remaining: Optional[int] = None
    requests_reset_at: Optional[float] = None  # Unix timestamp

    # Token limits (per minute)
    tokens_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_reset_at: Optional[float] = None  # Unix timestamp

    fetched_at: float = field(default_factory=time.time)
    status: str = "success"  # "success" or "no_data"

    @property
    def is_stale(self) -> bool:
        """Check if this snapshot is stale."""
        return time.time() - self.fetched_at > QUOTA_STALE_THRESHOLD_SECONDS

    @property
    def requests_used_percent(self) -> Optional[float]:
        """Calculate request usage percentage (0-100)."""
        if self.requests_limit is None or self.requests_remaining is None:
            return None
        if self.requests_limit == 0:
            return 100.0
        used = self.requests_limit - self.requests_remaining
        return round(used / self.requests_limit * 100, 1)

    @property
    def tokens_used_percent(self) -> Optional[float]:
        """Calculate token usage percentage (0-100)."""
        if self.tokens_limit is None or self.tokens_remaining is None:
            return None
        if self.tokens_limit == 0:
            return 100.0
        used = self.tokens_limit - self.tokens_remaining
        return round(used / self.tokens_limit * 100, 1)

    @property
    def is_request_limited(self) -> bool:
        """Check if request quota is exhausted."""
        return self.requests_remaining is not None and self.requests_remaining <= 0

    @property
    def is_token_limited(self) -> bool:
        """Check if token quota is exhausted."""
        return self.tokens_remaining is not None and self.tokens_remaining <= 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result: Dict[str, Any] = {
            "identifier": self.identifier,
            "fetched_at": self.fetched_at,
            "is_stale": self.is_stale,
            "status": self.status,
        }

        if self.requests_limit is not None:
            result["requests"] = {
                "limit": self.requests_limit,
                "remaining": self.requests_remaining,
                "used_percent": self.requests_used_percent,
                "reset_at": self.requests_reset_at,
                "is_exhausted": self.is_request_limited,
            }

        if self.tokens_limit is not None:
            result["tokens"] = {
                "limit": self.tokens_limit,
                "remaining": self.tokens_remaining,
                "used_percent": self.tokens_used_percent,
                "reset_at": self.tokens_reset_at,
                "is_exhausted": self.is_token_limited,
            }

        return result


# =============================================================================
# HEADER PARSING
# =============================================================================


def parse_anthropic_rate_limit_headers(
    headers: Dict[str, str],
) -> AnthropicRateLimitSnapshot:
    """
    Parse rate limit information from Anthropic API response headers.

    Args:
        headers: Response headers dict (keys should be lowercase)

    Returns:
        AnthropicRateLimitSnapshot with parsed rate limit data
    """
    requests_limit = _parse_int_header(headers, HEADER_REQUESTS_LIMIT)
    requests_remaining = _parse_int_header(headers, HEADER_REQUESTS_REMAINING)
    tokens_limit = _parse_int_header(headers, HEADER_TOKENS_LIMIT)
    tokens_remaining = _parse_int_header(headers, HEADER_TOKENS_REMAINING)

    requests_reset_at = None
    reset_str = headers.get(HEADER_REQUESTS_RESET)
    if reset_str:
        requests_reset_at = _parse_iso_timestamp(reset_str)

    tokens_reset_at = None
    reset_str = headers.get(HEADER_TOKENS_RESET)
    if reset_str:
        tokens_reset_at = _parse_iso_timestamp(reset_str)

    has_data = any(
        v is not None
        for v in [requests_limit, requests_remaining, tokens_limit, tokens_remaining]
    )

    return AnthropicRateLimitSnapshot(
        credential_path="",
        identifier="",
        requests_limit=requests_limit,
        requests_remaining=requests_remaining,
        requests_reset_at=requests_reset_at,
        tokens_limit=tokens_limit,
        tokens_remaining=tokens_remaining,
        tokens_reset_at=tokens_reset_at,
        fetched_at=time.time(),
        status="success" if has_data else "no_data",
    )


def _parse_int_header(headers: Dict[str, str], key: str) -> Optional[int]:
    """Parse an integer value from a header, returning None on failure."""
    value = headers.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# =============================================================================
# QUOTA TRACKER MIXIN
# =============================================================================


class AnthropicQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Anthropic provider.

    Capabilities:
    - Parse rate limit headers from streaming/non-streaming responses
    - Cache quota snapshots per credential
    - Push quota data to UsageManager for TUI display
    - Get structured quota info for /quota-stats endpoint

    Usage:
        class AnthropicProvider(AnthropicOAuthBase, AnthropicQuotaTracker, ProviderInterface):
            ...

    The provider class must call self._init_quota_tracker() in __init__.
    """

    # Type hints for attributes from provider
    _credentials_cache: Dict[str, Dict[str, Any]]
    _quota_cache: Dict[str, AnthropicRateLimitSnapshot]

    def _init_quota_tracker(self) -> None:
        """Initialize quota tracker state. Call from provider's __init__."""
        self._quota_cache: Dict[str, AnthropicRateLimitSnapshot] = {}
        self._usage_manager: Optional["UsageManager"] = None

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        """Set the UsageManager reference for pushing quota updates."""
        self._usage_manager = usage_manager

    # =========================================================================
    # HEADER-BASED QUOTA UPDATE
    # =========================================================================

    def update_quota_from_headers(
        self,
        credential_path: str,
        headers: Dict[str, str],
    ) -> Optional[AnthropicRateLimitSnapshot]:
        """
        Update cached quota info from response headers.

        Call this after each API response to keep quota cache up-to-date.
        Also pushes quota data to the UsageManager if available.

        Args:
            credential_path: Credential that made the request
            headers: Response headers dict

        Returns:
            Updated AnthropicRateLimitSnapshot or None if no rate limit headers
        """
        snapshot = parse_anthropic_rate_limit_headers(headers)

        if snapshot.status == "no_data":
            return None

        snapshot.credential_path = credential_path
        snapshot.identifier = _get_credential_identifier(credential_path)

        self._quota_cache[credential_path] = snapshot

        # Log quota info
        parts = []
        if snapshot.requests_remaining is not None:
            parts.append(
                f"requests={snapshot.requests_remaining}/{snapshot.requests_limit}"
            )
        if snapshot.tokens_remaining is not None:
            parts.append(
                f"tokens={snapshot.tokens_remaining}/{snapshot.tokens_limit}"
            )
        if parts:
            lib_logger.debug(
                f"Anthropic rate limits ({snapshot.identifier}): {', '.join(parts)}"
            )

        # Push to UsageManager
        if self._usage_manager:
            self._push_quota_to_usage_manager(credential_path, snapshot)

        return snapshot

    def _push_quota_to_usage_manager(
        self,
        credential_path: str,
        snapshot: AnthropicRateLimitSnapshot,
    ) -> None:
        """
        Push parsed quota snapshot to the UsageManager.

        Translates rate limit data into update_quota_baseline calls
        so the TUI and /quota-stats can display quota status.

        Anthropic's rate limits are per-minute, so we map:
        - Request limits → quota_max_requests with reset timestamp
        """
        if not self._usage_manager:
            return

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return

        async def _push() -> None:
            try:
                if snapshot.requests_limit is not None:
                    requests_used = (
                        snapshot.requests_limit - (snapshot.requests_remaining or 0)
                    )
                    await self._usage_manager.update_quota_baseline(
                        accessor=credential_path,
                        model="anthropic/_rpm",
                        quota_max_requests=snapshot.requests_limit,
                        quota_reset_ts=snapshot.requests_reset_at,
                        quota_used=requests_used,
                        quota_group="anthropic-rpm",
                        force=True,
                        apply_exhaustion=snapshot.is_request_limited,
                    )
            except Exception as e:
                lib_logger.debug(
                    f"Failed to push Anthropic quota to UsageManager: {e}"
                )

        # Schedule the async push
        if loop.is_running():
            asyncio.ensure_future(_push())
        else:
            loop.run_until_complete(_push())

    # =========================================================================
    # CACHE ACCESS
    # =========================================================================

    def get_cached_quota(
        self,
        credential_path: str,
    ) -> Optional[AnthropicRateLimitSnapshot]:
        """
        Get cached quota snapshot for a credential.

        Args:
            credential_path: Credential to look up

        Returns:
            Cached AnthropicRateLimitSnapshot or None if not cached
        """
        return self._quota_cache.get(credential_path)

    # =========================================================================
    # QUOTA INFO AGGREGATION (for /quota-stats)
    # =========================================================================

    def get_all_quota_info(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Any]:
        """
        Get cached quota info for all credentials.

        Since Anthropic doesn't have a dedicated /usage API endpoint,
        this only returns data from response header captures.

        Args:
            credential_paths: List of credential paths to report on

        Returns:
            Structured quota info dict for /quota-stats endpoint
        """
        results = {}
        exhausted_count = 0

        for cred_path in credential_paths:
            identifier = _get_credential_identifier(cred_path)
            cached = self._quota_cache.get(cred_path)

            if cached:
                entry = cached.to_dict()
                entry["file_path"] = (
                    cred_path if not cred_path.startswith("env://") else None
                )
                if cached.is_request_limited or cached.is_token_limited:
                    exhausted_count += 1
            else:
                entry = {
                    "identifier": identifier,
                    "file_path": (
                        cred_path if not cred_path.startswith("env://") else None
                    ),
                    "status": "no_data",
                    "fetched_at": None,
                    "is_stale": True,
                }

            results[identifier] = entry

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(credential_paths),
                "exhausted_count": exhausted_count,
                "data_source": "response_headers",
            },
            "timestamp": time.time(),
        }
