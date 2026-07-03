# src/rotator_library/providers/utilities/anthropic_quota_tracker.py
"""
Anthropic Quota Tracking Mixin

Provides quota tracking functionality for the Anthropic provider by:
1. Fetching utilization data from the /api/oauth/usage endpoint
2. Caching quota snapshots per credential
3. Pushing quota data to UsageManager for TUI and /quota-stats display

Anthropic OAuth Usage API Response:
{
  "five_hour": { "utilization": 23.0, "resets_at": "ISO8601" },
  "seven_day": { "utilization": 15.0, "resets_at": "ISO8601" } | null,
  ...
}

Required from provider:
    - self._credentials_cache: Dict[str, Dict[str, Any]]
    - self.get_anthropic_auth_header(credential_path) -> Dict[str, str]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage import UsageManager

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# CONFIGURATION
# =============================================================================

ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_BETA_HEADER = "oauth-2025-04-20"

# Stale threshold - snapshots older than this are considered stale (10 minutes)
QUOTA_STALE_THRESHOLD_SECONDS = 600

# When the usage endpoint returns 429 and carries no Retry-After header, back
# off future polls by this many seconds so we stop hammering a rate-limited
# account every refresh cycle (the background job runs every ~300s).
USAGE_RATE_LIMIT_BACKOFF_SECONDS = 900  # 15 minutes

# Never back off longer than this even if the server asks for more, so a bogus
# Retry-After can't freeze quota polling indefinitely.
USAGE_RATE_LIMIT_BACKOFF_MAX_SECONDS = 3600  # 1 hour


# =============================================================================
# HELPERS
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


def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header into seconds from now.

    Supports both forms defined by RFC 7231: a delta in seconds (e.g. "60")
    and an HTTP-date (e.g. "Wed, 21 Oct 2026 07:28:00 GMT"). Returns None if
    the value is missing or unparseable.
    """
    if not header_value:
        return None

    value = header_value.strip()

    # Form 1: integer seconds
    try:
        return max(0.0, float(int(value)))
    except (ValueError, TypeError):
        pass

    # Form 2: HTTP-date
    try:
        retry_dt = parsedate_to_datetime(value)
        if retry_dt is None:
            return None
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        delta = retry_dt.timestamp() - time.time()
        return delta if delta > 0 else None
    except (TypeError, ValueError):
        return None




# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class AnthropicQuotaWindow:
    """A single quota window (e.g., 5-hour or 7-day)."""

    utilization: float  # Percentage used (0-100)
    resets_at: Optional[float] = None  # Unix timestamp

    @property
    def remaining_percent(self) -> float:
        """Remaining quota as percentage (0-100)."""
        return max(0.0, 100.0 - self.utilization)

    @property
    def is_exhausted(self) -> bool:
        """Check if quota is fully used."""
        return self.utilization >= 100.0


@dataclass
class AnthropicQuotaSnapshot:
    """Complete quota snapshot for an Anthropic credential."""

    credential_path: str
    identifier: str

    # From /api/oauth/usage endpoint
    five_hour: Optional[AnthropicQuotaWindow] = None
    seven_day: Optional[AnthropicQuotaWindow] = None

    fetched_at: float = field(default_factory=time.time)
    status: str = "success"  # "success", "error", "no_data"
    error: Optional[str] = None

    @property
    def is_stale(self) -> bool:
        """Check if this snapshot is stale."""
        return time.time() - self.fetched_at > QUOTA_STALE_THRESHOLD_SECONDS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result: Dict[str, Any] = {
            "identifier": self.identifier,
            "fetched_at": self.fetched_at,
            "is_stale": self.is_stale,
            "status": self.status,
        }

        if self.five_hour:
            result["five_hour"] = {
                "utilization": self.five_hour.utilization,
                "remaining_percent": self.five_hour.remaining_percent,
                "resets_at": self.five_hour.resets_at,
                "is_exhausted": self.five_hour.is_exhausted,
            }

        if self.seven_day:
            result["seven_day"] = {
                "utilization": self.seven_day.utilization,
                "remaining_percent": self.seven_day.remaining_percent,
                "resets_at": self.seven_day.resets_at,
                "is_exhausted": self.seven_day.is_exhausted,
            }


        if self.error:
            result["error"] = self.error

        return result


# =============================================================================
# QUOTA TRACKER MIXIN
# =============================================================================


class AnthropicQuotaTracker:
    """
    Mixin class providing quota tracking functionality for Anthropic provider.

    Capabilities:
    - Fetch quota utilization from /api/oauth/usage endpoint
    - Cache quota snapshots per credential
    - Push quota data to UsageManager for TUI display

    Usage:
        class AnthropicProvider(AnthropicOAuthBase, AnthropicQuotaTracker, ProviderInterface):
            ...

    The provider class must call self._init_quota_tracker() in __init__.
    """

    # Type hints for attributes from provider
    _credentials_cache: Dict[str, Dict[str, Any]]
    _quota_cache: Dict[str, AnthropicQuotaSnapshot]
    _quota_refresh_interval: int

    def _init_quota_tracker(self) -> None:
        """Initialize quota tracker state. Call from provider's __init__."""
        self._quota_cache: Dict[str, AnthropicQuotaSnapshot] = {}
        self._quota_refresh_interval: int = 300  # 5 min default
        self._usage_manager: Optional["UsageManager"] = None
        # Per-credential timestamp until which usage polling is suppressed after
        # a 429 from the usage endpoint (see fetch_quota_from_api). Manual force
        # refreshes bypass this cooldown.
        self._usage_cooldown_until: Dict[str, float] = {}

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        """Set the UsageManager reference for pushing quota updates."""
        self._usage_manager = usage_manager

    # =========================================================================
    # API-BASED QUOTA FETCH
    # =========================================================================

    async def fetch_quota_from_api(
        self,
        credential_path: str,
        force: bool = False,
    ) -> AnthropicQuotaSnapshot:
        """
        Fetch quota utilization from the Anthropic /api/oauth/usage endpoint.

        Args:
            credential_path: Path to OAuth credential file
            force: If True, bypass the post-429 rate-limit cooldown (used for
                manual/user-initiated refreshes). Background polls pass False.

        Returns:
            AnthropicQuotaSnapshot with utilization data
        """
        identifier = _get_credential_identifier(credential_path)

        # Respect a prior 429 backoff: skip the HTTP call entirely while cooling
        # down so we don't keep hammering a rate-limited account. A manual force
        # refresh bypasses this.
        cooldown_until = self._usage_cooldown_until.get(credential_path, 0.0)
        if not force and time.time() < cooldown_until:
            remaining = int(cooldown_until - time.time())
            lib_logger.debug(
                f"Skipping Anthropic usage fetch for {identifier}: "
                f"rate-limit cooldown, {remaining}s remaining"
            )
            return AnthropicQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                status="error",
                error=f"rate_limited (cooldown {remaining}s)",
            )

        try:
            # Get auth header from the OAuth base class
            auth_headers = await self.get_anthropic_auth_header(credential_path)

            # Claude Code's User-Agent is required here: without it Anthropic
            # aggressively rate-limits /api/oauth/usage and returns persistent
            # HTTP 429s (anthropics/claude-code#31021). Imported lazily to avoid
            # a circular import (anthropic_provider imports this mixin).
            from ..anthropic_provider import ANTHROPIC_USER_AGENT

            proxy_kwargs = {}
            if hasattr(self, "_build_proxy_client_kwargs"):
                proxy_kwargs = self._build_proxy_client_kwargs(credential_path)
            async with httpx.AsyncClient(**proxy_kwargs) as client:
                response = await client.get(
                    ANTHROPIC_USAGE_URL,
                    headers={
                        **auth_headers,
                        "anthropic-beta": ANTHROPIC_BETA_HEADER,
                        "user-agent": ANTHROPIC_USER_AGENT,
                    },
                    timeout=5.0,
                )

            if response.status_code != 200:
                lib_logger.debug(
                    f"Anthropic usage API returned {response.status_code} "
                    f"for {identifier}: {response.text[:200]}"
                )
                if response.status_code == 429:
                    retry_after = _parse_retry_after(
                        response.headers.get("retry-after")
                    )
                    backoff = (
                        retry_after
                        if retry_after is not None
                        else USAGE_RATE_LIMIT_BACKOFF_SECONDS
                    )
                    backoff = min(backoff, USAGE_RATE_LIMIT_BACKOFF_MAX_SECONDS)
                    self._usage_cooldown_until[credential_path] = (
                        time.time() + backoff
                    )
                    lib_logger.debug(
                        f"Anthropic usage API rate-limited {identifier}; "
                        f"suppressing polls for {int(backoff)}s"
                    )
                return AnthropicQuotaSnapshot(
                    credential_path=credential_path,
                    identifier=identifier,
                    status="error",
                    error=f"HTTP {response.status_code}",
                )

            # Successful fetch — clear any prior rate-limit cooldown.
            self._usage_cooldown_until.pop(credential_path, None)

            data = response.json()

            # Parse five_hour window
            five_hour = None
            fh_data = data.get("five_hour")
            if fh_data and isinstance(fh_data, dict):
                utilization = fh_data.get("utilization")
                if utilization is not None:
                    resets_at = None
                    if fh_data.get("resets_at"):
                        resets_at = _parse_iso_timestamp(fh_data["resets_at"])
                    five_hour = AnthropicQuotaWindow(
                        utilization=float(utilization),
                        resets_at=resets_at,
                    )

            # Parse seven_day window
            seven_day = None
            sd_data = data.get("seven_day")
            if sd_data and isinstance(sd_data, dict):
                utilization = sd_data.get("utilization")
                if utilization is not None:
                    resets_at = None
                    if sd_data.get("resets_at"):
                        resets_at = _parse_iso_timestamp(sd_data["resets_at"])
                    seven_day = AnthropicQuotaWindow(
                        utilization=float(utilization),
                        resets_at=resets_at,
                    )

            snapshot = AnthropicQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                five_hour=five_hour,
                seven_day=seven_day,
                status="success",
            )

            # Log
            parts = []
            if five_hour:
                parts.append(f"5h={five_hour.utilization:.0f}%")
            if seven_day:
                parts.append(f"7d={seven_day.utilization:.0f}%")
            lib_logger.debug(
                f"Anthropic usage API ({identifier}): {', '.join(parts) or 'no windows'}"
            )

            # Cache and push (skip auto-push when force=True; the caller in
            # force_refresh_quota awaits apply_snapshot_to_usage_manager
            # directly so we avoid a double-push).
            self._quota_cache[credential_path] = snapshot
            if self._usage_manager and not force:
                self._push_quota_to_usage_manager(credential_path, snapshot)

            return snapshot

        except Exception as e:
            lib_logger.debug(
                f"Failed to fetch Anthropic usage for {identifier}: {e}"
            )
            return AnthropicQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                status="error",
                error=str(e),
            )


    # =========================================================================
    # USAGE MANAGER INTEGRATION
    # =========================================================================

    async def apply_snapshot_to_usage_manager(
        self,
        credential_path: str,
        snapshot: AnthropicQuotaSnapshot,
    ) -> None:
        """
        Apply a quota snapshot to the UsageManager (awaitable).

        Follows the Codex pattern: treats utilization percentage as
        quota_used on a 100-scale (quota_max_requests=100).

        Awaiting this guarantees the baselines are persisted before returning,
        which the force-refresh path relies on so the refreshed numbers are
        visible immediately after the request completes.
        """
        if not self._usage_manager:
            return

        try:
            if snapshot.five_hour:
                quota_used = int(snapshot.five_hour.utilization)
                await self._usage_manager.update_quota_baseline(
                    accessor=credential_path,
                    model="anthropic/_5h_window",
                    quota_max_requests=100,
                    quota_reset_ts=snapshot.five_hour.resets_at,
                    quota_used=quota_used,
                    quota_group="5h-limit",
                    force=True,
                    apply_exhaustion=snapshot.five_hour.is_exhausted,
                )

            if snapshot.seven_day:
                quota_used = int(snapshot.seven_day.utilization)
                await self._usage_manager.update_quota_baseline(
                    accessor=credential_path,
                    model="anthropic/_weekly_window",
                    quota_max_requests=100,
                    quota_reset_ts=snapshot.seven_day.resets_at,
                    quota_used=quota_used,
                    quota_group="weekly-limit",
                    force=True,
                    apply_exhaustion=snapshot.seven_day.is_exhausted,
                )
        except Exception as e:
            lib_logger.debug(
                f"Failed to push Anthropic quota to UsageManager: {e}"
            )

    def _push_quota_to_usage_manager(
        self,
        credential_path: str,
        snapshot: AnthropicQuotaSnapshot,
    ) -> None:
        """
        Schedule a (fire-and-forget) push of a quota snapshot to the
        UsageManager. Used by the background poll path where the caller does
        not await the result.
        """
        if not self._usage_manager:
            return

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return

        if loop.is_running():
            asyncio.ensure_future(
                self.apply_snapshot_to_usage_manager(credential_path, snapshot)
            )
        else:
            loop.run_until_complete(
                self.apply_snapshot_to_usage_manager(credential_path, snapshot)
            )

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Return configuration for quota refresh background job.

        Returns:
            Background job config dict
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "anthropic_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Execute periodic quota refresh for active credentials.

        Called by BackgroundRefresher at the configured interval.

        Args:
            usage_manager: UsageManager instance
            credentials: List of credential paths for this provider
        """
        if usage_manager and not self._usage_manager:
            self._usage_manager = usage_manager

        if not credentials:
            return

        # Filter to OAuth credentials only
        oauth_creds = [c for c in credentials if _is_oauth_path(c)]

        if not oauth_creds:
            lib_logger.debug("No OAuth Anthropic credentials to refresh quota for")
            return

        lib_logger.debug(
            f"Refreshing Anthropic quota for {len(oauth_creds)} OAuth credentials"
        )

        # Fetch quotas with limited concurrency
        semaphore = asyncio.Semaphore(3)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return await self.fetch_quota_from_api(cred_path)

        tasks = [fetch_with_semaphore(cred) for cred in oauth_creds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(
            1
            for r in results
            if isinstance(r, AnthropicQuotaSnapshot) and r.status == "success"
        )

        lib_logger.debug(
            f"Anthropic quota refresh complete: {success_count}/{len(oauth_creds)} successful"
        )

    # =========================================================================
    # CACHE ACCESS
    # =========================================================================

    def get_cached_quota(
        self,
        credential_path: str,
    ) -> Optional[AnthropicQuotaSnapshot]:
        """Get cached quota snapshot for a credential."""
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
                if cached.five_hour and cached.five_hour.is_exhausted:
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
                "data_source": "oauth_usage_api",
            },
            "timestamp": time.time(),
        }


def _is_oauth_path(path: str) -> bool:
    """Check if a credential path is for an OAuth credential."""
    return "oauth" in path.lower() or path.startswith("env://anthropic/")
