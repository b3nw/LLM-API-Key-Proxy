# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Perchai Dollar-Based Quota Tracking Mixin

Provides quota tracking for the Perchai provider using a dollar-based
monthly allowance system.  Perchai's Pro plan includes a monthly included
usage allowance that resets on the first day of each calendar month at
00:00 UTC.

The usage endpoint at ``GET/POST {appUrl}/api/perch-terminal/usage`` returns
the current month's dollar usage (and other limits).  Perchai's wire
response shape is not yet pinned, so this implementation walks several
candidate field paths defensively and falls back to a zero-valued baseline
if nothing matches.

Usage::

    class PerchaiProvider(PerchaiQuotaTracker, ProviderInterface):
        ...

The provider class must initialize these instance attributes in ``__init__``:
    - self._balance_cache: Dict[str, Dict[str, Any]] = {}
    - self._quota_refresh_interval: int = 300
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx

from ..provider_interface import UsageResetConfigDef

# Forward-reference UsageManager to avoid circular imports at runtime.
if TYPE_CHECKING:
    from ...usage import UsageManager

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


# Scale factor: dollars -> integer cents for UsageManager compatibility.
CENTS_PER_DOLLAR: int = 100

# Concurrency limit for parallel usage fetches (matches chutes pattern).
USAGE_FETCH_CONCURRENCY: int = 5

# HTTP timeout for usage fetch (per request).
USAGE_FETCH_TIMEOUT: float = 15.0


# =========================================================================
# CREDENTIAL RESOLUTION HELPERS (module-level; used by mixin methods)
# =========================================================================


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce ``value`` to ``int``; return ``default`` on failure/None."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_env_path(credential: str) -> bool:
    """Return True if ``credential`` is a virtual ``env://perchai/N`` path."""
    return isinstance(credential, str) and credential.startswith("env://perchai/")


def _parse_env_index(credential: str) -> Optional[int]:
    """Extract the numeric index from an ``env://perchai/N`` path.

    Returns ``None`` if the path doesn't match the expected shape.  The
    legacy single-credential env-var form (index 0) is preserved by
    returning ``0`` for ``env://perchai/0``.
    """
    if not _is_env_path(credential):
        return None
    parts = credential[len("env://perchai/"):].split("/")
    if not parts or not parts[0]:
        return None
    idx = _safe_int(parts[0], 0)
    return idx


def _get_credential_identifier(credential: str) -> str:
    """Return a short identifier for logs (env path stays full; file path -> basename)."""
    if credential.startswith("env://"):
        return credential
    return Path(credential).name


def _resolve_env_credentials(credential: str) -> Optional[Tuple[str, str]]:
    """Resolve ``env://perchai/N`` to ``(access_token, app_url)``.

    Reads:
        - ``PERCHAI_N_ACCESS_TOKEN``  (or ``PERCHAI_ACCESS_TOKEN`` for index 0)
        - ``PERCHAI_N_APP_URL``       (or ``PERCHAI_APP_URL`` for index 0)

    Returns ``None`` if either token or app URL is missing (caller treats
    that as "skip this credential" and logs a warning).
    """
    idx = _parse_env_index(credential)
    if idx is None:
        return None

    if idx == 0:
        token = os.getenv("PERCHAI_ACCESS_TOKEN", "").strip()
        app_url = (
            os.getenv("PERCHAI_APP_URL", "").strip()
            or os.getenv("PERCH_CLI_APP_URL", "").strip()
            or os.getenv("PERCH_MODEL_CALL_PROXY_URL", "").strip()
        )
    else:
        token = os.getenv(f"PERCHAI_{idx}_ACCESS_TOKEN", "").strip()
        app_url = (
            os.getenv(f"PERCHAI_{idx}_APP_URL", "").strip()
            or os.getenv(f"PERCHAI_{idx}_PERCH_CLI_APP_URL", "").strip()
        )

    if not token or not app_url:
        return None
    return token, app_url


def _read_session_file(credential: str) -> Optional[Tuple[str, str]]:
    """Read an OAuth session file path and extract ``(access_token, app_url)``.

    Returns ``None`` if the file is missing, unreadable, or malformed.  All
    exceptions are swallowed and converted to ``None`` so background
    polling never crashes the host.
    """
    try:
        path = Path(credential).expanduser()
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    access_token = data.get("accessToken")
    app_url = data.get("appUrl")
    if not isinstance(access_token, str) or not access_token:
        return None
    if not isinstance(app_url, str) or not app_url:
        return None
    return access_token, app_url


class PerchaiQuotaTracker:
    """
    Mixin class providing dollar-based quota tracking for the Perchai provider.

    This mixin adds:
    - Model quota groups for the TUI (``monthly($)`` group)
    - Calendar-month-UTC usage reset configuration
    - Background job config polled every 5 minutes by the executor
    - Real ``run_background_job()`` that fetches per-credential usage from
      the ``/api/perch-terminal/usage`` endpoint and pushes baselines into
      the ``UsageManager`` as cents-based counters.

    Usage::

        class PerchaiProvider(PerchaiQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in ``__init__``:
        self._balance_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300
    """

    # Type hints for attributes provided by the provider instance
    _balance_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # =========================================================================
    # QUOTA GROUPING
    # =========================================================================

    # Single monthly dollar quota group - TUI surfaces this under "monthly($)".
    model_quota_groups: Dict[str, List[str]] = {
        "monthly($)": ["_balance_monthly"],
    }

    # =========================================================================
    # USAGE RESET CONFIG (Calendar Month UTC)
    # =========================================================================

    # ``window_seconds`` here is a placeholder; the runtime value is computed
    # per-call by ``get_usage_reset_config()`` based on time-to-next-month UTC.
    usage_reset_configs: Dict[Any, Any] = {
        "default": UsageResetConfigDef(
            window_seconds=2592000,  # ~30d placeholder; actual value computed at runtime
            mode="per_model",
            description="Perchai monthly Pro quota",
            field_name="monthly",
        ),
    }

    @staticmethod
    def _seconds_until_next_month_utc() -> int:
        """
        Return the number of seconds until the next 1st-of-month 00:00 UTC.

        Used by ``get_usage_reset_config`` so the runtime window always
        reflects the time remaining in the current calendar month rather
        than a fixed 30-day guess.

        Returns:
            Seconds until the next month boundary (UTC).  Always > 0.
        """
        now_utc = datetime.now(timezone.utc)
        year = now_utc.year
        month = now_utc.month

        # First day of next month at 00:00 UTC
        if month == 12:
            next_month_start = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month_start = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        delta = next_month_start - now_utc
        # Guard against sub-second drift producing a zero/negative value.
        seconds = int(delta.total_seconds())
        return seconds if seconds > 0 else 1

    # =========================================================================
    # PROVIDER INTERFACE OVERRIDES
    # =========================================================================

    def get_usage_reset_config(
        self, credential: str
    ) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Perchai credentials.

        Pro plan resets on the calendar month boundary (00:00 UTC on the
        1st of each month).  The window is reported in ``per_model`` mode
        so UsageManager tracks each model group against its own window.

        Args:
            credential: API key / identifier for the credential being checked.
                Currently unused - Perchai uses a single global reset window
                shared across all Pro credentials.

        Returns:
            Dict with ``mode`` and ``window_seconds`` keys, or None if
            usage tracking is disabled for this credential.
        """
        return {
            "mode": "per_model",
            "window_seconds": self._seconds_until_next_month_utc(),
        }

    @staticmethod
    def get_background_job_config() -> Optional[Dict[str, Any]]:
        """
        Return configuration for the periodic Perchai quota refresh job.

        Returns:
            None if no background job, otherwise a dict with ``interval``,
            ``name``, and ``run_on_start`` keys.

        Notes:
            Exposed as ``@staticmethod`` so the executor can read it
            without instantiating the mixin.  The mixin is intended to be
            composed into the concrete provider via MRO
            ``(PerchaiQuotaTracker, ProviderInterface)``.
        """
        return {
            "interval": 300,
            "name": "perchai_quota_refresh",
            "run_on_start": True,
        }

    # =========================================================================
    # BACKGROUND JOB (REAL IMPLEMENTATION)
    # =========================================================================

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh Perchai usage baselines for each credential.

        For every credential path or ``env://perchai/N`` virtual path, this
        resolves the bearer token + app URL, GETs the upstream
        ``/api/perch-terminal/usage`` endpoint, parses the monthly dollar
        usage + cap + reset timestamp, and pushes a cents-based baseline
        into the shared ``UsageManager`` under the ``perchai/_balance_monthly``
        quota group.

        All errors are caught and logged; this method never raises into
        the background refresher loop.

        Args:
            usage_manager: The shared ``UsageManager`` instance.
            credentials: List of credential paths or ``env://perchai/N``
                virtual paths.
        """
        if not credentials:
            return

        semaphore = asyncio.Semaphore(USAGE_FETCH_CONCURRENCY)

        async def fetch_one(credential: str) -> None:
            async with semaphore:
                await self._refresh_one_credential(usage_manager, credential)

        tasks = [fetch_one(credential) for credential in credentials]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh_one_credential(
        self,
        usage_manager: "UsageManager",
        credential: str,
    ) -> None:
        """Fetch usage for a single credential and push to UsageManager.

        Errors are caught and logged at warning level - this method never
        raises into the background refresher loop.
        """
        identifier = _get_credential_identifier(credential)
        try:
            resolved = self._resolve_credential_path(credential)
            if resolved is None:
                lib_logger.debug(
                    f"Perchai: skipping credential {identifier} - "
                    "no usable token/appUrl"
                )
                return

            token, app_url = resolved
            data = await self._fetch_usage_data(credential, token, app_url)
            if data is None:
                return

            used_cents, cap_cents, reset_ts = self._extract_dollar_fields(data)

            self._balance_cache[credential] = {
                "status": "success",
                "monthly": {
                    "usage_cents": used_cents,
                    "cap_cents": cap_cents,
                    "reset_ts": reset_ts,
                },
                "fetched_at": _now_seconds(),
            }

            await usage_manager.update_quota_baseline(
                accessor=credential,
                model="perchai/_balance_monthly",
                quota_max_requests=cap_cents,
                quota_used=used_cents,
                quota_reset_ts=reset_ts,
                quota_group="monthly($)",
                force=True,
            )

            lib_logger.debug(
                f"Perchai quota refresh ({identifier}): "
                f"monthly=${used_cents / CENTS_PER_DOLLAR:.2f}"
                f"/${cap_cents / CENTS_PER_DOLLAR:.2f}, "
                f"reset_ts={reset_ts}"
            )

        except Exception as exc:
            lib_logger.warning(
                f"Perchai quota refresh failed for {identifier}: {exc}"
            )

    def _resolve_credential_path(
        self, credential: str
    ) -> Optional[Tuple[str, str]]:
        """Resolve ``credential`` to ``(access_token, app_url)``.

        Handles two shapes:
        - ``env://perchai/N`` - read from environment variables
        - file path           - parse the perchai session file

        Returns ``None`` if no usable credentials can be found (caller
        treats this as "skip").
        """
        if _is_env_path(credential):
            return _resolve_env_credentials(credential)
        return _read_session_file(credential)

    async def _fetch_usage_data(
        self, credential: str, token: str, app_url: str
    ) -> Optional[Dict[str, Any]]:
        """GET ``{appUrl}/api/perch-terminal/usage`` with Bearer auth.

        Returns the parsed JSON dict on 2xx, ``None`` on any error.  All
        exceptions are swallowed at this layer so the caller can treat
        None as "skip this baseline update".
        """
        identifier = _get_credential_identifier(credential)
        url = f"{app_url.rstrip('/')}/api/perch-terminal/usage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=USAGE_FETCH_TIMEOUT) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            lib_logger.warning(
                f"Perchai usage fetch for {identifier} returned "
                f"HTTP {exc.response.status_code}: {exc.response.text[:200]!r}"
            )
            return None
        except Exception as exc:
            lib_logger.warning(
                f"Perchai usage fetch for {identifier} failed: {exc}"
            )
            return None

    @staticmethod
    def _extract_dollar_fields(
        data: Dict[str, Any],
    ) -> Tuple[int, int, Optional[float]]:
        """Pull monthly ``(used_cents, cap_cents, reset_ts)`` from the usage response.

        The upstream response shape is not yet pinned, so this walks a set
        of candidate field paths.  If nothing matches, returns ``(0, 0,
        None)`` so the provider doesn't lose its baseline.

        Args:
            data: Parsed JSON body from ``/api/perch-terminal/usage``.

        Returns:
            Tuple of ``(used_cents, cap_cents, reset_ts)``.  All values
            default to 0/None on parse failure.
        """
        used_dollars = 0.0
        cap_dollars = 0.0
        reset_ts: Optional[float] = None

        # Walk candidate paths for "monthly" sub-object.
        monthly_obj: Any = None
        for path in (("monthly",), ("data", "monthly"), ("usage", "monthly")):
            current: Any = data
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
                if current is None:
                    break
            if current is not None:
                monthly_obj = current
                break

        if isinstance(monthly_obj, dict):
            for key in ("usageUsd", "usd", "usage", "used", "spend"):
                val = monthly_obj.get(key)
                if isinstance(val, (int, float)):
                    used_dollars = float(val)
                    break
            for key in ("limitUsd", "capUsd", "cap", "limit", "allowanceUsd"):
                val = monthly_obj.get(key)
                if isinstance(val, (int, float)):
                    cap_dollars = float(val)
                    break
            for key in ("resetAt", "reset_at", "resets_at", "nextResetAt"):
                val = monthly_obj.get(key)
                if isinstance(val, str):
                    reset_ts = _parse_iso_to_unix(val)
                    if reset_ts is not None:
                        break

        if used_dollars == 0.0 and cap_dollars == 0.0:
            for key in ("usedUsd", "used", "usage", "balance"):
                val = data.get(key)
                if isinstance(val, (int, float)):
                    used_dollars = float(val)
                    break
            for key in ("capUsd", "limit", "cap", "allowance"):
                val = data.get(key)
                if isinstance(val, (int, float)):
                    cap_dollars = float(val)
                    break

        # If we still don't have a reset_ts, use the calendar-month boundary.
        if reset_ts is None:
            reset_ts = float(PerchaiQuotaTracker._seconds_until_next_month_utc() + _now_seconds())

        used_cents = _safe_int(round(used_dollars * CENTS_PER_DOLLAR))
        cap_cents = _safe_int(round(cap_dollars * CENTS_PER_DOLLAR))
        return used_cents, cap_cents, reset_ts


def _parse_iso_to_unix(iso_string: str) -> Optional[float]:
    """Parse an ISO 8601 timestamp to Unix seconds; None on failure."""
    if not iso_string:
        return None
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _now_seconds() -> float:
    return time.time()
