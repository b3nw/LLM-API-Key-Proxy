# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
ClinePass Quota Tracking Mixin.

Fetches ClinePass subscription usage from the Cline API:

    GET https://api.cline.bot/api/v1/users/me/plan/usage-limits
    GET https://api.cline.bot/api/v1/users/me/plan

The dashboard exposes three percent-based windows:

    five_hour — rolling 5-hour window
    weekly    — calendar week
    monthly   — calendar month

Each window is its own quota group so the WebUI shows them independently
(consistent with the ollama-cloud session/weekly pattern).

Credentials are Bearer API keys (``CLINE_PASS_API_KEY_N``) or account auth
tokens; both are accepted at the upstream ``/api/v1`` surface.

Quota model:
- Polling endpoint: ``GET /api/v1/users/me/plan/usage-limits``
  Response: ``{"success": true, "data": {"limits": [{"type": ..., "percentUsed": 0..100}, ...]}}``
- Plan context: ``GET /api/v1/users/me/plan``
  - ``displayName`` (e.g. "ClinePass")
  - ``currentPeriodStart`` / ``currentPeriodEnd`` (ISO timestamps)
  - ``entitlements.cline_pass.inferenceCapThreshold`` (USD cost cap; may be
    a sentinel on internal plans — surfaced as ``plan_hard_cap_usd`` but not
    used for exhaustion).
- Polling cadence: 900 s default (15 min). Limits are percentage-based, not
  per-request headers, so a poll interval shorter than the smallest window
  gives timely cooldown signalling without being wasteful.
- 401/403 from the quota endpoint → standard error-cooldown path on the
  affected credential; the tracker flags the snapshot ``status="auth_error"``
  so the WebUI can render "auth required" rather than "exhausted".

Env vars:
    CLINE_PASS_API_KEY_1, _2, ...   — Bearer keys
    CLINE_PASS_API_BASE              — override (default https://api.cline.bot/api/v1)
    CLINE_PASS_QUOTA_REFRESH_INTERVAL — seconds (default 900)
    CLINE_PASS_QUOTA_EXHAUSTION_PCT   — exhaustion threshold (default 95.0)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage.manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

CLINE_PASS_API_BASE_DEFAULT = "https://api.cline.bot/api/v1"
DEFAULT_QUOTA_REFRESH_INTERVAL = 900  # 15 min — matches cline.md recommendation
DEFAULT_EXHAUSTION_PCT = 95.0
QUOTA_FETCH_CONCURRENCY = 4

# Window key -> quota group name. The ``type`` field from the upstream payload
# is the canonical key; we map it to a friendlier display group for the UI.
WINDOW_TYPE_TO_GROUP: Dict[str, str] = {
    "five_hour": "5h",
    "weekly": "weekly",
    "monthly": "monthly",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_credential_identifier(credential_path: str) -> str:
    """Short, log-safe identifier for a credential."""
    if credential_path.startswith("env://"):
        return credential_path
    if len(credential_path) <= 8:
        return credential_path
    return f"{credential_path[:4]}...{credential_path[-4:]}"


def _parse_iso_to_ts(value: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 timestamp string to a Unix epoch (seconds)."""
    if not value or not isinstance(value, str):
        return None
    try:
        # Tolerate trailing Z (RFC 3339) by converting to +00:00
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _coerce_percent(value: Any) -> Optional[float]:
    """Coerce an upstream ``percentUsed`` value to ``[0, 100]`` float.

    The Cline API returns ints or floats; defensive coercion handles strings
    and out-of-range negatives that have appeared in earlier dashboard versions.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            pct = float(value)
        elif isinstance(value, str) and value.strip():
            pct = float(value.strip())
        else:
            return None
    except (TypeError, ValueError):
        return None
    # Clamp to the documented 0..100 range
    if pct < 0.0:
        return 0.0
    if pct > 100.0:
        return 100.0
    return pct


def _as_optional_float(value: Any) -> Optional[float]:
    """Coerce a possibly-numeric value to ``float``; ``None`` on failure."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value.strip())
    except (TypeError, ValueError):
        return None
    return None


def _resolve_api_base() -> str:
    """Resolve the Cline API base (allows override for testing).

    The upstream is rooted at https://api.cline.bot/api/v1 — every
    endpoint (chat completions, models, usage-limits, plan) lives
    under that ``/api/v1`` prefix. We do **not** strip the trailing
    ``/v1``: doing so produces ``https://api.cline.bot/api/...`` which
    404s on the upstream (caught in deployment 2026-07-11).
    """
    # ``or`` (not the second arg of getenv) so an empty string falls back to
    # the default — operators occasionally set ``CLINE_PASS_API_BASE=""`` to
    # "reset" and we shouldn't break the tracker when they do.
    return (
        os.getenv("CLINE_PASS_API_BASE") or CLINE_PASS_API_BASE_DEFAULT
    ).rstrip("/")


def _build_billing_url(path: str) -> str:
    """Join the API base with a path.

    The Cline API is a flat ``/api/v1`` namespace; no path rewriting is
    needed. Path may be passed with or without a leading slash.
    """
    base = _resolve_api_base()
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


def parse_usage_limits_payload(data: Any) -> Dict[str, Optional[float]]:
    """Parse the ``GET /users/me/plan/usage-limits`` response.

    Accepts both the documented envelope (``{"success": true, "data": {"limits": [...]}}``)
    and a flat ``{"limits": [...]}`` shape in case the upstream version changes.

    Returns:
        Dict keyed by window type (``five_hour``/``weekly``/``monthly``) with
        ``percentUsed`` floats (0..100) or ``None`` when absent.
    """
    if not isinstance(data, dict):
        return {}

    inner: Any = data
    # Unwrap common envelopes: success/data, data/, payload/
    for envelope in ("data", "payload", "result"):
        candidate = data.get(envelope)
        if isinstance(candidate, dict):
            inner = candidate
            break

    raw_limits = inner.get("limits") if isinstance(inner, dict) else None
    if not isinstance(raw_limits, list):
        return {}

    parsed: Dict[str, Optional[float]] = {
        "five_hour": None,
        "weekly": None,
        "monthly": None,
    }
    for entry in raw_limits:
        if not isinstance(entry, dict):
            continue
        window = entry.get("type")
        if not isinstance(window, str):
            continue
        pct = _coerce_percent(entry.get("percentUsed"))
        # Map known window types; ignore unknown ones (forward compatibility)
        if window in parsed:
            parsed[window] = pct
    return parsed


def parse_plan_payload(data: Any) -> Dict[str, Any]:
    """Parse the ``GET /users/me/plan`` response.

    Extracts:
        - display_name
        - current_period_start_ts / current_period_end_ts (Unix seconds)
        - plan_hard_cap_usd (from entitlements.cline_pass.inferenceCapThreshold)
    """
    if not isinstance(data, dict):
        return {}

    # The plan endpoint is documented to return the plan object directly
    # (no envelope). Defensive unwrap in case that changes.
    plan: Any = data
    for envelope in ("data", "plan", "result"):
        candidate = data.get(envelope)
        if isinstance(candidate, dict):
            plan = candidate
            break

    if not isinstance(plan, dict):
        return {}

    display_name = plan.get("displayName") or plan.get("name")
    period_start = _parse_iso_to_ts(
        plan.get("currentPeriodStart") or plan.get("current_period_start")
    )
    period_end = _parse_iso_to_ts(
        plan.get("currentPeriodEnd") or plan.get("current_period_end")
    )

    plan_hard_cap_usd: Optional[float] = None
    entitlements = plan.get("entitlements")
    if isinstance(entitlements, dict):
        cline_pass = entitlements.get("cline_pass") or entitlements.get("clinePass")
        if isinstance(cline_pass, dict):
            cap = cline_pass.get("inferenceCapThreshold")
            if cap is not None:
                plan_hard_cap_usd = _as_optional_float(cap)
        # Some versions may expose it directly
        if plan_hard_cap_usd is None:
            cap = entitlements.get("inferenceCapThreshold")
            if cap is not None:
                plan_hard_cap_usd = _as_optional_float(cap)

    return {
        "display_name": display_name if isinstance(display_name, str) else None,
        "current_period_start_ts": period_start,
        "current_period_end_ts": period_end,
        "plan_hard_cap_usd": plan_hard_cap_usd,
    }


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClinePassQuotaSnapshot:
    """One credential's most recent quota observation."""

    credential_path: str
    identifier: str
    # Window percentages (0..100) keyed by window type
    five_hour_pct: Optional[float] = None
    weekly_pct: Optional[float] = None
    monthly_pct: Optional[float] = None
    # Plan context
    display_name: Optional[str] = None
    current_period_start_ts: Optional[float] = None
    current_period_end_ts: Optional[float] = None
    plan_hard_cap_usd: Optional[float] = None
    # Fetch metadata
    fetched_at: float = field(default_factory=time.time)
    status: str = "pending"  # "success" | "error" | "auth_error" | "no_key"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class ClinePassQuotaTracker:
    """
    Mixin for ClinePass provider: percent-window quota tracking via the
    ``/users/me/plan/usage-limits`` polling endpoint.

    Usage::

        class ClinePassProvider(ClinePassQuotaTracker, ProviderInterface):
            ...
    """

    _quota_cache: Dict[str, ClinePassQuotaSnapshot]
    _quota_refresh_interval: int
    _exhaustion_pct: float
    _usage_manager: Optional["UsageManager"]
    _initial_baselines_fetched: bool

    def _init_quota_tracker(self) -> None:
        self._quota_cache = {}
        self._quota_refresh_interval = int(
            os.getenv(
                "CLINE_PASS_QUOTA_REFRESH_INTERVAL",
                str(DEFAULT_QUOTA_REFRESH_INTERVAL),
            )
        )
        self._exhaustion_pct = float(
            os.getenv(
                "CLINE_PASS_QUOTA_EXHAUSTION_PCT",
                str(DEFAULT_EXHAUSTION_PCT),
            )
        )
        self._usage_manager = None
        self._initial_baselines_fetched = False

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        self._usage_manager = usage_manager

    # ------------------------------------------------------------------ Fetch

    def _build_proxy_client_kwargs(self, credential_path: str) -> Dict[str, Any]:
        """Hook for the proxy mixin (if the project provides one). No-op by default."""
        return {}

    async def _fetch_quota_for_credential(
        self, credential_path: str
    ) -> ClinePassQuotaSnapshot:
        """Fetch and parse the latest ClinePass quota snapshot for one credential."""
        identifier = _get_credential_identifier(credential_path)

        if not credential_path:
            return ClinePassQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                fetched_at=time.time(),
                status="no_key",
                error="Empty credential identifier",
            )

        # ``env://``-style credentials: we still need the raw key to call the
        # upstream endpoint, so resolve to the env var value here.
        bearer = self._resolve_bearer(credential_path)
        if not bearer:
            return ClinePassQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                fetched_at=time.time(),
                status="no_key",
                error="Could not resolve Bearer key for ClinePass credential",
            )

        headers = {
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/json",
        }
        proxy_kwargs = self._build_proxy_client_kwargs(credential_path)

        limits_url = _build_billing_url("/users/me/plan/usage-limits")
        plan_url = _build_billing_url("/users/me/plan")

        try:
            async with httpx.AsyncClient(timeout=30.0, **proxy_kwargs) as client:
                limits_resp = await client.get(limits_url, headers=headers)
                if limits_resp.status_code in (401, 403):
                    err_text = limits_resp.text[:200] if limits_resp.text else ""
                    lib_logger.warning(
                        f"ClinePass quota auth failed for {identifier}: "
                        f"HTTP {limits_resp.status_code} {err_text}"
                    )
                    return ClinePassQuotaSnapshot(
                        credential_path=credential_path,
                        identifier=identifier,
                        fetched_at=time.time(),
                        status="auth_error",
                        error=f"HTTP {limits_resp.status_code} on usage-limits",
                    )
                limits_resp.raise_for_status()
                limits_data = limits_resp.json()

                # Plan metadata is best-effort — if it 404s or errors we still
                # want to surface the percent windows.
                plan_data: Dict[str, Any] = {}
                try:
                    plan_resp = await client.get(plan_url, headers=headers)
                    if plan_resp.status_code == 200:
                        plan_data = plan_resp.json()
                except Exception as e:
                    lib_logger.debug(
                        f"ClinePass plan fetch failed for {identifier} (non-fatal): {e}"
                    )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            err_text = e.response.text[:200] if e.response.text else ""
            lib_logger.warning(
                f"ClinePass quota fetch failed for {identifier}: "
                f"HTTP {status_code} {err_text}"
            )
            return ClinePassQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                fetched_at=time.time(),
                status="error",
                error=f"HTTP {status_code}: {err_text}",
            )
        except Exception as e:
            err = str(e)
            lib_logger.warning(
                f"ClinePass quota fetch failed for {identifier}: {err}"
            )
            return ClinePassQuotaSnapshot(
                credential_path=credential_path,
                identifier=identifier,
                fetched_at=time.time(),
                status="error",
                error=err,
            )

        windows = parse_usage_limits_payload(limits_data)
        plan_ctx = parse_plan_payload(plan_data)

        snapshot = ClinePassQuotaSnapshot(
            credential_path=credential_path,
            identifier=identifier,
            five_hour_pct=windows.get("five_hour"),
            weekly_pct=windows.get("weekly"),
            monthly_pct=windows.get("monthly"),
            display_name=plan_ctx.get("display_name"),
            current_period_start_ts=plan_ctx.get("current_period_start_ts"),
            current_period_end_ts=plan_ctx.get("current_period_end_ts"),
            plan_hard_cap_usd=plan_ctx.get("plan_hard_cap_usd"),
            fetched_at=time.time(),
            status="success",
            error=None,
        )
        self._quota_cache[credential_path] = snapshot
        return snapshot

    async def fetch_initial_baselines(
        self, credential_paths: List[str]
    ) -> Dict[str, ClinePassQuotaSnapshot]:
        """Batch fetch quota snapshots for all credentials concurrently."""
        results: Dict[str, ClinePassQuotaSnapshot] = {}
        if not credential_paths:
            return results

        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def fetch_one(cred_path: str):
            async with semaphore:
                return cred_path, await self._fetch_quota_for_credential(cred_path)

        tasks = [fetch_one(c) for c in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for item in fetch_results:
            if isinstance(item, BaseException):
                lib_logger.warning(f"ClinePass baseline fetch error: {item}")
                continue
            cred_path, snapshot = item
            results[cred_path] = snapshot

        success_count = sum(1 for s in results.values() if s.status == "success")
        lib_logger.info(
            f"ClinePass: fetched {success_count}/{len(credential_paths)} quota baselines"
        )
        return results

    # ----------------------------------------------------------------- Storage

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, ClinePassQuotaSnapshot],
        usage_manager: "UsageManager",
        force: bool = False,
        is_initial_fetch: bool = False,
    ) -> int:
        """Push snapshots into the UsageManager as three window baselines."""
        stored_count = 0
        provider_prefix = getattr(self, "provider_env_name", "cline_pass")
        exhaustion_threshold = self._exhaustion_pct

        for cred_path, snapshot in quota_results.items():
            if snapshot.status != "success":
                continue

            # Each window is its own quota group so the WebUI renders them
            # independently. ``quota_max_requests=100`` because the upstream
            # is percent-based; ``quota_used`` carries the integer percentage.
            windows = (
                ("five_hour", snapshot.five_hour_pct, "5h"),
                ("weekly", snapshot.weekly_pct, "weekly"),
                ("monthly", snapshot.monthly_pct, "monthly"),
            )
            for window_type, pct, group in windows:
                if pct is None:
                    continue
                reset_ts = snapshot.current_period_end_ts
                # The 5h window is rolling, not period-bound, so don't apply
                # the calendar reset timestamp to it.
                if window_type == "five_hour":
                    reset_ts = None
                pct_int = int(round(pct))
                is_exhausted = pct >= exhaustion_threshold
                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=f"{provider_prefix}/_{window_type}",
                        quota_max_requests=100,
                        quota_reset_ts=reset_ts,
                        quota_used=pct_int,
                        quota_group=group,
                        force=force,
                        apply_exhaustion=is_exhausted and is_initial_fetch,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store ClinePass {window_type} baseline "
                        f"for {snapshot.identifier}: {e}"
                    )
        return stored_count

    # --------------------------------------------------------- Background job

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        return {
            "interval": self._quota_refresh_interval,
            "name": "cline_pass_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        if not credentials:
            return
        self._usage_manager = usage_manager
        quota_results = await self.fetch_initial_baselines(credentials)
        is_initial = not self._initial_baselines_fetched
        stored = await self._store_baselines_to_usage_manager(
            quota_results,
            usage_manager,
            force=True,
            is_initial_fetch=is_initial,
        )
        if stored > 0:
            self._initial_baselines_fetched = True
        elif any(
            r.status == "success" for r in quota_results.values() if hasattr(r, "status")
        ):
            lib_logger.warning(
                "ClinePass: quota fetch succeeded but no baselines were stored "
                "(check percent window parsing)"
            )

    # ----------------------------------------------------------- Public access

    def get_cached_quota(
        self, credential_path: str
    ) -> Optional[ClinePassQuotaSnapshot]:
        return self._quota_cache.get(credential_path)

    # ----------------------------------------------------- Internal helpers

    def _resolve_bearer(self, credential_path: str) -> Optional[str]:
        """Resolve a credential identifier to a raw Bearer API key.

        The env-var loader at ``proxy_app/main.py`` produces keys like
        ``env://cline_pass/1`` for numbered env vars and the raw value for
        un-numbered. We handle both shapes.
        """
        if not credential_path:
            return None
        # ``env://cline_pass/N`` → CLINE_PASS_API_KEY_N
        if credential_path.startswith("env://"):
            rest = credential_path[len("env://"):]
            parts = rest.split("/", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return os.getenv(f"CLINE_PASS_API_KEY_{parts[1]}")
            return os.getenv(f"CLINE_PASS_API_KEY")
        # Raw value (already a Bearer key)
        if credential_path.startswith("cv-") or "_API_KEY" not in credential_path:
            return credential_path
        # Defensive: if a credential path looks like an env-var name, resolve it
        if credential_path.isupper() and credential_path.startswith("CLINE_PASS"):
            return os.getenv(credential_path)
        return credential_path
