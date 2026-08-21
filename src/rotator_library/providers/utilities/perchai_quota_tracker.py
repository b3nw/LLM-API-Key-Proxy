# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kévin Cojean

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


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_env_path(credential: str) -> bool:
    return isinstance(credential, str) and credential.startswith("env://perchai/")


def _parse_env_index(credential: str) -> Optional[int]:
    if not _is_env_path(credential):
        return None
    parts = credential[len("env://perchai/"):].split("/")
    if not parts or not parts[0]:
        return None
    idx = _safe_int(parts[0], 0)
    return idx


def _get_credential_identifier(credential: str) -> str:
    if credential.startswith("env://"):
        return credential
    return Path(credential).name


def _resolve_env_credentials(credential: str) -> Optional[Tuple[str, str]]:
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
    # Swallow exceptions to None so background polling never crashes the host.
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

    # Type hints for attributes provided by the provider instance
    _balance_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # Single monthly dollar quota group - TUI surfaces this under "monthly($)".
    model_quota_groups: Dict[str, List[str]] = {
        "monthly($)": ["_balance_monthly"],
    }

    # ``window_seconds`` is a placeholder; runtime value is computed per-call
    # by ``get_usage_reset_config()`` based on time-to-next-month UTC.
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
        now_utc = datetime.now(timezone.utc)
        year = now_utc.year
        month = now_utc.month

        if month == 12:
            next_month_start = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month_start = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        delta = next_month_start - now_utc
        seconds = int(delta.total_seconds())
        return seconds if seconds > 0 else 1

    def get_usage_reset_config(
        self, credential: str
    ) -> Optional[Dict[str, Any]]:
        return {
            "mode": "per_model",
            "window_seconds": self._seconds_until_next_month_utc(),
        }

    @staticmethod
    def get_background_job_config() -> Optional[Dict[str, Any]]:
        return {
            "interval": 300,
            "name": "perchai_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
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
        if _is_env_path(credential):
            return _resolve_env_credentials(credential)
        return _read_session_file(credential)

    async def _fetch_usage_data(
        self, credential: str, token: str, app_url: str
    ) -> Optional[Dict[str, Any]]:
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
        # Upstream response shape not yet pinned; walk candidate paths and
        # fall back to (0, 0, None) so the baseline isn't lost on mismatch.
        used_dollars = 0.0
        cap_dollars = 0.0
        reset_ts: Optional[float] = None

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

        if reset_ts is None:
            reset_ts = float(PerchaiQuotaTracker._seconds_until_next_month_utc() + _now_seconds())

        used_cents = _safe_int(round(used_dollars * CENTS_PER_DOLLAR))
        cap_cents = _safe_int(round(cap_dollars * CENTS_PER_DOLLAR))
        return used_cents, cap_cents, reset_ts


def _parse_iso_to_unix(iso_string: str) -> Optional[float]:
    if not iso_string:
        return None
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _now_seconds() -> float:
    return time.time()
