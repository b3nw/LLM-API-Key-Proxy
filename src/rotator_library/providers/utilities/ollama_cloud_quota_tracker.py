# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Ollama Cloud Quota Tracking Mixin

Fetches session + weekly usage percentages by scraping the Ollama Cloud
settings page (https://ollama.com/settings) using a __Secure-session cookie.

Ollama Cloud has no public JSON API for quota data (multiple community
requests: ollama/ollama#15663, #15132, #16448 — all closed without action).

Quota model:
- Session usage: resets every ~6 hours (observed), percentage-based
- Weekly usage: resets weekly, percentage-based
- Plans: free, pro, max — higher plans have higher limits
- Per-model cost tiers (1–4 slots visible on model pages)
- 429 responses indicate quota exhaustion

Credential config:
- OLLAMA_CLOUD_API_KEY_N: API key for LLM requests (standard env var pattern)
- OLLAMA_CLOUD_SESSION_COOKIE_N: browser session cookie for quota scraping (separate)
  Falls back to OLLAMA_CLOUD_SESSION_COOKIE (unnumbered) if no _N variant is set.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ...usage.manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

OLLAMA_CLOUD_BASE = "https://ollama.com"
OLLAMA_SETTINGS_URL = f"{OLLAMA_CLOUD_BASE}/settings"

QUOTA_FETCH_CONCURRENCY = 3
OLLAMA_QUOTA_REFRESH_INTERVAL_DEFAULT = 300


@dataclass
class OllamaCloudQuotaSnapshot:
    """Server-reported quota state scraped from ollama.com/settings."""

    credential_path: str
    identifier: str
    plan: Optional[str]  # "free" | "pro" | "max" | None
    session_pct: Optional[float]  # 0.0–100.0
    weekly_pct: Optional[float]  # 0.0–100.0
    session_reset: Optional[str]  # "4 hours", "2 hours", etc.
    weekly_reset: Optional[str]  # "3 days", "6 days", etc.
    session_breakdown: List[Dict[str, Any]]  # per-model usage
    weekly_breakdown: List[Dict[str, Any]]  # per-model usage
    fetched_at: float
    status: str  # "success" | "error" | "no_cookie"
    error: Optional[str]


def _get_credential_identifier(credential: str) -> str:
    """Return a short, log-safe identifier for a credential."""
    if credential.startswith("env://"):
        return credential
    if len(credential) <= 8:
        return credential
    return f"{credential[:4]}...{credential[-4:]}"


def _resolve_credential_index(credential_path: str) -> Optional[int]:
    """
    Determine which OLLAMA_CLOUD_API_KEY_N env var holds this credential value.

    Scans OLLAMA_CLOUD_API_KEY_1 through _9 and returns the matching index,
    or None if the credential doesn't match any env var (e.g. added at runtime).
    """
    if credential_path.startswith("env://"):
        parts = credential_path[6:].split("/")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return None

    for i in range(1, 10):
        env_val = os.getenv(f"OLLAMA_CLOUD_API_KEY_{i}", "").strip()
        if env_val and env_val == credential_path:
            return i

    return None


def _extract_session_cookie(credential_path: str) -> Optional[str]:
    """
    Get the session cookie for a specific credential from environment variables.

    Resolves the credential's index (from OLLAMA_CLOUD_API_KEY_N) and looks up
    the corresponding OLLAMA_CLOUD_SESSION_COOKIE_N. Falls back to the
    unnumbered OLLAMA_CLOUD_SESSION_COOKIE if no per-index match is found.
    """
    idx = _resolve_credential_index(credential_path)
    if idx is not None:
        per_cred = os.getenv(f"OLLAMA_CLOUD_SESSION_COOKIE_{idx}", "").strip()
        if per_cred:
            return per_cred

    # Fallback to unnumbered
    return os.getenv("OLLAMA_CLOUD_SESSION_COOKIE", "").strip() or None


def _parse_settings_html(html: str) -> Dict[str, Any]:
    """
    Extract usage data from the Ollama settings page HTML.

    Parses patterns like:
      Session usage ... 4.6% used ... Resets in 4 hours
      Weekly usage ... 30.9% used ... Resets in 3 days
      Cloud Usage ... pro (plan badge)
      data-usage-track with per-model breakdown buttons
    """
    result: Dict[str, Any] = {}

    # Session usage percentage
    session_label = re.search(
        r'Session usage.*?(\d+(?:\.\d+)?)%\s*used', html, re.DOTALL
    )
    if session_label:
        result["session_pct"] = float(session_label.group(1))

    # Weekly usage percentage
    weekly_label = re.search(
        r'Weekly usage.*?(\d+(?:\.\d+)?)%\s*used', html, re.DOTALL
    )
    if weekly_label:
        result["weekly_pct"] = float(weekly_label.group(1))

    # Fallback: grab all "X.X% used" in order if labels weren't found
    if "session_pct" not in result or "weekly_pct" not in result:
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)%\s*used', html)
        if "session_pct" not in result and len(pct_matches) >= 1:
            result["session_pct"] = float(pct_matches[0])
        if "weekly_pct" not in result and len(pct_matches) >= 2:
            result["weekly_pct"] = float(pct_matches[1])

    # Reset timers
    reset_matches = re.findall(r'Resets in ([^<\n]+)', html)
    if len(reset_matches) >= 1:
        result["session_reset"] = reset_matches[0].strip()
    if len(reset_matches) >= 2:
        result["weekly_reset"] = reset_matches[1].strip()

    # Plan detection
    plan_match = re.search(
        r'Cloud Usage\s*</span>\s*<span[^>]*>\s*(pro|max|free|team|starter)\s*</span',
        html, re.IGNORECASE,
    )
    if not plan_match:
        plan_match = re.search(
            r'class="[^"]*capitalize[^"]*">\s*(pro|max|free|team|starter)\s*</span',
            html, re.IGNORECASE,
        )
    if plan_match:
        result["plan"] = plan_match.group(1).strip().lower()

    # Per-model breakdown from data-usage-track containers
    session_breakdown: List[Dict[str, Any]] = []
    weekly_breakdown: List[Dict[str, Any]] = []

    usage_tracks = re.findall(
        r'data-usage-track[^>]*aria-label="([^"]*usage[^"]*)"[^>]*>(.*?)</div>\s*</div>',
        html, re.DOTALL | re.IGNORECASE,
    )

    for aria_label, track_html in usage_tracks:
        button_pattern = re.compile(
            r'<button[^>]*data-usage-segment[^>]*>', re.DOTALL
        )
        buttons = button_pattern.findall(track_html)

        breakdown = []
        for btn in buttons:
            model_match = re.search(r'data-model="([^"]+)"', btn)
            req_match = re.search(r'data-requests="(\d+)"', btn)
            width_match = re.search(r'width:\s*([\d.]+)%', btn)
            if model_match and req_match and width_match:
                breakdown.append({
                    "model": model_match.group(1),
                    "requests": int(req_match.group(1)),
                    "pct": float(width_match.group(1)),
                })

        if "session" in aria_label.lower():
            session_breakdown = breakdown
        elif "weekly" in aria_label.lower():
            weekly_breakdown = breakdown

    result["session_breakdown"] = session_breakdown
    result["weekly_breakdown"] = weekly_breakdown

    return result


def _error_snapshot(
    credential_path: str, identifier: str, error_msg: str, status: str = "error"
) -> OllamaCloudQuotaSnapshot:
    """Return a snapshot representing a failed or skipped fetch."""
    return OllamaCloudQuotaSnapshot(
        credential_path=credential_path,
        identifier=identifier,
        plan=None,
        session_pct=None,
        weekly_pct=None,
        session_reset=None,
        weekly_reset=None,
        session_breakdown=[],
        weekly_breakdown=[],
        fetched_at=time.time(),
        status=status,
        error=error_msg,
    )


class OllamaCloudQuotaTracker:
    """
    Mixin class providing quota tracking for the Ollama Cloud provider.

    Scrapes https://ollama.com/settings using __Secure-session cookie to
    extract session/weekly usage percentages and per-model breakdowns.

    Usage:
        class OllamaCloudProvider(OllamaCloudQuotaTracker, ProviderInterface):
            ...
    """

    _quota_cache: Dict[str, OllamaCloudQuotaSnapshot]
    _quota_refresh_interval: int
    _usage_manager: Optional["UsageManager"]
    _initial_baselines_fetched: bool

    def _init_quota_tracker(self) -> None:
        self._quota_cache = {}
        self._quota_refresh_interval = int(
            os.getenv(
                "OLLAMA_CLOUD_QUOTA_REFRESH_INTERVAL",
                str(OLLAMA_QUOTA_REFRESH_INTERVAL_DEFAULT),
            )
        )
        self._usage_manager = None
        self._initial_baselines_fetched = False

    def set_usage_manager(self, usage_manager: "UsageManager") -> None:
        self._usage_manager = usage_manager

    async def _fetch_usage_for_credential(
        self, credential_path: str
    ) -> OllamaCloudQuotaSnapshot:
        """
        Scrape quota from ollama.com/settings for a single credential.

        Requires a session cookie in the credential or env var.
        """
        identifier = _get_credential_identifier(credential_path)
        session_cookie = _extract_session_cookie(credential_path)

        if not session_cookie:
            return _error_snapshot(
                credential_path, identifier,
                "No session cookie configured. Set OLLAMA_CLOUD_SESSION_COOKIE "
                "or use api_key:session_cookie format.",
                status="no_cookie",
            )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    OLLAMA_SETTINGS_URL,
                    follow_redirects=True,
                    cookies={"__Secure-session": session_cookie},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
                        ),
                    },
                )

                if resp.status_code == 200:
                    parsed = _parse_settings_html(resp.text)
                    if parsed.get("session_pct") is not None or parsed.get("weekly_pct") is not None:
                        snapshot = OllamaCloudQuotaSnapshot(
                            credential_path=credential_path,
                            identifier=identifier,
                            plan=parsed.get("plan"),
                            session_pct=parsed.get("session_pct"),
                            weekly_pct=parsed.get("weekly_pct"),
                            session_reset=parsed.get("session_reset"),
                            weekly_reset=parsed.get("weekly_reset"),
                            session_breakdown=parsed.get("session_breakdown", []),
                            weekly_breakdown=parsed.get("weekly_breakdown", []),
                            fetched_at=time.time(),
                            status="success",
                            error=None,
                        )
                        self._quota_cache[credential_path] = snapshot
                        lib_logger.debug(
                            f"Ollama Cloud quota fetched for {identifier}: "
                            f"plan={snapshot.plan}, session={snapshot.session_pct}%, "
                            f"weekly={snapshot.weekly_pct}%"
                        )
                        return snapshot
                    return _error_snapshot(
                        credential_path, identifier,
                        "Could not parse usage data from settings page. "
                        "Cookie may be expired or page structure changed.",
                    )
                elif resp.status_code in (401, 302):
                    return _error_snapshot(
                        credential_path, identifier,
                        "Session cookie expired or invalid.",
                    )
                else:
                    return _error_snapshot(
                        credential_path, identifier,
                        f"Unexpected HTTP {resp.status_code} from ollama.com/settings",
                    )

        except Exception as e:
            lib_logger.warning(
                f"Ollama Cloud quota fetch failed for {identifier}: {e}"
            )
            return _error_snapshot(credential_path, identifier, str(e))

    async def fetch_initial_baselines(
        self, credential_paths: List[str]
    ) -> Dict[str, OllamaCloudQuotaSnapshot]:
        """Batch fetch quota baselines for all credentials."""
        results: Dict[str, OllamaCloudQuotaSnapshot] = {}
        if not credential_paths:
            return results

        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def fetch_one(cred_path: str) -> Tuple[str, OllamaCloudQuotaSnapshot]:
            async with semaphore:
                snapshot = await self._fetch_usage_for_credential(cred_path)
                return cred_path, snapshot

        tasks = [fetch_one(c) for c in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in fetch_results:
            if isinstance(item, BaseException):
                lib_logger.warning(f"Ollama Cloud baseline fetch error: {item}")
                continue
            assert isinstance(item, tuple)
            cred_path, snapshot = item
            results[cred_path] = snapshot

        success_count = sum(1 for s in results.values() if s.status == "success")
        lib_logger.info(
            f"Ollama Cloud: fetched {success_count}/{len(credential_paths)} quota baselines"
        )
        return results

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, OllamaCloudQuotaSnapshot],
        usage_manager: "UsageManager",
        force: bool = True,
        is_initial_fetch: bool = False,
    ) -> int:
        """
        Push quota snapshots to the UsageManager.

        Stores session and weekly usage as separate virtual model baselines.
        apply_exhaustion=True when usage >= 95% to trigger fallback routing.
        """
        stored_count = 0
        provider_prefix = getattr(self, "provider_env_name", "ollama_cloud")

        for cred_path, snapshot in quota_results.items():
            if snapshot.status != "success":
                continue

            # Store session usage as a baseline
            if snapshot.session_pct is not None:
                session_reset_ts = self._parse_reset_to_ts(snapshot.session_reset)
                session_used = int(snapshot.session_pct)
                is_exhausted = snapshot.session_pct >= 95.0

                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=f"{provider_prefix}/_session",
                        quota_max_requests=100,
                        quota_reset_ts=session_reset_ts,
                        quota_used=session_used,
                        quota_group="session",
                        force=force,
                        apply_exhaustion=is_exhausted,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Ollama Cloud session baseline "
                        f"for {snapshot.identifier}: {e}"
                    )

            # Store weekly usage as a baseline (same "session" group — both
            # contribute to the unified exhaustion check for this credential)
            if snapshot.weekly_pct is not None:
                weekly_reset_ts = self._parse_reset_to_ts(snapshot.weekly_reset)
                weekly_used = int(snapshot.weekly_pct)
                is_exhausted = snapshot.weekly_pct >= 95.0

                try:
                    await usage_manager.update_quota_baseline(
                        accessor=cred_path,
                        model=f"{provider_prefix}/_weekly",
                        quota_max_requests=100,
                        quota_reset_ts=weekly_reset_ts,
                        quota_used=weekly_used,
                        quota_group="session",
                        force=force,
                        apply_exhaustion=is_exhausted,
                    )
                    stored_count += 1
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to store Ollama Cloud weekly baseline "
                        f"for {snapshot.identifier}: {e}"
                    )

        return stored_count

    @staticmethod
    def _parse_reset_to_ts(reset_str: Optional[str]) -> Optional[float]:
        """Convert 'Resets in X hours/days' to a Unix timestamp."""
        if not reset_str:
            return None
        now = time.time()
        hours_match = re.search(r'(\d+)\s*hour', reset_str.lower())
        days_match = re.search(r'(\d+)\s*day', reset_str.lower())
        minutes_match = re.search(r'(\d+)\s*min', reset_str.lower())

        seconds = 0
        if days_match:
            seconds += int(days_match.group(1)) * 86400
        if hours_match:
            seconds += int(hours_match.group(1)) * 3600
        if minutes_match:
            seconds += int(minutes_match.group(1)) * 60

        return (now + seconds) if seconds > 0 else None

    def get_cached_quota(
        self, credential_path: str
    ) -> Optional[OllamaCloudQuotaSnapshot]:
        """Return the most recently fetched snapshot for a credential."""
        return self._quota_cache.get(credential_path)

    async def get_all_quota_info(
        self,
        credential_paths: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Get aggregated quota info for all credentials."""
        if force_refresh:
            results = await self.fetch_initial_baselines(credential_paths)
        else:
            results = {
                p: self._quota_cache[p]
                for p in credential_paths
                if p in self._quota_cache
            }

        credentials_out: Dict[str, Any] = {}
        for path, snapshot in results.items():
            credentials_out[path] = {
                "identifier": snapshot.identifier,
                "status": snapshot.status,
                "plan": snapshot.plan,
                "session_pct": snapshot.session_pct,
                "weekly_pct": snapshot.weekly_pct,
                "session_reset": snapshot.session_reset,
                "weekly_reset": snapshot.weekly_reset,
                "session_breakdown": snapshot.session_breakdown,
                "fetched_at": snapshot.fetched_at,
                "error": snapshot.error,
            }

        return {
            "credentials": credentials_out,
            "summary": {
                "total_credentials": len(credential_paths),
                "fetched": sum(
                    1 for s in results.values() if s.status == "success"
                ),
            },
            "timestamp": time.time(),
        }

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Configure periodic quota refresh."""
        return {
            "interval": self._quota_refresh_interval,
            "name": "ollama_cloud_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """Periodic refresh cycle: fetch → push."""
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
