# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/perchai_auth_base.py
"""
Perchai CLI authentication base class.

Reads the OAuth session file written by the perchai CLI
(`~/.perch/cli-auth-session.json`) and provides reactive token refresh
on HTTP 401 responses.

Known limitations:
- No file locking on the session file. Concurrent processes (e.g. two proxy
  instances pointed at the same session) can race during refresh and the
  last writer wins. Single-process use is safe.
- Login flow is NOT implemented. Users must run `perch login` externally
  to populate the session file before this class is usable.
- Token refresh is reactive (triggered by 401), not proactive. Tokens that
  expire between requests will fail the first call before being refreshed.

Session file format (camelCase, written by perchai CLI v2.4.87):
    {
        "version": 1,
        "appUrl": "https://app.perchai.app",
        "accessToken": "...",
        "refreshToken": "...",
        "expiresAt": 1234567890,   # unix seconds, optional
        "userId": "user_xxx",      # optional
        "email": "user@example",   # optional
        "updatedAt": 1234567890    # unix seconds
    }

Perchai hosts its auth on Supabase GoTrue. The Supabase URL and anon
key are not stored in the session file; they are discovered once per
process via a public config endpoint and cached on the instance.

Config discovery (GET {appUrl}/api/perch-terminal/cli-auth/config):
    Response body: {"ok": true,
                    "appUrl": "https://app.perchai.app",
                    "supabaseUrl": "https://<id>.supabase.co",
                    "supabaseAnonKey": "sb_publishable_...",
                    "providers": ["google", "github"],
                    "redirectHost": "127.0.0.1"}

Refresh endpoint (POST {supabaseUrl}/auth/v1/token?grant_type=refresh_token):
    Request headers:
        apikey:        <supabaseAnonKey>     (required, public anon key)
        Authorization: Bearer <current_access_token>   (recommended)
        Content-Type:  application/json
    Request body:  {"refresh_token": "<refresh_token>"}   (snake_case, single key)
    Response body (flat snake_case, NOT wrapped in a ``session`` key):
        {"access_token":  "eyJ...",
         "token_type":    "bearer",
         "expires_in":    3600,
         "expires_at":    1787250252,
         "refresh_token": "z5kin7gtpbxm",
         "user":          {"id": "user_xxx", "email": "..."}}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Final, List, Optional, TypedDict, final

import httpx


class PerchaiSession(TypedDict, total=False):
    """Session data shape, matches the perchai CLI session file (camelCase)."""

    version: int
    appUrl: str
    accessToken: str
    refreshToken: str
    expiresAt: Optional[str]
    userId: Optional[str]


class PerchaiAuthError(Exception):
    """Raised for missing, corrupted, or unrefreshable perchai sessions."""


lib_logger = logging.getLogger("rotator_library")
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())
lib_logger.propagate = False


# Module-level lazy lock. Created on first call to _get_lock() so that
# __init__ (which runs before the asyncio loop starts) doesn't trip the
# Python 3.10+ warning about Locks created outside a running loop.
_refresh_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    """Return the module-shared single-flight refresh lock, creating it lazily."""
    global _refresh_lock
    if _refresh_lock is None:
        _refresh_lock = asyncio.Lock()
    return _refresh_lock


def _resolve_session_file() -> Path:
    """Resolve the session file path, honoring PERCHAI_OAUTH_<N> overrides.

    Order:
    1. First PERCHAI_OAUTH_<N> env var whose value is an existing file path.
    2. PERCH_CLI_AUTH_DIR env var (CLI convention) joined with the filename.
    3. Default ~/.perch/cli-auth-session.json.
    """
    filename = "cli-auth-session.json"

    for key, value in os.environ.items():
        if key.startswith("PERCHAI_OAUTH_") and value:
            candidate = Path(value).expanduser()
            if candidate.is_file():
                return candidate

    base = os.environ.get("PERCH_CLI_AUTH_DIR", "").strip()
    if base:
        return Path(base).expanduser() / filename

    return Path.home() / ".perch" / filename


@final
class PerchaiAuthBase:
    """
    Perchai CLI OAuth session reader with reactive token refresh.

    NOT a ProviderInterface subclass. Used by PerchaiProvider via composition.
    Shares state with other instances via the module-level refresh lock so
    that concurrent 401s on the same credential trigger only one refresh.
    """

    SESSION_FILE: Final[Path] = Path.home() / ".perch" / "cli-auth-session.json"
    DEFAULT_APP_URL: Final[str] = "https://app.perchai.app"
    CONFIG_PATH: Final[str] = "/api/perch-terminal/cli-auth/config"
    REFRESH_PATH: Final[str] = "/auth/v1/token"
    REFRESH_TIMEOUT: Final[float] = 30.0
    CONFIG_TIMEOUT: Final[float] = 15.0

    def __init__(self) -> None:
        self._session: Optional[PerchaiSession] = None
        self._refresh_lock: Optional[asyncio.Lock] = None
        self._model_cache: Dict[str, List[str]] = {}
        self._model_cache_ttl: float = 300.0
        self._model_cache_filled_at: float = 0.0
        self._supabase_url: Optional[str] = None
        self._supabase_anon_key: Optional[str] = None

    # =========================================================================
    # SESSION LOADING
    # =========================================================================

    def load_session(self) -> PerchaiSession:
        """Load and validate the perchai CLI session file.

        Reads JSON from the resolved session path (default
        `~/.perch/cli-auth-session.json`, overridable via PERCHAI_OAUTH_<N>
        or PERCH_CLI_AUTH_DIR). Caches the result on `self._session`.

        Returns the validated PerchaiSession dict.

        Raises:
            PerchaiAuthError: If the file is missing, unreadable, malformed,
                or missing required fields. Messages are actionable.
        """
        if self._session is not None:
            return self._session

        session_path = _resolve_session_file()

        if not session_path.is_file():
            raise PerchaiAuthError(
                f"Perchai session file not found at {session_path}. "
                f"Run `perch login` to authenticate, or set "
                f"PERCHAI_OAUTH_1=/path/to/cli-auth-session.json."
            )

        try:
            raw = session_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PerchaiAuthError(
                f"Perchai session file at {session_path} is corrupted "
                f"(invalid JSON at line {exc.lineno} column {exc.colno}): {exc.msg}. "
                f"Run `perch login` to re-authenticate."
            ) from exc
        except OSError as exc:
            raise PerchaiAuthError(
                f"Could not read perchai session file at {session_path}: {exc}. "
                f"Check file permissions or run `perch login` again."
            ) from exc

        if not isinstance(data, dict):
            raise PerchaiAuthError(
                f"Perchai session file at {session_path} is malformed: "
                f"expected a JSON object, got {type(data).__name__}. "
                f"Run `perch login` to re-authenticate."
            )

        access_token = data.get("accessToken")
        refresh_token = data.get("refreshToken")
        app_url = data.get("appUrl")

        missing = [
            name
            for name, value in (
                ("accessToken", access_token),
                ("refreshToken", refresh_token),
                ("appUrl", app_url),
            )
            if not value or not isinstance(value, str)
        ]
        if missing:
            raise PerchaiAuthError(
                f"Perchai session file at {session_path} is missing required "
                f"field(s): {', '.join(missing)}. Run `perch login` to "
                f"re-authenticate."
            )

        session: PerchaiSession = {
            "version": int(data.get("version", 1)),
            "appUrl": app_url,
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": data.get("expiresAt"),
            "userId": data.get("userId"),
        }
        self._session = session
        lib_logger.debug(
            f"Loaded perchai session from {session_path} "
            f"(userId={session['userId']!r}, expiresAt={session['expiresAt']!r})"
        )
        return session

    # =========================================================================
    # AUTH HEADER
    # =========================================================================

    def get_auth_header(self, credential_identifier: str = "") -> Dict[str, str]:
        """Return an Authorization header for an upstream API call.

        Args:
            credential_identifier: If non-empty, used as the bearer token
                directly (e.g. caller has the raw token). Otherwise the
                session's cached accessToken is used.
        """
        token = credential_identifier if credential_identifier else self._get_access_token()
        return {"Authorization": f"Bearer {token}"}

    def get_app_url(self) -> str:
        """Return the app URL from the loaded session, or DEFAULT_APP_URL."""
        session = self._ensure_session()
        return session.get("appUrl") or self.DEFAULT_APP_URL

    def _ensure_session(self) -> PerchaiSession:
        if self._session is None:
            return self.load_session()
        return self._session

    def _get_access_token(self) -> str:
        session = self._ensure_session()
        token = session.get("accessToken")
        if not token:
            raise PerchaiAuthError(
                "Perchai session has no accessToken. "
                "Run `perch login` to re-authenticate."
            )
        return token

    # =========================================================================
    # TOKEN REFRESH
    # =========================================================================

    async def _ensure_supabase_config(self) -> None:
        """Populate ``_supabase_url`` and ``_supabase_anon_key`` if not yet cached.

        Perchai does not embed the Supabase endpoint or anon key in the
        session file; they are served by a public config endpoint on the
        app host. The first refresh in this process pays the discovery
        round-trip; every subsequent refresh reuses the cached values.

        Raises:
            PerchaiAuthError: If the config endpoint is unreachable, returns
                a non-200 status, returns malformed JSON, or omits one of
                the two required fields.
        """
        if self._supabase_url and self._supabase_anon_key:
            return

        session = self._ensure_session()
        app_url = session.get("appUrl") or self.DEFAULT_APP_URL
        config_url = f"{app_url.rstrip('/')}{self.CONFIG_PATH}"

        try:
            async with httpx.AsyncClient(timeout=self.CONFIG_TIMEOUT) as client:
                response = await client.get(
                    config_url,
                    headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            raise PerchaiAuthError(
                f"Perchai Supabase config discovery failed at {config_url}: "
                f"{exc}. Run `perch login` to re-authenticate."
            ) from exc

        if response.status_code != 200:
            snippet = response.text[:200] if response.text else "<empty>"
            raise PerchaiAuthError(
                f"Perchai Supabase config endpoint returned HTTP "
                f"{response.status_code}: {snippet}. Run `perch login` to "
                f"re-authenticate."
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise PerchaiAuthError(
                f"Perchai Supabase config endpoint returned invalid JSON: "
                f"{exc}. Run `perch login` to re-authenticate."
            ) from exc

        if not isinstance(payload, dict):
            raise PerchaiAuthError(
                "Perchai Supabase config response is not a JSON object. "
                "Run `perch login` to re-authenticate."
            )

        supabase_url = payload.get("supabaseUrl")
        supabase_anon_key = payload.get("supabaseAnonKey")
        if (
            not isinstance(supabase_url, str)
            or not supabase_url
            or not isinstance(supabase_anon_key, str)
            or not supabase_anon_key
        ):
            raise PerchaiAuthError(
                "Perchai Supabase config response is missing 'supabaseUrl' "
                "or 'supabaseAnonKey'. Run `perch login` to re-authenticate."
            )

        self._supabase_url = supabase_url
        self._supabase_anon_key = supabase_anon_key
        lib_logger.debug(
            f"Discovered perchai Supabase config "
            f"(supabaseUrl={supabase_url!r})"
        )

    async def refresh_token(self) -> str:
        """Refresh the access token using the cached refresh token.

        Discovers perchai's Supabase GoTrue endpoint via the public config
        endpoint (once per process, then cached), then POSTs the refresh
        grant to ``{supabaseUrl}/auth/v1/token?grant_type=refresh_token``
        with the ``apikey`` header. Parses the flat snake_case response,
        writes the new tokens back to the session file atomically, updates
        the in-memory cache, and returns the new access token.

        Returns:
            The new access token string.

        Raises:
            PerchaiAuthError: If no session is loaded, the Supabase config
                or refresh network call fails, or the response is malformed.
        """
        session = self._ensure_session()
        refresh_token = session.get("refreshToken")
        if not refresh_token:
            raise PerchaiAuthError(
                "Perchai session has no refreshToken. "
                "Run `perch login` to re-authenticate."
            )

        await self._ensure_supabase_config()

        assert self._supabase_url and self._supabase_anon_key
        refresh_url = (
            f"{self._supabase_url.rstrip('/')}{self.REFRESH_PATH}"
            f"?grant_type=refresh_token"
        )
        lib_logger.debug(f"Refreshing perchai token via {refresh_url}")

        try:
            async with httpx.AsyncClient(timeout=self.REFRESH_TIMEOUT) as client:
                response = await client.post(
                    refresh_url,
                    json={"refresh_token": refresh_token},
                    headers={
                        "apikey": self._supabase_anon_key,
                        "Authorization": f"Bearer {self._get_access_token()}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            raise PerchaiAuthError(
                f"Perchai token refresh network error: {exc}. "
                f"The session may be expired; run `perch login` to "
                f"re-authenticate."
            ) from exc

        if response.status_code != 200:
            snippet = response.text[:200] if response.text else "<empty>"
            raise PerchaiAuthError(
                f"Perchai token refresh failed with HTTP {response.status_code}: "
                f"{snippet}. Run `perch login` to re-authenticate."
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise PerchaiAuthError(
                f"Perchai token refresh returned invalid JSON: {exc}. "
                f"Run `perch login` to re-authenticate."
            ) from exc

        if not isinstance(payload, dict):
            raise PerchaiAuthError(
                "Perchai token refresh response is not a JSON object. "
                "Run `perch login` to re-authenticate."
            )

        new_access = payload.get("access_token")
        new_refresh = payload.get("refresh_token", refresh_token)
        new_expires_at = payload.get("expires_at")
        if new_expires_at is None:
            expires_in = payload.get("expires_in")
            if isinstance(expires_in, (int, float)) and expires_in > 0:
                new_expires_at = int(time.time() + expires_in)

        user_payload = payload.get("user")
        new_user_id = (
            user_payload.get("id")
            if isinstance(user_payload, dict)
            else None
        ) or session.get("userId")

        if not new_access or not isinstance(new_access, str):
            raise PerchaiAuthError(
                "Perchai token refresh response is missing 'access_token'. "
                "Run `perch login` to re-authenticate."
            )

        updated: PerchaiSession = {
            "version": session.get("version", 1),
            "appUrl": session.get("appUrl") or self.DEFAULT_APP_URL,
            "accessToken": new_access,
            "refreshToken": new_refresh,
            "expiresAt": new_expires_at,
            "userId": new_user_id,
        }
        self._session = updated
        self._persist_session(updated)
        lib_logger.debug(
            f"Perchai token refresh succeeded "
            f"(userId={updated['userId']!r}, expiresAt={updated['expiresAt']!r})"
        )
        return new_access

    async def refresh_on_401(
        self, client: httpx.AsyncClient, expired_token: str
    ) -> str:
        """Reactively refresh the token after a 401 response.

        Uses a module-level asyncio.Lock to single-flight concurrent
        refreshes. If another coroutine already refreshed the token while
        we were waiting on the lock, this method returns the new token
        without making a redundant network call.

        Args:
            client: The httpx client that observed the 401 (kept for
                signature symmetry with refresh_on_401 callers; not used
                here because the refresh endpoint must use a fresh client
                with no stale connection state).
            expired_token: The access token that just failed. Used only
                to detect whether another coroutine already replaced it.

        Returns:
            The current (possibly newly-refreshed) access token.

        Raises:
            PerchaiAuthError: If the refresh itself fails.
        """
        del client  # signature parity; refresh uses its own client

        lock = _get_lock()
        async with lock:
            session = self._ensure_session()
            current = session.get("accessToken")
            if current and current != expired_token:
                lib_logger.debug(
                    "Perchai token already refreshed by another coroutine; "
                    "skipping redundant refresh."
                )
                return current
            return await self.refresh_token()

    # =========================================================================
    # SESSION PERSISTENCE
    # =========================================================================

    def _persist_session(self, session: PerchaiSession) -> None:
        """Write the session dict back to disk atomically.

        Writes to `<file>.json.tmp` then renames over the original with
        `os.replace` to avoid leaving a half-written file if the process
        is killed mid-write.
        """
        session_path = _resolve_session_file()
        try:
            session_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = session_path.with_suffix(session_path.suffix + ".tmp")
            payload = dict(session)
            payload["updatedAt"] = int(time.time())
            tmp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp_path, session_path)
        except OSError as exc:
            lib_logger.warning(
                f"Could not persist refreshed perchai session to {session_path}: "
                f"{exc}. The in-memory session is updated; the next refresh "
                f"will overwrite this state."
            )
