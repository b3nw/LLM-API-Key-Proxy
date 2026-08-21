# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

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
    version: int
    appUrl: str
    accessToken: str
    refreshToken: str
    expiresAt: Optional[str]
    userId: Optional[str]


class PerchaiAuthError(Exception):
    pass


lib_logger = logging.getLogger("rotator_library")
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())
lib_logger.propagate = False


# Lazy module-level lock: created on first call so __init__ (before the
# asyncio loop starts) doesn't trip the Python 3.10+ warning.
_refresh_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    global _refresh_lock
    if _refresh_lock is None:
        _refresh_lock = asyncio.Lock()
    return _refresh_lock


def _resolve_session_file() -> Path:
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

    def load_session(self) -> PerchaiSession:
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

    def get_auth_header(self, credential_identifier: str = "") -> Dict[str, str]:
        token = credential_identifier if credential_identifier else self._get_access_token()
        return {"Authorization": f"Bearer {token}"}

    def get_app_url(self) -> str:
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

    async def _ensure_supabase_config(self) -> None:
        # Perchai does not embed Supabase config in the session file;
        # discover it once per process and cache.
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
        # Single-flight: if another coroutine already refreshed while we
        # waited on the lock, return the current token without re-hitting the network.
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

    def _persist_session(self, session: PerchaiSession) -> None:
        # Atomic: write to .tmp then os.replace to avoid half-written files
        # if killed mid-write.
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
