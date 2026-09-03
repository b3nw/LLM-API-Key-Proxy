# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kévin Cojean

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, TypedDict, final

import httpx

from ..utils.paths import get_oauth_dir


class PerchaiSession(TypedDict, total=False):
    version: int
    appUrl: str
    accessToken: str
    refreshToken: str
    expiresAt: Optional[str]
    userId: Optional[str]


class PerchaiAuthError(Exception):
    pass


class PerchaiCredentialKind(StrEnum):
    SESSION_FILE = "session_file"
    ENV_VIRTUAL = "env_virtual"
    PASSWORD = "password"
    RAW_TOKEN = "raw_token"


class PerchaiTicketSurface(StrEnum):
    CLI = "cli"
    BROWSER = "browser"
    FILESYSTEM = "filesystem"
    GMAIL = "gmail"


class PerchaiTicketProfile(StrEnum):
    STANDARD = "standard"
    FLOCK = "flock"


class PerchaiTurnTicket(TypedDict):
    token: str
    ticket_id: str
    run_id: str
    expires_at: float
    profile: str


class PerchaiTicketRateLimitError(PerchaiAuthError):
    pass


lib_logger = logging.getLogger("rotator_library")
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())
lib_logger.propagate = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    # The Perch CLI (Db()) keeps using a token until 90s from expiry and
    # force-rotates on its own whenever it starts a turn. Refreshing earlier
    # than it does not stagger the two consumers, it just adds a second
    # independent rotation per hour to a single-use chain, and every extra
    # rotation is another chance to revoke the family and force `perch login`.
    REFRESH_EXPIRY_BUFFER_SECONDS: Final[int] = 90
    ADOPT_MIN_REMAINING_SECONDS: Final[int] = 15
    REFRESH_TOKEN_ALREADY_USED: Final[str] = "refresh_token_already_used"
    PASSWORD_SCHEME: Final[str] = "password://"
    PASSWORD_GRANT_PATH: Final[str] = "/auth/v1/token?grant_type=password"
    PASSWORD_CACHE_PREFIX: Final[str] = "perchai_password_"
    TURN_TICKET_PATH: Final[str] = "/api/perch-terminal/turn-ticket"
    TICKET_TTL_FALLBACK_SECONDS: Final[int] = 300
    TICKET_RENEW_MARGIN_SECONDS: Final[int] = 30
    TICKET_MINT_TIMEOUT_SECONDS: Final[float] = 8.0
    # Perch's server fingerprints direct API access vs the CLI by User-Agent.
    # The CLI bundle reads process.env.PERCH_CLI_VERSION and prepends
    # "perchai-cli/"; we mirror that exactly. "unknown" is the CLI's own
    # fallback when the env var is unset.
    USER_AGENT_PREFIX: Final[str] = "perchai-cli/"
    USER_AGENT_VERSION_ENV: Final[str] = "PERCHAI_CLI_VERSION"
    USER_AGENT_VERSION_FALLBACK: Final[str] = "unknown"

    def __init__(self, credential_path: str = "") -> None:
        self._session: Optional[PerchaiSession] = None
        self._refresh_lock: Optional[asyncio.Lock] = None
        self._model_cache: Dict[str, List[str]] = {}
        self._model_cache_ttl: float = 300.0
        self._model_cache_filled_at: float = 0.0
        self._supabase_url: Optional[str] = None
        self._supabase_anon_key: Optional[str] = None
        self._turn_ticket: Optional[PerchaiTurnTicket] = None
        # When set, load_session / _persist_session use this path instead of
        # auto-discovering. Supports file paths and env:// virtual paths.
        self._credential_path: str = credential_path

    def _get_lock(self) -> asyncio.Lock:
        # Lazy: __init__ can run before an event loop exists.
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    def _user_agent(self) -> str:
        version = os.getenv(
            self.USER_AGENT_VERSION_ENV, self.USER_AGENT_VERSION_FALLBACK
        ).strip() or self.USER_AGENT_VERSION_FALLBACK
        return f"{self.USER_AGENT_PREFIX}{version}"

    def user_agent(self) -> str:
        return self._user_agent()

    def _is_token_expired(self, session: PerchaiSession) -> bool:
        expires_at = session.get("expiresAt")
        if not isinstance(expires_at, (int, float)):
            return False
        return float(expires_at) < time.time() + self.REFRESH_EXPIRY_BUFFER_SECONDS

    def _is_token_usable(self, session: PerchaiSession) -> bool:
        expires_at = session.get("expiresAt")
        if not isinstance(expires_at, (int, float)):
            return True
        return float(expires_at) > time.time() + self.ADOPT_MIN_REMAINING_SECONDS

    def load_session(self) -> PerchaiSession:
        if self._credential_path:
            return self._load_session_from_path(self._credential_path)

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
        lib_logger.debug(
            f"Loaded perchai session from {session_path} "
            f"(userId={session['userId']!r}, expiresAt={session['expiresAt']!r})"
        )
        return session

    async def get_auth_header(
        self, credential_identifier: str = ""
    ) -> Dict[str, str]:
        auth = self
        if credential_identifier and credential_identifier != self._credential_path:
            auth = PerchaiAuthBase(credential_identifier)
        token = await auth.ensure_access_token()
        return {"Authorization": f"Bearer {token}"}

    def _credential_kind(self) -> PerchaiCredentialKind:
        credential = self._credential_path
        if not credential:
            return PerchaiCredentialKind.SESSION_FILE
        if credential.startswith(self.PASSWORD_SCHEME):
            return PerchaiCredentialKind.PASSWORD
        if credential.startswith("env://"):
            return PerchaiCredentialKind.ENV_VIRTUAL
        if credential.endswith(".json") or Path(credential).expanduser().is_file():
            return PerchaiCredentialKind.SESSION_FILE
        return PerchaiCredentialKind.RAW_TOKEN

    async def ensure_access_token(self) -> str:
        kind = self._credential_kind()
        if kind is PerchaiCredentialKind.RAW_TOKEN:
            return self._credential_path
        if kind is PerchaiCredentialKind.PASSWORD:
            return await self._ensure_password_session()

        session = self.load_session()
        if not self._is_token_expired(session):
            self._session = session
            return self._access_token_or_raise(session)
        return await self.refresh_token()

    def get_app_url(self) -> str:
        session = self._ensure_session()
        return session.get("appUrl") or self.DEFAULT_APP_URL

    def _ensure_session(self) -> PerchaiSession:
        if self._session is None:
            return self.load_session()
        return self._session

    @staticmethod
    def _access_token_or_raise(session: PerchaiSession) -> str:
        token = session.get("accessToken")
        if not token:
            raise PerchaiAuthError(
                "Perchai session has no accessToken. "
                "Run `perch login` to re-authenticate."
            )
        return token

    @staticmethod
    def _parse_env_credential_path(path: str) -> Optional[str]:
        if not path.startswith("env://"):
            return None
        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    @staticmethod
    def _load_session_from_env(index: str) -> PerchaiSession:
        prefix = f"PERCHAI_{index}" if index and index != "0" else "PERCHAI"
        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        if not access_token:
            raise PerchaiAuthError(
                f"Environment variable {prefix}_ACCESS_TOKEN not set. "
                f"Run `perch login` to re-authenticate."
            )
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN", "")
        app_url = os.getenv("PERCHAI_APP_URL", PerchaiAuthBase.DEFAULT_APP_URL)
        return PerchaiSession(
            version=1,
            appUrl=app_url,
            accessToken=access_token,
            refreshToken=refresh_token,
            expiresAt=None,
            userId=None,
        )

    def _load_session_from_path(self, path: str) -> PerchaiSession:
        env_index = self._parse_env_credential_path(path)
        if env_index is not None:
            return self._load_session_from_env(env_index)

        session_path = Path(path).expanduser()
        if not session_path.is_file():
            raise PerchaiAuthError(
                f"Perchai credential file not found at {session_path}. "
                f"Run `perch login` to re-authenticate."
            )
        try:
            raw = session_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PerchaiAuthError(
                f"Perchai credential file at {session_path} is corrupted "
                f"(invalid JSON at line {exc.lineno} column {exc.colno}): {exc.msg}. "
                f"Run `perch login` to re-authenticate."
            ) from exc
        except OSError as exc:
            raise PerchaiAuthError(
                f"Could not read perchai credential file at {session_path}: {exc}. "
                f"Check file permissions or run `perch login` again."
            ) from exc

        if not isinstance(data, dict):
            raise PerchaiAuthError(
                f"Perchai credential file at {session_path} is malformed: "
                f"expected a JSON object, got {type(data).__name__}. "
                f"Run `perch login` to re-authenticate."
            )

        access_token = data.get("accessToken")
        if not access_token or not isinstance(access_token, str):
            raise PerchaiAuthError(
                f"Perchai credential file at {session_path} is missing accessToken. "
                f"Run `perch login` to re-authenticate."
            )

        session: PerchaiSession = {
            "version": int(data.get("version", 1)),
            "appUrl": data.get("appUrl") or self.DEFAULT_APP_URL,
            "accessToken": access_token,
            "refreshToken": data.get("refreshToken", ""),
            "expiresAt": data.get("expiresAt"),
            "userId": data.get("userId"),
        }
        lib_logger.debug(
            f"Loaded perchai session from {session_path} "
            f"(userId={session['userId']!r}, expiresAt={session['expiresAt']!r})"
        )
        return session

    async def _ensure_supabase_config(
        self, app_url_override: Optional[str] = None
    ) -> None:
        # Perchai does not embed Supabase config in the session file;
        # discover it once per process and cache.
        if self._supabase_url and self._supabase_anon_key:
            return

        if app_url_override is not None:
            app_url = app_url_override
        else:
            session = self._ensure_session()
            app_url = session.get("appUrl") or self.DEFAULT_APP_URL
        config_url = f"{app_url.rstrip('/')}{self.CONFIG_PATH}"

        try:
            async with httpx.AsyncClient(timeout=self.CONFIG_TIMEOUT) as client:
                response = await client.get(
                    config_url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": self._user_agent(),
                    },
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
        async with self._get_lock():
            return await self._rotate_refresh_token(self.load_session())

    async def _rotate_refresh_token(self, session: PerchaiSession) -> str:
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
                        "Authorization": f"Bearer {self._access_token_or_raise(session)}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": self._user_agent(),
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
            try:
                err_payload = response.json()
                err_code = err_payload.get("error_code", "") if isinstance(err_payload, dict) else ""
            except Exception:
                err_code = ""
            if err_code == self.REFRESH_TOKEN_ALREADY_USED:
                adopted = self._adopt_persisted_session(refresh_token)
                if adopted is not None:
                    return adopted
                lib_logger.error(
                    "Perchai refresh token has been consumed/expired. "
                    "The session file or env var contains a stale refresh token "
                    "that was already used in a prior refresh, and no newer "
                    "session was persisted by another consumer. "
                    "Run `perch login` to obtain fresh credentials, "
                    "then restart the proxy or redeploy."
                )
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
        new_refresh = payload.get("refresh_token")
        if not new_refresh or not isinstance(new_refresh, str):
            raise PerchaiAuthError(
                "Perchai token refresh response is missing 'refresh_token'. "
                "The refresh token is single-use and must be rotated. "
                "Run `perch login` to re-authenticate."
            )
        new_expires_at = self._as_epoch_seconds(payload.get("expires_at"))
        if new_expires_at is None:
            expires_in = self._as_epoch_seconds(payload.get("expires_in"))
            if expires_in is not None and expires_in > 0:
                new_expires_at = int(time.time()) + expires_in

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
        del client  # signature parity; rotation uses its own client

        async with self._get_lock():
            session = self.load_session()
            current = session.get("accessToken")
            if (
                current
                and current != expired_token
                and not self._is_token_expired(session)
            ):
                lib_logger.debug(
                    "Perchai token already refreshed by another worker; "
                    "skipping redundant refresh."
                )
                self._session = session
                return current
            return await self._rotate_refresh_token(session)

    def _password_index(self) -> str:
        tail = self._credential_path[len(self.PASSWORD_SCHEME):]
        parts = [p for p in tail.split("/") if p]
        return parts[-1] if parts else "1"

    def _password_app_url(self) -> str:
        return os.getenv("PERCHAI_APP_URL", "").strip() or self.DEFAULT_APP_URL

    def _session_cache_path(self) -> Path:
        filename = f"{self.PASSWORD_CACHE_PREFIX}{self._password_index()}.json"
        return get_oauth_dir() / filename

    def _load_cached_session(self) -> Optional[PerchaiSession]:
        path = self._session_cache_path()
        if not path.is_file():
            return None
        raw = self._read_raw_session(path)
        access_token = raw.get("accessToken")
        if not isinstance(access_token, str) or not access_token:
            return None
        return PerchaiSession(
            version=int(raw.get("version", 1)),
            appUrl=raw.get("appUrl") or self._password_app_url(),
            accessToken=access_token,
            refreshToken=raw.get("refreshToken", ""),
            expiresAt=raw.get("expiresAt"),
            userId=raw.get("userId"),
        )

    async def _ensure_password_session(self) -> str:
        cached = self._load_cached_session()
        if cached is not None:
            self._session = cached
            if not self._is_token_expired(cached):
                return self._access_token_or_raise(cached)
            if cached.get("refreshToken"):
                try:
                    async with self._get_lock():
                        return await self._rotate_refresh_token(cached)
                except PerchaiAuthError:
                    lib_logger.info(
                        "Perchai password credential: refresh chain dead, "
                        "re-minting from stored password."
                    )
        return await self._sign_in_with_password()

    async def _sign_in_with_password(self) -> str:
        index = self._password_index()
        email = os.getenv(f"PERCHAI_EMAIL_{index}", "").strip()
        password = os.getenv(f"PERCHAI_PASSWORD_{index}")
        if not email or not password:
            raise PerchaiAuthError(
                f"Perchai password credential {self._credential_path} needs "
                f"PERCHAI_EMAIL_{index} and PERCHAI_PASSWORD_{index} to be set."
            )

        app_url = self._password_app_url()
        await self._ensure_supabase_config(app_url_override=app_url)
        assert self._supabase_url and self._supabase_anon_key
        sign_in_url = f"{self._supabase_url.rstrip('/')}{self.PASSWORD_GRANT_PATH}"

        try:
            async with httpx.AsyncClient(timeout=self.REFRESH_TIMEOUT) as client:
                response = await client.post(
                    sign_in_url,
                    json={"email": email, "password": password},
                    headers={
                        "apikey": self._supabase_anon_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": self._user_agent(),
                    },
                )
        except httpx.HTTPError as exc:
            raise PerchaiAuthError(
                f"Perchai password sign-in network error: {exc}."
            ) from exc

        if response.status_code != 200:
            snippet = response.text[:200] if response.text else "<empty>"
            raise PerchaiAuthError(
                f"Perchai password sign-in failed with HTTP "
                f"{response.status_code}: {snippet}. Check PERCHAI_EMAIL_{index} "
                f"and PERCHAI_PASSWORD_{index}."
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise PerchaiAuthError(
                f"Perchai password sign-in returned invalid JSON: {exc}."
            ) from exc

        new_access = payload.get("access_token")
        new_refresh = payload.get("refresh_token")
        if not isinstance(new_access, str) or not new_access:
            raise PerchaiAuthError(
                "Perchai password sign-in response is missing 'access_token'."
            )
        if not isinstance(new_refresh, str) or not new_refresh:
            raise PerchaiAuthError(
                "Perchai password sign-in response is missing 'refresh_token'."
            )
        new_expires_at = self._as_epoch_seconds(payload.get("expires_at"))
        if new_expires_at is None:
            expires_in = self._as_epoch_seconds(payload.get("expires_in"))
            if expires_in is not None and expires_in > 0:
                new_expires_at = int(time.time()) + expires_in
        user_payload = payload.get("user")
        user_id = user_payload.get("id") if isinstance(user_payload, dict) else None

        session: PerchaiSession = {
            "version": 1,
            "appUrl": app_url,
            "accessToken": new_access,
            "refreshToken": new_refresh,
            "expiresAt": new_expires_at,
            "userId": user_id,
        }
        self._session = session
        self._persist_session(session)
        lib_logger.info(
            f"Perchai minted an independent session via password for index "
            f"{index} (userId={user_id!r})."
        )
        return new_access

    async def ensure_turn_ticket(self, access_token: str) -> str:
        async with self._get_lock():
            now = time.time()
            cached = self._turn_ticket
            if (
                cached is not None
                and cached["expires_at"] > now + self.TICKET_RENEW_MARGIN_SECONDS
            ):
                return cached["token"]

            # Only a ticket that is still alive can be renewed. The CLI renews
            # inside the 30s margin and mints fresh otherwise; asking Perch to
            # renew a ticketId that already died is refused by the surface gate.
            renewable = (
                cached if cached is not None and cached["expires_at"] > now else None
            )

            token = access_token
            response = await self._post_turn_ticket(token, renewable)
            if response.status_code == 401:
                token = await self._rotate_refresh_token(self.load_session())
                response = await self._post_turn_ticket(token, renewable)
            if renewable is not None and response.status_code not in (200, 429):
                self._turn_ticket = None
                response = await self._post_turn_ticket(token, None)

            minted = self._parse_turn_ticket_response(response)
            self._turn_ticket = minted
            return minted["token"]

    def invalidate_turn_ticket(self) -> None:
        self._turn_ticket = None

    async def _post_turn_ticket(
        self, access_token: str, cached: Optional[PerchaiTurnTicket]
    ) -> httpx.Response:
        session = self._ensure_session()
        app_url = session.get("appUrl") or self.DEFAULT_APP_URL
        url = f"{app_url.rstrip('/')}{self.TURN_TICKET_PATH}"
        body: Dict[str, Any] = (
            {"renew": True, "ticketId": cached["ticket_id"]}
            if cached is not None
            else {
                "surface": PerchaiTicketSurface.CLI.value,
                "profile": PerchaiTicketProfile.STANDARD.value,
            }
        )
        try:
            async with httpx.AsyncClient(
                timeout=self.TICKET_MINT_TIMEOUT_SECONDS
            ) as client:
                return await client.post(
                    url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": self._user_agent(),
                    },
                )
        except httpx.HTTPError as exc:
            raise PerchaiAuthError(
                f"Perchai turn-ticket request failed: {exc}."
            ) from exc

    def _parse_turn_ticket_response(
        self, response: httpx.Response
    ) -> PerchaiTurnTicket:
        if response.status_code == 429:
            try:
                err_payload = response.json()
            except json.JSONDecodeError:
                err_payload = {}
            if isinstance(err_payload, dict) and err_payload.get("enforced"):
                raise PerchaiTicketRateLimitError(
                    str(err_payload.get("error") or "Perchai turn rate limited")
                )

        if response.status_code != 200:
            snippet = response.text[:200] if response.text else "<empty>"
            raise PerchaiAuthError(
                f"Perchai turn-ticket request returned HTTP "
                f"{response.status_code}: {snippet}."
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise PerchaiAuthError(
                f"Perchai turn-ticket response is invalid JSON: {exc}."
            ) from exc

        if (
            not isinstance(payload, dict)
            or not payload.get("ok")
            or not payload.get("ticket")
            or not payload.get("ticketId")
        ):
            raise PerchaiAuthError(
                "Perchai turn-ticket response missing ok/ticket/ticketId."
            )

        return PerchaiTurnTicket(
            token=payload["ticket"],
            ticket_id=payload["ticketId"],
            run_id=str(payload.get("runId") or ""),
            expires_at=self._parse_ticket_expiry(payload.get("expiresAt")),
            profile=str(payload.get("profile") or PerchaiTicketProfile.STANDARD.value),
        )

    @staticmethod
    def _parse_ticket_expiry(value: Any) -> float:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        return time.time() + PerchaiAuthBase.TICKET_TTL_FALLBACK_SECONDS

    def _adopt_persisted_session(self, presented: str) -> Optional[str]:
        # Another consumer rotated the family and persisted its session while
        # our request was in flight. Its token is the valid one; ours is not.
        try:
            session = self.load_session()
        except PerchaiAuthError:
            return None
        if session.get("refreshToken") == presented or not self._is_token_usable(
            session
        ):
            return None
        access_token = session.get("accessToken")
        if not access_token:
            return None
        self._session = session
        lib_logger.info(
            "Perchai rotation lost the race; adopted the session another "
            "consumer persisted."
        )
        return access_token

    @staticmethod
    def _as_epoch_seconds(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        return None

    @staticmethod
    def _read_raw_session(session_path: Path) -> Dict[str, Any]:
        try:
            data = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _persist_session(self, session: PerchaiSession) -> None:
        if self._credential_path and self._credential_path.startswith("env://"):
            return

        if self._credential_kind() is PerchaiCredentialKind.PASSWORD:
            session_path = self._session_cache_path()
        elif self._credential_path:
            session_path = Path(self._credential_path).expanduser()
        else:
            session_path = _resolve_session_file()

        # The Perch CLI owns this file too and stores fields the proxy does not
        # model (email, and anything a future CLI adds). Rewrite ours over what
        # is on disk rather than replacing the file with our narrower view.
        payload = self._read_raw_session(session_path)
        payload.update(session)
        payload["updatedAt"] = _utc_now_iso()
        content = json.dumps(payload, indent=2, sort_keys=True)

        try:
            session_path.parent.mkdir(parents=True, exist_ok=True)
            # In-place write, deliberately not tempfile + rename: the Perch CLI
            # rotates this same file, and renaming onto a single-file Docker
            # bind mount fails with EBUSY, which silently loses the rotation
            # and revokes the whole token family.
            with open(session_path, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            lib_logger.error(
                f"Could not persist refreshed perchai session to {session_path}: "
                f"{exc}. The next consumer to use the old refresh token will hit "
                f"reuse detection; make this path writable."
            )
