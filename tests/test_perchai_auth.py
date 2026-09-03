# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kévin Cojean
"""Auth lifecycle tests for the Perchai provider.

These run against a real local HTTP server emulating the two endpoints the
Perch CLI talks to: Supabase config discovery and the GoTrue refresh token
endpoint. GoTrue issues single-use rotating refresh tokens, so the emulator
enforces the same rule. Presenting anything other than the currently valid
token fails with error_code "refresh_token_already_used", which is what
revokes the whole token family upstream.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import pytest

from rotator_library.providers.perchai_auth_base import PerchaiAuthBase
from rotator_library.providers.perchai_provider import (
    PerchaiProvider,
)
from rotator_library.providers.provider_interface import SingletonABCMeta


@pytest.fixture(autouse=True)
def reset_perchai_singleton() -> Any:
    SingletonABCMeta._instances.pop(PerchaiProvider, None)
    yield
    SingletonABCMeta._instances.pop(PerchaiProvider, None)

CONFIG_PATH = "/api/perch-terminal/cli-auth/config"
TOKEN_PATH = "/auth/v1/token"
MODEL_CALL_PATH = "/api/perch-terminal/model-call"
TURN_TICKET_PATH = "/api/perch-terminal/turn-ticket"
TURN_TICKET_HEADER = "x-perch-turn-ticket"
USER_ID = "11111111-2222-3333-4444-555555555555"
ALREADY_USED = "refresh_token_already_used"
SURFACE_REQUIRED = "perch_surface_required"
TURN_RATE_LIMITED = "turn_rate_limited"


class FakePerchaiAuth:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current_refresh: Optional[str] = None
        self.issued_access: str = "access-0"
        self.rotations: int = 0
        self.presented: List[str] = []
        self._queue: List[str] = []
        self._seq: int = 0
        self.ttl: int = 3600
        self.on_reject: Optional[Callable[[], None]] = None
        self.account_email: str = "proxy@example.invalid"
        self.account_password: str = "correct-horse-battery-staple"
        self.password_signins: int = 0
        self.completion_text: str = "hello"
        self.model_calls: List[str] = []
        # Turn-ticket state. The fake requires a valid ticket on every
        # model-call, matching the live gate this test suite regression-tests.
        self._ticket_seq: int = 0
        self.current_ticket_token: Optional[str] = None
        self.current_ticket_id: Optional[str] = None
        self.ticket_mints: int = 0
        self.ticket_renews: List[str] = []
        self.ticket_ttl_seconds: int = 300
        self.ticket_rate_limited: bool = False
        self.refuse_renew: bool = False
        self.refused_renews: List[str] = []
        self.tickets_presented: List[str] = []
        # Every inbound request's User-Agent header is recorded here so tests
        # can assert the proxy mimics the Perch CLI's User-Agent on every
        # outbound HTTP request (turn-ticket, model-call, GoTrue, config).
        self.user_agents: List[str] = []
        self.config_user_agents: List[str] = []
        self.token_user_agents: List[str] = []
        self.turn_ticket_user_agents: List[str] = []
        self.model_call_user_agents: List[str] = []

    def issue_ticket(self, *, renew: bool, ticket_id: str) -> Dict[str, Any]:
        with self._lock:
            self._ticket_seq += 1
            seq = self._ticket_seq
            token = f"ticket-{seq}"
            new_ticket_id = f"ticket-id-{seq}"
            self.current_ticket_token = token
            self.current_ticket_id = new_ticket_id
            if renew:
                self.ticket_renews.append(ticket_id)
            else:
                self.ticket_mints += 1
            expires_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() + self.ticket_ttl_seconds),
            )
            return {
                "ok": True,
                "ticket": token,
                "ticketId": new_ticket_id,
                "runId": f"tkt-cli-{seq}",
                "surface": "cli",
                "profile": "standard",
                "expiresAt": expires_at,
            }

    def adopt_initial_refresh(self, refresh_token: str) -> None:
        with self._lock:
            self.current_refresh = refresh_token

    def queue_next_refreshes(self, *refresh_tokens: str) -> None:
        with self._lock:
            self._queue.extend(refresh_tokens)

    def rotate_as_other_consumer(self, next_refresh: str) -> Tuple[str, int]:
        with self._lock:
            self._seq += 1
            self.current_refresh = next_refresh
            self.issued_access = f"access-{self._seq}"
            return self.issued_access, int(time.time()) + self.ttl

    def sign_in_password(self, email: str, password: str) -> Tuple[int, Dict[str, Any]]:
        with self._lock:
            if email != self.account_email or password != self.account_password:
                return 400, {
                    "error": "invalid_grant",
                    "error_description": "Invalid login credentials",
                }
            self._seq += 1
            self.password_signins += 1
            self.issued_access = f"access-pw-{self._seq}"
            self.current_refresh = f"refresh-pw-{self._seq}"
            return 200, {
                "access_token": self.issued_access,
                "refresh_token": self.current_refresh,
                "expires_in": self.ttl,
                "expires_at": int(time.time()) + self.ttl,
                "user": {"id": USER_ID, "email": email},
            }

    def exchange(self, presented: str) -> Tuple[int, Dict[str, Any]]:
        rejection: Optional[Callable[[], None]] = None
        with self._lock:
            self.presented.append(presented)
            if presented != self.current_refresh:
                rejection = self.on_reject
                status, payload = 400, {
                    "code": "refresh_token_mismatch",
                    "error_code": ALREADY_USED,
                    "msg": "Refresh token already used",
                }
            else:
                self._seq += 1
                self.rotations += 1
                self.issued_access = f"access-{self._seq}"
                if self._queue:
                    self.current_refresh = self._queue.pop(0)
                else:
                    self.current_refresh = f"refresh-{self._seq}"
                status, payload = 200, {
                    "access_token": self.issued_access,
                    "refresh_token": self.current_refresh,
                    "expires_in": self.ttl,
                    "expires_at": int(time.time()) + self.ttl,
                    "user": {"id": USER_ID, "email": "test@example.invalid"},
                }
        if rejection:
            rejection()
        return status, payload


class _Handler(BaseHTTPRequestHandler):
    state: FakePerchaiAuth
    public_url: str

    def log_message(self, *args: Any) -> None:
        pass

    def _reply(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        ua = self.headers.get("User-Agent")
        self.state.user_agents.append(ua or "")
        if path == CONFIG_PATH:
            self.state.config_user_agents.append(ua or "")
            self._reply(
                200,
                {
                    "supabaseUrl": self.public_url,
                    "supabaseAnonKey": "test-anon-key",
                },
            )
            return
        self._reply(404, {"error": "unexpected_get"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        ua = self.headers.get("User-Agent")
        self.state.user_agents.append(ua or "")
        if parsed.path == MODEL_CALL_PATH:
            self.state.model_call_user_agents.append(ua or "")
            self._model_call()
            return
        if parsed.path == TURN_TICKET_PATH:
            self.state.turn_ticket_user_agents.append(ua or "")
            self._turn_ticket()
            return
        if parsed.path != TOKEN_PATH:
            self._reply(404, {"error": "unexpected_post"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._reply(400, {"error": "invalid_json"})
            return
        if "grant_type=password" in (urlparse(self.path).query or ""):
            status, payload = self.state.sign_in_password(
                str(body.get("email") or ""), str(body.get("password") or "")
            )
            self.state.token_user_agents.append(ua or "")
            self._reply(status, payload)
            return
        status, payload = self.state.exchange(str(body.get("refresh_token") or ""))
        self.state.token_user_agents.append(ua or "")
        self._reply(status, payload)

    def _turn_ticket(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._reply(400, {"ok": False, "error": "invalid_json"})
            return
        if self.state.ticket_rate_limited:
            self._reply(
                429,
                {
                    "enforced": True,
                    "errorCode": TURN_RATE_LIMITED,
                    "error": "Too many turns started",
                },
            )
            return
        renew = bool(body.get("renew"))
        ticket_id = str(body.get("ticketId") or "")
        if renew and self.state.refuse_renew:
            # Perch answers an unrecognised renewal with the surface-gate body.
            self.state.refused_renews.append(ticket_id)
            self._reply(
                403,
                {
                    "ok": False,
                    "error": (
                        "Your plan includes Perch-hosted models for use in "
                        "Perch AI Web, Desktop, and CLI only. Direct API "
                        "access is not included."
                    ),
                    "errorCode": SURFACE_REQUIRED,
                },
            )
            return
        payload = self.state.issue_ticket(renew=renew, ticket_id=ticket_id)
        self._reply(200, payload)

    def _model_call(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        presented = str(self.headers.get("Authorization") or "")
        self.state.model_calls.append(presented)
        if presented != f"Bearer {self.state.issued_access}":
            self._reply(401, {"ok": False, "error": "invalid_token"})
            return
        ticket_presented = str(self.headers.get(TURN_TICKET_HEADER) or "")
        self.state.tickets_presented.append(ticket_presented)
        if ticket_presented != (self.state.current_ticket_token or ""):
            self._reply(
                403,
                {
                    "ok": False,
                    "error": (
                        "Your plan includes Perch-hosted models for use in "
                        "Perch AI Web, Desktop, and CLI only. Direct API "
                        "access is not included."
                    ),
                    "errorCode": SURFACE_REQUIRED,
                },
            )
            return
        if "text/event-stream" in str(self.headers.get("Accept") or ""):
            text = self.state.completion_text
            body = (
                f'data: {json.dumps({"type": "text_delta", "text": text})}\n\n'
                f'data: {json.dumps({"type": "done"})}\n\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._reply(200, {"ok": True, "text": self.state.completion_text})


class AuthServer:
    def __init__(self, httpd: ThreadingHTTPServer, state: FakePerchaiAuth, url: str) -> None:
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self.state = state
        self.app_url = url
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def auth_server() -> Any:
    state = FakePerchaiAuth()
    handler = type("BoundHandler", (_Handler,), {"state": state, "public_url": ""})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    handler.public_url = url
    server = AuthServer(httpd, state, url)
    try:
        yield server
    finally:
        server.stop()


def write_session(
    path: Path,
    *,
    app_url: str,
    access: str,
    refresh: str,
    expires_at: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "appUrl": app_url,
                "accessToken": access,
                "refreshToken": refresh,
                "expiresAt": expires_at,
                "userId": USER_ID,
                "email": "test@example.invalid",
                "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ),
        encoding="utf-8",
    )


def read_session(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def session_file(tmp_path: Path) -> Path:
    return tmp_path / "cli-auth-session.json"


MODEL = "perchai/nemotron-3.5-lightning"
MESSAGES = [{"role": "user", "content": "Say hello in one word"}]


def given_session_with_valid_access_token(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access=auth_server.state.issued_access,
        refresh="refresh-initial",
        expires_at=int(time.time()) + 3000,
    )
    auth_server.state.adopt_initial_refresh("refresh-initial")


def given_password_credentials(monkeypatch: Any, auth_server: AuthServer) -> str:
    monkeypatch.setenv("PERCHAI_APP_URL", auth_server.app_url)
    monkeypatch.setenv("PERCHAI_EMAIL_1", auth_server.state.account_email)
    monkeypatch.setenv("PERCHAI_PASSWORD_1", auth_server.state.account_password)
    return "password://perchai/1"


async def test_model_call_401_refreshes_and_retries(
    auth_server: AuthServer, session_file: Path
) -> None:
    """Real seam: 401 from model-call must trigger one refresh and one retry."""
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-expired",
        refresh="refresh-initial",
        expires_at=int(time.time()) + 3000,
    )
    auth_server.state.adopt_initial_refresh("refresh-initial")

    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        when_response = await given_provider.acompletion(
            given_client,
            model=MODEL,
            messages=MESSAGES,
            credential_identifier=str(session_file),
            stream=False,
        )

    then_state = auth_server.state
    assert when_response.choices[0].message.content == then_state.completion_text
    assert then_state.rotations == 1, (
        f"expected exactly one refresh to recover from the 401, got {then_state.rotations}"
    )
    assert then_state.model_calls[-1] == f"Bearer {then_state.issued_access}", (
        "the retry must carry the freshly issued access token"
    )


async def test_turn_ticket_lifecycle(
    auth_server: AuthServer, session_file: Path, monkeypatch: Any
) -> None:
    """Real seam: ticket reused across completions; expired tickets mint fresh;
    refused renewals fall back to a fresh mint. Covers the full ticket lifecycle
    in one test rather than five."""
    monkeypatch.delenv("PERCHAI_CLI_VERSION", raising=False)
    given_session_with_valid_access_token(auth_server, session_file)
    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        # First completion: mints a ticket.
        await given_provider.acompletion(
            given_client,
            model=MODEL, messages=MESSAGES,
            credential_identifier=str(session_file), stream=False,
        )
        first_ticket = auth_server.state.current_ticket_token
        first_mints = auth_server.state.ticket_mints

        # Second completion: reuses the still-alive ticket, no fresh mint.
        await given_provider.acompletion(
            given_client,
            model=MODEL, messages=MESSAGES,
            credential_identifier=str(session_file), stream=False,
        )
        assert auth_server.state.ticket_mints == first_mints, (
            "ticket reused across completions: must not re-mint while alive"
        )

        # Force the cached ticket to expire on the server side and refuse
        # renewals: next completion must mint fresh, not try to renew.
        auth_server.state.ticket_ttl_seconds = -1
        auth_server.state.refuse_renew = True
        given_provider._auth_base_cache.clear()
        await given_provider.acompletion(
            given_client,
            model=MODEL, messages=MESSAGES,
            credential_identifier=str(session_file), stream=False,
        )
        assert auth_server.state.ticket_mints > first_mints, (
            "expired ticket with refused renew must fall back to a fresh mint"
        )
        assert auth_server.state.current_ticket_token != first_ticket


async def test_surface_required_403_retries_with_fresh_ticket_and_succeeds(
    auth_server: AuthServer, session_file: Path
) -> None:
    """Real seam: 403 perch_surface_required must invalidate cache and re-mint,
    not exhaust the credential."""
    given_session_with_valid_access_token(auth_server, session_file)

    # The first completion mints a ticket; the fake server validates it.
    # The second completion is told the ticket has been invalidated server-side
    # (simulating another consumer burning it), so the model-call returns 403
    # with perch_surface_required - the proxy must drop + re-mint + retry.
    auth_server.state.refuse_renew = True
    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        when_response = await given_provider.acompletion(
            given_client,
            model=MODEL, messages=MESSAGES,
            credential_identifier=str(session_file), stream=False,
        )

    assert when_response.choices[0].message.content == auth_server.state.completion_text, (
        "proxy must recover from a 403 by re-minting the ticket, "
        f"got {when_response.choices[0].message.content!r}"
    )


async def test_password_credential_mints_and_completes_without_a_session_file(
    auth_server: AuthServer, tmp_path: Path, monkeypatch: Any
) -> None:
    """Real seam: cold-start with password env vars; no ~/.perch read or write."""
    monkeypatch.setattr(
        "rotator_library.utils.paths.get_default_root", lambda: tmp_path
    )
    given_credential = given_password_credentials(monkeypatch, auth_server)

    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        when_response = await given_provider.acompletion(
            given_client,
            model=MODEL,
            messages=MESSAGES,
            credential_identifier=given_credential,
            stream=False,
        )

    assert when_response.choices[0].message.content == auth_server.state.completion_text
    assert auth_server.state.password_signins == 1, (
        "with no session file the proxy must mint its own via grant_type=password"
    )
    then_cache = tmp_path / "oauth_creds" / "perchai_password_1.json"
    assert then_cache.is_file(), (
        "the minted session must be cached in the proxy-owned oauth_creds dir"
    )


async def test_password_session_remints_when_refresh_chain_is_dead(
    auth_server: AuthServer, tmp_path: Path, monkeypatch: Any
) -> None:
    """Real seam: cached session with a consumed refresh must re-mint via password,
    not raise 'run perch login'."""
    monkeypatch.setattr(
        "rotator_library.utils.paths.get_default_root", lambda: tmp_path
    )
    given_credential = given_password_credentials(monkeypatch, auth_server)
    given_cache_dir = tmp_path / "oauth_creds"
    given_cache_dir.mkdir()
    write_session(
        given_cache_dir / "perchai_password_1.json",
        app_url=auth_server.app_url,
        access="access-stale",
        refresh="refresh-revoked",
        expires_at=int(time.time()) - 10,
    )
    auth_server.state.adopt_initial_refresh("refresh-held-by-someone-else")

    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        when_response = await given_provider.acompletion(
            given_client,
            model=MODEL,
            messages=MESSAGES,
            credential_identifier=given_credential,
            stream=False,
        )

    assert when_response.choices[0].message.content == auth_server.state.completion_text
    assert auth_server.state.password_signins == 1, (
        "a dead refresh chain must re-mint via password, not raise perch login"
    )


def test_user_agent_helper_uses_env_version(monkeypatch: Any) -> None:
    monkeypatch.setenv("PERCHAI_CLI_VERSION", "9.9.9")
    assert PerchaiAuthBase().user_agent() == "perchai-cli/9.9.9"


def test_user_agent_helper_falls_back_to_unknown(monkeypatch: Any) -> None:
    monkeypatch.delenv("PERCHAI_CLI_VERSION", raising=False)
    assert PerchaiAuthBase().user_agent() == "perchai-cli/unknown"


async def test_outbound_perchai_requests_send_cli_user_agent(
    auth_server: AuthServer, tmp_path: Path, monkeypatch: Any
) -> None:
    """Every outbound Perch HTTP request must carry User-Agent: perchai-cli/<version>.

    Perch's server fingerprints direct API access by User-Agent and rejects
    with 403 perch_surface_required regardless of which endpoint the request
    hits. The CLI bundle sends perchai-cli/<PERCHAI_CLI_VERSION||"unknown">;
    the proxy mirrors that pattern on every outbound call.
    """
    monkeypatch.setenv("PERCHAI_CLI_VERSION", "9.9.9")
    monkeypatch.setattr(
        "rotator_library.utils.paths.get_default_root", lambda: tmp_path
    )
    given_password_credentials(monkeypatch, auth_server)

    given_session_with_valid_access_token(auth_server, tmp_path / "session.json")
    given_auth = PerchaiAuthBase(str(tmp_path / "session.json"))

    async with httpx.AsyncClient() as given_client:
        # Trigger every outbound call type. Run one consistent session-file
        # chain first (refresh + turn-ticket + model call), then password
        # signin last - password signin changes the fake's `issued_access`
        # and would invalidate the session-file chain mid-test.
        await given_auth.refresh_token()
        await given_auth.ensure_turn_ticket(auth_server.state.issued_access)
        given_provider = PerchaiProvider()
        await given_provider.acompletion(
            given_client,
            model=MODEL, messages=MESSAGES,
            credential_identifier=str(tmp_path / "session.json"),
            stream=False,
        )
        await PerchaiAuthBase("password://perchai/1").ensure_access_token()

    expected = "perchai-cli/9.9.9"
    assert auth_server.state.turn_ticket_user_agents, (
        "expected at least one turn-ticket request"
    )
    assert all(ua == expected for ua in auth_server.state.turn_ticket_user_agents), (
        f"turn-ticket mint must send {expected!r}, "
        f"got {auth_server.state.turn_ticket_user_agents!r}"
    )
    assert auth_server.state.token_user_agents, (
        "expected at least one Supabase token endpoint call"
    )
    assert all(ua == expected for ua in auth_server.state.token_user_agents), (
        f"token endpoint calls must send {expected!r}, "
        f"got {auth_server.state.token_user_agents!r}"
    )
    assert auth_server.state.config_user_agents, (
        "expected at least one Supabase config GET"
    )
    assert all(ua == expected for ua in auth_server.state.config_user_agents), (
        f"config discovery must send {expected!r}, "
        f"got {auth_server.state.config_user_agents!r}"
    )
    assert auth_server.state.model_call_user_agents == [expected], (
        f"model-call must send {expected!r}, "
        f"got {auth_server.state.model_call_user_agents!r}"
    )


async def test_password_credential_refreshes_on_401_via_cached_session(
    auth_server: AuthServer, tmp_path: Path, monkeypatch: Any
) -> None:
    """Real seam: cached password session + server-side 401 must refresh via the cached refresh chain, not crash with 'credential file not found at password:/perchai/1'."""
    monkeypatch.setattr(
        "rotator_library.utils.paths.get_default_root", lambda: tmp_path
    )
    given_credential = given_password_credentials(monkeypatch, auth_server)

    given_cache_dir = tmp_path / "oauth_creds"
    given_cache_dir.mkdir()
    write_session(
        given_cache_dir / "perchai_password_1.json",
        app_url=auth_server.app_url,
        access="access-cached",
        refresh="refresh-cached",
        expires_at=int(time.time()) + 3000,
    )
    auth_server.state.adopt_initial_refresh("refresh-cached")

    given_provider = PerchaiProvider()
    async with httpx.AsyncClient() as given_client:
        when_response = await given_provider.acompletion(
            given_client,
            model=MODEL,
            messages=MESSAGES,
            credential_identifier=given_credential,
            stream=False,
        )

    assert when_response.choices[0].message.content == auth_server.state.completion_text, (
        "a 401 from the server on the first model call must trigger refresh "
        "and retry, not bubble 'credential file not found'"
    )
    assert auth_server.state.rotations == 1, (
        "exactly one refresh must run to recover from the 401, "
        f"got {auth_server.state.rotations}"
    )
    assert auth_server.state.model_calls[-1] == f"Bearer {auth_server.state.issued_access}", (
        "the retry must carry the freshly issued access token, not the cached one"
    )
    assert auth_server.state.password_signins == 0, (
        "refresh must succeed via the cached refresh chain - password "
        "signin is only the dead-refresh fallback, not the 401 fallback"
    )

