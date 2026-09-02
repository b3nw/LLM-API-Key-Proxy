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
from rotator_library.providers.perchai_provider import PerchaiProvider

CONFIG_PATH = "/api/perch-terminal/cli-auth/config"
TOKEN_PATH = "/auth/v1/token"
MODEL_CALL_PATH = "/api/perch-terminal/model-call"
USER_ID = "11111111-2222-3333-4444-555555555555"
ALREADY_USED = "refresh_token_already_used"


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
        self.completion_text: str = "hello"
        self.model_calls: List[str] = []

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
        if urlparse(self.path).path == CONFIG_PATH:
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
        if parsed.path == MODEL_CALL_PATH:
            self._model_call()
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
        status, payload = self.state.exchange(str(body.get("refresh_token") or ""))
        self._reply(status, payload)

    def _model_call(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        presented = str(self.headers.get("Authorization") or "")
        self.state.model_calls.append(presented)
        if presented != f"Bearer {self.state.issued_access}":
            self._reply(401, {"ok": False, "error": "invalid_token"})
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


async def test_proactively_refresh_rotates_token_nearing_expiry(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    given_expires_at = int(time.time()) + 30
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-initial",
        refresh="refresh-initial",
        expires_at=given_expires_at,
    )
    auth_server.state.adopt_initial_refresh("refresh-initial")
    auth_server.state.queue_next_refreshes("refresh-rotated")

    given_provider = PerchaiProvider()
    await given_provider.proactively_refresh(str(session_file))

    then_state = auth_server.state
    assert then_state.rotations == 1, (
        "a token expiring in 30s must be rotated by the background refresher "
        f"hook, got {then_state.rotations} rotations"
    )
    assert read_session(session_file)["refreshToken"] == "refresh-rotated", (
        "the rotated refresh token must be persisted so the next caller does "
        "not present an already-used token"
    )


async def test_proactively_refresh_leaves_healthy_token_alone(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-healthy",
        refresh="refresh-healthy",
        expires_at=int(time.time()) + 3000,
    )
    auth_server.state.adopt_initial_refresh("refresh-healthy")

    given_provider = PerchaiProvider()
    await given_provider.proactively_refresh(str(session_file))

    assert auth_server.state.rotations == 0, (
        "a token with 50 minutes of validity left must not be rotated; "
        "needless rotation burns single-use refresh tokens"
    )
    assert read_session(session_file)["refreshToken"] == "refresh-healthy"


async def test_refresh_presents_token_rotated_by_perch_login(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-first",
        refresh="refresh-first",
        expires_at=int(time.time()) + 30,
    )
    auth_server.state.adopt_initial_refresh("refresh-first")
    auth_server.state.queue_next_refreshes("refresh-second")

    given_auth = PerchaiAuthBase(credential_path=str(session_file))
    await given_auth.get_auth_header(str(session_file))

    new_access, _ = auth_server.state.rotate_as_other_consumer("refresh-login")
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access=new_access,
        refresh="refresh-login",
        expires_at=int(time.time()) + 30,
    )

    when_header = await given_auth.get_auth_header(str(session_file))

    then_state = auth_server.state
    assert then_state.presented[-1] == "refresh-login", (
        "refresh must present the token currently on disk, not the one cached "
        f"in memory, otherwise reuse detection revokes the family; "
        f"presented {then_state.presented[-1]!r}"
    )
    assert when_header == {"Authorization": f"Bearer {then_state.issued_access}"}, (
        f"expected the newly issued access token, got {when_header!r}"
    )


async def test_conflicting_refresh_adopts_session_written_by_other_consumer(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-loser",
        refresh="refresh-loser",
        expires_at=int(time.time()) + 30,
    )
    winner_access, winner_expiry = auth_server.state.rotate_as_other_consumer(
        "refresh-winner"
    )
    auth_server.state.on_reject = lambda: write_session(
        session_file,
        app_url=auth_server.app_url,
        access=winner_access,
        refresh="refresh-winner",
        expires_at=winner_expiry,
    )

    given_auth = PerchaiAuthBase(credential_path=str(session_file))
    when_header = await given_auth.get_auth_header(str(session_file))

    assert when_header == {"Authorization": f"Bearer {winner_access}"}, (
        "losing the rotation race must adopt the session the winner persisted "
        f"instead of failing, got {when_header!r}"
    )
    assert auth_server.state.rotations == 0, (
        "adopting must not issue another rotation on top of the winner's"
    )
    assert read_session(session_file)["refreshToken"] == "refresh-winner", (
        "adopting must not clobber the winner's persisted session"
    )


async def test_conflicting_refresh_without_persisted_session_raises(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    from rotator_library.providers.perchai_auth_base import PerchaiAuthError

    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-loser",
        refresh="refresh-loser",
        expires_at=int(time.time()) + 30,
    )
    auth_server.state.rotate_as_other_consumer("refresh-winner")

    given_auth = PerchaiAuthBase(credential_path=str(session_file))

    with pytest.raises(PerchaiAuthError) as when_error:
        await given_auth.get_auth_header(str(session_file))

    then_message = str(when_error.value).lower()
    assert "perch login" in then_message, (
        "an unrecoverable rotation conflict must tell the operator how to fix "
        f"it, got: {when_error.value!r}"
    )


async def test_persisted_session_keeps_inode_and_cli_contract(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-initial",
        refresh="refresh-initial",
        expires_at=int(time.time()) + 30,
    )
    auth_server.state.adopt_initial_refresh("refresh-initial")
    given_inode = session_file.stat().st_ino

    given_auth = PerchaiAuthBase(credential_path=str(session_file))
    await given_auth.get_auth_header(str(session_file))

    then_session = read_session(session_file)
    assert session_file.stat().st_ino == given_inode, (
        "the session file must be rewritten in place: renaming it onto itself "
        "fails with EBUSY through a single-file bind mount and silently loses "
        "the rotation"
    )
    assert then_session["version"] == 1, "Perch CLI rejects any version other than 1"
    assert then_session["appUrl"] == auth_server.app_url, (
        "appUrl must survive the rotation or the CLI rejects the session"
    )
    assert then_session["accessToken"], "accessToken must not be empty"
    assert then_session["updatedAt"], (
        "Perch CLI treats a session without updatedAt as invalid"
    )
    assert isinstance(then_session["expiresAt"], int), (
        "the CLI only keeps a numeric expiresAt; a string would be dropped and "
        "the CLI would then never refresh again"
    )


async def test_auth_context_falls_back_to_default_session_file(
    auth_server: AuthServer,
    session_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rotator_library.providers import perchai_auth_base as perchai_auth

    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-default",
        refresh="refresh-default",
        expires_at=int(time.time()) + 3000,
    )
    monkeypatch.setattr(perchai_auth, "_resolve_session_file", lambda: session_file)

    given_provider = PerchaiProvider()
    then_app_url, then_token = await given_provider._auth_context("")

    assert then_token == "access-default", (
        "an empty credential identifier must resolve the auto-discovered "
        f"session file instead of being sent as a bearer token, got {then_token!r}"
    )
    assert then_app_url == auth_server.app_url, (
        f"expected appUrl from the session file, got {then_app_url!r}"
    )


MODEL = "perchai/nemotron-3.5-lightning"
MESSAGES = [{"role": "user", "content": "Say hello in one word"}]


def given_session_with_dead_access_token(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    write_session(
        session_file,
        app_url=auth_server.app_url,
        access="access-expired",
        refresh="refresh-initial",
        expires_at=int(time.time()) + 3000,
    )
    auth_server.state.adopt_initial_refresh("refresh-initial")


async def test_model_call_401_refreshes_and_retries(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    given_session_with_dead_access_token(auth_server, session_file)

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
    assert when_response.choices[0].message.content == then_state.completion_text, (
        "a 401 from the model endpoint must be retried after a refresh, got "
        f"{when_response.choices[0].message.content!r}"
    )
    assert then_state.rotations == 1, (
        f"expected exactly one refresh to recover from the 401, got {then_state.rotations}"
    )
    assert then_state.model_calls[-1] == f"Bearer {then_state.issued_access}", (
        "the retry must carry the freshly issued access token, got "
        f"{then_state.model_calls[-1]!r}"
    )


async def test_model_call_401_refreshes_and_retries_stream(
    auth_server: AuthServer,
    session_file: Path,
) -> None:
    given_session_with_dead_access_token(auth_server, session_file)

    given_provider = PerchaiProvider()
    collected: List[str] = []
    async with httpx.AsyncClient() as given_client:
        when_stream = await given_provider.acompletion(
            given_client,
            model=MODEL,
            messages=MESSAGES,
            credential_identifier=str(session_file),
            stream=True,
        )
        async for given_chunk in when_stream:
            if not given_chunk.choices:
                continue
            collected.append(given_chunk.choices[0].delta.content or "")

    then_text = "".join(collected)
    then_state = auth_server.state
    assert then_state.completion_text in then_text, (
        "streaming must recover from a 401 the same way non-streaming does, "
        f"collected {then_text!r}"
    )
    assert then_state.rotations == 1, (
        f"expected exactly one refresh to recover from the 401, got {then_state.rotations}"
    )
