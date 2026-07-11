# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""Tests for ForwardedForAccessLogMiddleware.

Verifies that the access log middleware:
- Includes forwarded_for="<ip>" when X-Forwarded-For is present
- Omits the forwarded_for suffix when the header is absent
- Passes websockets and lifespan events through untouched
- Produces well-formed log lines matching the expected format
"""

import logging
from unittest.mock import AsyncMock

import pytest

from proxy_app.access_log_middleware import ForwardedForAccessLogMiddleware


def _make_http_scope(
    method="GET",
    path="/v1/health",
    query_string=b"",
    client=("192.168.4.4", 2671),
    headers=None,
    http_version="1.1",
):
    """Build a minimal ASGI HTTP scope for testing."""
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query_string,
        "client": client,
        "headers": headers or [],
        "http_version": http_version,
        "scheme": "http",
        "root_path": "",
        "server": ("0.0.0.0", 8000),
    }


@pytest.fixture
def captured_logs(caplog):
    """Capture log records from the access middleware's logger."""
    caplog.set_level(logging.INFO, logger="proxy_app.access")
    return caplog


async def _ok_inner_app(scope, receive, send):
    """Minimal ASGI app that returns a 200 response."""
    await send({"type": "http.response.start", "status": 200})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.asyncio
async def test_forwarded_for_in_log(captured_logs):
    """X-Forwarded-For value appears in the access log."""
    middleware = ForwardedForAccessLogMiddleware(_ok_inner_app)

    scope = _make_http_scope(
        headers=[(b"x-forwarded-for", b"136.62.87.30")],
    )

    async def send(message):
        pass

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)

    # Log record was produced
    assert len(captured_logs.records) == 1
    record = captured_logs.records[0]
    msg = record.getMessage()
    assert 'forwarded_for="136.62.87.30"' in msg
    assert "192.168.4.4:2671" in msg
    assert '"GET /v1/health HTTP/1.1"' in msg
    assert "200 OK" in msg


@pytest.mark.asyncio
async def test_no_forwarded_for_omits_suffix(captured_logs):
    """Without X-Forwarded-For, the forwarded_for suffix is omitted."""
    inner = AsyncMock()
    middleware = ForwardedForAccessLogMiddleware(inner)

    scope = _make_http_scope()  # no headers

    async def send(message):
        pass

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)

    assert len(captured_logs.records) == 1
    msg = captured_logs.records[0].getMessage()
    assert "forwarded_for" not in msg
    assert "192.168.4.4:2671" in msg


@pytest.mark.asyncio
async def test_forwarded_for_chain_takes_first(captured_logs):
    """Only the first (original client) IP from a comma-separated chain is logged."""
    inner = AsyncMock()
    middleware = ForwardedForAccessLogMiddleware(inner)

    scope = _make_http_scope(
        headers=[(b"x-forwarded-for", b"136.62.87.30, 10.0.0.1, 10.0.0.2")],
    )

    async def send(message):
        pass

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)

    msg = captured_logs.records[0].getMessage()
    assert 'forwarded_for="136.62.87.30"' in msg
    # The proxies should not appear
    assert "10.0.0.1" not in msg


@pytest.mark.asyncio
async def test_query_string_in_path(captured_logs):
    """Query string is included in the logged request path."""
    inner = AsyncMock()
    middleware = ForwardedForAccessLogMiddleware(inner)

    scope = _make_http_scope(
        path="/v1/health/errors",
        query_string=b"limit=10",
    )

    async def send(message):
        pass

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)

    msg = captured_logs.records[0].getMessage()
    assert '"GET /v1/health/errors?limit=10 HTTP/1.1"' in msg


@pytest.mark.asyncio
async def test_websocket_not_logged(captured_logs):
    """WebSocket connections are passed through without logging."""
    inner = AsyncMock()
    middleware = ForwardedForAccessLogMiddleware(inner)

    scope = {"type": "websocket"}

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        pass

    await middleware(scope, receive, send)

    assert inner.called
    assert len(captured_logs.records) == 0


@pytest.mark.asyncio
async def test_lifespan_not_logged(captured_logs):
    """Lifespan events are passed through without logging."""
    inner = AsyncMock()
    middleware = ForwardedForAccessLogMiddleware(inner)

    scope = {"type": "lifespan"}

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        pass

    await middleware(scope, receive, send)

    assert inner.called
    assert len(captured_logs.records) == 0


@pytest.mark.asyncio
async def test_404_status_logged(captured_logs):
    """Non-200 status codes are logged with their phrase."""
    inner = AsyncMock()

    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 404})
        await send({"type": "http.response.body", "body": b"not found"})

    middleware = ForwardedForAccessLogMiddleware(inner_app)

    scope = _make_http_scope(path="/api/tags")

    async def send(message):
        pass

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)

    msg = captured_logs.records[0].getMessage()
    assert "404 Not Found" in msg
