# SPDX-License-Identifier: MIT
# Copyright (c) 2026 COJEAN Kévin

import asyncio
import hashlib
import socketserver
import sys
import threading
from pathlib import Path
from typing import Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rotator_library import RotatingClient
from rotator_library.proxy_config import ProxyConfig, ProxySpec


# Deterministic synthetic credential and its 12-hex-char stable_id.
TEST_API_KEY = "sk-fake-test-key-for-proxy-routing-0000"
TEST_STABLE_ID = hashlib.sha256(TEST_API_KEY.encode()).hexdigest()[:12]
assert len(TEST_STABLE_ID) == 12


class _ProxyTrace:
    """Thread-safe buffer of CONNECT request lines the fake proxy received."""

    def __init__(self) -> None:
        self.lines = []
        self._lock = threading.Lock()

    def record(self, line: str, peer: Tuple[str, int]) -> None:
        with self._lock:
            self.lines.append((line, peer))


class _FakeProxyHandler(socketserver.BaseRequestHandler):
    """Accept one HTTP request line, send ``200 Connection Established``
    so a CONNECT-aware client proceeds with the tunnel, then close. Anything
    that uses a CONNECT-style proxy will leave a trace of the target host."""

    def handle(self) -> None:
        trace = self.server.trace  # type: ignore[attr-defined]
        try:
            data = self.request.recv(4096)
            first_line = data.split(b"\r\n", 1)[0] if data else b""
            trace.record(
                first_line.decode("latin-1", errors="replace"),
                self.client_address,
            )
            self.request.sendall(
                b"HTTP/1.1 200 Connection Established\r\n"
                b"Proxy-Agent: fake-test-proxy\r\n"
                b"\r\n"
            )
            try:
                self.request.settimeout(0.5)
                self.request.recv(4096)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            try:
                self.request.close()
            except Exception:
                pass


class _FakeProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    trace: _ProxyTrace


@pytest.fixture
def fake_proxy():
    """Spin up a real TCP proxy on localhost, yield ``{trace, url}``, tear down."""
    trace = _ProxyTrace()
    server = _FakeProxyServer(("127.0.0.1", 0), _FakeProxyHandler)
    server.trace = trace
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        proxy_url = f"http://user:pass@{host}:{port}/"
        yield {"trace": trace, "url": proxy_url}
    finally:
        server.shutdown()
        server.server_close()


def _build_client(
    tmp_path: Path,
    proxy_url: str,
    api_key: str = TEST_API_KEY,
    provider: str = "opencode_go",
) -> RotatingClient:
    """Build a RotatingClient for ``provider`` with the proxy URL bound to ``api_key``'s stable_id."""
    stable_id = hashlib.sha256(api_key.encode()).hexdigest()[:12]
    proxy_config = ProxyConfig(
        credential_proxies={stable_id: ProxySpec(url=proxy_url)},
    )
    return RotatingClient(
        api_keys={provider: [api_key]},
        proxy_config=proxy_config,
        data_dir=tmp_path,
        configure_logging=False,
        global_timeout=5,
    )


@pytest.mark.asyncio
async def test_opencode_go_request_traverses_configured_proxy(
    tmp_path, fake_proxy,
):
    """RED: the fake proxy is never contacted before the fix.
    GREEN: after executor.py passes the proxied client to plugin.acompletion,
    the fake proxy records a CONNECT targeting the opencode host.
    """
    client = _build_client(tmp_path, fake_proxy["url"])
    try:
        # Upstream call will fail: the fake proxy does not forward to the real
        # OpenCode API and the API key is synthetic. We don't care about the
        # final error; only that the fake proxy was contacted first.
        with pytest.raises(Exception):
            await asyncio.wait_for(
                client.acompletion(
                    model="opencode_go/kimi-k2.6",
                    messages=[{"role": "user", "content": "ping"}],
                ),
                timeout=10,
            )
    finally:
        await client.close()

    trace = fake_proxy["trace"]
    assert trace.lines, (
        "Configured per-credential proxy was never contacted. The executor "
        "is passing its unproxied shared client (self._http_client) to the "
        "opencode_go plugin instead of the proxy-aware client resolved "
        "from ProxiedClientPool.get_client()."
    )
    targets = [
        parts[1]
        for line, _ in trace.lines
        for parts in [line.split(" ")]
        if len(parts) >= 2
    ]
    assert any("opencode" in t.lower() for t in targets), (
        f"Expected a CONNECT targeting opencode host; observed targets={targets}"
    )


DEEPSEEK_TEST_API_KEY = "sk-fake-deepseek-key-for-proxy-routing-0000"
DEEPSEEK_TEST_STABLE_ID = hashlib.sha256(DEEPSEEK_TEST_API_KEY.encode()).hexdigest()[:12]


@pytest.mark.asyncio
async def test_deepseek_request_traverses_configured_proxy(tmp_path, fake_proxy):
    """Pattern B: deepseek plugin calls ``client.post(...)`` and
    ``_stream_completion(client=client, ...)`` directly rather than wrapping
    the executor's client in ``openai.AsyncOpenAI``. The executor fix must
    route the proxy-aware client through this path too.
    """
    client = _build_client(
        tmp_path=tmp_path,
        proxy_url=fake_proxy["url"],
        api_key=DEEPSEEK_TEST_API_KEY,
        provider="deepseek",
    )
    try:
        with pytest.raises(Exception):
            await asyncio.wait_for(
                client.acompletion(
                    model="deepseek/deepseek-chat",
                    messages=[{"role": "user", "content": "ping"}],
                ),
                timeout=10,
            )
    finally:
        await client.close()

    trace = fake_proxy["trace"]
    assert trace.lines, (
        "Pattern B (direct client.post): the configured proxy was never "
        "contacted. The executor fix must propagate the proxy-aware client "
        "to plugins that use httpx directly, not only to those that wrap "
        "the client in openai.AsyncOpenAI."
    )
