"""Access log middleware that exposes X-Forwarded-For information.

Replaces uvicorn's default access log with one that includes the real client
IP from the X-Forwarded-For header alongside the TCP connection IP, so the
originating client is visible when the proxy sits behind a reverse proxy.
"""

import logging
from http import HTTPStatus

logger = logging.getLogger("proxy_app.access")


class ForwardedForAccessLogMiddleware:
    """Pure ASGI middleware that logs HTTP access with X-Forwarded-For info.

    Logs each HTTP request with both the TCP connection address and the
    X-Forwarded-For header value (if present), so the real client IP is
    visible behind reverse proxies.

    Log format::

        <client_ip>:<client_port> - "<method> <path>?<query> HTTP/<ver>" <status> <phrase> forwarded_for="<ip>"

    When no X-Forwarded-For header is present the ``forwarded_for`` suffix is
    omitted, matching the original uvicorn format.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Only log HTTP requests; pass websockets and lifespan through untouched.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        # --- Build the access log line ---
        client = scope.get("client")
        client_addr = f"{client[0]}:{client[1]}" if client else "-"

        method = scope.get("method", "")
        path = scope.get("path", "")
        query_string = scope.get("query_string", b"")
        if query_string:
            full_path = f"{path}?{query_string.decode('ascii', errors='replace')}"
        else:
            full_path = path
        http_version = scope.get("http_version", "1.1")
        request_line = f"{method} {full_path} HTTP/{http_version}"

        try:
            status_phrase = HTTPStatus(status_code).phrase
        except ValueError:
            status_phrase = ""

        # Extract the first (original) IP from X-Forwarded-For.
        # A single header may contain a comma-separated chain: "client, proxy1, proxy2".
        forwarded_for = ""
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-for":
                forwarded_for = (
                    value.decode("utf-8", errors="replace")
                    .split(",")[0]
                    .strip()
                )
                break

        if forwarded_for:
            logger.info(
                '%s - "%s" %s %s forwarded_for="%s"',
                client_addr,
                request_line,
                status_code,
                status_phrase,
                forwarded_for,
            )
        else:
            logger.info(
                '%s - "%s" %s %s',
                client_addr,
                request_line,
                status_code,
                status_phrase,
            )
