# Feature: access-log

**Stack subject:** `feat(access-log): expose X-Forwarded-For in access logs behind reverse proxy`
**Feature ID:** `access-log`

## Problem

When the proxy runs behind a reverse proxy (e.g. Traefik, Nginx), uvicorn's
default access log only shows the TCP connection IP — which is always the
reverse proxy's address, not the real client. The real client IP is available
in the `X-Forwarded-For` HTTP header but never surfaced in the logs.

## Solution

Added a pure ASGI middleware (`ForwardedForAccessLogMiddleware`) that
intercepts every HTTP request and logs:

```
<client_ip>:<client_port> - "<method> <path>?<query> HTTP/<ver>" <status> <phrase> forwarded_for="<ip>"
```

When no `X-Forwarded-For` header is present, the `forwarded_for` suffix is
omitted and the log line matches the original uvicorn format.

### Changes

- **New file:** `src/proxy_app/access_log_middleware.py` — the middleware
- **Modified:** `src/proxy_app/main.py` — wire the middleware into the app,
  silence `uvicorn.access` logger, pass `proxy_headers=True` and
  `access_log=False` to `uvicorn.run()`
- **New tests:** `tests/test_access_log_middleware.py` — 7 tests covering
  forwarded-for presence/absence, comma-separated chains, query strings,
  websocket/lifespan passthrough, and 404 status

### Branch

`feat/access-log-forwarded` — push to `origin` and open PR against `dev`.

### Verification

```bash
uv run --with pytest --with pytest-asyncio --with pytest-mock \
  python3 -m pytest tests/test_access_log_middleware.py -v
```

All 7 tests pass. Lint clean (`py_compile` + `ruff F401,F811,F821,E9`).
