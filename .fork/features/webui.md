# webui — Admin Web UI

## 2026-06-24 — WebSocket auth: handle client disconnect without double-close

Target: `feat(webui): add React web UI with admin dashboard, quota viewer, log explorer, and settings`
Files:
- `src/proxy_app/main.py`
- `tests/test_websocket_auth_disconnect.py`

Working commits before autosquash:
- `da7e48b3 fixup! feat(webui): add React web UI with admin dashboard, quota viewer, log explorer, and settings`

Verification:
- `uv run python3 -m py_compile src/proxy_app/main.py` — passed
- `uv run ruff check src/proxy_app/main.py --select F401,F811,F821,E9` — passed
- `uv run python3 -m pytest tests/test_websocket_auth_disconnect.py -q` — passed

Notes:
- `/v1/ws` lives in `main.py` from the WebUI real-time quota feed; early client
  disconnect during PROXY_API_KEY auth must not call `close()` on an already-closed socket.
- `WebSocketDisconnect` returns quietly; timeouts use `_safe_ws_close()`.