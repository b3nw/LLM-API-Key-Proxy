# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""Regression: early WebSocket disconnect during auth must not double-close (ASGI RuntimeError)."""

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.testclient import TestClient


async def _auth_ws_handler(websocket: WebSocket, api_key: str):
    """Minimal mirror of proxy_app.main websocket auth handshake."""
    await websocket.accept()
    if api_key:
        try:
            await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        except WebSocketDisconnect:
            return
        except asyncio.TimeoutError:
            try:
                await websocket.close(code=4001, reason="Auth timeout")
            except RuntimeError:
                pass
            return


def test_ws_disconnect_during_auth_does_not_raise():
    app = FastAPI()

    @app.websocket("/v1/ws")
    async def ws(websocket: WebSocket):
        await _auth_ws_handler(websocket, "secret")

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws") as _ws:
            pass  # disconnect immediately — must not surface ASGI RuntimeError