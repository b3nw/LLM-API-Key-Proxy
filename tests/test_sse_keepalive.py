# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

import asyncio

from proxy_app import sse_keepalive


class FakeRequest:
    async def is_disconnected(self):
        return False


async def test_stream_with_sse_keepalives_emits_comments_during_quiet_gap(monkeypatch):
    monkeypatch.setattr(sse_keepalive, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0.01)

    async def quiet_then_data():
        await asyncio.sleep(0.025)
        yield 'data: {"ok": true}\n\n'

    chunks = [
        chunk
        async for chunk in sse_keepalive.stream_with_sse_keepalives(
            FakeRequest(), quiet_then_data(), "responses"
        )
    ]

    keepalives = [chunk for chunk in chunks if chunk.startswith(": keepalive responses")]
    assert len(keepalives) >= 2
    assert chunks[-1] == 'data: {"ok": true}\n\n'


async def test_stream_with_sse_keepalives_can_be_disabled(monkeypatch):
    monkeypatch.setattr(sse_keepalive, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0)

    async def quiet_then_data():
        await asyncio.sleep(0.01)
        yield 'data: {"ok": true}\n\n'

    chunks = [
        chunk
        async for chunk in sse_keepalive.stream_with_sse_keepalives(
            FakeRequest(), quiet_then_data(), "responses"
        )
    ]

    assert chunks == ['data: {"ok": true}\n\n']
