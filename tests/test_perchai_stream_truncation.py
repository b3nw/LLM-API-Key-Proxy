# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel
from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from rotator_library.providers.perchai_provider import PerchaiProvider


class MockStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockAsyncClient:
    def __init__(self, response: MockStreamResponse):
        self._response = response

    def stream(self, method: str, url: str, **kwargs) -> MockStreamResponse:
        return self._response


@pytest.mark.asyncio
async def test_perchai_stream_normal_termination():
    provider = PerchaiProvider()
    client = MagicMock(spec=httpx.AsyncClient)

    chunk = {
        "type": "text_delta",
        "text": "Hello",
    }

    lines = [
        f"data: {json.dumps(chunk)}",
        "data: [DONE]",
    ]
    response = MockStreamResponse(lines)
    client.stream = MagicMock(return_value=response)

    file_logger = AsyncMock()
    stream = provider._stream_completion(
        client=client,
        url="https://api.perchai.com/v1/chat/completions",
        build_headers=lambda token: {"Authorization": f"Bearer {token}"},
        token="test-token",
        payload={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
        model="perchai/test-model",
        file_logger=file_logger,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) >= 1
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_perchai_stream_incomplete_raises_error():
    provider = PerchaiProvider()
    client = MagicMock(spec=httpx.AsyncClient)

    chunk = {
        "type": "text_delta",
        "text": "Hello",
    }

    lines = [
        f"data: {json.dumps(chunk)}",
    ]
    response = MockStreamResponse(lines)
    client.stream = MagicMock(return_value=response)

    file_logger = AsyncMock()
    stream = provider._stream_completion(
        client=client,
        url="https://api.perchai.com/v1/chat/completions",
        build_headers=lambda token: {"Authorization": f"Bearer {token}"},
        token="test-token",
        payload={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
        model="perchai/test-model",
        file_logger=file_logger,
    )

    chunks = []
    with pytest.raises(RuntimeError, match="prematurely|truncated"):
        async for chunk in stream:
            chunks.append(chunk)


@pytest.mark.asyncio
async def test_perchai_stream_done_event_termination():
    provider = PerchaiProvider()
    client = MagicMock(spec=httpx.AsyncClient)

    text_chunk = {
        "type": "text_delta",
        "text": "Hello",
    }
    done_event = {
        "type": "done",
        "finishReason": "stop",
        "ok": True,
    }

    lines = [
        f"data: {json.dumps(text_chunk)}",
        f"data: {json.dumps(done_event)}",
    ]
    response = MockStreamResponse(lines)
    client.stream = MagicMock(return_value=response)

    file_logger = AsyncMock()
    stream = provider._stream_completion(
        client=client,
        url="https://api.perchai.com/v1/chat/completions",
        build_headers=lambda token: {"Authorization": f"Bearer {token}"},
        token="test-token",
        payload={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
        model="perchai/test-model",
        file_logger=file_logger,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert chunks[-1].choices[0].finish_reason == "stop"
