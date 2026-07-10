# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""Helpers for keeping server-sent event streams byte-active."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import Protocol


class DisconnectAwareRequest(Protocol):
    """Subset of FastAPI Request needed by stream keepalive wrapping."""

    async def is_disconnected(self) -> bool: ...


def read_non_negative_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logging.warning("Invalid %s=%r; using default %ss", name, raw, default)
        return default
    if value < 0:
        logging.warning("Invalid %s=%r; using default %ss", name, raw, default)
        return default
    return value


# Some clients abort an SSE stream if no bytes are received for a fixed idle
# window. Long reasoning/tool gaps can be valid, so emit SSE comments during
# quiet periods. Comments are ignored by SSE parsers but reset read-idle timers.
SSE_KEEPALIVE_INTERVAL_SECONDS = read_non_negative_float_env(
    "SSE_KEEPALIVE_INTERVAL_SECONDS", 30.0
)


async def stream_with_sse_keepalives(
    request: DisconnectAwareRequest,
    response_stream: AsyncGenerator[str, None],
    context: str,
) -> AsyncGenerator[str, None]:
    """Yield upstream SSE chunks, inserting comment keepalives during quiet gaps."""
    interval = SSE_KEEPALIVE_INTERVAL_SECONDS
    iterator = response_stream.__aiter__()
    pending: asyncio.Task[str] | None = None
    keepalive_count = 0

    try:
        while True:
            if await request.is_disconnected():
                logging.warning("Client disconnected, stopping %s stream.", context)
                break

            if pending is None:
                pending = asyncio.create_task(iterator.__anext__())

            if interval > 0:
                done, _ = await asyncio.wait({pending}, timeout=interval)
            else:
                done, _ = await asyncio.wait({pending})

            if not done:
                keepalive_count += 1
                yield f": keepalive {context} {keepalive_count}\n\n"
                continue

            try:
                chunk = pending.result()
            except StopAsyncIteration:
                break
            finally:
                pending = None

            yield chunk
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception as e:
                logging.debug("Error closing %s stream: %s", context, e)
