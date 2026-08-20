#!/usr/bin/env python3
"""
Perchai pricing probe.

Perchai does not expose per-model pricing via the API. What it does
expose:

1. ``GET /api/perch-terminal/usage`` - monthly aggregate
   (``monthly.usageUsd``, ``monthly.limitUsd``, ``monthly.resetAt``).
   Aggregate only, no per-model breakdown.

2. ``POST /api/perch-terminal/model-call`` response ``usage`` block -
   ``inputTokens``, ``outputTokens``, ``totalTokens``,
   ``cacheReadInputTokens``. No cost field.

So per-model pricing has to be inferred from perchai's website, the
perch CLI bundle if a cost field ever appears, or experimental
measurement (call the model with N input tokens, divide the monthly
cost delta by N - approximate only).

This script dumps whatever fields the two endpoints return so you can
see for yourself what's available.

Usage::

    uv run python3 scripts/perchai_probe_prices.py [option-id]

Defaults to ``bedrock-mantle-moonshotai-kimi-k2-5`` (the default model).
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

from scripts.perchai_discover_models import get_fresh_token

APP_URL = "https://app.perchai.app"

USAGE_FIELDS = (
    "monthly",
    "usedUsd",
    "used",
    "usage",
    "balance",
    "capUsd",
    "limit",
    "cap",
    "allowance",
)

COST_KEY_HINTS = ("cost", "usd", "price", "charge")


def _show_usage_response(body: object) -> None:
    if isinstance(body, dict):
        relevant = {k: body[k] for k in body if k in USAGE_FIELDS}
        print("  body (known fields):")
        print(json.dumps(relevant, indent=2, default=str))
    else:
        print(f"  body: {body!r}")


def _show_model_call_response(body: object) -> None:
    if not isinstance(body, dict):
        print(f"  raw: {body!r}")
        return
    trimmed = {k: v for k, v in body.items() if k != "text"}
    print("  response (excluding text):")
    print(json.dumps(trimmed, indent=2, default=str))
    usage = trimmed.get("usage") or {}
    if not usage:
        print("  no usage object")
        return
    print(f"\n  usage keys: {sorted(usage.keys())}")
    cost_keys = [k for k in usage if any(h in k.lower() for h in COST_KEY_HINTS)]
    if cost_keys:
        print(f"  cost-related keys present: {cost_keys}")
    else:
        print("  no cost-related keys in usage")


async def main(opt_id: str) -> None:
    token = get_fresh_token()
    print("Token OK.\n")

    url_usage = f"{APP_URL}/api/perch-terminal/usage"
    print(f"GET {url_usage}")
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            url_usage,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        print(f"  status: {r.status_code}")
        try:
            body: object = r.json()
        except Exception:
            body = r.text
        _show_usage_response(body)

    payload = {
        "request": {
            "model": "probe",
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": 8,
            "stream": False,
        },
        "runId": None,
        "lane": "chat",
        "preferredModelId": None,
        "manualModelOptionId": opt_id,
    }
    url_call = f"{APP_URL}/api/perch-terminal/model-call"
    print(f"\nPOST {url_call}  manualModelOptionId={opt_id!r}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            url_call,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            call_body: object = r.json()
        except Exception:
            call_body = r.text
        _show_model_call_response(call_body)


if __name__ == "__main__":
    opt_id = sys.argv[1] if len(sys.argv) > 1 else "bedrock-mantle-moonshotai-kimi-k2-5"
    asyncio.run(main(opt_id))