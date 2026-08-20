#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel
"""
Discover Perchai option IDs by reverse-engineering the Perch CLI bundle.

Perchai does not publish a public model catalog. Option IDs are derived from
(provider, upstream_model) pairs via the ``Tl(provider, model)`` helper in
``perch.mjs`` v2.4.87. This script enumerates every ``Tl(...)`` call site,
computes each option ID, and probes them in parallel batches of 10 against
``POST /api/perch-terminal/model-call``.

Usage: ``uv run python3 scripts/perchai_discover_models.py``

Prereqs: ``perch login`` (writes ``~/.perch/cli-auth-session.json``).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from typing import List, Tuple

import httpx

APP_URL = "https://app.perchai.app"
SESSION_PATH = "/home/dehi/.perch/cli-auth-session.json"
BATCH_SIZE = 10

TL_PAIRS: List[Tuple[str, str]] = [
    ("wandb", "deepseek-ai/DeepSeek-V4-Flash"),
    ("wandb", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    ("wandb", "deepseek-ai/DeepSeek-V4-Pro"),
    ("wandb", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"),
    ("wandb", "Qwen/Qwen3.8-27B"),
    ("bedrock_mantle", "deepseek.v3.2"),
    ("bedrock_mantle", "google.gemma-3-12b-it"),
    ("bedrock_mantle", "google.gemma-3-27b-it"),
    ("bedrock_mantle", "google.gemma-4-31b"),
    ("bedrock_mantle", "google.gemma-4-e2b"),
    ("bedrock_mantle", "minimax.minimax-m2"),
    ("bedrock_mantle", "moonshotai.kimi-k2.5"),
    ("bedrock_mantle", "nvidia.nemotron-super-3-120b"),
    ("bedrock_mantle", "openai.gpt-oss-120b"),
    ("bedrock_mantle", "qwen.qwen3-coder-480b-a35b-instruct"),
    ("bedrock_mantle", "qwen.qwen3-vl-235b-a22b-instruct"),
    ("bedrock_mantle", "xai.grok-4.3"),
    ("bedrock_mantle", "zai.glm-5"),
    ("ai_gateway", "anthropic/claude-3-haiku"),
    ("ai_gateway", "anthropic/claude-haiku-4.5"),
    ("ai_gateway", "xai/grok-4.5"),
    ("claude_code_oauth", "claude-haiku-4-5-20251001"),
    ("claude_code_oauth", "claude-opus-4-8"),
    ("claude_code_oauth", "claude-sonnet-4-6"),
    ("codex_oauth", "gpt-5.4"),
    ("codex_oauth", "gpt-5.4-mini"),
    ("codex_oauth", "gpt-5.5"),
    ("cohere", "command-a-reasoning-08-2025"),
    ("fireworks", "accounts/fireworks/models/deepseek-v4-flash"),
    ("fireworks", "accounts/fireworks/models/inkling"),
    ("fireworks", "accounts/fireworks/models/qwen3p7-plus"),
    ("grok_oauth", "grok-4.5"),
    ("grok_oauth", "grok-4.6"),
    ("meta", "muse-spark-1.1"),
    ("meta", "muse-spark-1.2"),
    ("meta", "muse-spark-1.2-contributor"),
    ("nvidia_nim", "meta/llama-3.2-11b-vision-instruct"),
    ("nvidia_nim", "mistralai/ministral-14b-instruct-2512"),
    ("nvidia_nim", "nvidia/nemotron-nano-12b-v2-vl"),
]


def tl(provider: str, model: str) -> str:
    """Mirror of ``Tl(e, t)`` in perch.mjs v2.4.87."""
    p = provider.replace("_", "-")
    m = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()
    return f"{p}-{m}"


def get_fresh_token() -> str:
    """Mint a fresh Perch access token via Supabase GoTrue refresh."""
    refresh = subprocess.check_output(
        ["jq", "-r", ".refreshToken", SESSION_PATH], text=True
    ).strip()
    config = httpx.get(f"{APP_URL}/api/perch-terminal/cli-auth/config").json()
    resp = httpx.post(
        f"{config['supabaseUrl']}/auth/v1/token?grant_type=refresh_token",
        json={"refresh_token": refresh},
        headers={"apikey": config["supabaseAnonKey"]},
    ).json()
    return resp["access_token"]


async def probe_one(
    client: httpx.AsyncClient, token: str, opt_id: str, sem: asyncio.Semaphore
) -> str:
    """Probe one option ID. A response matching the default upstream model
    means the option ID is not currently wired."""
    payload = {
        "request": {
            "model": "probe",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": False,
        },
        "runId": None,
        "lane": "chat",
        "preferredModelId": None,
        "manualModelOptionId": opt_id,
    }
    async with sem:
        try:
            r = await client.post(
                f"{APP_URL}/api/perch-terminal/model-call",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0,
            )
            body = r.json()
            if body.get("ok"):
                m, p = body.get("model", "?"), body.get("provider", "?")
                fallback = m == "moonshotai.kimi-k2.5" and p == "bedrock_mantle"
                marker = "ROUTED  " if not fallback else "fallback"
                return f"  {marker}  {opt_id:55s} -> {p:18s} {m}"
            return f"  ERROR   {opt_id:55s} -> {body.get('error', '?')[:60]}"
        except Exception as exc:
            return f"  EXC     {opt_id:55s} -> {str(exc)[:60]}"


async def probe_batch(
    client: httpx.AsyncClient, token: str, opt_ids: List[str]
) -> List[str]:
    """Probe a list of option IDs in parallel. 10-in-flight is empirically
    the fastest rate below the 401 lockout threshold."""
    sem = asyncio.Semaphore(BATCH_SIZE)
    tasks = [probe_one(client, token, oid, sem) for oid in opt_ids]
    return list(await asyncio.gather(*tasks))


def main() -> int:
    ids = [(p, m, tl(p, m)) for p, m in TL_PAIRS]
    print(f"Computed {len(ids)} option IDs from {len(TL_PAIRS)} (provider, model) pairs")
    for p, m, oid in ids:
        print(f"  Tl({p:18s}, {m:50s}) = {oid}")
    print()

    print("Refreshing access token...")
    token = get_fresh_token()
    print("Token OK, probing in parallel batches of 10...\n")

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            for i in range(0, len(ids), BATCH_SIZE):
                batch = ids[i : i + BATCH_SIZE]
                print(f"=== Batch {i // BATCH_SIZE + 1}: {len(batch)} IDs ===")
                for r in await probe_batch(client, token, [oid for _, _, oid in batch]):
                    print(r)
                print()

    asyncio.run(run())
    print(f"Total: {len(ids)} requests, ~{len(ids) * 2} tokens burned")
    return 0


if __name__ == "__main__":
    sys.exit(main())