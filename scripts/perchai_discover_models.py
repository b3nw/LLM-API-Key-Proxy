#!/usr/bin/env python3
"""
Perchai model discovery.

Perchai does not publish a public model catalog. The option IDs that
``app.perchai.app`` honors must be reverse-engineered from the Perch CLI
bundle (``perch.mjs`` v2.4.87) and verified via live probes.

This script does both:

1. Compute the cartesian product of (provider, upstream_model) pairs that
   the bundle references via the ``Tl(provider, model)`` function.
2. Probe each option ID against ``POST /api/perch-terminal/model-call``
   in parallel batches of 10 (this is the rate the bundle will tolerate
   without triggering a 2-hour auth cooldown).

A successful probe returns ``{"ok": true, "model": "...", "provider":
"..."}`` where ``model`` is the upstream model the option ID routes to.
A failed probe (or one whose response matches the default ``bedrock_mantle:
moonshotai.kimi-k2.5``) means the option ID is not currently wired.

Usage::

    uv run python3 scripts/perchai_discover_models.py

Prereqs:
- ``perch login`` (writes ``~/.perch/cli-auth-session.json``)
- ``uv`` environment with ``httpx`` available

The (provider, model) pairs below were extracted from every ``Tl(...)``
call site in the perch CLI bundle. Add new pairs here as perch adds new
models. To add a pair from first principles:

- ``provider`` is the perch-internal routing label (e.g. ``wandb``,
  ``bedrock_mantle``, ``ai_gateway``, ``claude_code_oauth``).
- ``model`` is the upstream model name as perchai would log it
  (e.g. ``moonshotai/Kimi-K2.5``, ``deepseek-ai/DeepSeek-V4-Flash``).
  Sometimes the upstream is referenced with dots instead of slashes
  (e.g. ``moonshotai.kimi-k2.5``) - use whatever form the bundle shows.
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

# All (provider, upstream_model) pairs extracted from `Tl(...)` calls
# in perch.mjs v2.4.87. The full list is 39 entries; 21 of them route
# to real upstream models (the rest either error or fall back to the
# default kimi-k2.5 model).
TL_PAIRS: List[Tuple[str, str]] = [
    # --- wandb (Weights & Biases inference) ---
    ("wandb", "deepseek-ai/DeepSeek-V4-Flash"),
    ("wandb", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    ("wandb", "deepseek-ai/DeepSeek-V4-Pro"),
    ("wandb", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"),
    ("wandb", "Qwen/Qwen3.8-27B"),
    # --- bedrock_mantle (AWS Bedrock via Mantle) ---
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
    # --- ai_gateway (Perch's own Anthropic/xAI gateway) ---
    ("ai_gateway", "anthropic/claude-3-haiku"),
    ("ai_gateway", "anthropic/claude-haiku-4.5"),
    ("ai_gateway", "xai/grok-4.5"),
    # --- claude_code_oauth (Anthropic via Claude Code OAuth) ---
    ("claude_code_oauth", "claude-haiku-4-5-20251001"),
    ("claude_code_oauth", "claude-opus-4-8"),
    ("claude_code_oauth", "claude-sonnet-4-6"),
    # --- codex_oauth (OpenAI via Codex OAuth) ---
    ("codex_oauth", "gpt-5.4"),
    ("codex_oauth", "gpt-5.4-mini"),
    ("codex_oauth", "gpt-5.5"),
    # --- cohere ---
    ("cohere", "command-a-reasoning-08-2025"),
    # --- fireworks ---
    ("fireworks", "accounts/fireworks/models/deepseek-v4-flash"),
    ("fireworks", "accounts/fireworks/models/inkling"),
    ("fireworks", "accounts/fireworks/models/qwen3p7-plus"),
    # --- grok_oauth (xAI via Grok OAuth) ---
    ("grok_oauth", "grok-4.5"),
    ("grok_oauth", "grok-4.6"),
    # --- meta (Meta Muse family) ---
    ("meta", "muse-spark-1.1"),
    ("meta", "muse-spark-1.2"),
    ("meta", "muse-spark-1.2-contributor"),
    # --- nvidia_nim (NVIDIA NIM catalog) ---
    ("nvidia_nim", "meta/llama-3.2-11b-vision-instruct"),
    ("nvidia_nim", "mistralai/ministral-14b-instruct-2512"),
    ("nvidia_nim", "nvidia/nemotron-nano-12b-v2-vl"),
]


def tl(provider: str, model: str) -> str:
    """
    Compute an option ID from a (provider, upstream_model) pair.

    Mirrors the `Tl(e, t)` function from perch.mjs v2.4.87 exactly::

        function Tl(e, t) {
            return e.replace(/_/g, "-") + "-" +
                   t.replace(/[^a-zA-Z0-9]+/g, "-")
                    .replace(/^-+|-+$/g, "")
                    .toLowerCase();
        }

    Steps:
    - Replace ``_`` with ``-`` in the provider name.
    - Replace every run of non-alphanumeric characters in the model
      name with ``-`` and strip leading/trailing dashes.
    - Lowercase the result.
    - Join with ``-``.

    Examples::

        Tl("wandb", "deepseek-ai/DeepSeek-V4-Flash")
            == "wandb-deepseek-ai-deepseek-v4-flash"
        Tl("bedrock_mantle", "moonshotai.kimi-k2.5")
            == "bedrock-mantle-moonshotai-kimi-k2-5"
        Tl("ai_gateway", "anthropic/claude-3-haiku")
            == "ai-gateway-anthropic-claude-3-haiku"
    """
    p = provider.replace("_", "-")
    m = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()
    return f"{p}-{m}"


def get_fresh_token(session_path: str = SESSION_PATH) -> str:
    """
    Mint a fresh Perch access token from the saved refresh token.

    Perchai uses Supabase GoTrue for token refresh::

        POST {supabaseUrl}/auth/v1/token?grant_type=refresh_token

    The Supabase URL and anon key are discovered via
    ``GET /api/perch-terminal/cli-auth/config``.
    """
    refresh = subprocess.check_output(
        ["jq", "-r", ".refreshToken", session_path], text=True
    ).strip()
    config = httpx.get(f"{APP_URL}/api/perch-terminal/cli-auth/config").json()
    supabase = config["supabaseUrl"]
    anon = config["supabaseAnonKey"]
    resp = httpx.post(
        f"{supabase}/auth/v1/token?grant_type=refresh_token",
        json={"refresh_token": refresh},
        headers={"apikey": anon},
    ).json()
    return resp["access_token"]


async def probe_one(
    client: httpx.AsyncClient,
    token: str,
    opt_id: str,
    sem: asyncio.Semaphore,
) -> str:
    """
    Probe a single option ID. Returns a formatted one-line result.

    A probe is a minimal model-call request with ``max_tokens=2`` and
    a one-word user message. The response reveals whether the option ID
    actually routes to a specific upstream model (``ROUTED``) or fell
    back to the default ``bedrock_mantle:moonshotai.kimi-k2.5``
    (``fallback``).
    """
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
                m = body.get("model", "?")
                p = body.get("provider", "?")
                routed = not (m == "moonshotai.kimi-k2.5" and p == "bedrock_mantle")
                marker = "ROUTED  " if routed else "fallback"
                return f"  {marker}  {opt_id:55s} -> {p:18s} {m}"
            err = body.get("error", "?")[:60]
            return f"  ERROR   {opt_id:55s} -> {err}"
        except Exception as exc:
            return f"  EXC     {opt_id:55s} -> {str(exc)[:60]}"


async def probe_batch(
    client: httpx.AsyncClient,
    token: str,
    opt_ids: List[str],
    batch_size: int = 10,
) -> List[str]:
    """
    Probe a list of option IDs in parallel batches of ``batch_size``.

    10-in-flight is the empirical sweet spot: faster than serial
    (one round-trip per batch instead of one per ID) but well below
    the rate at which perchai starts responding with 401s and locking
    the account for ~2 hours.
    """
    sem = asyncio.Semaphore(batch_size)
    tasks = [probe_one(client, token, oid, sem) for oid in opt_ids]
    return list(await asyncio.gather(*tasks))


def main() -> int:
    print(f"Computed {len(TL_PAIRS)} (provider, model) pairs from perch.mjs")
    print(f"Each pair -> one option ID via Tl(provider, model)")
    print()

    ids: List[str] = [(p, m, tl(p, m)) for p, m in TL_PAIRS]
    for p, m, oid in ids:
        print(f"  Tl({p:18s}, {m:50s}) = {oid}")

    print()
    print("Refreshing access token...")
    token = get_fresh_token()
    print("Token OK, probing in parallel batches of 10...")
    print()

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            for i in range(0, len(ids), 10):
                batch = ids[i : i + 10]
                print(f"=== Batch {i // 10 + 1}: {len(batch)} IDs ===")
                opt_ids = [oid for _, _, oid in batch]
                results = await probe_batch(client, token, opt_ids)
                for r in results:
                    print(r)
                print()

    asyncio.run(run())

    print(f"Total: {len(ids)} requests, ~{len(ids) * 2} tokens burned")
    print()
    print("Interpretation:")
    print("  ROUTED   - option ID accepted, response routed to the upstream model shown")
    print("  fallback - option ID silently fell back to the default kimi-k2.5 model")
    print("  ERROR    - perchai returned an explicit error (auth, exhausted, etc.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
