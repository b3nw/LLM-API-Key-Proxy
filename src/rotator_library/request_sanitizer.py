# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if "dimensions" in payload and not model.startswith("openai/text-embedding-3"):
        del payload["dimensions"]

    # Strip top-level thinking key for models that don't support it
    if "thinking" in payload:
        is_gemini = model.startswith("gemini/") or "gemini-" in model
        is_anthropic = model.startswith("anthropic/") or "claude-" in model
        if not (is_gemini or is_anthropic):
            del payload["thinking"]

    # Also strip thinking injected into extra_body (e.g. by the global
    # _guard_thinking_tool_calls transform) for non-supporting models.
    # Mistral's strict input validation rejects the thinking field on
    # models like ministral-14b that don't support it.
    extra_body = payload.get("extra_body")
    if isinstance(extra_body, dict) and "thinking" in extra_body:
        is_gemini = model.startswith("gemini/") or "gemini-" in model
        is_anthropic = model.startswith("anthropic/") or "claude-" in model
        is_mistral_reasoning = any(
            p in model for p in ("mistral-medium", "mistral-small")
        )
        if not (is_gemini or is_anthropic or is_mistral_reasoning):
            del extra_body["thinking"]

    return payload
