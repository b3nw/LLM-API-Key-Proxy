# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if "dimensions" in payload and "embedding" not in model:
        del payload["dimensions"]

    # Note: thinking / reasoning parameter filtering is handled per-provider.
    # Each provider's acompletion() method filters to its own SUPPORTED_PARAMS
    # or converts thinking-style params (e.g. Lightning AI converts `thinking`
    # and `reasoning` dicts to `reasoning_effort`).  The previous hardcoded
    # whitelist here caused thinking to be silently dropped for non-Anthropic /
    # non-Gemini providers like Lightning AI.

    return payload
