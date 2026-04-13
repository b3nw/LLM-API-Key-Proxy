# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any

def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if "dimensions" in payload and not model.startswith("openai/text-embedding-3"):
        del payload["dimensions"]
        
    if "thinking" in payload:
        is_gemini = model.startswith("gemini/") or "gemini-" in model
        is_anthropic = model.startswith("anthropic/") or "claude-" in model
        if not (is_gemini or is_anthropic):
            del payload["thinking"]
            
    return payload
