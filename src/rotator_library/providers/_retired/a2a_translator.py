# SPDX-License-Identifier: LGPL-3.0-only
# Retired A2A translator (archived — see _retired/README.md)
# Original: Translator between OpenAI chat completion format and A2A protocol.

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator

from .a2a_client import A2AEvent

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# OPENAI → A2A TRANSLATION
# =============================================================================


def openai_messages_to_a2a_prompt(
    messages: List[Dict[str, Any]],
    is_new_session: bool = True,
) -> str:
    """
    Convert OpenAI messages array to a single A2A prompt string.

    Strategy:
        - For new sessions: concatenate full history with role prefixes
        - For follow-up turns: only send the latest user message
          (previous messages are already in the A2A task's context)

    Args:
        messages: OpenAI-format messages array.
        is_new_session: If True, include full history. If False, only latest user.

    Returns:
        A single prompt string for the A2A message.
    """
    if not is_new_session:
        # Only send the latest user message for follow-up turns
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return _extract_text_content(msg)
        return ""

    # For new sessions, concatenate full history
    parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = _extract_text_content(msg)

        if not content:
            continue

        if role == "system":
            parts.append(f"[System Instructions]\n{content}\n")
        elif role == "user":
            parts.append(f"[User]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}\n")
        elif role == "tool":
            # Include tool results as context
            tool_name = msg.get("name", "tool")
            parts.append(f"[Tool Result: {tool_name}]\n{content}\n")

    return "\n".join(parts)


def _extract_text_content(msg: Dict[str, Any]) -> str:
    """Extract text content from an OpenAI message."""
    content = msg.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # Skip image_url and other non-text content
        return " ".join(text_parts)

    return str(content) if content else ""


# =============================================================================
# A2A → OPENAI TRANSLATION
# =============================================================================


def create_openai_chunk(
    chunk_id: str,
    model: str,
    content: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a single OpenAI streaming chunk.

    Args:
        chunk_id: The completion ID (shared across all chunks).
        model: The model name.
        content: Text content for the delta.
        role: Role for the delta (only set on first chunk).
        finish_reason: Set to "stop" on the final chunk.

    Returns:
        OpenAI chat.completion.chunk format dict.
    """
    delta: Dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content

    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def create_openai_response(
    chunk_id: str,
    model: str,
    content: str,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """
    Create a non-streaming OpenAI completion response.

    Args:
        chunk_id: The completion ID.
        model: The model name.
        content: Full response text.
        finish_reason: Finish reason (default: "stop").

    Returns:
        OpenAI chat.completion format dict.
    """
    return {
        "id": chunk_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def a2a_events_to_openai_stream(
    events: AsyncGenerator[A2AEvent, None],
    model: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Convert a stream of A2A events to OpenAI streaming chunks.

    Filters events to only yield text content and state changes.
    Handles the first chunk (with role) and final chunk (with finish_reason).

    Args:
        events: AsyncGenerator of A2AEvent objects.
        model: Model name for the response.

    Yields:
        OpenAI chat.completion.chunk format dicts.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    first_chunk = True
    has_content = False

    async for event in events:
        if event.kind == "error":
            # Emit the error as content and stop
            yield create_openai_chunk(
                chunk_id=chunk_id,
                model=model,
                content=event.text or "A2A server error",
                role="assistant" if first_chunk else None,
                finish_reason="stop",
            )
            return

        # A2A events carry text in status-update events, not as "text-content"
        if event.text and event.state not in ("failed",):
            # Emit text content
            yield create_openai_chunk(
                chunk_id=chunk_id,
                model=model,
                content=event.text,
                role="assistant" if first_chunk else None,
            )
            first_chunk = False
            has_content = True

        if event.kind == "thought" and event.text:
            # Optionally include thought text as content
            # (some clients may want to see the agent's reasoning)
            pass  # Skip thoughts for now; they're noisy

        if event.is_final or event.state in ("completed", "input-required"):
            # Emit final chunk with finish_reason
            if not has_content:
                # No content was emitted — send a single empty-content response
                yield create_openai_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    content="",
                    role="assistant",
                    finish_reason="stop",
                )
            else:
                yield create_openai_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    finish_reason="stop",
                )
            return

    # Stream ended without a final event — emit stop
    if has_content:
        yield create_openai_chunk(
            chunk_id=chunk_id,
            model=model,
            finish_reason="stop",
        )
    else:
        yield create_openai_chunk(
            chunk_id=chunk_id,
            model=model,
            content="",
            role="assistant",
            finish_reason="stop",
        )
