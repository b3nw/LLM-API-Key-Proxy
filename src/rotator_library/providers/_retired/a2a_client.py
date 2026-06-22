# SPDX-License-Identifier: LGPL-3.0-only
# Retired A2A JSON-RPC client (archived — see _retired/README.md)
# Original: A2A JSON-RPC client for communicating with the Gemini CLI A2A sidecar server.

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Optional, Any

import httpx

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# EVENT TYPES
# =============================================================================


@dataclass
class A2AEvent:
    """Parsed A2A event from the SSE stream."""

    kind: str  # "text-content", "thought", "state-change", "tool-call-update", etc.
    text: Optional[str] = None  # Extracted text content
    state: Optional[str] = None  # Task state (working, completed, failed, input-required)
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    is_final: bool = False  # Whether this is the final event in the stream
    raw: Dict[str, Any] = field(default_factory=dict)  # Full raw event data


# =============================================================================
# SSE PARSER
# =============================================================================


def _parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single SSE data line into a dict.

    SSE format from A2A server:
        data: {"jsonrpc":"2.0","id":"...","result":{...}}

    Returns None for non-data lines (comments, empty lines, event: lines).
    """
    line = line.strip()
    if not line or line.startswith(":"):
        return None

    if line.startswith("data:"):
        data_str = line[5:].strip()
        if not data_str:
            return None
        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            lib_logger.warning(f"[A2A] Failed to parse SSE data: {e}")
            return None

    return None


def _extract_event(rpc_response: Dict[str, Any]) -> Optional[A2AEvent]:
    """
    Extract a high-level A2AEvent from a JSON-RPC response.

    The JSON-RPC response wraps an A2A TaskStatusUpdateEvent or similar.
    """
    result = rpc_response.get("result")
    if not result:
        # Check for error
        error = rpc_response.get("error")
        if error:
            return A2AEvent(
                kind="error",
                text=error.get("message", "Unknown A2A error"),
                is_final=True,
                raw=rpc_response,
            )
        return None

    kind = result.get("kind", "")
    task_id = result.get("taskId")
    context_id = result.get("contextId")
    is_final = result.get("final", False)

    status = result.get("status", {})
    state = status.get("state")
    message = status.get("message", {})

    # Extract coder agent event kind from metadata
    metadata = message.get("metadata", {})
    coder_agent = metadata.get("coderAgent", {})
    coder_kind = coder_agent.get("kind", "")

    # Extract text from message parts
    text_parts = []
    for part in message.get("parts", []):
        if part.get("kind") == "text":
            text_parts.append(part.get("text", ""))

    text = "".join(text_parts) if text_parts else None

    # Determine the event kind we care about
    event_kind = coder_kind or kind or "unknown"

    return A2AEvent(
        kind=event_kind,
        text=text,
        state=state,
        task_id=task_id,
        context_id=context_id,
        is_final=is_final,
        raw=rpc_response,
    )


# =============================================================================
# A2A CLIENT
# =============================================================================


class A2AClient:
    """
    JSON-RPC client for the Gemini CLI A2A server.

    Communicates with the local A2A sidecar via HTTP on localhost.
    Uses JSON-RPC 2.0 with the message/stream method for streaming responses.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,  # Long timeout for streaming responses
                    write=10.0,
                    pool=10.0,
                ),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_agent_card(self) -> Dict[str, Any]:
        """
        Fetch the agent card from the A2A server.

        Returns:
            Agent card dict with capabilities, skills, etc.
        """
        client = await self._get_client()
        response = await client.get(
            f"{self.base_url}/.well-known/agent-card.json",
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        """
        Check if the A2A server is healthy and responding.

        Returns:
            True if the server responds to the agent card request.
        """
        try:
            await self.get_agent_card()
            return True
        except Exception:
            return False

    async def send_message_stream(
        self,
        text: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        workspace_path: str = "/tmp/workspace",
        auto_execute: bool = True,
    ) -> AsyncGenerator[A2AEvent, None]:
        """
        Send a message via JSON-RPC message/stream and yield parsed events.

        Args:
            text: The user message text to send.
            context_id: Optional context ID for session continuity.
            task_id: Optional task ID to continue an existing task.
            workspace_path: Workspace path for the agent (default: /tmp/workspace).
            auto_execute: Whether tools should auto-execute (YOLO mode).

        Yields:
            A2AEvent objects parsed from the SSE stream.
        """
        message_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Build the message
        message: Dict[str, Any] = {
            "kind": "message",
            "role": "user",
            "messageId": message_id,
            "parts": [{"kind": "text", "text": text}],
        }

        # Add context/task IDs for session continuity
        if context_id:
            message["contextId"] = context_id
        if task_id:
            message["taskId"] = task_id

        # Add agent settings metadata
        message["metadata"] = {
            "coderAgent": {
                "kind": "agent-settings",
                "workspacePath": workspace_path,
                "autoExecute": auto_execute,
            }
        }

        # Build JSON-RPC request
        rpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/stream",
            "params": {
                "message": message,
            },
        }

        client = await self._get_client()

        lib_logger.debug(
            f"[A2A] Sending message/stream request (msgId={message_id[:8]}, "
            f"ctxId={context_id[:8] if context_id else 'new'}, "
            f"taskId={task_id[:8] if task_id else 'new'})"
        )

        async with client.stream(
            "POST",
            self.base_url,
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                lib_logger.error(
                    f"[A2A] message/stream failed: HTTP {response.status_code}: {body.decode()}"
                )
                yield A2AEvent(
                    kind="error",
                    text=f"A2A server returned HTTP {response.status_code}",
                    is_final=True,
                )
                return

            # Parse SSE stream
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    parsed = _parse_sse_line(line)
                    if parsed:
                        event = _extract_event(parsed)
                        if event:
                            yield event
                            if event.is_final:
                                return

            # Handle any remaining data in buffer
            if buffer.strip():
                parsed = _parse_sse_line(buffer)
                if parsed:
                    event = _extract_event(parsed)
                    if event:
                        yield event
