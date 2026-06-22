# SPDX-License-Identifier: LGPL-3.0-only
# Retired A2A session manager (archived — see _retired/README.md)
# Original: Session manager for mapping OpenAI conversations to A2A task sessions.

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

lib_logger = logging.getLogger("rotator_library")


@dataclass
class A2ASession:
    """Represents an active A2A conversation session."""

    context_id: str
    task_id: Optional[str] = None  # Set after first response from server
    fingerprint: str = ""
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    message_count: int = 0

    def touch(self):
        """Update last_used_at timestamp and increment message count."""
        self.last_used_at = time.time()
        self.message_count += 1


class A2ASessionManager:
    """
    Maps OpenAI conversations to A2A contextId/taskId pairs.

    Uses a conversation fingerprint (hash of first user message) to identify
    returning conversations. Sessions are invalidated on credential rotation
    (server restart) since the A2A server loses all in-memory task state.

    Session lifecycle:
        1. First request with new fingerprint → new context_id, no task_id
        2. A2A server responds with task_id → stored in session
        3. Follow-up requests → reuse context_id and task_id
        4. Credential rotation → invalidate_all() clears everything
        5. TTL expiry → cleaned up periodically
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Args:
            ttl_seconds: Time-to-live for sessions in seconds (default: 1 hour).
        """
        self._sessions: Dict[str, A2ASession] = {}
        self._ttl = ttl_seconds

    def generate_fingerprint(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a conversation fingerprint from the OpenAI messages array.

        Uses SHA256 of the first user message content to create a deterministic
        identifier. This matches the approach in GeminiCliProvider._generate_stable_session_id().

        Args:
            messages: OpenAI-format messages array.

        Returns:
            Hex digest string, or a random UUID if no user message found.
        """
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multi-part content - extract text parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = " ".join(text_parts)

                if content:
                    return hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Fallback to random UUID if no user message found
        return str(uuid.uuid4())

    def get_or_create(self, fingerprint: str) -> A2ASession:
        """
        Look up an existing session by fingerprint, or create a new one.

        Args:
            fingerprint: Conversation fingerprint from generate_fingerprint().

        Returns:
            Existing or newly created A2ASession.
        """
        # Clean up expired sessions periodically
        self._cleanup_expired()

        session = self._sessions.get(fingerprint)
        if session:
            session.touch()
            lib_logger.debug(
                f"[A2A Session] Resuming session ctx={session.context_id[:8]}, "
                f"task={session.task_id[:8] if session.task_id else 'pending'}, "
                f"msgs={session.message_count}"
            )
            return session

        # Create new session
        context_id = str(uuid.uuid4())
        session = A2ASession(
            context_id=context_id,
            fingerprint=fingerprint,
        )
        self._sessions[fingerprint] = session
        lib_logger.debug(
            f"[A2A Session] Created new session ctx={context_id[:8]}"
        )
        return session

    def update_task_id(self, fingerprint: str, task_id: str):
        """
        Update the task_id for a session after receiving first response.

        Args:
            fingerprint: Conversation fingerprint.
            task_id: Task ID from A2A server response.
        """
        session = self._sessions.get(fingerprint)
        if session and not session.task_id:
            session.task_id = task_id
            lib_logger.debug(
                f"[A2A Session] Set task_id={task_id[:8]} for ctx={session.context_id[:8]}"
            )

    def invalidate_all(self):
        """
        Invalidate all sessions.

        Called on credential rotation when the A2A server restarts
        and loses all in-memory task state.
        """
        count = len(self._sessions)
        self._sessions.clear()
        if count > 0:
            lib_logger.info(
                f"[A2A Session] Invalidated {count} sessions (credential rotation)"
            )

    def _cleanup_expired(self):
        """Remove sessions older than TTL."""
        now = time.time()
        expired = [
            fp
            for fp, session in self._sessions.items()
            if (now - session.last_used_at) > self._ttl
        ]
        for fp in expired:
            del self._sessions[fp]

        if expired:
            lib_logger.debug(f"[A2A Session] Cleaned up {len(expired)} expired sessions")

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)
