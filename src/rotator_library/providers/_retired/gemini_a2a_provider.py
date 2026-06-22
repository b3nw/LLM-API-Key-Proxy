# SPDX-License-Identifier: LGPL-3.0-only
# Retired Gemini A2A Provider
#
# This module is archived for historical reference only. The Gemini A2A provider
# is no longer registered, discovered, or exposed by the proxy. Do not import or
# rewire this module into active runtime paths unless it is deliberately un-retired.
#
# Original: Routes OpenAI requests through the Gemini CLI A2A sidecar.

import logging
import os
import time
import uuid
from typing import List, Dict, Any, AsyncGenerator, Union, Optional

import httpx
import litellm
from litellm.exceptions import RateLimitError

from ..provider_interface import ProviderInterface, QuotaGroupMap, UsageResetConfigDef
from .a2a_client import A2AClient
from .a2a_session_manager import A2ASessionManager
from .a2a_translator import (
    openai_messages_to_a2a_prompt,
)
from .a2a_sidecar_manager import A2ASidecarManager
from ..utilities.gemini_shared_utils import (
    TIER_PRIORITIES,
    DEFAULT_TIER_PRIORITY,
)
from .utilities.gemini_credential_manager import GeminiCredentialManager
from .gemini_auth_base import GeminiAuthBase

lib_logger = logging.getLogger("rotator_library")

# Models available through the A2A server
# (same as GeminiCliProvider — the A2A server uses the same Code Assist backend)
AVAILABLE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
]


class GeminiA2AProvider(
    GeminiAuthBase,
    GeminiCredentialManager,
    ProviderInterface,
):
    """
    Provider that routes OpenAI-format requests through a Gemini CLI A2A sidecar.

    Architecture:
        1. The A2A server runs as a separate Docker sidecar container
           (or local subprocess for development)
        2. Each request is translated from OpenAI format → A2A JSON-RPC message
        3. The SSE response stream is translated back → OpenAI streaming chunks
        4. On 429 errors, the sidecar is restarted with the next credential

    Key differences from GeminiCliProvider:
        - No direct Code Assist API calls — goes through the A2A server
        - No tool schema transformation — the A2A agent handles tools internally
        - Stateful sessions — conversation fingerprinting for multi-turn support
        - YOLO sandbox mode — agent tools auto-execute in /tmp workspace
    """

    skip_cost_calculation = True

    # LOCKED: Sequential only — one credential active until exhausted
    default_rotation_mode: str = "sequential"

    # Provider name for env var lookups
    provider_env_name: str = "gemini_a2a"

    # Reuse tier configuration from GeminiCliProvider
    tier_priorities = TIER_PRIORITIES
    default_tier_priority: int = DEFAULT_TIER_PRIORITY

    # Usage reset configs (same as GeminiCliProvider)
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=24 * 60 * 60,
            mode="per_model",
            description="24-hour per-model window (all tiers)",
            field_name="models",
        ),
    }

    # Model quota groups (same as GeminiCliProvider)
    model_quota_groups: QuotaGroupMap = {
        "pro": ["gemini-2.5-pro", "gemini-3-pro-preview", "gemini-3.1-pro-preview"],
        "25-flash": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "3-flash": ["gemini-3-flash-preview"],
    }

    # Concurrency multipliers
    default_priority_multipliers = {1: 2, 2: 1}
    default_sequential_fallback_multiplier = 1

    def __init__(self):
        super().__init__()

        # A2A components
        a2a_url = os.getenv("A2A_SERVER_URL", "http://localhost:8080")
        a2a_backend = os.getenv("A2A_BACKEND", "local")
        session_ttl = int(os.getenv("A2A_SESSION_TTL", "3600"))
        workspace_path = os.getenv("A2A_WORKSPACE_PATH", "/tmp/workspace")

        self._a2a_client = A2AClient(base_url=a2a_url)
        self._sidecar = A2ASidecarManager(
            base_url=a2a_url,
            backend=a2a_backend,
        )
        self._sessions = A2ASessionManager(ttl_seconds=session_ttl)
        self._workspace_path = workspace_path

        # Track whether the sidecar has been started
        self._sidecar_initialized = False

        lib_logger.info(
            f"[GeminiA2A] Initialized: backend={a2a_backend}, "
            f"url={a2a_url}, session_ttl={session_ttl}s"
        )

    # =========================================================================
    # PROVIDER INTERFACE METHODS
    # =========================================================================

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        return AVAILABLE_MODELS

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse quota errors from A2A server responses.

        The A2A server forwards Gemini quota errors in its SSE events.
        """
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                try:
                    body = error.response.text
                except Exception:
                    pass
            if not body and hasattr(error, "message"):
                body = str(error.message)
            if not body:
                body = str(error)

        if not body:
            return None

        from ..error_handler import extract_retry_after_from_body

        retry_after = extract_retry_after_from_body(body)
        if retry_after:
            return {
                "retry_after": retry_after,
                "reason": "RATE_LIMIT_EXCEEDED",
                "reset_timestamp": None,
                "quota_reset_timestamp": None,
            }
        return None

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """No model-specific tier restrictions."""
        return None

    # =========================================================================
    # CREDENTIAL INITIALIZATION
    # =========================================================================

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Called at startup to initialize provider with credentials.

        Sets up the sidecar manager with the credential list and
        starts the A2A server (local mode) or verifies it (sidecar mode).
        """
        await super().initialize_credentials(credential_paths)

        if not credential_paths:
            lib_logger.warning("[GeminiA2A] No credentials provided")
            return

        self._sidecar.set_credentials(credential_paths)

        lib_logger.info(
            f"[GeminiA2A] Configured {len(credential_paths)} credentials for rotation"
        )

    async def _ensure_sidecar_started(self):
        """Ensure the sidecar is started (lazy initialization)."""
        if self._sidecar_initialized:
            return

        try:
            started = await self._sidecar.start()
            if started:
                self._sidecar_initialized = True
                lib_logger.info("[GeminiA2A] Sidecar started and healthy")
            else:
                lib_logger.error(
                    "[GeminiA2A] Sidecar failed to start — check A2A server logs"
                )
        except Exception as e:
            lib_logger.error(f"[GeminiA2A] Sidecar start error: {e}")

    # =========================================================================
    # MAIN COMPLETION HANDLER
    # =========================================================================

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle a completion request via the A2A sidecar.

        Flow:
            1. Extract messages, model, stream flag from kwargs
            2. Generate conversation fingerprint → look up/create session
            3. Translate messages → A2A prompt text
            4. Send via A2AClient.send_message_stream()
            5. Translate SSE events → OpenAI streaming chunks
            6. On 429: trigger sidecar rotation, raise RateLimitError
        """
        model = kwargs["model"]
        # credential_identifier is passed by the rotating client
        credential_path = kwargs.pop("credential_identifier", None)
        transaction_context = kwargs.pop("transaction_context", None)
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        # Ensure sidecar is running
        await self._ensure_sidecar_started()

        if not self._sidecar.is_healthy:
            # Try health check
            if not await self._sidecar.health_check():
                raise RuntimeError(
                    "[GeminiA2A] A2A sidecar is not healthy. Check server logs."
                )

        # Session management
        fingerprint = self._sessions.generate_fingerprint(messages)
        session = self._sessions.get_or_create(fingerprint)
        is_new_session = session.message_count <= 1  # First message in this session

        # Translate OpenAI messages to A2A prompt
        prompt_text = openai_messages_to_a2a_prompt(messages, is_new_session=is_new_session)

        if not prompt_text.strip():
            lib_logger.warning("[GeminiA2A] Empty prompt text after translation")
            prompt_text = "(empty message)"

        # Strip model prefix if present
        model_name = model.split("/")[-1].replace(":thinking", "")

        lib_logger.debug(
            f"[GeminiA2A] Request: model={model_name}, "
            f"session={'existing' if not is_new_session else 'new'}, "
            f"prompt_len={len(prompt_text)}"
        )

        async def stream_handler():
            """Inner generator that produces litellm.ModelResponse chunks."""
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            first_chunk = True
            has_content = False

            try:
                async for event in self._a2a_client.send_message_stream(
                    text=prompt_text,
                    context_id=session.context_id,
                    task_id=session.task_id,
                    workspace_path=self._workspace_path,
                    auto_execute=True,
                ):
                    # Update task_id from first response
                    if event.task_id and not session.task_id:
                        self._sessions.update_task_id(fingerprint, event.task_id)

                    # Check for quota errors in the event
                    if event.kind == "error" and event.text:
                        if "429" in event.text or "RESOURCE_EXHAUSTED" in event.text:
                            await self._handle_quota_error(event.text, model)
                            return  # RateLimitError raised above

                    # Handle state changes indicating failure
                    if event.state == "failed":
                        error_text = event.text or "A2A task failed"
                        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                            await self._handle_quota_error(error_text, model)
                            return

                        # Non-quota failure — emit as content
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant" if first_chunk else None,
                                    "content": f"Error: {error_text}",
                                },
                                "finish_reason": "stop",
                            }],
                        }
                        # Clean None role
                        if chunk["choices"][0]["delta"].get("role") is None:
                            del chunk["choices"][0]["delta"]["role"]
                        yield litellm.ModelResponse(**chunk)
                        return

                    # Emit text content
                    # A2A events carry text in status-update events (state=working),
                    # not as a separate "text-content" kind
                    if event.text and event.state not in ("failed",):
                        delta = {"content": event.text}
                        if first_chunk:
                            delta["role"] = "assistant"
                            first_chunk = False
                        has_content = True

                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }],
                        }
                        yield litellm.ModelResponse(**chunk)

                    # Handle final events
                    if event.is_final or event.state in ("completed", "input-required"):
                        final_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 1,
                                "total_tokens": 1,
                            },
                        }

                        if not has_content:
                            final_chunk["choices"][0]["delta"] = {
                                "role": "assistant",
                                "content": "",
                            }

                        yield litellm.ModelResponse(**final_chunk)
                        return

                # Stream ended without final event
                if has_content:
                    yield litellm.ModelResponse(**{
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    })

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    error_body = e.response.text if e.response else ""
                    await self._handle_quota_error(error_body, model)
                raise

        if stream:
            return stream_handler()
        else:
            # Non-streaming: collect all chunks and assemble
            chunks = []
            async for chunk in stream_handler():
                chunks.append(chunk)
            return self._assemble_response(chunks, model_name)

    # =========================================================================
    # HELPERS
    # =========================================================================

    async def _handle_quota_error(self, error_text: str, model: str):
        """
        Handle a quota exhaustion error by rotating credentials and raising.

        This triggers:
            1. Session invalidation (all sessions become stale)
            2. Sidecar credential rotation (restart with next credential)
            3. RateLimitError raised (rotator will retry with next credential)
        """
        lib_logger.info(
            f"[GeminiA2A] Quota error detected, rotating credential. "
            f"Current: {self._sidecar.current_credential}"
        )

        # Invalidate all sessions (server restart loses task state)
        self._sessions.invalidate_all()

        # Rotate to next credential
        try:
            new_cred = await self._sidecar.rotate_credential()
            lib_logger.info(f"[GeminiA2A] Rotated to credential: {new_cred}")
        except Exception as e:
            lib_logger.error(f"[GeminiA2A] Credential rotation failed: {e}")

        # Parse retry_after from the error text
        from ..error_handler import extract_retry_after_from_body

        retry_after = extract_retry_after_from_body(error_text) or 60

        raise RateLimitError(
            message=f"Gemini A2A quota exceeded (rotating credential). {error_text}",
            llm_provider="gemini_a2a",
            model=model,
        )

    def _assemble_response(
        self, chunks: List[litellm.ModelResponse], model: str
    ) -> litellm.ModelResponse:
        """Assemble streaming chunks into a non-streaming response."""
        if not chunks:
            return litellm.ModelResponse(**{
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        # Collect all content from chunks
        content_parts = []
        for chunk in chunks:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].get("delta", {})
                if isinstance(delta, dict) and delta.get("content"):
                    content_parts.append(delta["content"])

        content = "".join(content_parts)

        return litellm.ModelResponse(**{
            "id": chunks[0].id if chunks else f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": chunks[0].created if chunks else int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
