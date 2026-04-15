# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
GitHub Copilot Provider - Custom API integration for Copilot Chat.

This provider implements the full Copilot Chat API integration including:
- Device Flow OAuth authentication (via CopilotAuthBase)
- Direct API calls to Copilot's OpenAI-compatible chat/completions endpoint
- Dynamic base URL from token's proxy-ep field
- X-Initiator header control (user vs agent mode, from pi-mono)
- Vision request support
- Both streaming and non-streaming responses
- Model policy enabling after Device Flow login

Based on:
- https://github.com/sst/opencode-copilot-auth
- https://github.com/badlogic/pi-mono (packages/ai/src/providers/github-copilot-headers.ts)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

import httpx
import litellm

from .provider_interface import ProviderInterface
from .copilot_auth_base import CopilotAuthBase, COPILOT_HEADERS
from .copilot_plan_mapping import (
    fetch_plan_mapping,
    filter_models_for_plan,
    get_plan_for_sku,
)
from .utilities.copilot_quota_tracker import (
    CopilotQuotaTracker,
    COPILOT_USER_URL,
)

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# DEFAULT COPILOT MODELS
# =============================================================================

# Default model list advertised to clients when the plan mapping is
# unavailable (scrape failed, no cache).  Only include models that are
# confirmed to exist on the Copilot API — speculative/future model IDs
# will 404 and produce client-facing errors.
#
# Last validated against the live plan cache and litellm model registry.
# When adding new entries, verify the model ID against the Copilot API
# (check .copilot_plan_cache.json after a fresh scrape).
DEFAULT_COPILOT_MODELS = [
    # OpenAI models
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    # Anthropic models
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
    # Google models
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash",
    "gemini-3.1-pro",
    # xAI models
    "grok-code-fast-1",
    # Other models
    "raptor-mini",
    "goldeneye",
]


# =============================================================================
# COPILOT DYNAMIC HEADERS
# =============================================================================


def _infer_copilot_initiator(messages: List[Dict[str, Any]]) -> str:
    """
    Determine the X-Initiator header value based on message patterns.

    Extended from pi-mono's simple last-role check to also detect:
    - Tool results sent as role="user" with a tool_call_id field
    - Agent tool-call loops (assistant with tool_calls followed by results)

    All new paths only add "agent" classifications (quota-saving direction),
    never reclassify genuine user messages — so ban risk is unchanged.

    See docs/copilot-initiator-problem.md for full analysis.
    """
    if not messages:
        return "user"

    last = messages[-1]

    # Tool result disguised as role="user" — some clients do this
    if last.get("tool_call_id"):
        return "agent"

    # Previous assistant made tool calls → this is the loop continuation
    if len(messages) >= 2:
        prev = messages[-2]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            return "agent"

    # Non-user last message = agent continuation (original pi-mono logic)
    if last.get("role") != "user":
        return "agent"

    return "user"


def _has_copilot_vision_input(messages: List[Dict[str, Any]]) -> bool:
    """Check if request contains vision/image content."""
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in [
                    "image_url",
                    "input_image",
                ]:
                    return True
    return False


# =============================================================================
# MAIN PROVIDER CLASS
# =============================================================================


class CopilotProvider(CopilotAuthBase, CopilotQuotaTracker, ProviderInterface):
    """
    GitHub Copilot provider with custom API integration.

    Features:
    - Device Flow OAuth authentication
    - Direct Copilot Chat API calls (OpenAI-compatible endpoint)
    - Dynamic base URL from token's proxy-ep field
    - X-Initiator header (simple logic from pi-mono)
    - Vision request support
    - Both streaming and non-streaming responses
    - Plan-based model filtering (copilot_plan_mapping)

    Environment Variables:
    - COPILOT_1_GITHUB_TOKEN: GitHub OAuth token for first credential
    - COPILOT_2_GITHUB_TOKEN: GitHub OAuth token for second credential
    - COPILOT_GITHUB_TOKEN: Legacy single-credential format
    - COPILOT_MODELS: Comma-separated list of available models (optional)
    """

    # Provider identification for env var overrides and quota display
    provider_env_name: str = "copilot"

    skip_cost_calculation = True  # Copilot uses subscription, not token billing

    # Quota groups: models that share rate limits
    # Copilot doesn't expose a quota API, but groups help the TUI display
    # and enable fair-cycle rotation across related models.
    # premium_interactions maps to the quota bucket from /copilot_internal/user
    model_quota_groups = {
        "premium_interactions": [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5.1-codex-mini",
            "gpt-5.2",
            "gpt-5.2-codex",
            "gpt-5.3-codex",
            "gpt-5.4",
            "gpt-5.4-mini",
            "claude-sonnet-4",
            "claude-sonnet-4.5",
            "claude-sonnet-4.6",
            "claude-haiku-4.5",
            "claude-opus-4.5",
            "claude-opus-4.6",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-pro",
            "grok-code-fast-1",
            "raptor-mini",
            "goldeneye",
        ],
    }

    def __init__(self):
        super().__init__()
        self._init_quota_tracker()

        # Model configuration
        models_env = os.getenv("COPILOT_MODELS", "")
        if models_env:
            self._available_models = [
                m.strip() for m in models_env.split(",") if m.strip()
            ]
        else:
            self._available_models = DEFAULT_COPILOT_MODELS

        # Plan mapping (populated on first get_models call)
        self._plan_mapping: Dict[str, set] = {}
        self._plan_mapping_fetched = False

        lib_logger.debug(
            f"CopilotProvider initialized with {len(self._available_models)} models"
        )

    # =========================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # =========================================================================

    def has_custom_logic(self) -> bool:
        """Returns True - Copilot uses custom API calls, not LiteLLM."""
        return True

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Load all Copilot credentials at startup to populate the cache
        with SKU info needed for plan-based model filtering.

        Also fetches initial quota baselines from /copilot_internal/user
        so the TUI shows quota data from first startup.

        Called once by BackgroundRefresher before the main refresh loop.
        """
        for path in credential_paths:
            try:
                await self._load_credentials(path)
                lib_logger.debug(
                    f"Copilot credential loaded at startup: {Path(path).name}"
                )
            except Exception as e:
                lib_logger.warning(
                    f"Failed to load Copilot credential '{path}' at startup: {e}"
                )

        # Log discovered plan tiers
        plans_found = set()
        for cred_path, creds in self._credentials_cache.items():
            sku = creds.get("_proxy_metadata", {}).get("sku", "")
            if sku:
                plan = get_plan_for_sku(sku)
                if plan:
                    plans_found.add(plan)

        if plans_found:
            lib_logger.info(
                f"Copilot plan tiers discovered: {', '.join(sorted(plans_found))}"
            )
        else:
            lib_logger.info(
                "Copilot: no plan SKU info found in credentials "
                "(model filtering disabled, all models will be shown)"
            )

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Returns the plan tier name for a Copilot credential based on its SKU.

        Used for startup summary display (e.g., 'pro', 'business', 'free').
        """
        # Check cache first
        creds = self._credentials_cache.get(credential)
        if creds:
            sku = creds.get("_proxy_metadata", {}).get("sku", "")
            if sku:
                return get_plan_for_sku(sku)

        # Try lazy-loading from file (for credentials not yet in cache)
        if not credential.startswith("env://"):
            try:
                with open(credential, "r") as f:
                    data = json.load(f)
                sku = data.get("_proxy_metadata", {}).get("sku", "")
                if sku:
                    return get_plan_for_sku(sku)
            except Exception:
                pass

        return None

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return available Copilot models, filtered by plan tier.

        On first call, fetches the plan/model mapping from GitHub docs
        (cached for 24h). Then filters the default model list based on
        the union of plans across all credentials.
        """
        # Fetch plan mapping on first call
        if not self._plan_mapping_fetched:
            self._plan_mapping = await fetch_plan_mapping()
            self._plan_mapping_fetched = True

        # Ensure the passed credential is loaded into cache
        if api_key and api_key not in self._credentials_cache:
            try:
                await self._load_credentials(api_key)
            except Exception as e:
                lib_logger.debug(
                    f"Could not load copilot credential for model listing: {e}"
                )

        # Collect all unique plans across credentials
        plans: Set[str] = set()
        for cred_path in self._credentials_cache:
            creds = self._credentials_cache[cred_path]
            sku = creds.get("_proxy_metadata", {}).get("sku", "")
            if sku:
                plan = get_plan_for_sku(sku)
                if plan:
                    plans.add(plan)

        # Filter models: include if accessible under ANY credential's plan
        if plans and self._plan_mapping:
            filtered = set()
            for plan in plans:
                plan_models = filter_models_for_plan(
                    self._available_models, self._plan_mapping, plan
                )
                filtered.update(plan_models)
            # Preserve original ordering
            result_models = [m for m in self._available_models if m in filtered]
        else:
            # No plan info or mapping unavailable — return all models
            result_models = self._available_models

        return [f"copilot/{m}" for m in result_models]

    def get_credential_priority(self, credential: str) -> Optional[int]:
        """All Copilot credentials are treated equally."""
        return 1

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """Copilot doesn't restrict by tier."""
        return None

    # =========================================================================
    # API COMPLETION
    # =========================================================================

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion requests to Copilot API.

        This method:
        1. Gets fresh Copilot API token
        2. Resolves base URL from token's proxy-ep field
        3. Builds request with proper headers (X-Initiator, Vision, Copilot headers)
        4. Makes direct API call to Copilot's OpenAI-compatible endpoint
        5. Parses response into LiteLLM format
        """
        credential_path = kwargs.pop("credential_identifier", "")
        model = kwargs.get("model", "gpt-4o")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        # Remove internal context before processing
        kwargs.pop("transaction_context", None)
        kwargs.pop("_anthropic_payload", None)

        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/")[-1]

        # Get fresh credentials and token
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_copilot_token(credential_path, creds)

        access_token = creds.get("access_token", "")
        base_url = creds.get(
            "copilot_base_url",
            "https://api.individual.githubcopilot.com",
        )

        # Determine dynamic headers
        initiator = _infer_copilot_initiator(messages)
        is_vision = _has_copilot_vision_input(messages)

        headers = {
            **COPILOT_HEADERS,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Openai-Intent": "conversation-edits",
            "X-Initiator": initiator,
        }

        if is_vision:
            headers["Copilot-Vision-Request"] = "true"

        # Build request body (OpenAI-compatible format)
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Add optional parameters
        for key in ("temperature", "max_tokens", "top_p", "stop",
                     "tools", "tool_choice", "response_format",
                     "presence_penalty", "frequency_penalty",
                     "n", "seed"):
            if kwargs.get(key) is not None:
                body[key] = kwargs[key]

        lib_logger.debug(
            f"Copilot request: model={model}, initiator={initiator}, "
            f"stream={stream}, vision={is_vision}"
        )

        if stream:
            return self._handle_streaming_response(
                client, base_url, headers, body, model, credential_path
            )
        else:
            return await self._handle_non_streaming_response(
                client, base_url, headers, body, model, credential_path
            )

    # =========================================================================
    # RATE LIMIT HANDLING
    # =========================================================================

    def _parse_rate_limit_headers(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Parse rate limit info from Copilot API response headers.

        Copilot uses standard x-ratelimit-* headers when rate limiting.
        Returns parsed info or None if no rate limit headers present.
        """
        remaining = headers.get("x-ratelimit-remaining")
        limit = headers.get("x-ratelimit-limit")
        reset = headers.get("x-ratelimit-reset")
        retry_after = headers.get("retry-after")

        if not any([remaining, limit, reset, retry_after]):
            return None

        result = {}
        if remaining is not None:
            try:
                result["remaining"] = int(remaining)
            except (ValueError, TypeError):
                pass
        if limit is not None:
            try:
                result["limit"] = int(limit)
            except (ValueError, TypeError):
                pass
        if reset is not None:
            try:
                result["reset_at"] = int(reset)
            except (ValueError, TypeError):
                pass
        if retry_after is not None:
            try:
                result["retry_after_seconds"] = int(retry_after)
            except (ValueError, TypeError):
                try:
                    # Retry-After can be an HTTP date
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(retry_after)
                    result["retry_after_seconds"] = int(dt.timestamp() - time.time())
                except Exception:
                    pass

        return result if result else None

    async def _handle_rate_limit_response(
        self,
        status_code: int,
        headers: Dict[str, str],
        credential_path: str,
        model: str,
    ) -> None:
        """
        Handle a 429 rate limit response by pushing info to the UsageManager.

        This ensures the TUI quota display reflects rate-limited credentials
        and applies cooldown so the credential is skipped until reset.
        """
        if status_code != 429:
            return

        rl_info = self._parse_rate_limit_headers(headers)
        retry_seconds = 60  # Default 1 minute cooldown

        if rl_info:
            retry_seconds = rl_info.get("retry_after_seconds", 60)
            if retry_seconds <= 0:
                retry_seconds = 60

            reset_at = rl_info.get("reset_at")
            if reset_at and reset_at > time.time():
                retry_seconds = max(retry_seconds, int(reset_at - time.time()))

            lib_logger.info(
                f"Copilot rate limited for {model}: "
                f"remaining={rl_info.get('remaining', '?')}, "
                f"limit={rl_info.get('limit', '?')}, "
                f"retry_after={retry_seconds}s"
            )
        else:
            lib_logger.warning(
                f"Copilot rate limited for {model} (no rate limit headers), "
                f"applying default {retry_seconds}s cooldown"
            )

        # Determine quota group for this model
        clean_model = model.split("/")[-1] if "/" in model else model
        quota_group = self._find_model_quota_group(clean_model) or clean_model

        # Apply cooldown via UsageManager if available
        if self._usage_manager:
            try:
                await self._usage_manager.apply_cooldown(
                    accessor=credential_path,
                    duration=retry_seconds,
                    reason="rate_limited",
                    model_or_group=quota_group,
                )
            except Exception as e:
                lib_logger.debug(f"Failed to apply cooldown via UsageManager: {e}")

    # =========================================================================
    # STREAMING / NON-STREAMING HANDLERS
    # =========================================================================

    async def _handle_non_streaming_response(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model: str,
        credential_path: str = "",
    ) -> litellm.ModelResponse:
        """Handle non-streaming Copilot API response."""
        url = f"{base_url}/chat/completions"

        try:
            response = await client.post(
                url,
                headers=headers,
                json=body,
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            return self._convert_to_litellm_response(data, model)

        except httpx.HTTPStatusError as e:
            # Handle rate limiting
            if e.response.status_code == 429:
                await self._handle_rate_limit_response(
                    e.response.status_code,
                    dict(e.response.headers),
                    credential_path,
                    model,
                )
            lib_logger.error(
                f"Copilot API error (HTTP {e.response.status_code}): "
                f"{e.response.text}"
            )
            raise
        except Exception as e:
            lib_logger.error(f"Copilot request failed: {e}")
            raise

    async def _handle_streaming_response(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model: str,
        credential_path: str = "",
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming Copilot API response."""
        url = f"{base_url}/chat/completions"

        try:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
                timeout=300.0,
            ) as response:
                # Must read the body before raise_for_status() so that
                # e.response.text is populated on error.  Streaming
                # responses are not consumed until iterated, so without
                # this the error body would be empty.
                if response.status_code >= 400:
                    await response.aread()
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        yield self._convert_to_litellm_chunk(
                            chunk_data, model
                        )
                    except json.JSONDecodeError:
                        continue

        except httpx.HTTPStatusError as e:
            # Handle rate limiting
            if e.response.status_code == 429:
                await self._handle_rate_limit_response(
                    e.response.status_code,
                    dict(e.response.headers),
                    credential_path,
                    model,
                )
            lib_logger.error(
                f"Copilot streaming error (HTTP {e.response.status_code}): "
                f"{e.response.text}"
            )
            raise
        except Exception as e:
            lib_logger.error(f"Copilot streaming failed: {e}")
            raise

    # =========================================================================
    # LITELLM FORMAT CONVERSION
    # =========================================================================

    def _convert_to_litellm_response(
        self, data: Dict[str, Any], model: str
    ) -> litellm.ModelResponse:
        """Convert Copilot API response to LiteLLM ModelResponse format."""
        choices = []
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            litellm_choice = litellm.Choices(
                index=choice.get("index", 0),
                message=litellm.Message(
                    role=message.get("role", "assistant"),
                    content=message.get("content", ""),
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )

            # Handle tool calls
            if message.get("tool_calls"):
                litellm_choice.message.tool_calls = message["tool_calls"]

            choices.append(litellm_choice)

        usage = data.get("usage", {})
        return litellm.ModelResponse(
            id=data.get("id", f"copilot-{uuid.uuid4()}"),
            choices=choices,
            created=data.get("created", int(time.time())),
            model=f"copilot/{model}",
            usage=litellm.Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    def _convert_to_litellm_chunk(
        self, chunk_data: Dict[str, Any], model: str
    ) -> litellm.ModelResponse:
        """Convert Copilot streaming chunk to LiteLLM format."""
        choices = []
        for choice in chunk_data.get("choices", []):
            delta = choice.get("delta", {})
            delta_dict = {
                "role": delta.get("role"),
                "content": delta.get("content"),
            }
            if delta.get("tool_calls"):
                delta_dict["tool_calls"] = delta["tool_calls"]

            choices.append({
                "index": choice.get("index", 0),
                "delta": delta_dict,
                "finish_reason": choice.get("finish_reason"),
            })

        return litellm.ModelResponse(
            id=chunk_data.get("id", f"copilot-{uuid.uuid4()}"),
            choices=choices,
            created=chunk_data.get("created", int(time.time())),
            model=f"copilot/{model}",
            object="chat.completion.chunk",
        )

    # =========================================================================
    # EMBEDDINGS (NOT SUPPORTED)
    # =========================================================================

    async def aembedding(
        self, client: httpx.AsyncClient, **kwargs
    ) -> litellm.EmbeddingResponse:
        """Copilot doesn't support embeddings API."""
        raise NotImplementedError("Copilot does not support embeddings API")
