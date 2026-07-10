# SPDX-License-Identifier: LGPL-3.0-only

# src/rotator_library/providers/xai_provider.py
"""
xAI Grok Provider

Provider for xAI Grok models via OAuth2 authentication (SuperGrok / X Premium+).
Routes all requests through the CLI chat proxy at cli-chat-proxy.grok.com/v1,
which supports every chat-capable model and ensures traffic is attributed to the
Grok Build / subscription billing track.

Model discovery merges results from both the standard API (api.x.ai) and the
CLI proxy to build the complete list.  Non-chat models (image generation,
video generation, multi-agent) are excluded since they are not usable via
the /chat/completions endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, List, Optional, Union

import httpx
import litellm
import openai

from .provider_interface import ProviderInterface
from .x_ai_auth_base import XAiAuthBase
from .utilities.x_ai_quota_tracker import XAiQuotaTracker
from ..model_definitions import ModelDefinitions
from ..error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION
# =============================================================================

XAI_API_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
XAI_CLI_PROXY_BASE = os.getenv(
    "XAI_CLI_PROXY_BASE", "https://cli-chat-proxy.grok.com/v1"
)

# Minimum CLI version the proxy accepts (426 Upgrade Required otherwise)
XAI_CLI_VERSION = os.getenv("XAI_CLI_VERSION", "0.1.202")

# Models that are not usable via /chat/completions (image gen, video gen,
# multi-agent orchestration).  Excluded from the advertised model list.
_NON_CHAT_MODEL_PREFIXES = (
    "grok-imagine-",
    "grok-4.20-multi-agent",
)

# Params accepted by litellm.acompletion for xAI (OpenAI-compatible)
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "extra_headers",
    "extra_body",
    "api_key",
    "api_base",
    "custom_llm_provider",
    "client",
}


def _is_chat_model(model_id: str) -> bool:
    """Return False for image/video/multi-agent models."""
    return not any(model_id.startswith(p) for p in _NON_CHAT_MODEL_PREFIXES)


class XAiProvider(XAiAuthBase, XAiQuotaTracker, ProviderInterface):
    """
    Provider for xAI Grok models using OAuth2 credentials.

    Authentication:
      - OAuth credentials stored as JSON files (via XaiAuthBase PKCE/Device flow)
      - Access token injected as Bearer auth for the OpenAI-compatible API

    Model routing:
      - All chat completions route through cli-chat-proxy.grok.com
      - CLI proxy headers (User-Agent, x-xai-token-auth, x-grok-client-version)
        are injected on every request
      - Model discovery merges both api.x.ai and CLI proxy, filtered to chat models
    """

    provider_env_name = "x-ai"

    model_quota_groups = {
        "monthly-limit": ["_billing_monthly"],
        "on-demand($)": ["_billing_ondemand"],
    }

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """Map chat models to monthly billing pool; virtual buckets for display."""
        clean = model.split("/")[-1] if "/" in model else model
        if clean == "_billing_ondemand":
            return "on-demand($)"
        if clean == "_billing_monthly":
            return "monthly-limit"
        return "monthly-limit"

    def __init__(self):
        super().__init__()
        self._init_quota_tracker()
        self.api_base = XAI_API_BASE
        self.cli_proxy_base = XAI_CLI_PROXY_BASE
        self._cli_version = XAI_CLI_VERSION
        self.model_definitions = ModelDefinitions()
        # Context window metadata from CLI proxy discovery
        # Maps bare model id -> context_window (e.g. {"grok-build": 512000})
        self._cli_proxy_metadata: dict = {}
        lib_logger.debug(
            f"XAiProvider initialized: cli_proxy={self.cli_proxy_base}"
        )

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return the list of available xAI chat models.

        Discovery order:
        1. Static override from environment / model_definitions
        2. Live fetch from both xAI standard API and CLI proxy, merged
        3. Hardcoded fallback

        Non-chat models (image gen, video gen, multi-agent) are excluded.
        """
        static_models = self.model_definitions.get_all_provider_models("x-ai")
        if static_models:
            return static_models

        try:
            auth_header = await self.get_auth_header(api_key)
            token = auth_header.get("Authorization", "").replace("Bearer ", "")
        except Exception as e:
            lib_logger.warning(f"Failed to resolve xAI OAuth token for model discovery: {e}")
            return ["x-ai/grok-3", "x-ai/grok-3-mini"]

        all_ids: set = set()

        # Fetch from standard API (api.x.ai) — broadest catalog
        try:
            response = await client.get(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            standard_ids = {
                m["id"] for m in data.get("data", []) if m.get("id")
            }
            all_ids.update(standard_ids)
            if standard_ids:
                lib_logger.info(
                    f"Discovered {len(standard_ids)} models from xAI standard API"
                )
        except Exception as e:
            lib_logger.warning(f"Failed to fetch xAI standard API models: {e}")

        # Fetch from CLI proxy — may have additional models + context metadata
        try:
            response = await client.get(
                f"{self.cli_proxy_base}/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            cli_count = 0
            for m in data.get("data", []):
                mid = m.get("id")
                if not mid:
                    continue
                all_ids.add(mid)
                cli_count += 1
                ctx = m.get("context_window")
                if ctx:
                    self._cli_proxy_metadata[mid] = int(ctx)
            if cli_count:
                lib_logger.info(
                    f"Discovered {cli_count} models from xAI CLI proxy"
                )
        except Exception as e:
            lib_logger.warning(f"Failed to fetch xAI CLI proxy models: {e}")

        # Filter to chat-capable models only
        chat_ids = {mid for mid in all_ids if _is_chat_model(mid)}
        excluded = all_ids - chat_ids
        if excluded:
            lib_logger.debug(
                f"xAI: excluded {len(excluded)} non-chat models: "
                f"{', '.join(sorted(excluded))}"
            )

        if chat_ids:
            return sorted(f"x-ai/{mid}" for mid in chat_ids)

        return ["x-ai/grok-3", "x-ai/grok-3-mini"]

    def get_model_context_overrides(self) -> dict:
        """
        Return context window overrides for xAI models discovered from
        the CLI proxy that don't have catalog metadata.

        Returns:
            Dict mapping full model ID (e.g. "x-ai/grok-build") to
            context_window size in tokens.
        """
        return {
            f"x-ai/{mid}": ctx
            for mid, ctx in self._cli_proxy_metadata.items()
        }

    def has_custom_logic(self) -> bool:
        """
        xAI requires custom logic to inject OAuth bearer token.

        The standard LiteLLM flow sets api_key = credential_path (file path),
        which won't work for OAuth providers. We override acompletion to
        resolve the credential file into an actual token.
        """
        return True

    def _get_cli_proxy_headers(self) -> dict:
        """Return headers for all CLI proxy requests.

        These headers identify the client as a Grok CLI session, which:
        1. Satisfies the CLI proxy's transport requirements
        2. Ensures xAI attributes traffic to the subscription billing track
           (not the pay-per-token API track)
        """
        ver = self._cli_version
        return {
            "User-Agent": f"grok/{ver}",
            "x-xai-token-auth": "xai-grok-cli",
            "x-grok-client-version": ver,
        }

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Make a chat completion request to xAI via LiteLLM.

        All requests route through cli-chat-proxy.grok.com with CLI headers.
        """
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        auth_header = await self.get_auth_header(credential)
        token = auth_header.get("Authorization", "").replace("Bearer ", "")

        if not token:
            raise ValueError(
                f"Failed to resolve xAI OAuth token from credential: "
                f"{mask_credential(credential)}"
            )

        kwargs["model"] = f"xai/{model_bare}"
        kwargs["api_key"] = token
        kwargs["api_base"] = self.cli_proxy_base
        kwargs["custom_llm_provider"] = "xai"

        existing_headers = kwargs.get("extra_headers") or {}
        kwargs["extra_headers"] = {**existing_headers, **self._get_cli_proxy_headers()}

        kwargs["client"] = openai.AsyncOpenAI(
            api_key=token,
            base_url=self.cli_proxy_base,
            http_client=client,
        )

        unsupported = set(kwargs.keys()) - SUPPORTED_PARAMS
        if unsupported:
            lib_logger.debug(
                f"xai: stripping unsupported params for {model}: {unsupported}"
            )
            kwargs = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        return await litellm.acompletion(**kwargs)

    async def aembedding(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> litellm.EmbeddingResponse:
        """
        Make an embedding request to xAI via LiteLLM.
        """
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        auth_header = await self.get_auth_header(credential)
        token = auth_header.get("Authorization", "").replace("Bearer ", "")

        if not token:
            raise ValueError(
                f"Failed to resolve xAI OAuth token for embedding: "
                f"{mask_credential(credential)}"
            )

        kwargs["model"] = f"xai/{model_bare}"
        kwargs["api_key"] = token
        kwargs["api_base"] = self.cli_proxy_base
        kwargs["custom_llm_provider"] = "xai"
        kwargs["extra_headers"] = {
            **(kwargs.get("extra_headers") or {}),
            **self._get_cli_proxy_headers(),
        }

        kwargs["client"] = openai.AsyncOpenAI(
            api_key=token,
            base_url=self.cli_proxy_base,
            http_client=client,
        )

        return await litellm.aembedding(**kwargs)
