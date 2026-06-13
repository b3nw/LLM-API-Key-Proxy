# SPDX-License-Identifier: LGPL-3.0-only

# src/rotator_library/providers/xai_provider.py
"""
xAI Grok Provider

Provider for xAI Grok models via OAuth2 authentication (SuperGrok / X Premium+).
Routes requests through LiteLLM's built-in xAI support (`xai/` prefix).

xAI's API is OpenAI-compatible, so LiteLLM handles the routing natively.
This provider's role is to:
  1. Load OAuth credentials from credential files
  2. Inject the OAuth access_token as api_key into LiteLLM requests
  3. Discover available models from the xAI /v1/models endpoint
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, List, Union

import httpx
import litellm
import openai

from .provider_interface import ProviderInterface
from .xai_auth_base import XaiAuthBase
from ..model_definitions import ModelDefinitions
from ..error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION
# =============================================================================

XAI_API_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")

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


class XaiProvider(XaiAuthBase, ProviderInterface):
    """
    Provider for xAI Grok models using OAuth2 credentials.

    Authentication:
      - OAuth credentials stored as JSON files (via XaiAuthBase PKCE/Device flow)
      - Access token injected as Bearer auth for the OpenAI-compatible API

    Model routing:
      - Uses LiteLLM's built-in `xai/` prefix for native routing
      - Model discovery from https://api.x.ai/v1/models
    """

    provider_env_name = "xai"

    def __init__(self):
        super().__init__()
        self.api_base = XAI_API_BASE
        self.model_definitions = ModelDefinitions()
        lib_logger.debug(
            f"XaiProvider initialized: base={self.api_base}"
        )

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return the list of available xAI models.

        Discovery order:
        1. Static override from environment / model_definitions
        2. Live fetch from xAI /v1/models endpoint
        3. Hardcoded fallback
        """
        # 1. Check static model definitions first
        static_models = self.model_definitions.get_all_provider_models("xai")
        if static_models:
            return static_models

        # 2. Try live model discovery
        try:
            # For OAuth credentials, api_key is a file path; resolve it
            auth_header = await self.get_auth_header(api_key)
            token = auth_header.get("Authorization", "").replace("Bearer ", "")

            response = await client.get(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()

            discovered = [
                f"xai/{m['id']}"
                for m in data.get("data", [])
                if m.get("id")
            ]
            if discovered:
                lib_logger.info(
                    f"Discovered {len(discovered)} models from xAI API: "
                    f"{', '.join(discovered[:5])}"
                    + (f"... (+{len(discovered)-5} more)" if len(discovered) > 5 else "")
                )
                return discovered

        except Exception as e:
            lib_logger.warning(f"Failed to fetch xAI models: {e}")

        # 3. Graceful fallback
        return [
            "xai/grok-3",
            "xai/grok-3-mini",
        ]

    def has_custom_logic(self) -> bool:
        """
        xAI requires custom logic to inject OAuth bearer token.

        The standard LiteLLM flow sets api_key = credential_path (file path),
        which won't work for OAuth providers. We override acompletion to
        resolve the credential file into an actual token.
        """
        return True

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Make a chat completion request to xAI via LiteLLM.

        Resolves the OAuth credential file path into a bearer token and
        routes through LiteLLM's native xAI support.
        """
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        # Resolve OAuth credential to access token
        auth_header = await self.get_auth_header(credential)
        token = auth_header.get("Authorization", "").replace("Bearer ", "")

        if not token:
            raise ValueError(
                f"Failed to resolve xAI OAuth token from credential: "
                f"{mask_credential(credential)}"
            )

        # Route through LiteLLM as xai/model
        kwargs["model"] = f"xai/{model_bare}"
        kwargs["api_key"] = token
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "xai"

        # Set up async OpenAI client for LiteLLM
        kwargs["client"] = openai.AsyncOpenAI(
            api_key=token,
            base_url=self.api_base,
            http_client=client,
        )

        # Strip unsupported params
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

        # Resolve OAuth credential to access token
        auth_header = await self.get_auth_header(credential)
        token = auth_header.get("Authorization", "").replace("Bearer ", "")

        if not token:
            raise ValueError(
                f"Failed to resolve xAI OAuth token for embedding: "
                f"{mask_credential(credential)}"
            )

        kwargs["model"] = f"xai/{model_bare}"
        kwargs["api_key"] = token
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "xai"

        kwargs["client"] = openai.AsyncOpenAI(
            api_key=token,
            base_url=self.api_base,
            http_client=client,
        )

        return await litellm.aembedding(**kwargs)
