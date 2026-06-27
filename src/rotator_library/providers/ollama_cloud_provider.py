# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Ollama Cloud Provider — routes to ollama.com's OpenAI-compatible API.

Ollama Cloud offers free and paid plans with session + weekly usage limits.
Models are accessed via https://ollama.com/v1/chat/completions with Bearer auth.

Credential config:
- OLLAMA_CLOUD_API_KEY_N: API key for LLM requests
- OLLAMA_CLOUD_SESSION_COOKIE_N: browser session cookie for quota scraping (separate env var)

Quota tracking uses HTML scraping of ollama.com/settings (no public JSON API).
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

import httpx
import litellm
import openai

from .provider_interface import ProviderInterface
from .utilities.ollama_cloud_quota_tracker import OllamaCloudQuotaTracker
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


OLLAMA_CLOUD_API_BASE = "https://ollama.com/v1"
OLLAMA_CLOUD_TAGS_URL = "https://ollama.com/api/tags"


class OllamaCloudProvider(OllamaCloudQuotaTracker, ProviderInterface):
    """
    Provider for Ollama Cloud — free/paid inference via ollama.com.

    OpenAI-compatible endpoint: https://ollama.com/v1/chat/completions
    Native Ollama endpoint: https://ollama.com/api/chat
    """

    provider_env_name = "ollama_cloud"

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
        "extra_headers",
        "extra_body",
        "api_key",
        "api_base",
        "custom_llm_provider",
        "client",
    }

    # Ollama Cloud uses session (~5hr reset) + weekly (7-day reset) quota windows.
    # Each window is its own quota group so the UI shows both independently.
    model_quota_groups = {
        "session": ["_session"],
        "weekly": ["_weekly"],
    }

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """Route synthetic baseline models to their window; real models to session."""
        clean = model.split("/")[-1] if "/" in model else model
        if clean == "_weekly":
            return "weekly"
        return "session"

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Ollama Cloud-specific quota errors.

        Ollama returns 429 when quota is exhausted. The response body may
        contain reset timing information.
        """
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                body = error.response.text
            elif hasattr(error, "body"):
                body = str(error.body) if not isinstance(error.body, str) else error.body
            else:
                body = str(error)

        body_lower = (body or "").lower()

        is_rate_limit = (
            "rate limit" in body_lower or "too many requests" in body_lower
        )
        is_weekly = "weekly" in body_lower
        is_session = "session" in body_lower

        if is_rate_limit or (is_session and "limit" in body_lower) or (is_weekly and "limit" in body_lower):
            retry_after = None
            # Anchor to reset phrasing ("in N days/hours/minutes", "after N ...")
            # to avoid capturing rate descriptions like "100 requests per hour".
            reset_match = re.search(
                r"(?:in|after)\s+(\d+)\s*(day|hour|minute|min)",
                body_lower,
            )
            if reset_match:
                value = int(reset_match.group(1))
                unit = reset_match.group(2)
                if unit.startswith("day"):
                    retry_after = value * 86400
                elif unit.startswith("hour"):
                    retry_after = value * 3600
                else:
                    retry_after = value * 60

            reason = "weekly_quota_exhausted" if is_weekly else "session_quota_exhausted"
            return {"retry_after": retry_after, "reason": reason}

        return None

    def __init__(self):
        super().__init__()
        self._init_quota_tracker()
        self.api_base = os.getenv("OLLAMA_CLOUD_API_BASE", OLLAMA_CLOUD_API_BASE)
        self.model_definitions = ModelDefinitions()
        self._models_cache: Optional[List[str]] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: float = 300.0

        lib_logger.debug(
            f"OllamaCloudProvider initialized: base={self.api_base}"
        )

    def _parse_credential(self, credential_identifier: str) -> Dict[str, str]:
        """
        Parse the credential identifier into component parts.

        The API key is the full credential string (from OLLAMA_CLOUD_API_KEY_N).
        The session cookie is always sourced from a separate env var
        (OLLAMA_CLOUD_SESSION_COOKIE), never embedded in the API key.
        """
        from .utilities.ollama_cloud_quota_tracker import _extract_session_cookie
        return {
            "api_key": credential_identifier,
            "session_cookie": _extract_session_cookie(credential_identifier),
        }

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns models from Ollama Cloud.

        Uses the /api/tags native endpoint (no auth required) or /v1/models.
        """
        import time

        # Check static definitions first
        static_models = self.model_definitions.get_all_provider_models("ollama_cloud")
        if static_models:
            return static_models

        # Check cache
        now = time.time()
        if self._models_cache and (now - self._models_cache_time < self._models_cache_ttl):
            return self._models_cache

        # Query Ollama Cloud /api/tags (no auth needed, returns native format)
        try:
            response = await client.get(OLLAMA_CLOUD_TAGS_URL, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            discovered = []
            for m in data.get("models", []):
                model_name = m.get("name") or m.get("model", "")
                if model_name:
                    # Strip -cloud suffix if present
                    clean = model_name.replace("-cloud", "")
                    discovered.append(f"ollama_cloud/{clean}")

            if discovered:
                self._models_cache = discovered
                self._models_cache_time = now
                lib_logger.info(
                    f"Discovered {len(discovered)} models from Ollama Cloud"
                )
                return discovered
        except Exception as e:
            lib_logger.warning(f"Failed to fetch Ollama Cloud models: {e}")

        # Minimal fallback — only long-lived flagship models.
        # Dynamic /api/tags discovery is the primary source.
        return [
            "ollama_cloud/deepseek-v3.2",
            "ollama_cloud/deepseek-v4-flash",
            "ollama_cloud/gemma4:31b",
            "ollama_cloud/qwen3.5:397b",
        ]

    def has_custom_logic(self) -> bool:
        return True

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential = kwargs.pop("credential_identifier", "")
        cred = self._parse_credential(credential)
        kwargs.pop("transaction_context", None)
        model = kwargs.get("model", "")

        # Strip provider prefix for upstream
        model_bare = model.split("/")[-1] if "/" in model else model
        kwargs["model"] = "openai/" + model_bare

        # Set up auth
        actual_key = cred["api_key"]
        kwargs["api_key"] = actual_key
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "openai"
        kwargs["client"] = openai.AsyncOpenAI(
            api_key=actual_key,
            base_url=self.api_base,
            http_client=client,
        )

        # Strip unsupported params
        unsupported = set(kwargs.keys()) - self.SUPPORTED_PARAMS
        if unsupported:
            lib_logger.debug(
                f"ollama_cloud: stripping unsupported params for {model}: {unsupported}"
            )
            kwargs = {k: v for k, v in kwargs.items() if k in self.SUPPORTED_PARAMS}

        return await litellm.acompletion(**kwargs)

    SUPPORTED_EMBEDDING_PARAMS = {
        "model", "input", "api_key", "api_base", "custom_llm_provider", "client",
        "encoding_format", "dimensions", "extra_headers", "extra_body",
    }

    async def aembedding(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> litellm.EmbeddingResponse:
        credential = kwargs.pop("credential_identifier", "")
        cred = self._parse_credential(credential)
        kwargs.pop("transaction_context", None)
        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        kwargs["model"] = "openai/" + model_bare
        kwargs["api_key"] = cred["api_key"]
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "openai"
        kwargs["client"] = openai.AsyncOpenAI(
            api_key=cred["api_key"],
            base_url=self.api_base,
            http_client=client,
        )

        kwargs = {k: v for k, v in kwargs.items() if k in self.SUPPORTED_EMBEDDING_PARAMS}

        return await litellm.aembedding(**kwargs)

    async def refresh_balance(
        self,
        api_key: str,
        credential_identifier: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """Refresh quota from session cookie scraping."""
        snapshot = await self._fetch_usage_for_credential(credential_identifier)
        if snapshot.status == "success":
            return {
                "status": "success",
                "plan": snapshot.plan,
                "session_pct": snapshot.session_pct,
                "weekly_pct": snapshot.weekly_pct,
                "session_reset": snapshot.session_reset,
                "weekly_reset": snapshot.weekly_reset,
                "fetched_at": snapshot.fetched_at,
            }
        return {"status": snapshot.status, "error": snapshot.error}

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> float:
        """Ollama Cloud is free/subscription — no per-token cost."""
        return 0.0
