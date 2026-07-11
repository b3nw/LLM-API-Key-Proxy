# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
ClinePass Provider

Provider for the ClinePass subscription tier of the Cline API
(https://api.cline.bot). ClinePass offers 10 curated open coding models under
``cline-pass/<model>`` identifiers with a single subscription (5h / weekly /
monthly usage windows).

Authentication:
    Bearer API key (Settings > API Keys) or account auth token. The proxy
    standardises on ``CLINE_PASS_API_KEY_N`` (or ``CLINE_PASS_API_KEY``).

Routing:
    - Chat completions route through ``https://api.cline.bot/api/v1/chat/completions``
      using LiteLLM's ``openai/`` custom provider.
    - Reasoning models surface ``delta.reasoning`` chunks (Cline relays the
      upstream provider's reasoning field verbatim). LiteLLM passes the field
      through as part of the streamed completion.

Quota:
    Mixin: ``ClinePassQuotaTracker`` — polls
    ``GET /api/v1/users/me/plan/usage-limits`` every 15 min by default and
    stores per-window baselines (5h / weekly / monthly).

Catalog:
    The ClinePass model list is small, curated, and changes infrequently.
    We ship a hardcoded default catalog in ``DEFAULT_CLINEPASS_MODELS`` but
    allow operators to override via the ``CLINE_PASS_MODELS`` env var (the
    same shape used by every other provider via ``ModelDefinitions``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
import openai

from .provider_interface import ProviderInterface
from .utilities.cline_pass_quota_tracker import ClinePassQuotaTracker
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Default ClinePass catalog (https://docs.cline.bot/getting-started/clinepass#models).
# Key = display name (shown in /v1/models and used in proxy model IDs).
# Value = {"id": upstream model id, "options": {...}}
#
# Upstream model IDs retain Cline's own naming (e.g. ``cline-pass/qwen3.7-plus``)
# so we don't lose the context window hints in their pricing table.
DEFAULT_CLINEPASS_MODELS: Dict[str, Dict[str, Any]] = {
    "glm-5.2": {"id": "cline-pass/glm-5.2"},
    "kimi-k2.7-code": {"id": "cline-pass/kimi-k2.7-code"},
    "kimi-k2.6": {"id": "cline-pass/kimi-k2.6"},
    "deepseek-v4-pro": {"id": "cline-pass/deepseek-v4-pro"},
    "deepseek-v4-flash": {"id": "cline-pass/deepseek-v4-flash"},
    "mimo-v2.5": {"id": "cline-pass/mimo-v2.5"},
    "mimo-v2.5-pro": {"id": "cline-pass/mimo-v2.5-pro"},
    "minimax-m3": {"id": "cline-pass/minimax-m3"},
    "qwen3.7-max": {"id": "cline-pass/qwen3.7-max"},
    "qwen3.7-plus": {"id": "cline-pass/qwen3.7-plus"},
}

# The Cline API is rooted at https://api.cline.bot/api/v1 — every endpoint
# (chat completions, models, usage-limits, plan) lives under that /api/v1
# prefix. Do NOT use the standard OpenAI ``/v1`` shape here: Cline's path is
# ``/api/v1/chat/completions``, not ``/v1/chat/completions``. (Deployment
# 2026-07-11: routing via ``/v1`` produced 404s.)
CLINE_PASS_DEFAULT_API_BASE = "https://api.cline.bot/api/v1"

# Litellm params accepted for the openai/ provider route. Mirrors the
# x_ai / ollama_cloud allowlists; conservative subset that the Cline API
# documents on its chat completions page.
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


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ClinePassProvider(ClinePassQuotaTracker, ProviderInterface):
    """
    Provider for the ClinePass subscription tier via the Cline API.

    Static catalog + dynamic discovery (when the Cline API exposes a
    ``/api/v1/models`` endpoint with subscription-tier filtering). Cline
    docs do not currently publish a stable catalog endpoint for the
    subscription tier, so we fall back to ``DEFAULT_CLINEPASS_MODELS``.
    """

    provider_env_name = "cline_pass"

    # All chat-capable ClinePass models share the same 5h/weekly/monthly
    # window quotas; we declare three groups so the WebUI renders each
    # window as its own bar.
    model_quota_groups = {
        "5h": ["_five_hour"],
        "weekly": ["_weekly"],
        "monthly": ["_monthly"],
    }

    # ClinePass is a flat subscription — no per-token cost calculation
    # needs LiteLLM's pricing DB (LiteLLM has no ClinePass pricing).
    skip_cost_calculation = True

    default_rotation_mode: str = "sequential"

    def __init__(self):
        super().__init__()
        self._init_quota_tracker()
        self.api_base = os.getenv(
            "CLINE_PASS_API_BASE", CLINE_PASS_DEFAULT_API_BASE
        )
        self.model_definitions = ModelDefinitions()
        # Upstream-id -> display-name reverse map (mirrors Umans pattern).
        # Built lazily from the active model catalog (env override first,
        # then defaults).
        self._id_to_display: Dict[str, str] = self._build_reverse_map()
        # Upstream context lengths (when the upstream /v1/models responds).
        self._upstream_context: Dict[str, int] = {}

    # ----------------------------------------------------------------- Models

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """Route synthetic baseline models to their window group."""
        clean = model.split("/")[-1] if "/" in model else model
        if clean == "_five_hour":
            return "5h"
        if clean == "_weekly":
            return "weekly"
        if clean == "_monthly":
            return "monthly"
        return "5h"  # real models default to the most restrictive window

    async def get_models(
        self, api_key: str, client: httpx.AsyncClient
    ) -> List[str]:
        """Return the ClinePass model catalog.

        Discovery order:
            1. Operator override from ``CLINE_PASS_MODELS`` env var
            2. ``DEFAULT_CLINEPASS_MODELS`` (shipped)
            3. Live ``GET /api/v1/models`` (best-effort, filters to
               ``cline-pass/`` namespace)
        """
        # 1. Env-var override (full catalog)
        static_models = self.model_definitions.get_all_provider_models("cline_pass")
        if static_models:
            # Rebuild reverse map when env override changes the catalog
            self._id_to_display = self._build_reverse_map()
            return static_models

        # 3. Live discovery (best-effort)
        try:
            response = await client.get(
                f"{self.api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            discovered: List[str] = []
            for entry in data.get("data", []):
                mid = entry.get("id")
                if not isinstance(mid, str):
                    continue
                # Filter to the ClinePass namespace only
                if not mid.startswith("cline-pass/"):
                    continue
                bare = mid[len("cline-pass/"):]
                discovered.append(f"cline_pass/{bare}")
                ctx = entry.get("context_length")
                if ctx:
                    self._upstream_context[f"cline_pass/{bare}"] = int(ctx)
            if discovered:
                self._id_to_display = self._build_reverse_map()
                lib_logger.info(
                    f"Discovered {len(discovered)} ClinePass models from /v1/models"
                )
                return sorted(set(discovered))
        except Exception as e:
            lib_logger.debug(
                f"ClinePass live model discovery failed (using defaults): {e}"
            )

        # 2. Fall back to shipped defaults
        return [
            f"cline_pass/{name}" for name in DEFAULT_CLINEPASS_MODELS.keys()
        ]

    def get_model_context_overrides(self) -> Dict[str, int]:
        """Expose upstream-authoritative context window sizes when known."""
        return dict(self._upstream_context)

    def normalize_model_for_tracking(self, model: str) -> str:
        """Map upstream ``cline-pass/<bare>`` IDs to display names.

        Cline upstream IDs already carry the ``cline-pass/`` prefix
        (e.g. ``cline-pass/glm-5.2``); the proxy exposes models as
        ``cline_pass/<bare>``. Callers may pass any of three shapes:

        - ``cline-pass/<bare>`` — raw upstream id from Cline error
          messages or quota breakdowns.
        - ``cline_pass/<bare>`` — proxy display name.
        - ``<bare>`` — bare display name without provider prefix.

        All three map to the canonical ``cline_pass/<bare>`` form so
        usage records land under the display name regardless of which
        form the caller started with.
        """
        if not model:
            return model
        # Direct hit on the reverse map (e.g. caller passed the raw
        # upstream id ``cline-pass/<bare>``). Stores keys WITHOUT the
        # proxy ``cline_pass/`` prefix — see ``_build_reverse_map``.
        if model in self._id_to_display:
            return self._id_to_display[model]
        # Caller passed the proxy display name ``cline_pass/<bare>`` —
        # the value in the reverse map.
        for upstream_key, display_name in self._id_to_display.items():
            if display_name == model:
                return display_name
        # Caller passed the bare display name ``<bare>`` — synthesise
        # the proxy form and look it up by value.
        if "/" not in model:
            for display_name in self._id_to_display.values():
                if display_name == f"cline_pass/{model}":
                    return display_name
        return model

    @staticmethod
    def _build_reverse_map() -> Dict[str, str]:
        """Build upstream-id -> display-name reverse map from the active catalog.

        Returns a dict whose **keys** are the raw upstream IDs (e.g.
        ``"cline-pass/glm-5.2"`` — already prefixed by the Cline docs)
        and whose **values** are the proxy display names
        (``"cline_pass/<bare>"``). Callers that pass either form to
        ``normalize_model_for_tracking()`` get the canonical display
        name back.

        The previous implementation wrapped the upstream id with
        ``f"cline_pass/{upstream_id}"`` again, producing double-prefixed
        keys like ``"cline_pass/cline-pass/glm-5.2"`` that never matched
        the upstream-id branch of the normaliser. Caught by Kilo Code
        review on PR #122.
        """
        defs = ModelDefinitions()
        provider_models = defs.get_provider_models("cline_pass")
        # If the operator didn't set CLINE_PASS_MODELS, use the shipped defaults
        if not provider_models:
            provider_models = DEFAULT_CLINEPASS_MODELS
        reverse: Dict[str, str] = {}
        for display_key, defn in provider_models.items():
            upstream_id = (
                defn.get("id", display_key)
                if isinstance(defn, dict)
                else display_key
            )
            if upstream_id != display_key:
                reverse[upstream_id] = f"cline_pass/{display_key}"
        return reverse

    # ------------------------------------------------------------------ Errors

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse ClinePass 429/403 responses for quota exhaustion signals.

        Cline relays the upstream provider's rate-limit response, so the
        body usually contains a ``quota_exceeded`` or ``rate_limit_exceeded``
        marker. We can't reliably attribute a 429 to a specific window
        from the body alone, so we return ``None`` and let the standard
        error-cooldown path apply a generic backoff.
        """
        body = error_body
        if not body:
            response = getattr(error, "response", None)
            if response is not None:
                body = getattr(response, "text", None)
            if not body:
                err_body = getattr(error, "body", None)
                body = str(err_body) if err_body is not None else None
            if not body:
                body = str(error)

        body_lower = (body or "").lower()
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)

        if status_code == 429 or "rate limit" in body_lower:
            return {"retry_after": None, "reason": "RATE_LIMITED"}
        if status_code == 403 or "forbidden" in body_lower:
            return {"retry_after": None, "reason": "FORBIDDEN"}
        return None

    # --------------------------------------------------------------- Routing

    def has_custom_logic(self) -> bool:
        """Use the LiteLLM openai/ path; we just need to inject Bearer."""
        return True

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[Any, None]]:
        """Route a chat completion through the Cline API via LiteLLM."""
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        if not model:
            raise ValueError("ClinePass acompletion: missing model")

        # Resolve the upstream model id (e.g. ``cline-pass/qwen3.7-plus``)
        bare = model.split("/")[-1] if "/" in model else model
        upstream_id = self._lookup_upstream_id(bare)
        if not upstream_id:
            raise ValueError(
                f"ClinePass: unknown model '{model}' "
                f"(set CLINE_PASS_MODELS or update DEFAULT_CLINEPASS_MODELS)"
            )

        # ``self.api_base`` is rooted at ``/api/v1`` — LiteLLM's openai/
        # provider appends ``/chat/completions`` to whatever base you give
        # it, so the resulting URL is ``https://api.cline.bot/api/v1/chat/completions``.
        kwargs["model"] = f"openai/{upstream_id}"
        kwargs["api_key"] = credential
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "openai"
        kwargs["client"] = openai.AsyncOpenAI(
            api_key=credential,
            base_url=self.api_base,
            http_client=client,
        )

        # Strip params the Cline API doesn't document
        unsupported = set(kwargs.keys()) - SUPPORTED_PARAMS
        if unsupported:
            lib_logger.debug(
                f"cline_pass: stripping unsupported params for {model}: {unsupported}"
            )
            kwargs = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        return await litellm.acompletion(**kwargs)

    async def aembedding(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> litellm.EmbeddingResponse:
        """ClinePass does not expose a separate embeddings endpoint.

        Raise a clear error rather than silently failing on an HTTP 404
        from the upstream; this lets the rotation engine try a fallback
        provider when the caller routes through ``MODEL_FALLBACK``.
        """
        raise NotImplementedError(
            "ClinePass does not currently expose an embeddings endpoint. "
            "Use a different provider or fall back via MODEL_FALLBACK."
        )

    # ----------------------------------------------------------- Internals

    def _lookup_upstream_id(self, bare: str) -> Optional[str]:
        """Return the upstream Cline model id for a bare display name.

        Looks first at the operator-supplied env catalog, then at the
        shipped default catalog. Falls back to ``cline-pass/<bare>`` if no
        override is registered (lets the upstream reject unknown models
        with a clear error rather than us silently mapping to the wrong id).
        """
        defs = ModelDefinitions()
        provider_models = defs.get_provider_models("cline_pass")
        if not provider_models:
            provider_models = DEFAULT_CLINEPASS_MODELS
        defn = provider_models.get(bare)
        if isinstance(defn, dict):
            return defn.get("id") or f"cline-pass/{bare}"
        if defn is not None:
            return f"cline-pass/{bare}"
        return None
