# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import os
import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, TYPE_CHECKING
import httpx
import litellm

if TYPE_CHECKING:
    from ..usage import UsageManager

from .provider_interface import ProviderInterface
from .utilities.opencode_quota_tracker import OpencodeQuotaTracker
from ..model_definitions import ModelDefinitions
from ..error_handler import mask_credential
from ..usage.types import ResetMode

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class OpencodeProvider(OpencodeQuotaTracker, ProviderInterface):
    """
    Provider for OpenCode 'Go' service - OpenAI-compatible API.
    """

    provider_env_name = "opencode_go"

    # Define the quota groups using the window names as model keys
    model_quota_groups = {
        "5hr": ["5hr"],
        "weekly": ["weekly"],
        "monthly": ["monthly"],
    }

    def __init__(self):
        super().__init__()
        self.api_base = os.getenv("OPENCODE_GO_API_BASE", "https://opencode.ai/zen/v1")
        self.global_workspace_id = os.getenv("OPENCODE_WORKSPACE_ID")
        self._balance_cache = {}
        self._quota_refresh_interval = 300
        self.model_definitions = ModelDefinitions()
        
        masked_wrk = mask_credential(self.global_workspace_id) if self.global_workspace_id else "None"
        lib_logger.debug(f"OpencodeProvider initialized: base={self.api_base}, global_wrk={masked_wrk}")

    def _get_headers(self, auth_cookie: Optional[str] = None) -> Dict[str, str]:
        """Return the custom headers required by OpenCode."""
        headers = {
            "HTTP-Referer": "https://opencode.ai/",
            "X-Title": "opencode",
        }
        if auth_cookie:
            headers["Cookie"] = f"auth={auth_cookie}; oc_locale=en"
        return headers

    def _parse_credential(self, credential_identifier: str) -> Dict[str, str]:
        """
        Parse the credential identifier into component parts.
        Format: sk-key (required) or api_key:workspace_id:auth_cookie (workspace and cookie optional)
        """
        result = {
            "api_key": credential_identifier,
            "workspace_id": self.global_workspace_id,
            "auth_cookie": None
        }

        if ":" in credential_identifier:
            parts = credential_identifier.split(":")
            # Part 0: API Key (Required)
            result["api_key"] = parts[0]
            
            # Part 1: Workspace ID (Optional)
            if len(parts) > 1 and parts[1]:
                result["workspace_id"] = parts[1]
                
            # Part 2: Auth Cookie (Optional)
            if len(parts) > 2 and parts[2]:
                rest = parts[2]
                if rest.startswith("auth="):
                    result["auth_cookie"] = rest[5:]
                else:
                    result["auth_cookie"] = rest
        
        # Fallback for simple Fe26.2** cookies passed as the only identifier
        if not result["auth_cookie"] and result["api_key"].startswith("Fe26.2**"):
            result["auth_cookie"] = result["api_key"]
            
        return result

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns static models and quota tracking virtual models.
        """
        models = []
        static_models = self.model_definitions.get_all_provider_models("opencode_go")
        if static_models:
            models = [m.split("/")[-1] if "/" in m else m for m in static_models]
        else:
            models = ["deepseek-v4-pro", "glm-5.1", "kimi-k2.6"]
        
        # Ensure quota models are present for registration
        for q_model in ["5hr", "weekly", "monthly"]:
             if q_model not in models:
                 models.append(q_model)
                 
        return models

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
        model_bare = model.split("/")[-1] if "/" in model else model
        kwargs["model"] = "openai/" + model_bare
        extra_headers = self._get_headers(cred["auth_cookie"])
        existing_headers = kwargs.get("extra_headers") or {}
        kwargs["extra_headers"] = {**existing_headers, **extra_headers}
        actual_key = cred["api_key"]
        if not actual_key or actual_key == "dummy":
            actual_key = cred["auth_cookie"]
        kwargs["api_key"] = actual_key
        api_base = self.api_base
        if "/zen/v1" in api_base and not "/zen/go/v1" in api_base:
             api_base = api_base.replace("/zen/v1", "/zen/go/v1")
        kwargs["api_base"] = api_base
        kwargs["custom_llm_provider"] = "openai"
        return await litellm.acompletion(**kwargs)

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
        extra_headers = self._get_headers(cred["auth_cookie"])
        existing_headers = kwargs.get("extra_headers") or {}
        kwargs["extra_headers"] = {**existing_headers, **extra_headers}
        actual_key = cred["api_key"]
        if not actual_key or actual_key == "dummy":
            actual_key = cred["auth_cookie"]
        kwargs["api_key"] = actual_key
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "openai"
        return await litellm.aembedding(**kwargs)

    async def refresh_balance(
        self,
        api_key: str,
        credential_identifier: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        cred = self._parse_credential(credential_identifier)
        auth_cookie = cred["auth_cookie"]
        workspace_id = cred["workspace_id"]
        if not auth_cookie or not workspace_id:
            return {"status": "skipped", "reason": "missing credentials"}
        return await super().refresh_balance(
            auth_cookie, credential_identifier, workspace_id=workspace_id, client=client
        )

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        return {
            "interval": self._quota_refresh_interval,
            "name": "opencode_go_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for ident in credentials:
                try:
                    balance_data = await self.refresh_balance(ident, ident, client=client)
                    if balance_data.get("status") == "success":
                        usage_raw = balance_data.get("usage_raw", {})
                        now = balance_data.get("fetched_at", time.time())
                        windows_map = {
                            "rollingUsage": "5hr",
                            "weeklyUsage": "weekly",
                            "monthlyUsage": "monthly"
                        }
                        for raw_key, model_key in windows_map.items():
                            win_data = usage_raw.get(raw_key, {})
                            if isinstance(win_data, dict):
                                usage_percent = win_data.get("usage_percent", 0)
                                reset_in = win_data.get("resetInSec")
                                reset_ts = now + (reset_in if reset_in is not None else 0)
                                
                                # Use WITHOUT opencode_go/ prefix in the model name passed to update_quota_baseline
                                # as UsageManager will add it during normalization.
                                # Force usage to be > 0 so TUI shows reset time
                                await usage_manager.update_quota_baseline(
                                    ident,
                                    model_key,
                                    quota_max_requests=100,
                                    quota_used=max(1, int(usage_percent)),
                                    quota_reset_ts=reset_ts,
                                    force=True
                                )
                except Exception as e:
                    lib_logger.warning(f"Failed to refresh OpenCode Go quota for {mask_credential(ident)}: {e}")

