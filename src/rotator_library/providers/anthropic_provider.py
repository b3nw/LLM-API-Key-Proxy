# src/rotator_library/providers/anthropic_provider.py
"""
Anthropic Provider

Dedicated provider for Anthropic Claude models with dual credential routing:
- OAuth credentials (Claude Pro/Max): Direct httpx calls to Anthropic Messages API
- API key credentials: Delegated to litellm.acompletion() (preserves existing behavior)

OAuth requests use:
- Bearer token authentication
- anthropic-beta headers for OAuth and interleaved thinking
- Tool name prefixing (mcp_) for OAuth path
- Streaming SSE event handling
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import httpx
import litellm

from .provider_interface import ProviderInterface, UsageResetConfigDef, QuotaGroupMap
from .anthropic_oauth_base import AnthropicOAuthBase
from .utilities.anthropic_quota_tracker import AnthropicQuotaTracker
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Anthropic API endpoints
ANTHROPIC_API_BASE = os.getenv(
    "ANTHROPIC_API_BASE", "https://api.anthropic.com"
)
ANTHROPIC_MESSAGES_ENDPOINT = f"{ANTHROPIC_API_BASE}/v1/messages"

# Required headers for OAuth requests
# Base betas for all OAuth requests — mirrors Claude Code's header set.
# The claude-code-20250219 beta is critical: it tells Anthropic's API that
# this is a Claude Code session (confirmed by pi-agent and @cgaravitoq).
_BASE_BETA_HEADERS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
    "prompt-caching-scope-2026-01-05",
    "context-management-2025-06-27",
]
# Additional betas for long-context (1M) models
_LONG_CONTEXT_BETAS = ["context-1m-2025-08-07", "effort-2025-11-24"]
_LONG_CONTEXT_MODEL_PREFIXES = (
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6", "claude-sonnet-5", "claude-fable-5",
)
# Haiku models exclude interleaved thinking
_HAIKU_PREFIX = "claude-haiku-"


def _compute_beta_header(model: str) -> str:
    """Compute the anthropic-beta header value based on the model.

    Mirrors the beta header computation used by pi-agent and the
    @cgaravitoq/pi-claude-code-auth extension.
    """
    betas = list(_BASE_BETA_HEADERS)

    # Add long-context betas for 1M context models
    if any(model.startswith(p) for p in _LONG_CONTEXT_MODEL_PREFIXES):
        betas.extend(_LONG_CONTEXT_BETAS)

    # Haiku models exclude interleaved thinking
    if model.startswith(_HAIKU_PREFIX):
        betas = [b for b in betas if b != "interleaved-thinking-2025-05-14"]

    return ",".join(betas)


# Kept for backward compatibility (used in token refresh requests)
ANTHROPIC_BETA_HEADER = _compute_beta_header("")
ANTHROPIC_VERSION = "2023-06-01"
# User agent for OAuth requests — must use claude-code/ prefix (claude-cli/ is blocked by Anthropic).
# Configurable via env var so operators can update without code changes.
# Latest Claude Code version: https://www.npmjs.com/package/@anthropic-ai/claude-code
ANTHROPIC_USER_AGENT = os.environ.get("ANTHROPIC_CLI_USER_AGENT", "claude-code/2.1.195")

# Tool name prefix for OAuth path
TOOL_PREFIX = "mcp_"


def _prefix_tool_name(name: str) -> str:
    """Ensure a tool name has the mcp_ prefix with PascalCase.

    Mirrors Claude Code's tool naming convention (e.g., read → mcp_Read).
    Anthropic's OAuth API requires that mcp_-prefixed tools have an uppercase
    letter immediately after the prefix — lowercase causes a 400 rejection.

    Handles both cases:
    - Tool without prefix: read → mcp_Read
    - Tool already prefixed but lowercase: mcp_read → mcp_Read
    """
    if not name:
        return name
    if name.startswith(TOOL_PREFIX):
        remainder = name[len(TOOL_PREFIX):]
        if remainder:
            return f"{TOOL_PREFIX}{remainder[0].upper()}{remainder[1:]}"
        return name
    return f"{TOOL_PREFIX}{name[0].upper()}{name[1:]}"

# =============================================================================
# CLAUDE CODE PROTOCOL SIGNALS (OAuth identity emulation)
# =============================================================================
# The following functions mirror the @cgaravitoq/pi-claude-code-auth extension
# to make OAuth requests look like genuine Claude Code sessions.
# Without these signals, Anthropic applies stricter rate limits (429s).

_BILLING_SALT = "59cf53e54c78"
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
_CLI_VERSION = os.environ.get("ANTHROPIC_CLI_VERSION", "2.1.195")
_CLAUDE_CODE_ENTRYPOINT = os.environ.get("CLAUDE_CODE_ENTRYPOINT", "cli")


def _extract_first_user_message_text(messages: List[Dict[str, Any]]) -> str:
    """Extract text content from the first user message in the list."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                return " ".join(texts) if texts else ""
            return str(content) if content else ""
    return ""


def _compute_billing_header(messages: List[Dict[str, Any]]) -> str:
    """Compute the Claude Code billing header (client attestation).

    Mirrors the @cgaravitoq/pi-claude-code-auth implementation:
    - cch: SHA256(first_user_message_text)[:5]
    - suffix: SHA256(salt + sampled_chars + version)[:3]
    - sampled_chars: chars at positions [4, 7, 20] of first user message text
    """
    text = _extract_first_user_message_text(messages)

    cch = hashlib.sha256(text.encode("utf-8")).hexdigest()[:5]

    sampled = "".join(
        text[i] if i < len(text) else "0"
        for i in [4, 7, 20]
    )
    suffix_input = f"{_BILLING_SALT}{sampled}{_CLI_VERSION}"
    suffix = hashlib.sha256(suffix_input.encode("utf-8")).hexdigest()[:3]

    return (
        f"x-anthropic-billing-header: "
        f"cc_version={_CLI_VERSION}.{suffix}; "
        f"cc_entrypoint={_CLAUDE_CODE_ENTRYPOINT}; "
        f"cch={cch};"
    )


def _build_claude_code_system(
    messages: List[Dict[str, Any]],
    original_system_prompt: Optional[str],
) -> tuple:
    """Build the system prompt array and potentially modify messages.

    Returns (system_entries, messages) where system_entries is a list of
    {"type": "text", "text": ...} entries for the Anthropic API system field,
    and messages may have been modified to relocate the original system prompt.

    The system array contains:
    1. Billing header (client attestation) -- first entry
    2. Claude Code identity string -- second entry
    The original system prompt (if any) is moved to the first user message.
    """
    billing = _compute_billing_header(messages)

    system_entries = [
        {"type": "text", "text": billing},
        {"type": "text", "text": _CLAUDE_CODE_IDENTITY},
    ]

    if original_system_prompt:
        relocated = False
        if messages and messages[0].get("role") == "user":
            first_content = messages[0].get("content", "")
            if isinstance(first_content, str):
                messages[0]["content"] = f"{original_system_prompt}\n\n{first_content}"
                relocated = True
            elif isinstance(first_content, list):
                messages[0]["content"] = [
                    {"type": "text", "text": original_system_prompt}
                ] + first_content
                relocated = True
        if not relocated:
            system_entries.append({"type": "text", "text": original_system_prompt})

    return system_entries, messages



# Models available via OAuth subscription (Claude Pro/Max)
OAUTH_MODELS = [
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
]

# Max output tokens per model family — used when caller doesn't specify max_tokens.
# Maps model prefix → max output tokens.
_MODEL_MAX_OUTPUT_TOKENS: Dict[str, int] = {
    "claude-fable-5": 128_000,
    "claude-opus-4-8": 128_000,
    "claude-opus-4-7": 128_000,
    "claude-opus-4-6": 128_000,
    "claude-sonnet-4-6": 64_000,
    "claude-opus-4-5": 64_000,
    "claude-sonnet-4-5": 64_000,
    "claude-haiku-4-5": 64_000,
}
_DEFAULT_MAX_TOKENS = 16_384  # Fallback for unknown models

# Token prefixes for identifying credential types
OAUTH_ACCESS_TOKEN_PREFIX = "sk-ant-oat"
OAUTH_REFRESH_TOKEN_PREFIX = "sk-ant-ort"
API_KEY_PREFIX = "sk-ant-api"

# =============================================================================
# DYNAMIC MODEL DISCOVERY via models.dev
# =============================================================================
# Fetches the Anthropic model catalog from https://models.dev/api.json at
# runtime, providing model IDs and max output tokens without code changes.
# Falls back to the hardcoded OAUTH_MODELS list on failure.
#
# Pattern follows the Codex provider's GitHub JSON catalog fetch:
# module-level cache, 3-tier fallback (fresh → stale → hardcoded).
#
# OAuth tokens (sk-ant-oat-*) cannot call Anthropic's GET /v1/models endpoint,
# so we use models.dev (a community-maintained catalog) instead. This is the
# same approach used by pi (earendil-works/pi) and opencode (sst/opencode).

_MODELS_DEV_URL = os.environ.get("MODELS_DEV_URL", "https://models.dev/api.json")
_MODELS_DEV_CACHE_TTL = int(os.environ.get("ANTHROPIC_MODELS_CACHE_TTL", "3600"))
_models_dev_cache: Optional[Dict[str, Any]] = None
_models_dev_cache_time: float = 0.0

# Model families excluded from the OAuth-eligible list:
# - claude-3-* : retired, no longer available via Anthropic API
# - claude-mythos-* : restricted to Project Glasswing participants
_OAUTH_EXCLUDED_PREFIXES = ("claude-3-", "claude-mythos-")


def _fetch_anthropic_models_from_models_dev() -> Optional[Dict[str, Any]]:
    """Fetch Anthropic models from the models.dev catalog.

    Returns a dict with 'models' (list of model IDs) and 'max_tokens'
    (dict of model_id → max output tokens), or None on failure.
    """
    import urllib.request

    try:
        req = urllib.request.Request(
            _MODELS_DEV_URL,
            headers={"User-Agent": "llm-proxy/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        anthropic_data = data.get("anthropic", {})
        models_data = anthropic_data.get("models", {})
        if not models_data:
            lib_logger.warning("[Anthropic] models.dev returned empty Anthropic models")
            return None

        models = []
        max_tokens: Dict[str, int] = {}

        for model_id, model_info in models_data.items():
            # Skip retired/restricted model families
            if any(model_id.startswith(p) for p in _OAUTH_EXCLUDED_PREFIXES):
                continue
            # Only include models with tool calling support (required by Claude Code)
            if not model_info.get("tool_call", False):
                continue

            models.append(model_id)

            # Extract max output tokens
            limit = model_info.get("limit", {})
            output = limit.get("output")
            if output:
                max_tokens[model_id] = output

        lib_logger.info(
            f"[Anthropic] Fetched {len(models)} models from models.dev: "
            f"{', '.join(models)}"
        )

        # Return None if all models were filtered out, so callers fall back
        # to hardcoded OAUTH_MODELS instead of receiving an empty truthy dict.
        if not models:
            lib_logger.warning(
                "[Anthropic] All models from models.dev were filtered out; "
                "falling back to hardcoded OAUTH_MODELS"
            )
            return None

        return {"models": models, "max_tokens": max_tokens}

    except json.JSONDecodeError as e:
        lib_logger.warning(f"[Anthropic] models.dev response is not valid JSON: {e}")
        return None
    except Exception as e:
        lib_logger.warning(f"[Anthropic] Failed to fetch models from models.dev: {e}")
        return None


def _get_dynamic_models() -> Optional[Dict[str, Any]]:
    """Get dynamic model data, fetching from models.dev if cache is stale.

    Returns dict with 'models' (list) and 'max_tokens' (dict), or None if
    no data is available (caller should fall back to hardcoded OAUTH_MODELS).
    """
    global _models_dev_cache, _models_dev_cache_time

    now = time.time()
    if _models_dev_cache is not None and (now - _models_dev_cache_time) < _MODELS_DEV_CACHE_TTL:
        return _models_dev_cache

    fetched = _fetch_anthropic_models_from_models_dev()
    if fetched is not None:
        _models_dev_cache = fetched
        _models_dev_cache_time = now
        return fetched

    # Use stale cache if available (network failed but we have old data)
    if _models_dev_cache is not None:
        lib_logger.info("[Anthropic] Using stale models.dev cache (fetch failed)")
        return _models_dev_cache

    return None


def _is_oauth_credential(credential: str) -> bool:
    """
    Determine if a credential identifier is an OAuth credential (file path or env:// URI)
    vs a raw API key.
    """
    # env:// paths are always OAuth
    if credential.startswith("env://"):
        return True
    # File paths (contain / or \\ or end in .json) are OAuth
    if "/" in credential or "\\" in credential or credential.endswith(".json"):
        return True
    # Raw OAuth access tokens
    if credential.startswith(OAUTH_ACCESS_TOKEN_PREFIX):
        return True
    # API keys start with sk-ant-api
    if credential.startswith(API_KEY_PREFIX):
        return False
    # Default: treat as API key
    return False


# =============================================================================
# MESSAGE FORMAT CONVERSION
# =============================================================================


def _convert_openai_to_anthropic_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert OpenAI chat format messages to Anthropic Messages format.

    Returns:
        Tuple of (system_prompt, anthropic_messages)
    """
    system_prompt = None
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "system":
            # Extract system message
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                # Handle multipart system content
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                system_prompt = "\n".join(texts)
            continue

        if role == "user":
            if isinstance(content, str):
                anthropic_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Convert multipart content
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                            if url.startswith("data:"):
                                try:
                                    header, data = url.split(",", 1)
                                    media_type = header.split(":")[1].split(";")[0]
                                    parts.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data,
                                        },
                                    })
                                except (ValueError, IndexError):
                                    lib_logger.debug(
                                        "Failed to parse data URI image in user message, skipping."
                                    )
                if parts:
                    anthropic_messages.append({"role": "user", "content": parts})
            continue

        if role == "assistant":
            content_blocks = []

            # Handle text content
            if isinstance(content, str) and content:
                content_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_blocks.append({"type": "text", "text": part.get("text", "")})

            # Handle tool calls
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("type") == "function":
                    func = tc.get("function", {})
                    arguments = func.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        input_data = arguments
                    else:
                        try:
                            input_data = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError):
                            input_data = {}

                    tool_name = func.get("name", "")
                    tool_name = _prefix_tool_name(tool_name)

                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tool_name,
                        "input": input_data,
                    })

            if content_blocks:
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue

        if role == "tool":
            # Tool result message
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = content
            if isinstance(tool_content, str):
                try:
                    tool_content = json.loads(tool_content)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Anthropic expects tool results as user messages with tool_result blocks
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(tool_content) if not isinstance(tool_content, str) else tool_content,
                }],
            })
            continue

    return system_prompt, anthropic_messages


def _convert_tools_to_anthropic_format(
    tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI tools format to Anthropic tool definitions."""
    if not tools:
        return None

    anthropic_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue

        name = _prefix_tool_name(name)

        anthropic_tools.append({
            "name": name,
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })

    return anthropic_tools if anthropic_tools else None


def _strip_tool_prefix(name: str) -> str:
    """Strip mcp_ prefix and reverse PascalCase applied by _prefix_tool_name.

    Restores the original tool name as sent by the client:
    - mcp_Read → read (original: read)
    - mcp_Outline_get_document → outline_get_document (original: mcp_outline_get_document)
    """
    if name.startswith(TOOL_PREFIX):
        remainder = name[len(TOOL_PREFIX):]
        if remainder and remainder[0].isupper():
            return f"{remainder[0].lower()}{remainder[1:]}"
        return remainder
    return name


# =============================================================================
# PROVIDER IMPLEMENTATION
# =============================================================================


class AnthropicProvider(AnthropicOAuthBase, AnthropicQuotaTracker, ProviderInterface):
    """
    Anthropic Provider with dual credential routing.

    - OAuth credentials: Direct httpx calls to Anthropic Messages API
    - API key credentials: Delegated to litellm.acompletion()
    """

    # Provider configuration
    provider_env_name: str = "anthropic"

    # Skip cost calculation for OAuth credentials only
    # (API key requests keep litellm cost tracking)
    skip_cost_calculation: bool = True

    # Sequential mode - use one OAuth cred until rate-limited
    default_rotation_mode: str = "sequential"

    # Tier configuration
    # OAuth credentials preferred over API keys
    tier_priorities: Dict[str, int] = {
        "pro": 1,
        "max_5": 1,
        "max_20": 1,
        "api_key": 2,
    }
    default_tier_priority: int = 2

    # Usage reset configuration
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=86400,  # 24 hours
            mode="per_model",
            description="Daily per-model reset",
            field_name="models",
        ),
    }

    # Model quota groups - Anthropic subscription windows
    # Mirrors Codex pattern: 5h-limit and weekly-limit windows
    # Synthetic models (anthropic/_5h_window, anthropic/_weekly_window) are used
    # for quota tracking via the /api/oauth/usage endpoint
    # "anthropic-global" ensures sequential rotation shares one sticky credential
    # across all models, maximizing prompt cache hits.
    model_quota_groups: QuotaGroupMap = {
        "5h-limit": list(OAUTH_MODELS),
        "weekly-limit": list(OAUTH_MODELS),
        "anthropic-global": list(OAUTH_MODELS),
    }

    def __init__(self):
        ProviderInterface.__init__(self)
        AnthropicOAuthBase.__init__(self)
        self.model_definitions = ModelDefinitions()
        # Track which credentials are API keys vs OAuth
        self._credential_types: Dict[str, str] = {}

        # Populate quota groups dynamically from models.dev, fall back to hardcoded
        dynamic = _get_dynamic_models()
        oauth_models = dynamic["models"] if dynamic else list(OAUTH_MODELS)
        self.model_quota_groups = {
            "5h-limit": list(oauth_models),
            "weekly-limit": list(oauth_models),
            "anthropic-global": list(oauth_models),
        }

        # Initialize quota tracker
        self._init_quota_tracker()

    def has_custom_logic(self) -> bool:
        """This provider uses custom logic for OAuth credentials."""
        return True

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """All Anthropic models share the anthropic-global quota group.

        This ensures dynamically discovered models (not yet in the quota
        groups list) are still tracked under the global group, matching
        the Codex provider pattern.
        """
        return "anthropic-global"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Return available Anthropic models."""
        models = set()

        # Try dynamic discovery first, fall back to hardcoded OAUTH_MODELS
        dynamic = _get_dynamic_models()
        oauth_models = dynamic["models"] if dynamic else OAUTH_MODELS

        for model in oauth_models:
            models.add(f"anthropic/{model}")

        # Get static model definitions from env var overrides
        static_models = self.model_definitions.get_all_provider_models("anthropic")
        if static_models:
            for model in static_models:
                models.add(model)

        return sorted(models)

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """Get tier name for a credential."""
        if not _is_oauth_credential(credential):
            return "api_key"

        # Check cached tier info from AnthropicOAuthBase
        tier_info = self.get_credential_tier_info(credential)
        if tier_info:
            sub_type = tier_info.get("subscription_type", "")
            rate_tier = tier_info.get("rate_limit_tier", "")
            if sub_type:
                return sub_type.lower()
            if rate_tier:
                return rate_tier.lower()

        # Check credentials cache for metadata
        creds = self._credentials_cache.get(credential)
        if creds:
            metadata = creds.get("_proxy_metadata", {})
            sub_type = metadata.get("subscription_type", "")
            if sub_type:
                return sub_type.lower()

        return "pro"  # Default assumption for OAuth credentials

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Get auth header for a credential.

        For OAuth credentials: Bearer token via AnthropicOAuthBase
        For API keys: x-api-key header
        """
        if _is_oauth_credential(credential_identifier):
            return await self.get_anthropic_auth_header(credential_identifier)
        else:
            return {"x-api-key": credential_identifier}

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion request.

        Routes based on credential type:
        - OAuth credential: Direct httpx call to Anthropic Messages API
        - API key: Delegate to litellm.acompletion()
        """
        credential = kwargs.pop("credential_identifier", "")
        stream = kwargs.pop("stream", False)

        if _is_oauth_credential(credential):
            return await self._oauth_completion(client, credential, stream, **kwargs)
        else:
            return await self._apikey_completion(credential, **kwargs)

    # =========================================================================
    # API KEY PATH (litellm passthrough)
    # =========================================================================

    async def _apikey_completion(
        self, api_key: str, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Delegate to litellm.acompletion() for API key credentials."""
        kwargs["api_key"] = api_key
        # Remove internal context before litellm call
        kwargs.pop("transaction_context", None)
        kwargs.pop("litellm_params", None)

        response = await litellm.acompletion(**kwargs)
        return response

    # =========================================================================
    # OAUTH PATH (direct Anthropic Messages API)
    # =========================================================================

    async def _oauth_completion(
        self,
        client: httpx.AsyncClient,
        credential_path: str,
        stream: bool,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Handle completion via OAuth credential using direct Anthropic Messages API."""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        # Derive max_tokens from model family if caller didn't specify
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None:
            model_bare = model.split("/", 1)[-1] if "/" in model else model
            max_tokens = _DEFAULT_MAX_TOKENS

            # Try dynamic data first (exact match from models.dev)
            dynamic = _get_dynamic_models()
            if dynamic and model_bare in dynamic["max_tokens"]:
                max_tokens = dynamic["max_tokens"][model_bare]
            else:
                # Fall back to hardcoded prefix matching
                for prefix, limit in _MODEL_MAX_OUTPUT_TOKENS.items():
                    if model_bare.startswith(prefix):
                        max_tokens = limit
                        break
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        stop = kwargs.get("stop")

        # Strip provider prefix
        if "/" in model:
            model = model.split("/", 1)[1]

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = _convert_openai_to_anthropic_messages(messages)

        # Convert tools
        anthropic_tools = _convert_tools_to_anthropic_format(tools)

        # Get auth headers
        auth_headers = await self.get_anthropic_auth_header(credential_path)

        # Build request headers
        headers = {
            **auth_headers,
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": _compute_beta_header(model),
            "user-agent": ANTHROPIC_USER_AGENT,
            "x-app": "cli",
        }

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }

        system_entries, anthropic_messages = _build_claude_code_system(
            anthropic_messages, system_prompt
        )
        payload["system"] = system_entries
        payload["messages"] = anthropic_messages

        if anthropic_tools:
            payload["tools"] = anthropic_tools

        if temperature is not None:
            payload["temperature"] = temperature

        if top_p is not None:
            payload["top_p"] = top_p

        if stop:
            payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        if stream:
            payload["stream"] = True

        # Add beta=true query param for OAuth
        url = f"{ANTHROPIC_MESSAGES_ENDPOINT}?beta=true"

        lib_logger.debug(f"Anthropic OAuth request to {model}: {json.dumps(payload, default=str)[:500]}...")

        if stream:
            return self._stream_response(client, url, headers, payload, model, credential_path)
        else:
            return await self._non_stream_response(client, url, headers, payload, model, credential_path)

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        credential_path: str = "",
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming response from Anthropic Messages API."""
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Track state
        thinking_text = ""
        sent_thinking = False
        current_tool_calls: Dict[int, Dict[str, Any]] = {}
        tool_index = 0
        input_tokens = 0
        output_tokens = 0

        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:

            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(f"Anthropic API error {response.status_code}: {error_text[:500]}")
                raise httpx.HTTPStatusError(
                    f"Anthropic API error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = evt.get("type")

                # Handle message_start - get response ID and usage
                if event_type == "message_start":
                    msg = evt.get("message", {})
                    if msg.get("id"):
                        response_id = msg["id"]
                    usage = msg.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    continue

                # Handle content_block_start
                if event_type == "content_block_start":
                    block = evt.get("content_block", {})
                    block_type = block.get("type")

                    if block_type == "tool_use":
                        # Start of a tool use block
                        current_tool_calls[tool_index] = {
                            "id": block.get("id", ""),
                            "name": _strip_tool_prefix(block.get("name", "")),
                            "arguments": "",
                        }
                    continue

                # Handle content_block_delta
                if event_type == "content_block_delta":
                    delta_obj = evt.get("delta", {})
                    delta_type = delta_obj.get("type")

                    if delta_type == "text_delta":
                        text = delta_obj.get("text", "")
                        if text:
                            # If we have accumulated thinking and haven't sent it, prepend
                            if not sent_thinking and thinking_text:
                                text = f"<think>{thinking_text}</think>{text}"
                                sent_thinking = True

                            chunk = litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=f"anthropic/{model}",
                                object="chat.completion.chunk",
                                choices=[{
                                    "index": 0,
                                    "delta": {"content": text, "role": "assistant"},
                                    "finish_reason": None,
                                }],
                            )
                            yield chunk

                    elif delta_type == "thinking_delta":
                        # Accumulate thinking text
                        thinking_text += delta_obj.get("thinking", "")

                    elif delta_type == "input_json_delta":
                        # Tool call argument delta
                        partial_json = delta_obj.get("partial_json", "")
                        if tool_index in current_tool_calls:
                            current_tool_calls[tool_index]["arguments"] += partial_json

                    continue

                # Handle content_block_stop
                if event_type == "content_block_stop":
                    # Check if this is a completed tool call
                    if tool_index in current_tool_calls:
                        tc = current_tool_calls[tool_index]
                        chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"anthropic/{model}",
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": tool_index,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["name"],
                                            "arguments": tc["arguments"],
                                        },
                                    }],
                                },
                                "finish_reason": None,
                            }],
                        )
                        yield chunk
                        tool_index += 1
                    continue

                # Handle message_delta (end of message)
                if event_type == "message_delta":
                    delta_obj = evt.get("delta", {})
                    stop_reason = delta_obj.get("stop_reason", "end_turn")
                    usage = evt.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                    # Map Anthropic stop reasons to OpenAI finish reasons
                    finish_reason_map = {
                        "end_turn": "stop",
                        "stop_sequence": "stop",
                        "tool_use": "tool_calls",
                        "max_tokens": "length",
                    }
                    finish_reason = finish_reason_map.get(stop_reason, "stop")

                    # Send any remaining thinking text
                    if not sent_thinking and thinking_text:
                        think_chunk = litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=f"anthropic/{model}",
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {"content": f"<think>{thinking_text}</think>", "role": "assistant"},
                                "finish_reason": None,
                            }],
                        )
                        yield think_chunk
                        sent_thinking = True

                    # Send final chunk
                    final_chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=f"anthropic/{model}",
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }],
                    )
                    final_chunk.usage = litellm.Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
                    yield final_chunk
                    break

    async def _non_stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        credential_path: str = "",
    ) -> litellm.ModelResponse:
        """Handle non-streaming response from Anthropic Messages API."""
        created = int(time.time())

        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )


        if response.status_code >= 400:
            error_text = response.text
            lib_logger.error(f"Anthropic API error {response.status_code}: {error_text[:500]}")
            raise httpx.HTTPStatusError(
                f"Anthropic API error: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()
        response_id = data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")

        # Extract content
        full_text = ""
        thinking_text = ""
        tool_calls = []

        for block in data.get("content", []):
            block_type = block.get("type")

            if block_type == "text":
                full_text += block.get("text", "")

            elif block_type == "thinking":
                thinking_text += block.get("thinking", "")

            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": _strip_tool_prefix(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        # Build message
        message: Dict[str, Any] = {"role": "assistant"}

        # Prepend thinking as <think> tags
        if thinking_text and full_text:
            message["content"] = f"<think>{thinking_text}</think>{full_text}"
        elif thinking_text:
            message["content"] = f"<think>{thinking_text}</think>"
        elif full_text:
            message["content"] = full_text
        else:
            message["content"] = None

        if tool_calls:
            message["tool_calls"] = tool_calls

        # Map stop reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")

        # Extract usage
        usage_data = data.get("usage", {})
        usage = litellm.Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        response_obj = litellm.ModelResponse(
            id=response_id,
            created=created,
            model=f"anthropic/{model}",
            object="chat.completion",
            choices=[{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        )
        response_obj.usage = usage

        return response_obj


    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse quota/rate-limit errors from Anthropic API."""
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                try:
                    body = error.response.text
                except Exception:
                    pass
            if not body and hasattr(error, "body"):
                body = str(error.body)
            if not body:
                body = str(error)

        if not body:
            return None

        # Check for rate limit / overloaded status
        status_code = None
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            status_code = error.response.status_code

        if status_code == 429 or status_code == 529:
            retry_after = 60  # Default

            # Try to extract retry-after from headers
            if hasattr(error, "response") and hasattr(error.response, "headers"):
                retry_header = error.response.headers.get("retry-after")
                if retry_header:
                    try:
                        retry_after = int(retry_header)
                    except ValueError:
                        pass

            reason = "RATE_LIMITED" if status_code == 429 else "OVERLOADED"

            return {
                "retry_after": retry_after,
                "reason": reason,
                "reset_timestamp": None,
                "quota_reset_timestamp": None,
            }

        # Try to parse JSON error body
        try:
            data = json.loads(body) if isinstance(body, str) else body
            error_obj = data.get("error", data)
            error_type = error_obj.get("type", "")

            if error_type in ("rate_limit_error", "overloaded_error"):
                return {
                    "retry_after": 60,
                    "reason": "RATE_LIMITED" if error_type == "rate_limit_error" else "OVERLOADED",
                    "reset_timestamp": None,
                    "quota_reset_timestamp": None,
                }
        except Exception:
            pass

        return None
