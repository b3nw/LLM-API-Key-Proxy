# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Perchai Provider

Provider implementation for Perch (Perchai) - an OAuth-authenticated LLM
gateway that exposes a multi-model catalog under a single Bearer token.

Auth:
    OAuth session is managed by ``PerchaiAuthBase`` (T1).  The access token
    is short-lived and refreshed from the local ``~/.perch/cli-auth-session.json``
    file (or its env-var equivalents).  ``PerchaiAuthBase.get_app_url()`` returns
    the canonical app URL (env override > saved session > default).

Discovery:
    Models are listed by the upstream ``GET {appUrl}/api/perchai/account``
    endpoint.  Response shape is not stable across bundle versions, so this
    provider walks a set of candidate field paths defensively and returns an
    empty list if none match.  Cached for ``MODEL_CACHE_TTL_SECONDS`` (default
    300s) per-token on the singleton instance.

Routing:
    Perchai has a single bearer token that is shared across all models, so
    there is no per-model credential scoping.  ``skip_cost_calculation=True``
    because the upstream reports its own usage and dollar cost, and the proxy
    should not double-count.  ``default_rotation_mode="sequential"`` because
    the upstream applies rate limits per session; rotating mid-stream breaks
    prompt caching.

Environment variables:
    PERCHAI_API_KEY_<N>           - not used (Perchai uses OAuth, not API keys)
    PERCH_MODEL_CALL_PROXY_URL    - optional app URL override
    PERCH_MODEL_CALL_PROXY_TOKEN  - optional access token override
    PERCH_CLI_APP_URL             - optional app URL override
    PERCH_APP_URL                 - optional app URL override
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Union, final, override

import httpx
import litellm
from litellm.exceptions import APIError as LitellmAPIError
from typing_extensions import AsyncGenerator as _AsyncGenerator

from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
from .provider_interface import ProviderInterface
from .utilities.perchai_quota_tracker import PerchaiQuotaTracker

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


# =========================================================================
# MODULE-LEVEL CONSTANTS
# =========================================================================


# Perchai's account endpoint is rate-limited and the model catalog changes
# infrequently - 5 minutes is a reasonable freshness window.
MODEL_CACHE_TTL_SECONDS: int = 300


MODEL_CALL_PATH: str = "/api/perch-terminal/model-call"


USAGE_PATH: str = "/api/perch-terminal/usage"


DEFAULT_LANE: str = "chat"


SUPPORTED_PARAMS: set[str] = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "stream_options",
    "stop",
    "tools",
    "tool_choice",
}


# =========================================================================
# TypedDicts describing the upstream Perchai wire format
# =========================================================================


class PerchaiRequestEnvelope(TypedDict, total=False):
    """Envelope sent by the upstream CLI to ``/api/perch-terminal/model-call``.

    Only the fields we need to round-trip are declared; extras pass through
    transparently.  ``total=False`` because every field is optional from the
    caller's perspective - we never validate the envelope, just shape it.
    """

    request: Dict[str, Any]
    lane: str
    preferredModelId: str
    avoidModelIds: List[str]
    attribution: Optional[Any]
    clientSurface: str
    promoOverflowAccepted: bool
    runId: Optional[str]


class PerchaiStreamEvent(TypedDict, total=False):
    """A single event in the SSE stream from Perch.

    Perch emits a heterogeneous event stream (``text_delta``,
    ``reasoning_delta``, ``tool_call_delta``, ``tool_use_end``, ``done``,
    ``error``).  We normalize them in ``_parse_sse_line``.
    """

    type: str
    text: Optional[str]
    delta: Optional[str]
    runId: Optional[str]
    finishReason: Optional[str]
    usage: Optional[Dict[str, Any]]
    error: Optional[str]
    toolCall: Optional[Dict[str, Any]]


class PerchaiErrorResponse(TypedDict, total=False):
    """Error body returned by Perch on quota / rate-limit / auth failures.

    Both fields are populated for quota errors; transient errors may only
    include ``error``.
    """

    error: str
    errorCode: str


_PERCHAI_MODEL_FIELD_PATHS: tuple[tuple[str, ...], ...] = (
    ("models",),
    ("data", "models"),
    ("session", "models"),
    ("account", "models"),
    ("result", "models"),
)


DEFAULT_RETRY_AFTER_SECONDS: int = 3600


@final
class PerchaiProvider(PerchaiQuotaTracker, ProviderInterface):
    """First-party Perch (Perchai) provider using its native Bearer-auth API.

    ``PerchaiQuotaTracker`` is mixed in FIRST so its ``__init__`` side
    effects (``_balance_cache`` / ``_quota_refresh_interval``) are
    established before the provider's own ``__init__`` runs.  ``ProviderInterface``
    comes second so its singleton metaclass still applies.

    The provider implements:
    - ``get_models()`` - defensive account-endpoint walker
    - ``acompletion()`` - non-stream + stream dispatch via the perchai envelope
    - ``parse_quota_error()`` - all 10 perchai error code mappings
    - Background quota refresh via ``PerchaiQuotaTracker.run_background_job``
    """

    # =========================================================================
    # CLASS-LEVEL CONFIGURATION (overrides ProviderInterface defaults)
    # =========================================================================

    # Env-var lookup prefix (QUOTA_GROUPS_PERCHAI_*, ROTATION_MODE_PERCHAI, etc.).
    provider_env_name: str = "perchai"

    skip_cost_calculation: bool = True

    default_rotation_mode: str = "sequential"

    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================

    def __init__(self) -> None:
        self._balance_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = int(
            __import__("os").environ.get("PERCHAI_QUOTA_REFRESH_INTERVAL", "300")
        )

        self._model_cache: Dict[str, List[str]] = {}
        self._model_cache_timestamps: Dict[str, float] = {}

    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================

    @override
    def has_custom_logic(self) -> bool:
        return True

    @override
    async def get_models(
        self, api_key: str, client: httpx.AsyncClient
    ) -> List[str]:
        cache_key = self._cache_key_for(api_key)
        if self._model_cache_is_valid(cache_key):
            return list(self._model_cache[cache_key])

        try:
            app_url = self._resolve_app_url()

            response = await client.get(
                f"{app_url.rstrip('/')}/api/perchai/account",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                timeout=TimeoutConfig.non_streaming(),
            )
            response.raise_for_status()
            data = response.json()

            model_ids = self._extract_model_ids(data)
            if not model_ids:
                lib_logger.debug(
                    "Perchai account endpoint returned no models "
                    f"(tried {len(_PERCHAI_MODEL_FIELD_PATHS)} field paths)"
                )
                return []

            prefixed = [f"perchai/{model_id}" for model_id in model_ids]
            self._model_cache[cache_key] = prefixed
            self._model_cache_timestamps[cache_key] = time.time()
            return prefixed

        except Exception as exc:
            lib_logger.debug(f"Failed to fetch Perchai models: {exc}")
            return []

    # =========================================================================
    # acompletion() - NON-STREAM + STREAM DISPATCH
    # =========================================================================

    @override
    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs: Any
    ) -> Union[
        litellm.ModelResponse,
        _AsyncGenerator[litellm.ModelResponseStream, None],
    ]:
        """Perchai non-stream + stream ``acompletion`` dispatcher.

        Pop ``credential_identifier``, ``transaction_context``, and
        ``api_base`` (executor / rotator plumbing), validate ``messages``,
        strip the ``perchai/`` prefix from the model id, build the
        envelope, then either POST + parse JSON or stream SSE depending on
        ``payload["stream"]``.

        Reactive 401 handling: on the first 401, refresh the access token
        via ``PerchaiAuthBase.refresh_on_401`` and retry once.  If the
        retry still 401s, raise ``AuthenticationError``.
        """
        # Lazy import to avoid a circular dependency with perchai_auth_base.
        from .perchai_auth_base import PerchaiAuthBase
        from litellm.exceptions import (
            AuthenticationError as LitellmAuthenticationError,
        )

        credential_identifier = kwargs.pop("credential_identifier", "")
        transaction_context = kwargs.pop("transaction_context", None)
        kwargs.pop("api_base", None)

        messages = kwargs.get("messages")
        if not messages:
            raise ValueError("messages cannot be empty")

        raw_model = kwargs.get("model", "")
        model_name = raw_model.split("/", 1)[1] if "/" in raw_model else raw_model
        payload = self._build_payload(model_name=model_name, kwargs=kwargs)
        app_url = self._resolve_app_url()
        url = f"{app_url.rstrip('/')}{MODEL_CALL_PATH}"

        file_logger = ProviderLogger(transaction_context)
        file_logger.log_request({"envelope_request": payload, "model": raw_model})

        stream_mode = bool(payload.get("stream"))
        token = credential_identifier or self._resolve_session_token()

        def _headers(using_token: str) -> Dict[str, str]:
            return {
                "Authorization": f"Bearer {using_token}",
                "Content-Type": "application/json",
                "Accept": (
                    "text/event-stream" if stream_mode else "application/json"
                ),
            }

        if stream_mode:
            return self._stream_completion(
                client=client,
                url=url,
                headers=_headers(token),
                payload=payload,
                model=raw_model,
                file_logger=file_logger,
            )

        return await self._non_stream_completion(
            client=client,
            url=url,
            build_headers=_headers,
            payload=payload,
            model=raw_model,
            file_logger=file_logger,
            credential_identifier=credential_identifier,
            auth_base_cls=PerchaiAuthBase,
            auth_error_cls=LitellmAuthenticationError,
        )

    async def _non_stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        build_headers: Any,
        payload: Dict[str, Any],
        model: str,
        file_logger: ProviderLogger,
        credential_identifier: str,
        auth_base_cls: Any,
        auth_error_cls: Any,
    ) -> litellm.ModelResponse:
        """Non-stream POST + JSON parse + litellm.ModelResponse wrap.

        Reactive 401 handling: refresh the token via
        ``auth_base_cls.refresh_on_401`` and retry once.  A persistent 401
        raises ``auth_error_cls`` (typically ``litellm.AuthenticationError``).

        Body validation: a success status with empty body or missing
        ``choices`` raises ``litellm.exceptions.APIError`` so the rotator
        can rotate the credential or surface a clean error to the caller.

        UTF-8: the envelope is pre-serialized with ``ensure_ascii=False``
        and sent as utf-8 bytes so unicode content (emoji, RTL,
        combining marks) round-trips without escaping.
        """
        envelope = self._build_envelope(model=model, payload=payload)
        token = credential_identifier or self._resolve_session_token()

        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")

        def _post(using_token: str) -> Any:
            headers = build_headers(using_token)
            headers["Content-Type"] = "application/json; charset=utf-8"
            return client.post(
                url,
                headers=headers,
                content=body,
                timeout=TimeoutConfig.non_streaming(),
            )

        response = await _post(token)

        if response.status_code == 401:
            auth = auth_base_cls()
            new_token = await auth.refresh_on_401(client, token)
            response = await _post(new_token)

        await self._raise_for_status(response, model)

        response_text = response.text
        if not response_text or not response_text.strip():
            raise LitellmAPIError(
                status_code=response.status_code,
                message=(
                    f"Perchai returned an empty response body for model {model} "
                    f"(status {response.status_code})"
                ),
                llm_provider="perchai",
                model=model,
                request=response.request,
            )

        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise LitellmAPIError(
                status_code=response.status_code,
                message=f"Perchai returned malformed JSON for model {model}: {exc}",
                llm_provider="perchai",
                model=model,
                request=response.request,
            ) from exc

        if not isinstance(response_data, dict) or response_data.get("ok") is not True:
            error_text = (
                json.dumps(response_data)
                if isinstance(response_data, dict)
                else str(response_data)
            )
            quota_info = PerchaiProvider.parse_quota_error(
                Exception(error_text), error_text
            )
            if quota_info:
                reason = quota_info.get("reason")
                retry_after = quota_info.get("retry_after")
                raise LitellmAPIError(
                    status_code=response.status_code,
                    message=(
                        f"Perchai rejected request for model {model} "
                        f"(reason={reason}, retry_after={retry_after}): {error_text}"
                    ),
                    llm_provider="perchai",
                    model=model,
                    request=response.request,
                )
            raise LitellmAPIError(
                status_code=response.status_code,
                message=(
                    f"Perchai returned unexpected response shape for model {model}: "
                    f"{response_data!r}"
                ),
                llm_provider="perchai",
                model=model,
                request=response.request,
            )

        # Perchai response shape (observed live, not OpenAI-compatible):
        # {
        #   "ok": true,
        #   "text": "..." | null,
        #   "content": [{"type": "text"|"reasoning"|..., "text": "..."}, ...],
        #   "reasoning": "..." | null,
        #   "toolCalls": [...],
        #   "provider": "meta",
        #   "model": "muse-spark-1.2",
        #   "usage": {
        #     "inputTokens": 12,
        #     "outputTokens": 549,
        #     "cacheReadInputTokens": 0,
        #     "totalTokens": 561,
        #   },
        #   ...
        # }
        content_text: str = response_data.get("text") or ""
        reasoning_text: str = (
            response_data.get("reasoning")
            or response_data.get("thinking")
            or ""
        )
        content_blocks = response_data.get("content")
        if not content_text and isinstance(content_blocks, list):
            parts_text: List[str] = []
            parts_reasoning: List[str] = []
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                block_text = block.get("text", "")
                block_type = block.get("type", "")
                if block_type in ("reasoning", "thinking", "reasoning_content"):
                    parts_reasoning.append(block_text)
                elif block_type in ("text",) or not block_type:
                    parts_text.append(block_text)
            content_text = "".join(parts_text)
            if not reasoning_text:
                reasoning_text = "".join(parts_reasoning)

        usage_data = response_data.get("usage", {}) or {}
        usage_kwargs: Dict[str, Any] = {}
        prompt_tokens = usage_data.get("inputTokens")
        completion_tokens = usage_data.get("outputTokens")
        total_tokens = usage_data.get("totalTokens")
        cache_read = usage_data.get("cacheReadInputTokens")
        if prompt_tokens is not None:
            usage_kwargs["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            usage_kwargs["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            usage_kwargs["total_tokens"] = total_tokens
        if cache_read is not None and prompt_tokens is not None:
            usage_kwargs["prompt_tokens_details"] = {"cached_tokens": cache_read}

        message_kwargs: Dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
        }
        if reasoning_text:
            message_kwargs["reasoning_content"] = reasoning_text

        choices_list = [
            litellm.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.Message(**message_kwargs),
            )
        ]

        model_response = litellm.ModelResponse(
            id=response_data.get("id") or str(uuid.uuid4()),
            choices=choices_list,
            # Preserve the user-requested model id.  Perchai's own "model"
            # field is the upstream it routed through (e.g. "meta-muse-spark-1-2")
            # and would silently overwrite what the caller asked for.
            model=model,
            usage=litellm.Usage(**usage_kwargs) if usage_kwargs else None,
        )

        file_logger.log_final_response(response_data)
        return model_response

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: ProviderLogger,
    ) -> _AsyncGenerator[litellm.ModelResponseStream, None]:
        """SSE streaming path: ``client.stream("POST", ...)`` + ``aiter_lines``.

        Yields one ``litellm.ModelResponseStream`` per upstream event.
        The upstream terminates with ``[DONE]``; we end the generator
        with a final chunk carrying ``finish_reason="stop"``.  If the
        stream ends without a ``[DONE]`` sentinel (truncated response,
        TCP reset) we still synthesize a final ``stop`` chunk so
        downstream consumers always observe a terminating finish reason.

        Note: this method does NOT do reactive 401 refresh - the executor
        rotates the credential on stream failure (proxy handles retries).
        """
        envelope = self._build_envelope(model=model, payload=payload)

        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        stream_headers = dict(headers)
        stream_headers["Content-Type"] = "application/json; charset=utf-8"

        saw_done = False
        stream_id = f"chatcmpl-perchai-stream-{int(time.time())}"

        async with client.stream(
            "POST",
            url,
            headers=stream_headers,
            content=body,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            await self._raise_for_status(response, model)

            async for line in response.aiter_lines():
                file_logger.log_response_chunk(line)

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    saw_done = True
                    yield litellm.ModelResponseStream(
                        id=stream_id,
                        created=int(time.time()),
                        model=model,
                        object="chat.completion.chunk",
                        choices=[
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    )
                    return

                stream_chunk = self._parse_sse_line(line, model)
                if stream_chunk is not None:
                    yield stream_chunk

        if not saw_done:
            lib_logger.debug(
                "Perchai stream ended without [DONE] sentinel; "
                "synthesizing final stop chunk"
            )
            yield litellm.ModelResponseStream(
                id=stream_id,
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            )

    # =========================================================================
    # PARSE_QUOTA_ERROR (T7)
    # =========================================================================

    @staticmethod
    @override
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Map a Perchai error to a structured quota-classification dict.

        Returns ``None`` for errors the proxy should classify generically
        (context overflow, api_error, timeout, aborted, vision_model_error).
        For known quota/auth errors returns::

            {"retry_after": int | None, "reason": str}

        The mapping table mirrors the upstream ``lKe`` function:
        - 429 without errorCode -> usage_limit_reached
        - 403 + "Upgrade to Pro" -> starter_model_blocked
        - error string containing "usage limit"/"pilot capacity"/"monthly allowance"
          -> usage_limit_reached
        - error string containing "abort" -> timeout
        """
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                body = error.response.text
            elif hasattr(error, "body"):
                body = (
                    error.body
                    if isinstance(error.body, str)
                    else str(error.body)
                )
            else:
                body = str(error) if error else ""

        parsed: Optional[Dict[str, Any]] = None
        if body:
            try:
                loaded = json.loads(body)
                if isinstance(loaded, dict):
                    parsed = loaded
            except (json.JSONDecodeError, TypeError):
                parsed = None

        error_code = ""
        error_text = ""
        if parsed is not None:
            error_code = str(parsed.get("errorCode") or "").strip()
            error_text = str(parsed.get("error") or "").strip()

        status_code: Optional[int] = None
        if hasattr(error, "status_code") and isinstance(error.status_code, int):
            status_code = error.status_code
        elif hasattr(error, "response") and hasattr(error.response, "status_code"):
            try:
                status_code = int(error.response.status_code)
            except (TypeError, ValueError):
                status_code = None
        else:
            # Defensive fallback: ``Exception("429")`` style synthetic
            # errors (e.g. proxy test fixtures) embed the status code as
            # the message.  Real litellm exceptions carry ``.status_code``
            # so this branch is only reached in tests.  Limited to
            # quota-relevant codes (429/403/402) so we don't accidentally
            # classify arbitrary 5xx errors as quota failures.
            args = getattr(error, "args", ()) or ()
            if args:
                try:
                    candidate = int(str(args[0]).strip())
                    if candidate in (402, 403, 429):
                        status_code = candidate
                except (TypeError, ValueError):
                    pass

        return PerchaiProvider._classify_perchai_error(
            error_code=error_code,
            error_text=error_text,
            status_code=status_code,
        )

    @staticmethod
    def _classify_perchai_error(
        error_code: str,
        error_text: str,
        status_code: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Inner pure mapping for ``parse_quota_error``.  No I/O.

        Mapping (mirrors upstream ``lKe``):
            usage_limit_reached     -> reason="rate_limit", retry_after=3600 (or parsed)
            promo_overflow_decision -> reason="rate_limit", retry_after=3600
            starter_model_blocked   -> reason="forbidden",  retry_after=None
            byo_feature_blocked     -> reason="forbidden",  retry_after=None
            not_authenticated       -> reason="authentication", retry_after=None
            context_overflow        -> None  (terminal: proxy classifies as context_window_exceeded)
            api_error               -> None  (proxy classifies as server_error)
            timeout                 -> None  (proxy classifies as api_connection)
            aborted                 -> None  (terminal)
            vision_model_error      -> None  (terminal: invalid_request)

        HTTP fallback (no errorCode):
            429 (no errorCode) -> reason="rate_limit", retry_after=3600
            403 + "Upgrade to Pro" -> reason="forbidden"
        """
        text_lower = (error_text or "").lower()

        if error_code == "usage_limit_reached":
            return {
                "retry_after": DEFAULT_RETRY_AFTER_SECONDS,
                "reason": "rate_limit",
            }

        if error_code == "promo_overflow_decision":
            return {
                "retry_after": DEFAULT_RETRY_AFTER_SECONDS,
                "reason": "rate_limit",
            }

        if error_code in ("starter_model_blocked", "byo_feature_blocked"):
            return {
                "retry_after": None,
                "reason": "forbidden",
            }

        if error_code == "not_authenticated":
            return {
                "retry_after": None,
                "reason": "authentication",
            }

        if error_code in (
            "context_overflow",
            "api_error",
            "timeout",
            "aborted",
            "vision_model_error",
        ):
            return None

        if not error_code:
            if status_code == 429:
                return {
                    "retry_after": DEFAULT_RETRY_AFTER_SECONDS,
                    "reason": "rate_limit",
                }
            if status_code == 403 and "upgrade to pro" in text_lower:
                return {
                    "retry_after": None,
                    "reason": "forbidden",
                }
            # Free-form text inference (mirrors upstream ``Ome``):
            if any(
                marker in text_lower
                for marker in (
                    "usage limit",
                    "pilot capacity",
                    "monthly allowance",
                )
            ):
                return {
                    "retry_after": DEFAULT_RETRY_AFTER_SECONDS,
                    "reason": "rate_limit",
                }
            if "abort" in text_lower:
                return None

        return None

    # =========================================================================
    # ENVELOPE + PAYLOAD HELPERS
    # =========================================================================

    @staticmethod
    def _build_payload(
        model_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Filter ``kwargs`` to Perchai-supported params and shape the inner ``request``.

        Mirrors DeepSeek's ``_build_payload``:
        - filter to ``SUPPORTED_PARAMS``
        - convert ``max_completion_tokens`` -> ``max_tokens`` when present
        - strip ``stream_options`` for non-stream requests
        - merge ``extra_body`` (if dict) onto the payload
        """
        payload: Dict[str, Any] = {
            key: value
            for key, value in kwargs.items()
            if key in SUPPORTED_PARAMS
        }
        payload["model"] = model_name

        if (
            "max_completion_tokens" in kwargs
            and "max_tokens" not in payload
        ):
            payload["max_tokens"] = kwargs["max_completion_tokens"]

        if not payload.get("stream"):
            payload.pop("stream_options", None)

        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        return payload

    @staticmethod
    def _build_envelope(
        model: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Wrap the inner ``request`` payload in a Perchai envelope.

        The envelope shape is taken from bundle grep evidence of the CLI's
        ``cKe`` build function::

            {
              "request": {...kwargs...},
              "lane": "chat",
              "preferredModelId": "<model>",
              "avoidModelIds": [],
              "attribution": null,
              "clientSurface": "cli",
              "promoOverflowAccepted": false
            }
        """
        stripped = model.split("/", 1)[1] if "/" in model else model
        return {
            "request": dict(payload),
            "runId": None,
            "lane": DEFAULT_LANE,
            "preferredModelId": stripped,
            "avoidModelIds": [],
            "attribution": None,
            "clientSurface": "cli",
            "promoOverflowAccepted": False,
        }

    @staticmethod
    def _parse_sse_line(
        line: str, model: str = "perchai"
    ) -> Optional[litellm.ModelResponseStream]:
        """Parse a raw SSE ``data:`` line into a ``ModelResponseStream``.

        Accepts the unparsed line (after ``aiter_lines`` yields it) and
        handles every defensive case before delegating to event dispatch:

        - non-``data:`` prefix -> ``None``
        - empty payload after stripping -> ``None``
        - ``[DONE]`` sentinel -> ``None`` (caller decides whether to
          synthesize a final stop chunk)
        - malformed JSON -> debug log + ``None``
        - non-dict JSON payload -> debug log + ``None``
        - unknown event type -> debug log + ``None``

        Per bundle grep evidence (``cKe`` SSE parser in
        ``/tmp/opencode/perchai-spike/package/dist/perch.mjs``), the raw
        wire format from ``/api/perch-terminal/model-call`` uses
        ``answer_delta`` for text content.  The CLI's higher-level ``bVe``
        orchestrator normalizes this to ``text_delta`` for its internal
        consumer callback.  We sit at the raw SSE layer, so we accept
        ``answer_delta`` as primary and keep ``text_delta`` as a defensive
        fallback in case a future server variant normalizes upstream.

        ``answer_delta`` / ``text_delta`` -> ``delta.content`` (rstripped)
        ``reasoning_delta`` -> ``delta.reasoning_content`` (rstripped)
        ``tool_call_delta`` / ``tool_use_end`` -> ``None`` (v1 strips tool use)
        events with ``finishReason`` -> yield with the given reason
        Anything else -> ``None`` (debug-logged)
        """
        if not isinstance(line, str) or not line.startswith("data:"):
            return None

        data_str = line[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            return None

        try:
            parsed_event = json.loads(data_str)
        except json.JSONDecodeError:
            lib_logger.debug(
                "Perchai stream: could not decode SSE line, skipping "
                f"(first 80 chars): {line[:80]!r}"
            )
            return None

        if not isinstance(parsed_event, dict):
            lib_logger.debug(
                f"Perchai stream: non-dict SSE payload, skipping: {parsed_event!r}"
            )
            return None

        event_type = parsed_event.get("type") or parsed_event.get("event") or ""

        if event_type in ("tool_call_delta", "tool_use_end"):
            lib_logger.debug(
                f"Perchai stream: stripping {event_type!r} event "
                "(tool calls not supported in v1)"
            )
            return None

        if event_type in ("answer_delta", "text_delta"):
            text = (parsed_event.get("text") or "").rstrip()
            return litellm.ModelResponseStream(
                id=f"chatcmpl-perchai-stream-{int(time.time())}",
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            )

        if event_type == "reasoning_delta":
            text = (parsed_event.get("text") or "").rstrip()
            return litellm.ModelResponseStream(
                id=f"chatcmpl-perchai-stream-{int(time.time())}",
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": 0,
                        "delta": {"reasoning_content": text},
                        "finish_reason": None,
                    }
                ],
            )

        finish_reason = parsed_event.get("finishReason") or parsed_event.get(
            "finish_reason"
        )
        if finish_reason:
            return litellm.ModelResponseStream(
                id=f"chatcmpl-perchai-stream-{int(time.time())}",
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            )

        lib_logger.debug(
            f"Perchai stream: ignoring unknown event type {event_type!r}"
        )
        return None

    async def _raise_for_status(
        self, response: httpx.Response, model: str
    ) -> None:
        """Translate non-OK responses into litellm exceptions.

        - 429 -> ``litellm.exceptions.RateLimitError``
        - 401 -> ``litellm.exceptions.AuthenticationError``
        - 400 -> ``litellm.exceptions.BadRequestError``
        - other non-OK -> ``httpx.HTTPStatusError``
        - structured quota info from ``parse_quota_error`` is embedded
          into the exception message where available.
        """
        if response.status_code < 400:
            return

        from litellm.exceptions import (
            AuthenticationError as LitellmAuthenticationError,
            BadRequestError as LitellmBadRequestError,
            RateLimitError as LitellmRateLimitError,
        )

        content = await response.aread()
        error_text = (
            content.decode("utf-8", errors="replace") if content else ""
        )

        quota_info = PerchaiProvider.parse_quota_error(
            Exception(error_text), error_text
        )
        suffix = ""
        if quota_info:
            reason = quota_info.get("reason")
            retry_after = quota_info.get("retry_after")
            suffix = f" [perchai-quota: reason={reason}, retry_after={retry_after}]"

        if response.status_code == 429:
            raise LitellmRateLimitError(
                f"Perchai rate limit exceeded: {error_text}{suffix}",
                llm_provider="perchai",
                model=model,
                response=response,
            )
        if response.status_code == 401:
            raise LitellmAuthenticationError(
                f"Perchai authentication failed: {error_text}{suffix}",
                llm_provider="perchai",
                model=model,
                response=response,
            )
        if response.status_code == 400:
            raise LitellmBadRequestError(
                f"Perchai bad request: {error_text}{suffix}",
                llm_provider="perchai",
                model=model,
                response=response,
            )

        raise httpx.HTTPStatusError(
            f"Perchai HTTP {response.status_code}: {error_text}{suffix}",
            request=response.request,
            response=response,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _cache_key_for(self, api_key: str) -> str:
        if not api_key:
            return ""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _redact_token(token: Optional[str]) -> str:
        """Return a safely-redacted form of an auth token for debug logs.

        Format: first 8 chars + ``...`` (e.g. ``abc12345...``).  Use this
        whenever you would otherwise log a raw Bearer token, refresh
        token, or session id.  Empty input returns ``<empty>``; very
        short input returns ``<redacted>`` so we never leak even the
        first 8 chars of a short secret.
        """
        if not token:
            return "<empty>"
        if len(token) <= 8:
            return "<redacted>"
        return f"{token[:8]}..."

    def _model_cache_is_valid(self, token_key: str) -> bool:
        if token_key not in self._model_cache:
            return False
        timestamp = self._model_cache_timestamps.get(token_key)
        if timestamp is None:
            return False
        return (time.time() - timestamp) < MODEL_CACHE_TTL_SECONDS

    def _resolve_app_url(self) -> str:
        from .perchai_auth_base import PerchaiAuthBase

        return PerchaiAuthBase().get_app_url()

    def _resolve_session_token(self) -> str:
        """Resolve the current access token from the session file.

        Used when the executor passes an empty ``credential_identifier``
        (e.g. during fallback or background refresh).  Raises
        ``PerchaiAuthError`` if no session is loaded - the rotator treats
        that as a credential failure and rotates.
        """
        from .perchai_auth_base import PerchaiAuthBase

        session = PerchaiAuthBase().load_session()
        token = session.get("accessToken")
        if not token:
            from .perchai_auth_base import PerchaiAuthError

            raise PerchaiAuthError(
                "Perchai session has no accessToken. "
                "Run `perch login` to re-authenticate."
            )
        return token

    def _extract_model_ids(self, payload: Any) -> List[str]:
        if not isinstance(payload, dict):
            return []

        for path in _PERCHAI_MODEL_FIELD_PATHS:
            models = self._dig(payload, path)
            ids = self._coerce_model_ids(models)
            if ids:
                return ids
        return []

    @staticmethod
    def _dig(payload: Dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
        return current

    @staticmethod
    def _coerce_model_ids(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []

        if not isinstance(value, list):
            return []

        ids: List[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    ids.append(stripped)
            elif isinstance(item, dict):
                candidate = (
                    item.get("id")
                    or item.get("modelId")
                    or item.get("model")
                )
                if isinstance(candidate, str) and candidate.strip():
                    ids.append(candidate.strip())
        return ids
