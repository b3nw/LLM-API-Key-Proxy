# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict, Union, final, override

import httpx
import litellm
from litellm.exceptions import APIError as LitellmAPIError
from litellm.types.utils import ChatCompletionMessageToolCall, Function

from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
from .provider_interface import ProviderInterface
from .utilities.perchai_quota_tracker import PerchaiQuotaTracker

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


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


class PerchaiRequestEnvelope(TypedDict, total=False):
    request: Dict[str, Any]
    lane: str
    preferredModelId: Optional[str]
    manualModelOptionId: Optional[str]
    avoidModelIds: List[str]
    attribution: Optional[Any]
    clientSurface: str
    promoOverflowAccepted: bool
    runId: Optional[str]


class PerchaiStreamEvent(TypedDict, total=False):
    type: str
    text: Optional[str]
    delta: Optional[str]
    runId: Optional[str]
    finishReason: Optional[str]
    usage: Optional[Dict[str, Any]]
    error: Optional[str]
    toolCall: Optional[Dict[str, Any]]


class PerchaiErrorResponse(TypedDict, total=False):
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
    provider_env_name: str = "perchai"

    skip_cost_calculation: bool = True

    default_rotation_mode: str = "sequential"

    def __init__(self) -> None:
        self._balance_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = int(
            os.environ.get("PERCHAI_QUOTA_REFRESH_INTERVAL", "300")
        )

        self._model_cache: Dict[str, List[str]] = {}
        self._model_cache_timestamps: Dict[str, float] = {}

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

    @override
    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs: Any
    ) -> Union[
        litellm.ModelResponse,
        AsyncGenerator[litellm.ModelResponseStream, None],
    ]:
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
                build_headers=_headers,
                token=token,
                payload=payload,
                model=raw_model,
                file_logger=file_logger,
                auth_base_cls=PerchaiAuthBase,
                auth_error_cls=LitellmAuthenticationError,
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

        # toolCalls[] is the canonical tool-call source; content[].tool_use
        # blocks duplicate the same data and are skipped during text accumulation.
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
                elif block_type in ("tool_use", "tool_call"):
                    continue
                elif block_type in ("text",) or not block_type:
                    parts_text.append(block_text)
            content_text = "".join(parts_text)
            if not reasoning_text:
                reasoning_text = "".join(parts_reasoning)

        # Tool calls: perchai may return arguments as dict or string; litellm
# requires a JSON-encoded string.
        raw_tool_calls = response_data.get("toolCalls")
        tool_calls_list: Optional[List[ChatCompletionMessageToolCall]] = None
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            tool_calls_list = []
            for tc in raw_tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                tc_name = tc.get("name")
                tc_arguments = tc.get("arguments", {})
                if not tc_id or not tc_name:
                    lib_logger.debug(
                        "Perchai non-stream: skipping malformed toolCall "
                        f"missing id or name: {tc!r}"
                    )
                    continue
                if isinstance(tc_arguments, str):
                    arguments_str = tc_arguments
                elif isinstance(tc_arguments, dict):
                    arguments_str = json.dumps(tc_arguments, ensure_ascii=False)
                else:
                    arguments_str = json.dumps({}, ensure_ascii=False)
                tool_calls_list.append(
                    ChatCompletionMessageToolCall(
                        id=tc_id,
                        type="function",
                        function=Function(
                            name=tc_name,
                            arguments=arguments_str,
                        ),
                    )
                )

        has_tool_calls = bool(tool_calls_list)
        finish_reason = "tool_calls" if has_tool_calls else "stop"

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
        if tool_calls_list:
            message_kwargs["tool_calls"] = tool_calls_list

        choices_list = [
            litellm.Choices(
                finish_reason=finish_reason,
                index=0,
                message=litellm.Message(**message_kwargs),
            )
        ]

        model_response = litellm.ModelResponse(
            id=response_data.get("id") or str(uuid.uuid4()),
            choices=choices_list,
            # Preserve the caller's model id; perchai's own "model" field
            # is the upstream it routed through and would silently overwrite it.
            model=model,
            usage=litellm.Usage(**usage_kwargs) if usage_kwargs else None,
        )

        file_logger.log_final_response(response_data)
        return model_response

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        build_headers: Any,
        token: str,
        payload: Dict[str, Any],
        model: str,
        file_logger: ProviderLogger,
        auth_base_cls: Any = None,
        auth_error_cls: Any = None,
    ) -> AsyncGenerator[litellm.ModelResponseStream, None]:
        envelope = self._build_envelope(model=model, payload=payload)

        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")

        saw_done = False
        saw_tool_call = False
        stream_id = f"chatcmpl-perchai-stream-{int(time.time())}"

        for attempt in range(2):
            stream_headers = dict(build_headers(token))
            stream_headers["Content-Type"] = "application/json; charset=utf-8"

            ctx = client.stream(
                "POST",
                url,
                headers=stream_headers,
                content=body,
                timeout=TimeoutConfig.streaming(),
            )
            response = await ctx.__aenter__()

            if (
                response.status_code == 401
                and attempt == 0
                and auth_base_cls is not None
            ):
                await response.aread()
                await ctx.__aexit__(None, None, None)
                auth = auth_base_cls()
                token = await auth.refresh_on_401(client, token)
                continue

            try:
                await self._raise_for_status(response, model)

                async for line in response.aiter_lines():
                    file_logger.log_response_chunk(line)

                    if not line.startswith("data:"):
                        continue

                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        saw_done = True
                        # tool_use_end already yielded the terminating
                        # tool_calls chunk; skip emitting a redundant stop.
                        if saw_tool_call:
                            return
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
                        # _parse_sse_line emits finish_reason="tool_calls"
                        # from tool_use_end events; track it to suppress
                        # the redundant stop chunk at [DONE].
                        chunk_finish_reason: Optional[str] = None
                        try:
                            chunk_choices = stream_chunk.choices
                            if chunk_choices:
                                first_choice = chunk_choices[0]
                                if isinstance(first_choice, dict):
                                    chunk_finish_reason = first_choice.get(
                                        "finish_reason"
                                    )
                                else:
                                    chunk_finish_reason = getattr(
                                        first_choice, "finish_reason", None
                                    )
                        except Exception:
                            chunk_finish_reason = None
                        if chunk_finish_reason == "tool_calls":
                            saw_tool_call = True
                        yield stream_chunk
            finally:
                await ctx.__aexit__(None, None, None)
            break

        if not saw_done:
            lib_logger.debug(
                "Perchai stream ended without [DONE] sentinel; "
                "synthesizing final stop chunk"
            )
            if saw_tool_call:
                return
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

    @staticmethod
    @override
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
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
            # Synthetic test errors may embed the status code as the message
            # (e.g. Exception("429")); only treat quota-relevant codes as such.
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

    @staticmethod
    def _build_payload(
        model_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        stripped = model.split("/", 1)[1] if "/" in model else model
        return {
            "request": dict(payload),
            "runId": None,
            "lane": DEFAULT_LANE,
            "preferredModelId": None,
            "manualModelOptionId": stripped,
            "avoidModelIds": [],
            "attribution": None,
            "clientSurface": "cli",
            "promoOverflowAccepted": False,
        }

    @staticmethod
    def _parse_sse_line(
        line: str, model: str = "perchai"
    ) -> Optional[litellm.ModelResponseStream]:
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

        if event_type == "tool_call_delta":
            # Defensive: perchai currently returns toolCalls[] only in the
            # non-stream response; this handles a hypothetical streaming variant.
            tc_id = parsed_event.get("id")
            tc_name = parsed_event.get("name")
            tc_arguments = parsed_event.get("arguments")
            tc_index = parsed_event.get("index", 0)
            if isinstance(tc_arguments, dict):
                arguments_str = json.dumps(tc_arguments, ensure_ascii=False)
            elif isinstance(tc_arguments, str):
                arguments_str = tc_arguments
            else:
                arguments_str = ""
            function_delta: Dict[str, Any] = {}
            if tc_name is not None:
                function_delta["name"] = tc_name
            if tc_arguments is not None:
                function_delta["arguments"] = arguments_str
            tool_call_delta: Dict[str, Any] = {
                "index": tc_index,
                "type": "function",
            }
            if tc_id is not None:
                tool_call_delta["id"] = tc_id
            tool_call_delta["function"] = function_delta
            return litellm.ModelResponseStream(
                id=f"chatcmpl-perchai-stream-{int(time.time())}",
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": 0,
                        "delta": {"tool_calls": [tool_call_delta]},
                        "finish_reason": None,
                    }
                ],
            )

        if event_type == "tool_use_end":
            # Final chunk of a tool-calling turn; empty delta + finish_reason
            # signals tool dispatch to downstream consumers.
            return litellm.ModelResponseStream(
                id=f"chatcmpl-perchai-stream-{int(time.time())}",
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            )

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

    def _cache_key_for(self, api_key: str) -> str:
        if not api_key:
            return ""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _redact_token(token: Optional[str]) -> str:
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
