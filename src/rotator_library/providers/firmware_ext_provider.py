# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/firmware_ext_provider.py

"""
Firmware Extension Provider — uses the Chrome Extension's internal API.

Reverse-engineered from the Firmware Chrome Extension (kilnglameccalibejghopglodbidhkmj).
Uses cookie-based JWT auth, Vercel AI SDK v6 message format, and custom SSE translation.

This provider coexists alongside the standard `firmware` provider which uses the
OpenAI-compatible API with Bearer token auth.

Environment variables:
    FIRMWARE_EXT_API_KEY_1: JWT token from firmware-token browser cookie
    FIRMWARE_EXT_API_KEY_2: Optional second JWT for credential rotation
"""

import copy
import json
import time
import random
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, AsyncGenerator, Union, Optional

import httpx
import litellm
from litellm.exceptions import RateLimitError

from .provider_interface import ProviderInterface, QuotaGroupMap
from ..transaction_logger import ProviderLogger
from ..timeout_config import TimeoutConfig

lib_logger = logging.getLogger("rotator_library")

FIRMWARE_STREAM_URL = "https://app.firmware.ai/api/chats/stream"
FIRMWARE_ORIGIN = "chrome-extension://kilnglameccalibejghopglodbidhkmj"

AVAILABLE_MODELS = [
    # Firmware proprietary
    "firmware-sidekick-pro",
    "firmware-sidekick-fast",
    "firmware-juris",
    "firmware-juris-lite",
    # ChatGPT
    "chatgpt-auto",
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    # Claude (via AWS Bedrock)
    "claude-auto",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-opus-4-5-20251101-v1:0",
    # Gemini
    "gemini-auto",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]


class FirmwareExtProvider(ProviderInterface):
    """
    Provider for Firmware.ai via the Chrome Extension's internal API.

    Uses cookie-based JWT authentication and Vercel AI SDK v6 message format.
    All models share a single per-credential rate limit pool.
    """

    skip_cost_calculation = True

    model_quota_groups: QuotaGroupMap = {
        "firmware_ext_global": ["firmware_ext/_quota"],
    }

    def _generate_object_id(self) -> str:
        """Generate a MongoDB ObjectId (24 hex chars: 8-char timestamp + 16-char random)."""
        ts = format(int(time.time()), "08x")
        rand = "".join(format(random.randint(0, 15), "x") for _ in range(16))
        return ts + rand

    def _get_request_headers(self, token: str) -> Dict[str, str]:
        """Build headers required by the extension API."""
        return {
            "Cookie": f"firmware-token={token}",
            "Origin": FIRMWARE_ORIGIN,
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Cache-Control": "no-cache",
        }

    def _transform_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI message format to Firmware Vercel AI SDK v6 parts format.

        Handles:
        - System messages: prepended to first user message's parts
        - User messages: text and image_url content types
        - Assistant messages: wrapped with step-start and state:done
        """
        messages = copy.deepcopy(messages)
        firmware_messages = []
        system_text = None

        # Extract system message if present
        if messages and messages[0].get("role") == "system":
            system_content = messages.pop(0).get("content", "")
            if system_content:
                system_text = system_content

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                parts = []

                # Prepend system text to first user message
                if system_text is not None:
                    parts.append({"type": "text", "text": system_text})
                    system_text = None  # Only prepend once

                if isinstance(content, str):
                    if content:
                        parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                parts.append({"type": "text", "text": text})
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                try:
                                    header, data = image_url.split(",", 1)
                                    media_type = header.split(":")[1].split(";")[0]
                                    parts.append({
                                        "type": "media",
                                        "data": data,
                                        "mediaType": media_type,
                                    })
                                except Exception as e:
                                    lib_logger.warning(
                                        f"Failed to parse image data URL: {e}"
                                    )
                            else:
                                lib_logger.warning(
                                    f"Non-data-URL images not supported: {image_url[:50]}..."
                                )

                firmware_messages.append({
                    "id": self._generate_object_id(),
                    "role": "user",
                    "parts": parts,
                    "metadata": {
                        "createdAt": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.000Z"
                        )
                    },
                })

            elif role == "assistant":
                parts = [{"type": "step-start"}]
                if isinstance(content, str) and content:
                    parts.append({
                        "type": "text",
                        "text": content,
                        "state": "done",
                    })
                firmware_messages.append({
                    "id": self._generate_object_id(),
                    "role": "assistant",
                    "parts": parts,
                })

        # If system text was never consumed (no user messages), add a user message
        if system_text is not None:
            firmware_messages.insert(0, {
                "id": self._generate_object_id(),
                "role": "user",
                "parts": [{"type": "text", "text": system_text}],
                "metadata": {
                    "createdAt": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z"
                    )
                },
            })

        return firmware_messages

    def has_custom_logic(self) -> bool:
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Handle completion calls via the Firmware Extension API."""
        token = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)

        model = kwargs.get("model", "")
        # Strip provider prefix: "firmware_ext/gpt-5.2" -> "gpt-5.2"
        model_id = model.split("/")[-1] if "/" in model else model

        file_logger = ProviderLogger(transaction_context)

        # Transform messages
        transformed_messages = self._transform_messages(kwargs.get("messages", []))

        # Build request body
        request_body = {
            "id": self._generate_object_id(),
            "messages": transformed_messages,
            "config": 2,
            "modelId": model_id,
            "metadata": {
                "browser": {
                    "currentTabId": 0,
                    "pageUrl": "",
                    "totalTabs": 1,
                    "windowId": 0,
                    "pageTitle": "",
                    "openTabIds": [],
                },
                "planMode": False,
            },
        }

        headers = self._get_request_headers(token)
        file_logger.log_request(request_body)

        async def stream_handler():
            try:
                async with client.stream(
                    "POST",
                    FIRMWARE_STREAM_URL,
                    headers=headers,
                    json=request_body,
                    timeout=TimeoutConfig.streaming(),
                ) as response:
                    # Handle error responses
                    if response.status_code >= 400:
                        error_body_bytes = await response.aread()
                        error_body = error_body_bytes.decode("utf-8", errors="replace")
                        file_logger.log_error(
                            f"API error {response.status_code}: {error_body}"
                        )

                        if response.status_code == 429:
                            # Parse rate limit info
                            retry_after = None
                            try:
                                error_data = json.loads(error_body)
                                retry_after = error_data.get("retryAfterSeconds")
                            except (json.JSONDecodeError, TypeError):
                                pass

                            retry_info = (
                                f" (retry after {retry_after}s)" if retry_after else ""
                            )
                            lib_logger.debug(
                                f"Firmware Ext 429 rate limit: retry_after={retry_after}s"
                            )
                            raise RateLimitError(
                                message=f"Firmware Ext rate limit exceeded{retry_info} | {error_body}",
                                llm_provider="firmware_ext",
                                model=model,
                                response=response,
                            )

                        if response.status_code == 401:
                            raise litellm.exceptions.AuthenticationError(
                                message=f"Firmware Ext authentication failed (expired JWT?): {error_body}",
                                llm_provider="firmware_ext",
                                model=model,
                                response=response,
                            )

                        # Other errors
                        raise httpx.HTTPStatusError(
                            message=f"Firmware Ext API error {response.status_code}: {error_body}",
                            request=response.request,
                            response=response,
                        )

                    # Parse SSE stream
                    response_id = f"chatcmpl-fwext-{time.time()}"
                    sent_role = False

                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)

                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            lib_logger.warning(
                                f"Could not decode JSON from Firmware Ext: {line}"
                            )
                            continue

                        event_type = data.get("type")
                        delta = {}

                        if event_type == "text-start":
                            if not sent_role:
                                delta["role"] = "assistant"
                                sent_role = True
                            else:
                                continue

                        elif event_type == "text-delta":
                            if not sent_role:
                                delta["role"] = "assistant"
                                sent_role = True
                            delta["content"] = data.get("delta", "")

                        elif event_type == "reasoning-start":
                            # No-op, just marks start of reasoning
                            continue

                        elif event_type == "reasoning-delta":
                            if not sent_role:
                                delta["role"] = "assistant"
                                sent_role = True
                            delta["reasoning_content"] = data.get("delta", "")

                        elif event_type == "text-end":
                            continue

                        elif event_type == "finish":
                            # Final chunk with finish_reason
                            finish_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop",
                                    }
                                ],
                            }
                            yield litellm.ModelResponse(**finish_chunk)
                            continue

                        elif event_type == "error":
                            error_msg = data.get("error", "Unknown error")
                            lib_logger.error(
                                f"Firmware Ext stream error: {error_msg}"
                            )
                            file_logger.log_error(f"Stream error: {error_msg}")
                            raise litellm.exceptions.APIError(
                                message=f"Firmware Ext stream error: {error_msg}",
                                llm_provider="firmware_ext",
                                model=model,
                                status_code=500,
                            )

                        else:
                            # Unknown event type, skip
                            continue

                        if not delta:
                            continue

                        openai_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                }
                            ],
                        }
                        yield litellm.ModelResponse(**openai_chunk)

            except httpx.HTTPStatusError:
                raise
            except RateLimitError:
                raise
            except litellm.exceptions.AuthenticationError:
                raise
            except litellm.exceptions.APIError:
                raise
            except Exception as e:
                file_logger.log_error(f"Stream handler exception: {str(e)}")
                raise

        async def logging_stream_wrapper():
            """Wraps the stream to log the final reassembled response."""
            openai_chunks = []
            try:
                async for chunk in stream_handler():
                    openai_chunks.append(chunk)
                    yield chunk
            finally:
                if openai_chunks:
                    final_response = self._stream_to_completion_response(openai_chunks)
                    file_logger.log_final_response(final_response.dict())

        if kwargs.get("stream", False):
            return logging_stream_wrapper()
        else:
            # Non-streaming: buffer all chunks and reassemble
            chunks = [chunk async for chunk in logging_stream_wrapper()]
            return self._stream_to_completion_response(chunks)

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        """
        Reassemble streaming chunks into a complete non-streaming response.

        Reuses the proven pattern from GeminiCliProvider.
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        final_message = {"role": "assistant"}
        chunk_finish_reason = None
        first_chunk = chunks[0]

        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.get("delta", {})

            if "content" in delta and delta["content"] is not None:
                if "content" not in final_message:
                    final_message["content"] = ""
                final_message["content"] += delta["content"]

            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                if "reasoning_content" not in final_message:
                    final_message["reasoning_content"] = ""
                final_message["reasoning_content"] += delta["reasoning_content"]

            if choice.get("finish_reason"):
                chunk_finish_reason = choice["finish_reason"]

        # Ensure standard fields
        for field in ["content", "tool_calls", "function_call"]:
            if field not in final_message:
                final_message[field] = None

        finish_reason = chunk_finish_reason or "stop"

        final_choice = {
            "index": 0,
            "message": final_message,
            "finish_reason": finish_reason,
        }

        return litellm.ModelResponse(
            **{
                "id": first_chunk.id,
                "object": "chat.completion",
                "created": first_chunk.created,
                "model": first_chunk.model,
                "choices": [final_choice],
                "usage": None,
            }
        )

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Firmware 429 rate limit responses.

        Example error body:
        {
            "error": "Usage limit reached. Try again 2026-02-07T06:00:30.644Z",
            "resetAt": "2026-02-07T06:00:30.644Z",
            "retryAfterSeconds": 6664
        }
        """
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                try:
                    body = error.response.text
                except Exception:
                    pass
            if not body and hasattr(error, "body"):
                body = str(error.body)
            if not body and hasattr(error, "message"):
                body = str(error.message)
            if not body:
                body = str(error)

        if not body:
            return None

        result = {
            "retry_after": None,
            "reason": "RATE_LIMITED",
            "reset_timestamp": None,
            "quota_reset_timestamp": None,
        }

        try:
            data = json.loads(body)

            retry_after = data.get("retryAfterSeconds")
            if retry_after is not None:
                result["retry_after"] = int(retry_after)

            reset_at = data.get("resetAt")
            if reset_at:
                result["reset_timestamp"] = reset_at
                # Parse ISO timestamp to unix timestamp
                try:
                    dt = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
                    result["quota_reset_timestamp"] = dt.timestamp()
                except (ValueError, AttributeError):
                    pass

        except (json.JSONDecodeError, TypeError):
            pass

        if result["retry_after"] is None and result["quota_reset_timestamp"] is None:
            return None

        return result

    async def get_models(
        self, api_key: str, client: httpx.AsyncClient
    ) -> List[str]:
        """Return hardcoded list of available models prefixed with firmware_ext/."""
        return [f"firmware_ext/{model}" for model in AVAILABLE_MODELS]

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """All models share a single per-credential rate limit pool."""
        return "firmware_ext_global"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """Get all models in a quota group."""
        if group == "firmware_ext_global":
            return ["firmware_ext/_quota"]
        return []

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """Return 5-hour rolling window config matching Firmware's rate limit behavior."""
        return {
            "mode": "per_model",
            "window_seconds": 18000,  # 5 hours
            "field_name": "models",
        }
