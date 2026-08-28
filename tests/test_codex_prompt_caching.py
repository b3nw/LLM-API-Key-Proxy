"""
Unit tests for Codex prompt caching fixes.

Verifies:
1. <think> and <thought> reasoning tags are stripped from assistant history.
2. Assistant input items include "type": "message".
3. Missing tool call IDs produce deterministic IDs rather than random UUIDs.
4. convert_responses_request_to_chat preserves prompt_cache_key and session_id.
5. executor._prepare_request_kwargs propagates context.session_id.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from rotator_library.providers.codex_provider import (
    _strip_think_tags,
    _convert_messages_to_responses_input,
)
from proxy_app.responses_compat import convert_responses_request_to_chat


def test_strip_think_tags_basic():
    text = "<think>Let me ponder this.</think>Hello world!"
    assert _strip_think_tags(text) == "Hello world!"


def test_strip_think_tags_multiline():
    text = "<think>\nStep 1: analyze\nStep 2: solve\n</think>\nHere is the answer."
    assert _strip_think_tags(text) == "Here is the answer."


def test_strip_thought_tags():
    text = "<thought>Internal reasoning</thought>Direct response."
    assert _strip_think_tags(text) == "Direct response."


def test_strip_think_tags_case_insensitive():
    text = "<THINK>Thinking deeply</THINK>Output."
    assert _strip_think_tags(text) == "Output."


def test_strip_think_tags_no_tags():
    text = "Just a regular assistant response."
    assert _strip_think_tags(text) == "Just a regular assistant response."


def test_strip_think_tags_only_thinking():
    text = "<think>Only internal reasoning, no text output</think>"
    assert _strip_think_tags(text) == ""


def test_convert_messages_strips_think_tags_from_assistant():
    messages = [
        {"role": "user", "content": "How do I reverse a string?"},
        {
            "role": "assistant",
            "content": "<think>Slicing is easiest</think>Use s[::-1]",
        },
        {"role": "user", "content": "Can you give another way?"},
    ]

    input_items, _ = _convert_messages_to_responses_input(messages)

    assert len(input_items) == 3
    # Check user message 1
    assert input_items[0]["type"] == "message"
    assert input_items[0]["role"] == "user"
    assert input_items[0]["content"] == [{"type": "input_text", "text": "How do I reverse a string?"}]

    # Check assistant message has type: message and stripped think tags
    assert input_items[1]["type"] == "message"
    assert input_items[1]["role"] == "assistant"
    assert input_items[1]["content"] == [{"type": "output_text", "text": "Use s[::-1]"}]

    # Check user message 2
    assert input_items[2]["type"] == "message"
    assert input_items[2]["role"] == "user"
    assert input_items[2]["content"] == [{"type": "input_text", "text": "Can you give another way?"}]


def test_convert_messages_handles_none_function_in_tool_calls():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": None}
            ],
        }
    ]
    input_items, _ = _convert_messages_to_responses_input(messages)
    assert len(input_items) == 1
    assert input_items[0]["type"] == "function_call"
    assert input_items[0]["name"] == ""
    assert input_items[0]["arguments"] == "{}"


def test_convert_messages_multipart_assistant_strips_think_tags():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>pondering</think>Part 1"},
                {"type": "output_text", "text": "Part 2"},
            ],
        }
    ]

    input_items, _ = _convert_messages_to_responses_input(messages)

    assert len(input_items) == 1
    assert input_items[0]["type"] == "message"
    assert input_items[0]["role"] == "assistant"
    assert input_items[0]["content"] == [
        {"type": "output_text", "text": "Part 1"},
        {"type": "output_text", "text": "Part 2"},
    ]


def test_convert_messages_deterministic_tool_call_ids():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "foo.py"}'},
                    # Notice: "id" is intentionally missing
                }
            ],
        }
    ]

    # Convert twice and ensure call_id is identical (deterministic)
    input_items_1, _ = _convert_messages_to_responses_input(messages)
    input_items_2, _ = _convert_messages_to_responses_input(messages)

    assert len(input_items_1) == 1
    assert len(input_items_2) == 1
    assert input_items_1[0]["type"] == "function_call"
    assert input_items_2[0]["type"] == "function_call"
    assert input_items_1[0]["call_id"] == input_items_2[0]["call_id"]
    assert input_items_1[0]["call_id"].startswith("call_gen_")


def test_responses_compat_preserves_prompt_cache_key_and_session_id():
    request_data = {
        "model": "gpt-5.3-codex",
        "input": "Hello",
        "prompt_cache_key": "my-stable-cache-key",
        "session_id": "session-12345",
    }

    cc_request = convert_responses_request_to_chat(request_data)

    assert cc_request["prompt_cache_key"] == "my-stable-cache-key"
    assert cc_request["session_id"] == "session-12345"


@pytest.mark.asyncio
async def test_executor_forwards_session_id_to_kwargs():
    from rotator_library.client.executor import RequestExecutor

    executor = RequestExecutor(
        usage_managers={},
        cooldown_manager=MagicMock(),
        credential_filter=MagicMock(),
        provider_transforms=MagicMock(),
        provider_plugins={},
        http_client=MagicMock(),
    )
    # Mock transforms.apply to return kwargs as-is
    executor._transforms.apply = AsyncMock(
        side_effect=lambda provider, model, cred, kwargs, **kw: kwargs
    )

    context = MagicMock()
    context.kwargs = {"model": "gpt-5.3-codex", "messages": []}
    context.provider_config = {}
    context.session_id = "inferred-session-abc"
    context.transaction_logger = None

    prepared = await executor._prepare_request_kwargs(
        "codex", "gpt-5.3-codex", "dummy-cred", context
    )

    assert prepared.get("session_id") == "inferred-session-abc"


def test_strip_think_tags_preserves_mid_sentence_tags():
    text = "Here is how models work: they emit <think> blocks. Also see docs."
    assert _strip_think_tags(text) == text


def test_strip_think_tags_preserves_exact_whitespace():
    text = "<think>reasoning</think>\nHere is the answer.\n"
    assert _strip_think_tags(text) == "Here is the answer.\n"


def test_strip_think_tags_non_string_safety():
    assert _strip_think_tags(None) == ""
    assert _strip_think_tags(123) == ""
    assert _strip_think_tags({"a": 1}) == ""


def test_convert_messages_tool_call_ids_no_collision_within_request():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "file1.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{}"},
                }
            ],
        },
    ]

    input_items, _ = _convert_messages_to_responses_input(messages)
    tool_items = [item for item in input_items if item.get("type") == "function_call"]

    assert len(tool_items) == 2
    assert tool_items[0]["call_id"] != tool_items[1]["call_id"]
    assert tool_items[0]["call_id"].startswith("call_gen_0_")
    assert tool_items[1]["call_id"].startswith("call_gen_1_")


def test_convert_messages_tool_call_ids_stable_across_turns():
    turn1_messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{}"},
                }
            ],
        }
    ]

    turn2_messages = turn1_messages + [
        {"role": "tool", "content": "file1.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "cat", "arguments": '{"file": "file1.txt"}'},
                }
            ],
        },
    ]

    input_items_t1, _ = _convert_messages_to_responses_input(turn1_messages)
    input_items_t2, _ = _convert_messages_to_responses_input(turn2_messages)

    # Prefix tool call in Turn 1 must have the EXACT same call_id in Turn 2
    assert input_items_t1[0]["call_id"] == input_items_t2[0]["call_id"]

