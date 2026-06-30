import json
from proxy_app.responses_compat import (
    convert_responses_input_to_messages,
    convert_tools_from_responses_format,
    convert_responses_request_to_chat,
    convert_chat_response_to_responses,
    ResponsesStreamConverter,
    _flatten_content,
    _convert_input_content,
    _flatten_output_content,
)

class MockChatResponse:
    def __init__(self, data):
        self.data = data
    def model_dump(self):
        return self.data

def test_convert_responses_input_to_messages_string():
    result = convert_responses_input_to_messages("Hello!", instructions="Be helpful.")
    assert result == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello!"}
    ]

def test_convert_responses_input_to_messages_easy_format():
    input_data = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "tool_output"},
    ]
    result = convert_responses_input_to_messages(input_data)
    assert result == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "tool_output"},
    ]

def test_convert_responses_input_to_messages_typed_format():
    input_data = [
        {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "sys"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call", "call_id": "call_2", "name": "get_time", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": {"result": "sunny"}},
    ]
    result = convert_responses_input_to_messages(input_data)
    assert result == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result": "sunny"}'},
    ]

def test_convert_tools_from_responses_format():
    tools = [
        {"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "function": {"name": "get_time", "description": "Get time"}},
        {"type": "web_search"},
        {"type": "custom_tool", "function": {"name": "custom", "description": "Custom"}},
    ]
    result = convert_tools_from_responses_format(tools)
    assert result == [
        {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_time", "description": "Get time"}},
        {"type": "custom_tool", "function": {"name": "custom", "description": "Custom"}},
    ]

    assert convert_tools_from_responses_format(None) is None

def test_convert_responses_request_to_chat():
    request_data = {
        "model": "test-model",
        "input": "hello",
        "instructions": "sys",
        "tools": [{"type": "function", "name": "f"}],
        "tool_choice": {"type": "function", "name": "f"},
        "parallel_tool_calls": True,
        "max_output_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "reasoning": {"effort": "high"},
        "text": {"format": {"type": "json_schema", "name": "schema", "strict": True, "schema": {}}},
        "service_tier": "auto",
        "user": "user1",
    }
    result = convert_responses_request_to_chat(request_data)
    assert result == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"}
        ],
        "stream": False,
        "tools": [{"type": "function", "function": {"name": "f", "description": "", "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
        "parallel_tool_calls": True,
        "max_completion_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "reasoning_effort": "high",
        "response_format": {"type": "json_schema", "json_schema": {"name": "schema", "strict": True, "schema": {}}},
        "service_tier": "auto",
        "user": "user1",
    }

    # Test fallback max_tokens and string tool choice
    request_data_2 = {
        "model": "test-model",
        "input": "hi",
        "max_tokens": 200,
        "tool_choice": "auto",
        "text": {"format": {"type": "json_object"}}
    }
    result_2 = convert_responses_request_to_chat(request_data_2)
    assert result_2["max_completion_tokens"] == 200
    assert result_2["tool_choice"] == "auto"
    assert result_2["response_format"] == {"type": "json_object"}

def test_convert_chat_response_to_responses():
    cc_response_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello world",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    cc_response = MockChatResponse(cc_response_data)
    request_data = {"model": "test-model", "metadata": {"key": "val"}}

    result = convert_chat_response_to_responses(cc_response, "resp_123", request_data)
    assert result["id"] == "resp_123"
    assert result["object"] == "response"
    assert result["created_at"] == 1234567890
    assert result["model"] == "test-model"
    assert result["status"] == "completed"
    assert len(result["output"]) == 2

    # Tool call output item
    assert result["output"][0]["type"] == "function_call"
    assert result["output"][0]["call_id"] == "call_1"
    assert result["output"][0]["name"] == "f"
    assert result["output"][0]["arguments"] == "{}"

    # Message output item
    assert result["output"][1]["type"] == "message"
    assert result["output"][1]["role"] == "assistant"
    assert result["output"][1]["content"] == [{"type": "output_text", "text": "hello world"}]

    assert result["usage"] == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    assert result["metadata"] == {"key": "val"}

def test_responses_stream_converter():
    converter = ResponsesStreamConverter("resp_123", "test-model")

    # Test creation chunk
    chunk1 = {"model": "test-model", "choices": [], "usage": None}
    events1 = converter.convert_chunk(f"data: {json.dumps(chunk1)}")
    assert "response.created" in events1
    assert "response.in_progress" in events1

    # Test reasoning delta
    chunk2 = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
    events2 = converter.convert_chunk(f"data: {json.dumps(chunk2)}")
    assert "response.reasoning_summary_text.delta" in events2
    assert "thinking..." in events2

    # Test content delta
    chunk3 = {"choices": [{"delta": {"content": "hello"}}]}
    events3 = converter.convert_chunk(f"data: {json.dumps(chunk3)}")
    assert "response.output_item.added" in events3
    assert "response.content_part.added" in events3
    assert "response.output_text.delta" in events3
    assert "hello" in events3

    # Test content delta 2
    chunk4 = {"choices": [{"delta": {"content": " world"}}]}
    events4 = converter.convert_chunk(f"data: {json.dumps(chunk4)}")
    assert "response.output_item.added" not in events4
    assert "response.output_text.delta" in events4
    assert " world" in events4

    # Test tool call delta
    chunk5 = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": "{"}}]}}]}
    events5 = converter.convert_chunk(f"data: {json.dumps(chunk5)}")
    assert "response.output_item.added" in events5
    assert "response.function_call_arguments.delta" in events5

    # Test finalization
    events_final = converter.convert_chunk("data: [DONE]")
    assert "response.function_call_arguments.done" in events_final
    assert "response.output_text.done" in events_final
    assert "response.content_part.done" in events_final
    assert "response.completed" in events_final

def test_flatten_content():
    assert _flatten_content("hello") == "hello"
    assert _flatten_content([{"type": "text", "text": "hi"}, "there"]) == "hi\nthere"
    assert _flatten_content(None) == ""
    assert _flatten_content(123) == "123"

def test_convert_input_content():
    assert _convert_input_content("hello") == "hello"
    assert _convert_input_content(None) == ""
    assert _convert_input_content([{"type": "input_text", "text": "hi"}, "there"]) == "hi\nthere"

    # Multimodal
    multimodal_input = [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": "url1"}]
    result = _convert_input_content(multimodal_input)
    assert result == [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "url1"}}]

def test_flatten_output_content():
    assert _flatten_output_content("hello") == "hello"
    assert _flatten_output_content([{"type": "output_text", "text": "hi"}, "there"]) == "hi\nthere"
    assert _flatten_output_content(None) == ""
