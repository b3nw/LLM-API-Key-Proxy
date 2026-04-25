import pytest
from unittest.mock import MagicMock

from src.rotator_library.failure_logger import log_failure

def test_log_failure_success(mocker):
    # Mock dependencies
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance

    mock_main_lib_logger = mocker.patch("src.rotator_library.failure_logger.main_lib_logger")

    mock_get_error_tracker = mocker.patch("src.rotator_library.failure_logger.get_error_tracker")
    mock_error_tracker_instance = MagicMock()
    mock_get_error_tracker.return_value = mock_error_tracker_instance

    # Mock mask_credential so we can verify output easily
    mocker.patch("src.rotator_library.failure_logger.mask_credential", return_value="masked_key")

    # Call function
    error = ValueError("Something went wrong")
    log_failure(
        api_key="sk-123456",
        model="openai/gpt-4",
        attempt=2,
        error=error,
        request_headers={"X-Test": "1"},
    )

    # Verify detailed log
    mock_failure_logger_instance.error.assert_called_once()
    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]
    assert detailed_log_data["api_key_ending"] == "masked_key"
    assert detailed_log_data["model"] == "openai/gpt-4"
    assert detailed_log_data["attempt_number"] == 2
    assert detailed_log_data["error_type"] == "ValueError"
    assert detailed_log_data["error_message"] == "Something went wrong"
    assert detailed_log_data["request_headers"] == {"X-Test": "1"}

    # Verify summary log
    mock_main_lib_logger.error.assert_called_once()
    summary_msg = mock_main_lib_logger.error.call_args[0][0]
    assert "openai/gpt-4" in summary_msg
    assert "masked_key" in summary_msg
    assert "ValueError" in summary_msg

    # Verify tracker
    mock_error_tracker_instance.record_error.assert_called_once_with(
        provider="openai",
        model="openai/gpt-4",
        error_type="ValueError",
        error_message="Something went wrong",
        credential_masked="masked_key",
        attempt=2,
        status_code=None,
    )

def test_log_failure_raw_response_precedence(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance

    mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    error = Exception("General error")

    # Should use the explicitly provided raw_response_text
    log_failure(
        api_key="test-key",
        model="test-model",
        attempt=1,
        error=error,
        request_headers={},
        raw_response_text="explicit raw text"
    )

    mock_failure_logger_instance.error.assert_called_once()
    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]
    assert detailed_log_data["raw_response"] == "explicit raw text"

def test_log_failure_error_chain(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance

    mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    # Create a nested exception chain
    root_error = ValueError("root cause")
    intermediate_error = RuntimeError("intermediate")
    intermediate_error.__cause__ = root_error
    top_error = Exception("top level")
    top_error.__context__ = intermediate_error

    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=top_error,
        request_headers={}
    )

    mock_failure_logger_instance.error.assert_called_once()
    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]

    error_chain = detailed_log_data["error_chain"]
    assert len(error_chain) == 3
    assert error_chain[0]["type"] == "Exception"
    assert error_chain[0]["message"] == "top level"
    assert error_chain[1]["type"] == "RuntimeError"
    assert error_chain[1]["message"] == "intermediate"
    assert error_chain[2]["type"] == "ValueError"
    assert error_chain[2]["message"] == "root cause"

def test_log_failure_error_chain_circular(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance

    mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    e1 = Exception("1")
    e2 = Exception("2")
    e1.__cause__ = e2
    e2.__cause__ = e1  # Circular reference!

    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=e1,
        request_headers={}
    )

    mock_failure_logger_instance.error.assert_called_once()
    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]

    error_chain = detailed_log_data["error_chain"]
    # It should detect the cycle and break out
    assert len(error_chain) == 2

def test_log_failure_logger_exception_resilience(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    # Make the logger throw an OSError to test resilience
    mock_failure_logger_instance.error.side_effect = OSError("Disk full")
    mock_get_failure_logger.return_value = mock_failure_logger_instance

    mock_main_lib_logger = mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mock_get_error_tracker = mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    # This should not raise an exception
    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=Exception("test"),
        request_headers={}
    )

    # Main logger and tracker should still be called
    mock_main_lib_logger.error.assert_called_once()
    mock_get_error_tracker().record_error.assert_called_once()

def test_log_failure_tracker_exception_resilience(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_main_lib_logger = mocker.patch("src.rotator_library.failure_logger.main_lib_logger")

    mock_get_error_tracker = mocker.patch("src.rotator_library.failure_logger.get_error_tracker")
    mock_error_tracker_instance = MagicMock()
    # Make tracker throw exception
    mock_error_tracker_instance.record_error.side_effect = Exception("Tracker error")
    mock_get_error_tracker.return_value = mock_error_tracker_instance

    # This should not raise an exception
    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=Exception("test"),
        request_headers={}
    )

    # Logger should still have been called
    mock_get_failure_logger().error.assert_called_once()
    mock_main_lib_logger.error.assert_called_once()

# =============================================================================
# Extraction Tests for _extract_response_body
# =============================================================================

from src.rotator_library.failure_logger import _extract_response_body
from src.rotator_library.core.errors import StreamedAPIError
import json

def test_extract_streamed_api_error_dict():
    # .data is a dict
    error = StreamedAPIError("Stream failed")
    error.data = {"error": "bad request", "code": 400}

    result = _extract_response_body(error)
    assert "bad request" in result
    assert "400" in result
    assert json.loads(result) == error.data

def test_extract_streamed_api_error_nested_exception():
    # .data is an Exception that itself has a message
    inner = Exception("Inner timeout")
    inner.message = "Inner timeout message"
    error = StreamedAPIError("Stream failed")
    error.data = inner

    # StreamedAPIError wraps it, _extract_response_body should recurse and get the string form
    result = _extract_response_body(error)
    assert result == "Inner timeout message"

def test_extract_httpx_text_response():
    class MockResponse:
        text = "This is text content"
        content = None

    class MockHTTPError(Exception):
        response = MockResponse()

    error = MockHTTPError("HTTP Error")
    assert _extract_response_body(error) == "This is text content"

def test_extract_httpx_content_response():
    class MockResponse:
        text = None
        content = b'This is byte content'

    class MockHTTPError(Exception):
        response = MockResponse()

    error = MockHTTPError("HTTP Error")
    assert _extract_response_body(error) == "This is byte content"

def test_extract_litellm_body():
    class LiteLLMError(Exception):
        body = "litellm error body"

    error = LiteLLMError("Some error")
    assert _extract_response_body(error) == "litellm error body"

def test_extract_message_attribute():
    class LegacyError(Exception):
        message = "legacy message format"

    error = LegacyError()
    assert _extract_response_body(error) == "legacy message format"

# =============================================================================
# Boundary Condition Tests
# =============================================================================

from src.rotator_library.failure_logger import (
    FAILURE_LOG_ERROR_CHAIN_LIMIT,
    FAILURE_LOG_ERROR_MESSAGE_LIMIT,
    FAILURE_LOG_FULL_MESSAGE_LIMIT,
    FAILURE_LOG_RAW_RESPONSE_LIMIT
)

def test_boundary_error_chain_limit(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance
    mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    # Create a chain of 7 exceptions
    e = Exception("root")
    for i in range(6):
        next_e = Exception(f"error {i}")
        next_e.__cause__ = e
        e = next_e

    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=e,
        request_headers={}
    )

    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]
    error_chain = detailed_log_data["error_chain"]
    # The code uses `>` limit so it captures up to `FAILURE_LOG_ERROR_CHAIN_LIMIT + 1` elements.
    # e.g., if len=6, and limit=5, it breaks, meaning max length is 6.
    assert len(error_chain) == FAILURE_LOG_ERROR_CHAIN_LIMIT + 1

def test_boundary_error_message_limit(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance
    mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    long_msg = "A" * (FAILURE_LOG_ERROR_MESSAGE_LIMIT + 500)
    root = Exception(long_msg)
    top = Exception("top")
    top.__cause__ = root

    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=top,
        request_headers={}
    )

    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]
    error_chain = detailed_log_data["error_chain"]
    root_logged_msg = error_chain[1]["message"]

    assert len(root_logged_msg) == FAILURE_LOG_ERROR_MESSAGE_LIMIT
    assert root_logged_msg.endswith("A")

def test_boundary_full_message_limit(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance
    mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    long_msg = "B" * (FAILURE_LOG_FULL_MESSAGE_LIMIT + 500)
    error = Exception(long_msg)

    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=error,
        request_headers={}
    )

    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]
    error_message = detailed_log_data["error_message"]

    assert len(error_message) == FAILURE_LOG_FULL_MESSAGE_LIMIT
    assert error_message.endswith("B")

def test_boundary_raw_response_limit(mocker):
    mock_get_failure_logger = mocker.patch("src.rotator_library.failure_logger.get_failure_logger")
    mock_failure_logger_instance = MagicMock()
    mock_get_failure_logger.return_value = mock_failure_logger_instance
    mocker.patch("src.rotator_library.failure_logger.main_lib_logger")
    mocker.patch("src.rotator_library.failure_logger.get_error_tracker")

    long_response = "C" * (FAILURE_LOG_RAW_RESPONSE_LIMIT + 500)

    log_failure(
        api_key="test",
        model="test",
        attempt=1,
        error=Exception("test"),
        request_headers={},
        raw_response_text=long_response
    )

    detailed_log_data = mock_failure_logger_instance.error.call_args[0][0]
    raw_response = detailed_log_data["raw_response"]

    assert len(raw_response) == FAILURE_LOG_RAW_RESPONSE_LIMIT
    assert raw_response.endswith("C")
