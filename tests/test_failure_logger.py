import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from rotator_library.failure_logger import (
    get_failure_logger,
    configure_failure_logger,
)
import rotator_library.failure_logger as failure_logger_module

@pytest.fixture(autouse=True)
def reset_failure_logger_state():
    """Reset the module-level state before and after each test."""
    original_logger = failure_logger_module._failure_logger
    original_configured_dir = failure_logger_module._configured_logs_dir

    failure_logger_module._failure_logger = None
    failure_logger_module._configured_logs_dir = None

    yield

    # Clear handlers from the failure logger to prevent accumulation
    logging.getLogger("failure_logger").handlers.clear()

    failure_logger_module._failure_logger = original_logger
    failure_logger_module._configured_logs_dir = original_configured_dir

def test_get_failure_logger_lazy_init(tmp_path):
    """Verify that get_failure_logger initializes the logger when called for the first time."""
    configure_failure_logger(tmp_path)

    assert failure_logger_module._failure_logger is None

    logger = get_failure_logger()

    assert isinstance(logger, logging.Logger)
    assert logger.name == "failure_logger"
    assert failure_logger_module._failure_logger is logger

def test_get_failure_logger_cached(tmp_path):
    """Verify that get_failure_logger returns the same cached logger instance on subsequent calls."""
    configure_failure_logger(tmp_path)

    logger1 = get_failure_logger()
    logger2 = get_failure_logger()

    assert logger1 is logger2

def test_get_failure_logger_with_configured_dir(tmp_path):
    """Verify that get_failure_logger respects the directory set by configure_failure_logger."""
    configure_failure_logger(tmp_path)

    with patch("rotator_library.failure_logger._setup_failure_logger") as mock_setup:
        mock_setup.return_value = logging.getLogger("failure_logger_mock")

        get_failure_logger()

        mock_setup.assert_called_once_with(tmp_path)

def test_get_failure_logger_with_get_logs_dir_fallback():
    """Verify that get_failure_logger falls back to get_logs_dir() if no directory has been configured."""
    fallback_dir = Path("/mock/logs/dir")

    # Ensure it's not configured
    failure_logger_module._configured_logs_dir = None

    with patch("rotator_library.failure_logger.get_logs_dir") as mock_get_logs_dir:
        mock_get_logs_dir.return_value = fallback_dir

        with patch("rotator_library.failure_logger._setup_failure_logger") as mock_setup:
            mock_setup.return_value = logging.getLogger("failure_logger_mock")

            get_failure_logger()

            mock_get_logs_dir.assert_called_once()
            mock_setup.assert_called_once_with(fallback_dir)

def test_get_failure_logger_directory_creation_failure(tmp_path):
    """Verify get_failure_logger adds a NullHandler if directory creation fails."""
    configure_failure_logger(tmp_path)

    with patch("pathlib.Path.mkdir") as mock_mkdir:
        mock_mkdir.side_effect = PermissionError("Permission denied")

        logger = get_failure_logger()

        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)
