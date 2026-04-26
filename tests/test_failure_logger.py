from pathlib import Path

import pytest

from rotator_library import failure_logger
from rotator_library.failure_logger import configure_failure_logger


@pytest.fixture(autouse=True)
def cleanup_failure_logger_globals():
    """Fixture to reset module-level globals before and after each test for isolation."""
    # Store initial state (likely None, but good practice)
    original_logs_dir = failure_logger._configured_logs_dir
    original_logger = failure_logger._failure_logger

    yield

    # Restore original state after test
    failure_logger._configured_logs_dir = original_logs_dir
    failure_logger._failure_logger = original_logger


class TestConfigureFailureLogger:
    def test_configure_with_string_path(self):
        """Test configuring with a string path."""
        configure_failure_logger("/tmp/test_logs")
        assert failure_logger._configured_logs_dir == Path("/tmp/test_logs")
        assert failure_logger._failure_logger is None

    def test_configure_with_path_object(self):
        """Test configuring with a Path object."""
        path = Path("/tmp/test_logs_path")
        configure_failure_logger(path)
        assert failure_logger._configured_logs_dir == path
        assert failure_logger._failure_logger is None

    def test_configure_with_none(self):
        """Test configuring with None resets the configured directory."""
        configure_failure_logger("/tmp/initial")
        assert failure_logger._configured_logs_dir is not None

        configure_failure_logger(None)
        assert failure_logger._configured_logs_dir is None
        assert failure_logger._failure_logger is None

    def test_configure_resets_logger(self):
        """Test that configuring always resets the _failure_logger instance."""
        # Set a dummy value to _failure_logger to simulate it being initialized
        failure_logger._failure_logger = "dummy_logger"

        configure_failure_logger("/tmp/another_path")

        # It should reset to None
        assert failure_logger._failure_logger is None
