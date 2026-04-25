import os
import json
import shutil
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest

from rotator_library.utils.resilient_io import safe_write_json, BufferedWriteRegistry


@pytest.fixture
def mock_logger():
    return logging.getLogger("test_logger")


def test_safe_write_json_atomic_happy_path(tmp_path, mock_logger):
    """Test basic atomic write creates file with correct content."""
    file_path = tmp_path / "test_atomic.json"
    data = {"key": "value", "number": 42}

    result = safe_write_json(
        path=file_path,
        data=data,
        logger=mock_logger,
        atomic=True
    )

    assert result is True
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == data


def test_safe_write_json_nonatomic_happy_path(tmp_path, mock_logger):
    """Test basic non-atomic write creates file with correct content."""
    file_path = tmp_path / "test_nonatomic.json"
    data = {"key": "value", "number": 42}

    result = safe_write_json(
        path=file_path,
        data=data,
        logger=mock_logger,
        atomic=False
    )

    assert result is True
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == data


def test_safe_write_json_secure_permissions_atomic(tmp_path, mock_logger):
    """Test secure permissions are set for atomic writes."""
    file_path = tmp_path / "test_secure_atomic.json"
    data = {"secret": "data"}

    with patch("os.chmod") as mock_chmod:
        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True,
            secure_permissions=True
        )

        assert result is True
        # os.chmod should be called at least once with 0o600
        mock_chmod.assert_any_call(mock_chmod.call_args[0][0], 0o600)


def test_safe_write_json_secure_permissions_nonatomic(tmp_path, mock_logger):
    """Test secure permissions are set for non-atomic writes."""
    file_path = tmp_path / "test_secure_nonatomic.json"
    data = {"secret": "data"}

    with patch("os.chmod") as mock_chmod:
        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=False,
            secure_permissions=True
        )

        assert result is True
        mock_chmod.assert_called_with(file_path, 0o600)


def test_safe_write_json_atomic_error_handling(tmp_path, mock_logger):
    """Test error handling when atomic write (shutil.move) fails."""
    file_path = tmp_path / "test_error.json"
    data = {"key": "value"}

    with patch("shutil.move", side_effect=OSError("Permission denied")):
        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True
        )

        assert result is False
        assert not file_path.exists()


def test_safe_write_json_nonatomic_error_handling(tmp_path, mock_logger):
    """Test error handling when non-atomic write fails."""
    file_path = tmp_path / "test_error_non_atomic.json"
    data = {"key": "value"}

    mock_open_file = mock_open()
    mock_open_file.side_effect = OSError("Permission denied")

    with patch("builtins.open", mock_open_file):
        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=False
        )

        assert result is False


def test_safe_write_json_buffer_on_failure(tmp_path, mock_logger):
    """Test failed write registers with BufferedWriteRegistry when buffer_on_failure=True."""
    file_path = tmp_path / "test_buffer.json"
    data = {"critical": "data"}

    mock_registry = MagicMock()

    with patch("shutil.move", side_effect=OSError("Disk full")), \
         patch.object(BufferedWriteRegistry, "get_instance", return_value=mock_registry):

        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True,
            buffer_on_failure=True,
            secure_permissions=True
        )

        assert result is False

        # Verify register_pending was called correctly
        mock_registry.register_pending.assert_called_once()
        args, kwargs = mock_registry.register_pending.call_args
        assert args[0] == file_path
        assert args[1] == data
        # args[2] is the serializer function
        assert callable(args[2])
        assert args[3] == {"secure_permissions": True}


def test_safe_write_json_cleanup_on_failure(tmp_path, mock_logger):
    """Test temporary file is cleaned up if atomic write fails."""
    file_path = tmp_path / "test_cleanup.json"
    data = {"key": "value"}

    with patch("shutil.move", side_effect=OSError("Failed to move")), \
         patch("os.unlink") as mock_unlink:

        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True
        )

        assert result is False
        # Ensure unlink was called to clean up the temporary file
        mock_unlink.assert_called_once()
