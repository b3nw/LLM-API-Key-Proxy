import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import importlib.util

# Load the module directly to bypass missing dependencies in __init__.py
spec = importlib.util.spec_from_file_location("paths", "src/rotator_library/utils/paths.py")
paths_module = importlib.util.module_from_spec(spec)
sys.modules["paths"] = paths_module
spec.loader.exec_module(paths_module)


def test_get_default_root_not_frozen():
    """Test get_default_root when sys.frozen is False (standard script/library)."""
    with patch("paths.sys") as mock_sys, patch("paths.Path.cwd") as mock_cwd:
        # Mock sys.frozen to be False
        mock_sys.frozen = False
        mock_cwd.return_value = Path("/mock/cwd")

        result = paths_module.get_default_root()

        assert result == Path("/mock/cwd")
        mock_cwd.assert_called_once()

def test_get_default_root_frozen():
    """Test get_default_root when sys.frozen is True (PyInstaller executable)."""
    with patch("paths.sys") as mock_sys:
        # Mock sys.frozen to be True and mock sys.executable
        mock_sys.frozen = True
        mock_sys.executable = "/mock/bin/executable"

        result = paths_module.get_default_root()

        assert result == Path("/mock/bin")

def test_get_default_root_no_frozen_attr():
    """Test get_default_root when sys has no 'frozen' attribute."""
    # Ensure sys.frozen is truly deleted/non-existent for this test
    with patch("paths.sys", spec=[]) as mock_sys, patch("paths.Path.cwd") as mock_cwd:
        # mock_sys won't have 'frozen'
        mock_cwd.return_value = Path("/mock/cwd2")

        result = paths_module.get_default_root()

        assert result == Path("/mock/cwd2")
        mock_cwd.assert_called_once()
