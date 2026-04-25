# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for TimeoutConfig centralized timeout handling.
"""

import os
from unittest import mock
import pytest
from rotator_library.timeout_config import TimeoutConfig

class TestTimeoutConfig:
    """Test TimeoutConfig reading from env vars and default fallbacks."""

    def test_timeout_config_value_error(self):
        """Test that ValueError is caught and handled when env var is not a float."""
        # Using TIMEOUT_CONNECT as it maps to _CONNECT
        with mock.patch.dict(os.environ, {"TIMEOUT_CONNECT": "not-a-float"}):
            with mock.patch("rotator_library.timeout_config.lib_logger.warning") as mock_warning:
                result = TimeoutConfig.connect()

                # Should return the default value
                assert result == TimeoutConfig._CONNECT

                # Should log a warning about the invalid value
                mock_warning.assert_called_once_with(
                    f"Invalid value for TIMEOUT_CONNECT: not-a-float. Using default: {TimeoutConfig._CONNECT}"
                )

    def test_timeout_config_custom_valid(self):
        """Test that valid custom float values from env vars are returned."""
        with mock.patch.dict(os.environ, {"TIMEOUT_CONNECT": "45.5"}):
            result = TimeoutConfig.connect()
            assert result == 45.5
