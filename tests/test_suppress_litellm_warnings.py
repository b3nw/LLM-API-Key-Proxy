# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for suppress_litellm_warnings utility.

Verifies that the warning suppression filter is correctly configured
to silence litellm's internal Pydantic serialization warnings.

NO network calls, NO API keys needed.
"""

import os
import re
import warnings

from rotator_library.utils.suppress_litellm_warnings import (
    suppress_litellm_serialization_warnings,
)


class TestSuppressLitellmWarnings:
    """Test that litellm serialization warnings are suppressed correctly."""

    def test_runs_without_error(self):
        """Function executes without raising."""
        suppress_litellm_serialization_warnings()

    def test_filter_installed_by_default(self):
        """Warning filter is installed when env var is unset (default '1')."""
        old_val = os.environ.pop("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", None)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                suppress_litellm_serialization_warnings()

                ignore_filters = [
                    f for f in warnings.filters
                    if f[0] == "ignore" and f[2] is UserWarning
                ]
                assert len(ignore_filters) > 0, (
                    "Expected an 'ignore' filter for UserWarning to be installed"
                )
        finally:
            if old_val is not None:
                os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = old_val

    def test_filter_installed_when_explicitly_enabled(self):
        """Warning filter is installed when env var is '1'."""
        old_val = os.environ.get("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS")
        os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = "1"
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                suppress_litellm_serialization_warnings()

                ignore_filters = [
                    f for f in warnings.filters
                    if f[0] == "ignore" and f[2] is UserWarning
                ]
                assert len(ignore_filters) > 0, (
                    "Expected an 'ignore' filter for UserWarning to be installed"
                )
        finally:
            if old_val is None:
                os.environ.pop("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", None)
            else:
                os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = old_val

    def test_no_filter_when_disabled(self):
        """No filter is installed when env var is '0'."""
        old_val = os.environ.get("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS")
        os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = "0"
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                before_count = len(warnings.filters)
                suppress_litellm_serialization_warnings()
                after_count = len(warnings.filters)
                assert after_count == before_count, (
                    "Expected no new filter when disabled, "
                    f"but filter count changed from {before_count} to {after_count}"
                )
        finally:
            if old_val is None:
                os.environ.pop("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", None)
            else:
                os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = old_val

    def test_filter_targets_pydantic_module(self):
        """Filter module regex matches 'pydantic.main'."""
        old_val = os.environ.pop("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", None)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                suppress_litellm_serialization_warnings()

                pydantic_filters = [
                    f for f in warnings.filters
                    if f[0] == "ignore" and f[2] is UserWarning
                ]
                assert len(pydantic_filters) > 0

                module_pattern = pydantic_filters[0][3]
                assert module_pattern is not None, "Expected module pattern to be set"
                assert re.search(module_pattern, "pydantic.main"), (
                    f"Module pattern {module_pattern!r} should match 'pydantic.main'"
                )
        finally:
            if old_val is not None:
                os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = old_val

    def test_filter_targets_serialization_message(self):
        """Filter message regex matches the Pydantic serialization warning text."""
        old_val = os.environ.pop("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", None)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                suppress_litellm_serialization_warnings()

                pydantic_filters = [
                    f for f in warnings.filters
                    if f[0] == "ignore" and f[2] is UserWarning
                ]
                assert len(pydantic_filters) > 0

                message_pattern = pydantic_filters[0][1]
                assert message_pattern is not None, "Expected message pattern to be set"
                test_msg = (
                    "Pydantic serializer warnings: "
                    "PydanticSerializationUnexpectedValue"
                )
                assert re.search(message_pattern, test_msg), (
                    f"Message pattern {message_pattern!r} should match "
                    f"the serialization warning message"
                )
        finally:
            if old_val is not None:
                os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = old_val

    def test_filter_does_not_match_unrelated_module(self):
        """Filter module regex does not match non-pydantic modules."""
        old_val = os.environ.pop("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", None)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                suppress_litellm_serialization_warnings()

                pydantic_filters = [
                    f for f in warnings.filters
                    if f[0] == "ignore" and f[2] is UserWarning
                ]
                assert len(pydantic_filters) > 0

                module_pattern = pydantic_filters[0][3]
                assert not re.search(module_pattern, "some_other_module"), (
                    "Filter should not match non-pydantic modules"
                )
        finally:
            if old_val is not None:
                os.environ["SUPPRESS_LITELLM_SERIALIZATION_WARNINGS"] = old_val
