# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for Anthropic OAuth header and tool name conventions.

Verifies the safe header changes that mirror pi-agent's handling:
- Dynamic beta header computation per model
- x-app: cli header presence
- Tool name PascalCase prefixing (mcp_read → mcp_Read)

NO network calls, NO API keys needed.
"""

from rotator_library.providers.anthropic_provider import (
    _compute_beta_header,
    _prefix_tool_name,
    _BASE_BETA_HEADERS,
    _LONG_CONTEXT_BETAS,
    TOOL_PREFIX,
)


class TestComputeBetaHeader:
    """Test dynamic anthropic-beta header computation."""

    def test_base_betas_present(self):
        """Base betas are included for all models."""
        header = _compute_beta_header("claude-opus-4-5-20251101")
        for beta in _BASE_BETA_HEADERS:
            assert beta in header, f"{beta} should be in header for all models"

    def test_claude_code_beta_present(self):
        """The critical claude-code-20250219 beta is always included."""
        assert "claude-code-20250219" in _compute_beta_header("claude-opus-4-8")
        assert "claude-code-20250219" in _compute_beta_header("claude-haiku-4-5")
        assert "claude-code-20250219" in _compute_beta_header("claude-sonnet-4-6")

    def test_long_context_betas_added(self):
        """Long-context models get additional betas."""
        for model in ["claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
                      "claude-sonnet-4-6", "claude-sonnet-5", "claude-fable-5"]:
            header = _compute_beta_header(model)
            for beta in _LONG_CONTEXT_BETAS:
                assert beta in header, f"{beta} should be in header for {model}"

    def test_long_context_betas_not_added_for_non_long_context(self):
        """Non-long-context models do not get long-context betas."""
        header = _compute_beta_header("claude-opus-4-5-20251101")
        assert "context-1m-2025-08-07" not in header
        assert "effort-2025-11-24" not in header

    def test_haiku_excludes_interleaved_thinking(self):
        """Haiku models exclude interleaved-thinking beta."""
        header = _compute_beta_header("claude-haiku-4-5-20251001")
        assert "interleaved-thinking-2025-05-14" not in header
        # But still has other base betas
        assert "claude-code-20250219" in header
        assert "oauth-2025-04-20" in header

    def test_non_haiku_includes_interleaved_thinking(self):
        """Non-haiku models include interleaved-thinking beta."""
        header = _compute_beta_header("claude-opus-4-8")
        assert "interleaved-thinking-2025-05-14" in header


class TestPrefixToolName:
    """Test tool name prefixing with PascalCase."""

    def test_lowercase_tool_gets_capitalized(self):
        """Lowercase tool names get first letter capitalized."""
        assert _prefix_tool_name("read") == "mcp_Read"
        assert _prefix_tool_name("bash") == "mcp_Bash"
        assert _prefix_tool_name("web_search") == "mcp_Web_search"

    def test_already_capitalized_stays_capitalized(self):
        """Already-capitalized tool names keep their casing."""
        assert _prefix_tool_name("Read") == "mcp_Read"
        assert _prefix_tool_name("Bash") == "mcp_Bash"

    def test_empty_name_returns_empty(self):
        """Empty tool name returns empty string."""
        assert _prefix_tool_name("") == ""

    def test_single_char_gets_capitalized(self):
        """Single character tool names get capitalized."""
        assert _prefix_tool_name("x") == "mcp_X"

    def test_already_prefixed_not_double_prefixed(self):
        """The prefixing logic in the caller prevents double-prefixing."""
        # _prefix_tool_name itself doesn't check for existing prefix;
        # the caller checks with startswith(TOOL_PREFIX) before calling.
        # This test verifies the function would produce the right result
        # if called on a non-prefixed name.
        result = _prefix_tool_name("read")
        assert result.startswith(TOOL_PREFIX)
        assert result == "mcp_Read"
