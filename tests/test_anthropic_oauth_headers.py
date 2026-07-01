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


class TestComputeBillingHeader:
    """Test the Claude Code billing header (client attestation) computation."""

    def test_billing_header_format(self):
        """Billing header has the expected format."""
        from rotator_library.providers.anthropic_provider import _compute_billing_header

        messages = [{"role": "user", "content": "Hello, Claude!"}]
        header = _compute_billing_header(messages)
        assert header.startswith("x-anthropic-billing-header: ")
        assert "cc_version=" in header
        assert "cc_entrypoint=" in header
        assert "cch=" in header
        assert header.endswith(";")

    def test_billing_header_cch_is_deterministic(self):
        """Same input produces same cch hash."""
        from rotator_library.providers.anthropic_provider import _compute_billing_header

        messages = [{"role": "user", "content": "Test message for hashing"}]
        header1 = _compute_billing_header(messages)
        header2 = _compute_billing_header(messages)
        assert header1 == header2

    def test_billing_header_different_messages_different_cch(self):
        """Different first user messages produce different cch hashes."""
        from rotator_library.providers.anthropic_provider import _compute_billing_header

        header1 = _compute_billing_header([{"role": "user", "content": "message one"}])
        header2 = _compute_billing_header([{"role": "user", "content": "message two"}])
        assert header1 != header2

    def test_billing_header_cch_is_5_chars(self):
        """cch is a 5-character SHA256 prefix."""
        import hashlib
        from rotator_library.providers.anthropic_provider import _compute_billing_header

        text = "test message"
        expected_cch = hashlib.sha256(text.encode()).hexdigest()[:5]
        header = _compute_billing_header([{"role": "user", "content": text}])
        assert f"cch={expected_cch};" in header

    def test_billing_header_extracts_from_list_content(self):
        """Billing header extracts text from multipart content lists."""
        from rotator_library.providers.anthropic_provider import _compute_billing_header

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello from list"},
                {"type": "image", "source": {"type": "base64"}},
            ]
        }]
        header = _compute_billing_header(messages)
        assert "cch=" in header  # Should not crash

    def test_billing_header_empty_messages(self):
        """Billing header handles empty message list gracefully."""
        from rotator_library.providers.anthropic_provider import _compute_billing_header

        header = _compute_billing_header([])
        assert "cch=" in header  # Uses empty string hash

    def test_billing_header_skips_assistant_messages(self):
        """Billing header uses first USER message, not first message."""
        import hashlib
        from rotator_library.providers.anthropic_provider import _compute_billing_header

        messages = [
            {"role": "assistant", "content": "ignored"},
            {"role": "user", "content": "actual user message"},
        ]
        expected_cch = hashlib.sha256("actual user message".encode()).hexdigest()[:5]
        header = _compute_billing_header(messages)
        assert f"cch={expected_cch};" in header


class TestBuildClaudeCodeSystem:
    """Test the Claude Code system prompt builder."""

    def test_system_array_has_two_entries(self):
        """System array contains billing header and identity."""
        from rotator_library.providers.anthropic_provider import _build_claude_code_system

        messages = [{"role": "user", "content": "test"}]
        system_entries, _ = _build_claude_code_system(messages, None)
        assert len(system_entries) == 2
        assert system_entries[0]["type"] == "text"
        assert "billing-header" in system_entries[0]["text"]
        assert system_entries[1]["type"] == "text"
        assert "Claude Code" in system_entries[1]["text"]

    def test_original_system_prompt_moved_to_first_user_message(self):
        """Original system prompt is prepended to first user message."""
        from rotator_library.providers.anthropic_provider import _build_claude_code_system

        messages = [{"role": "user", "content": "original user text"}]
        system_entries, modified_messages = _build_claude_code_system(
            messages, "You are a helpful assistant"
        )
        assert "You are a helpful assistant" in modified_messages[0]["content"]
        assert "original user text" in modified_messages[0]["content"]

    def test_no_system_prompt_no_relocation(self):
        """When no original system prompt, messages are unchanged."""
        from rotator_library.providers.anthropic_provider import _build_claude_code_system

        messages = [{"role": "user", "content": "hello"}]
        _, modified = _build_claude_code_system(messages, None)
        assert modified[0]["content"] == "hello"

    def test_system_prompt_inserted_when_no_user_message(self):
        """When no user message exists, original system prompt becomes one."""
        from rotator_library.providers.anthropic_provider import _build_claude_code_system

        messages = [{"role": "assistant", "content": "hi"}]
        _, modified = _build_claude_code_system(messages, "system instructions")
        assert modified[0]["role"] == "user"
        assert modified[0]["content"] == "system instructions"
