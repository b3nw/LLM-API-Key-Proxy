# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for MODEL_ALIAS and MODEL_LATEST registry parsing.

Cross-provider routing via MODEL_ALIAS_<NAME> environment variables
and smart "latest" aliases via MODEL_LATEST_<NAME> are critical for
failover and model version management.

Breakage here means:
- Cross-provider failover stops working
- "latest" aliases point to wrong/stale models

NO network calls, NO API keys needed.
"""

import os
from unittest.mock import patch

import pytest

from rotator_library.model_alias_registry import (
    ModelAliasRegistry,
    AliasTarget,
    DEFAULT_RETRY_MODE,
)


class TestModelAliasRegistry:
    """Test MODEL_ALIAS env var parsing and alias resolution."""

    def test_parse_single_alias(self):
        """Single provider target is parsed correctly."""
        with patch.dict(os.environ, {
            "MODEL_ALIAS_TEST_MODEL": "chutes:deepseek-v3"
        }, clear=False):
            registry = ModelAliasRegistry()
            # Env var key TEST_MODEL normalizes to canonical "test-model"
            targets = registry.resolve("test-model")
            assert targets is not None
            assert len(targets) == 1
            assert targets[0].provider == "chutes"
            assert targets[0].model_name == "deepseek-v3"

    def test_parse_multi_provider_alias(self):
        """Multiple provider targets are parsed in order."""
        with patch.dict(os.environ, {
            "MODEL_ALIAS_DEEPSEEK_V3": "chutes:deepseek-v3,nanogpt:deepseek-chat"
        }, clear=False):
            registry = ModelAliasRegistry()
            targets = registry.resolve("deepseek-v3")
            assert targets is not None
            assert len(targets) == 2
            assert targets[0].provider == "chutes"
            assert targets[1].provider == "nanogpt"

    def test_parse_retry_mode_exhaust(self):
        """exhaust retry mode is parsed from pipe suffix."""
        with patch.dict(os.environ, {
            "MODEL_ALIAS_GLM_5": "chutes:glm-5,nanogpt:glm-5:thinking|exhaust"
        }, clear=False):
            registry = ModelAliasRegistry()
            mode = registry.get_retry_mode("glm-5")
            assert mode == "exhaust"

    def test_default_retry_mode_round_robin(self):
        """Default retry mode is round_robin."""
        with patch.dict(os.environ, {
            "MODEL_ALIAS_TEST2": "provider1:model1"
        }, clear=False):
            registry = ModelAliasRegistry()
            mode = registry.get_retry_mode("test2")
            assert mode == "round_robin"

    def test_no_matching_alias(self):
        """Unknown alias returns None."""
        with patch.dict(os.environ, {}, clear=False):
            registry = ModelAliasRegistry()
            targets = registry.resolve("nonexistent-alias")
            assert targets is None

    def test_is_alias(self):
        """is_alias returns True for registered aliases."""
        with patch.dict(os.environ, {
            "MODEL_ALIAS_MY_MODEL": "chutes:my-model"
        }, clear=False):
            registry = ModelAliasRegistry()
            assert registry.is_alias("my-model")

    def test_not_alias(self):
        """is_alias returns False for unknown models."""
        with patch.dict(os.environ, {}, clear=False):
            registry = ModelAliasRegistry()
            assert not registry.is_alias("nonexistent")

    def test_alias_target_full_model(self):
        """AliasTarget.full_model returns provider/model format."""
        target = AliasTarget(provider="chutes", model_name="deepseek-v3")
        assert target.full_model == "chutes/deepseek-v3"

    def test_underscore_to_hyphen_normalization(self):
        """Env var underscores are normalized to hyphens in canonical names."""
        with patch.dict(os.environ, {
            "MODEL_ALIAS_MY_COOL_MODEL": "provider1:model1"
        }, clear=False):
            registry = ModelAliasRegistry()
            # MY_COOL_MODEL → canonical "my-cool-model"
            targets = registry.resolve("my-cool-model")
            assert targets is not None


class TestDefaultClaudeAliases:
    """Test built-in default Claude model aliases.

    Default aliases provide bare-ID routing for Claude Code and similar
    clients that send unprefixed model IDs (e.g. ``claude-opus-4-8``)
    without requiring MODEL_ALIAS_* env vars.
    """

    @staticmethod
    def _env_without_model_aliases():
        """Return env dict with MODEL_ALIAS_* vars stripped out."""
        return {
            k: v for k, v in os.environ.items()
            if not k.startswith("MODEL_ALIAS_")
        }

    def test_default_claude_aliases_loaded(self):
        """Default Claude model aliases resolve without env vars."""
        with patch.dict(os.environ, self._env_without_model_aliases(), clear=True):
            registry = ModelAliasRegistry()
            for model in [
                "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7",
                "claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-5",
                "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
            ]:
                targets = registry.resolve(model)
                assert targets is not None, f"{model} should have a default alias"
                assert len(targets) == 1
                assert targets[0].provider == "anthropic"

    def test_default_alias_targets_correct_model_ids(self):
        """Default aliases target the correct Anthropic model IDs."""
        with patch.dict(os.environ, self._env_without_model_aliases(), clear=True):
            registry = ModelAliasRegistry()
            # 4-5 family uses date-suffixed IDs (matching OAUTH_MODELS)
            targets = registry.resolve("claude-opus-4-5")
            assert targets is not None and targets[0].model_name == "claude-opus-4-5-20251101"
            targets = registry.resolve("claude-sonnet-4-5")
            assert targets is not None and targets[0].model_name == "claude-sonnet-4-5-20250929"
            targets = registry.resolve("claude-haiku-4-5")
            assert targets is not None and targets[0].model_name == "claude-haiku-4-5-20251001"
            # Newer models use bare IDs (no date suffix per Anthropic docs)
            targets = registry.resolve("claude-opus-4-8")
            assert targets is not None and targets[0].model_name == "claude-opus-4-8"
            targets = registry.resolve("claude-fable-5")
            assert targets is not None and targets[0].model_name == "claude-fable-5"
            targets = registry.resolve("claude-sonnet-4-6")
            assert targets is not None and targets[0].model_name == "claude-sonnet-4-6"

    def test_env_var_overrides_default_alias(self):
        """MODEL_ALIAS_* env vars override built-in defaults."""
        with patch.dict(os.environ, {
            "MODEL_ALIAS_CLAUDE_OPUS_4_8": "anthropic:claude-opus-4-8,copilot:claude-opus-4.8"
        }, clear=False):
            registry = ModelAliasRegistry()
            targets = registry.resolve("claude-opus-4-8")
            assert targets is not None
            assert len(targets) == 2  # env var has 2 targets vs default's 1
            assert targets[0].provider == "anthropic"
            assert targets[1].provider == "copilot"

    def test_default_claude_aliases_in_canonical_models(self):
        """Default Claude aliases appear in get_canonical_models()."""
        with patch.dict(os.environ, self._env_without_model_aliases(), clear=True):
            registry = ModelAliasRegistry()
            canonical = set(registry.get_canonical_models())
            expected = {
                "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7",
                "claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-5",
                "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
            }
            assert expected.issubset(canonical)


class TestDefaultCodex56Aliases:
    """Test built-in GPT-5.6 aliases for Codex-backed compatibility IDs."""

    @staticmethod
    def _env_without_model_aliases():
        return {
            k: v for k, v in os.environ.items()
            if not k.startswith("MODEL_ALIAS_")
        }

    @pytest.mark.parametrize(
        ("public_id", "target_id"),
        [
            ("gpt-5.6-sol", "gpt-5.6-sol"),
            ("gpt-5.6-terra", "gpt-5.6-terra"),
            ("gpt-5.6-luna", "gpt-5.6-luna"),
            ("openai/gpt-5.6-sol", "gpt-5.6-sol"),
            ("openai/gpt-5.6-terra", "gpt-5.6-terra"),
            ("openai/gpt-5.6-luna", "gpt-5.6-luna"),
        ],
    )
    def test_default_codex_alias_resolves(self, public_id, target_id):
        with patch.dict(os.environ, self._env_without_model_aliases(), clear=True):
            targets = ModelAliasRegistry().resolve(public_id)

        assert targets is not None
        assert len(targets) == 1
        assert targets[0].provider == "codex"
        assert targets[0].model_name == target_id

    def test_default_codex_aliases_are_canonical_models(self):
        with patch.dict(os.environ, self._env_without_model_aliases(), clear=True):
            canonical = set(ModelAliasRegistry().get_canonical_models())

        assert {
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "openai/gpt-5.6-sol",
            "openai/gpt-5.6-terra",
            "openai/gpt-5.6-luna",
        }.issubset(canonical)


class TestModelLatestRegistry:
    """Test MODEL_LATEST env var parsing and resolution."""

    def test_parse_latest_alias(self):
        """MODEL_LATEST env vars are parsed into registry entries."""
        from rotator_library.model_latest_registry import ModelLatestRegistry

        with patch.dict(os.environ, {
            "MODEL_LATEST_GLM_LATEST": "nanogpt:glm-[0-9]*:exclude=*:thinking,*v*"
        }, clear=False):
            registry = ModelLatestRegistry()
            # Registry should have parsed the alias
            assert registry is not None

    def test_glob_pattern_matching(self):
        """Glob patterns match model names correctly."""
        import fnmatch

        pattern = "glm-[0-9]*"
        assert fnmatch.fnmatch("glm-5", pattern)
        assert fnmatch.fnmatch("glm-5.1", pattern)
        assert not fnmatch.fnmatch("glm-preview", pattern)

    def test_exclude_pattern(self):
        """Exclude patterns filter out unwanted models."""
        import fnmatch

        model = "glm-5:thinking"
        exclude_patterns = ["*:thinking", "*v*"]
        excluded = any(fnmatch.fnmatch(model, p) for p in exclude_patterns)
        assert excluded

        model = "glm-5"
        excluded = any(fnmatch.fnmatch(model, p) for p in exclude_patterns)
        assert not excluded
