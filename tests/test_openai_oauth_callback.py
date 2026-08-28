# SPDX-License-Identifier: MIT

"""Tests for localhost OpenAI OAuth callback request classification."""

import pytest

from rotator_library.providers.openai_oauth_base import (
    _classify_oauth_callback_target,
)


CALLBACK_PATH = "/auth/callback"


@pytest.mark.parametrize("target", ["/", "/favicon.ico", "/other?code=probe"])
def test_non_callback_browser_requests_are_ignored(target):
    result = _classify_oauth_callback_target(target, CALLBACK_PATH)

    assert result.kind == "ignore"
    assert result.code is None
    assert result.error is None


def test_bare_callback_is_incomplete_not_error():
    result = _classify_oauth_callback_target(CALLBACK_PATH, CALLBACK_PATH)

    assert result.kind == "incomplete"
    assert result.code is None
    assert result.error is None


def test_callback_code_and_state_are_preserved():
    result = _classify_oauth_callback_target(
        f"{CALLBACK_PATH}?code=authorization-code&state=expected-state",
        CALLBACK_PATH,
    )

    assert result.kind == "code"
    assert result.code == "authorization-code"
    assert result.state == "expected-state"


def test_callback_without_state_remains_a_code_result():
    result = _classify_oauth_callback_target(
        f"{CALLBACK_PATH}?code=authorization-code",
        CALLBACK_PATH,
    )

    assert result.kind == "code"
    assert result.code == "authorization-code"
    assert result.state is None


def test_provider_error_is_preserved():
    result = _classify_oauth_callback_target(
        f"{CALLBACK_PATH}?error=access_denied",
        CALLBACK_PATH,
    )

    assert result.kind == "error"
    assert result.error == "access_denied"
