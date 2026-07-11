"""Unit tests for the ClinePass quota tracker (no network)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from rotator_library.providers.utilities.cline_pass_quota_tracker import (
    ClinePassQuotaSnapshot,
    ClinePassQuotaTracker,
    parse_plan_payload,
    parse_usage_limits_payload,
    _coerce_percent,
    _build_billing_url,
)


# ---------------------------------------------------------------------------
# Host stub
# ---------------------------------------------------------------------------


class _TrackerHost(ClinePassQuotaTracker):
    """Minimal host providing only what the mixin expects."""

    provider_env_name = "cline_pass"

    def __init__(self):
        self._init_quota_tracker()


# ---------------------------------------------------------------------------
# _coerce_percent
# ---------------------------------------------------------------------------


def test_coerce_percent_basic():
    assert _coerce_percent(42) == 42.0
    assert _coerce_percent(42.5) == 42.5
    assert _coerce_percent("37") == 37.0
    assert _coerce_percent(None) is None
    assert _coerce_percent("not-a-number") is None


def test_coerce_percent_clamps_out_of_range():
    # Negative clamps to 0
    assert _coerce_percent(-5) == 0.0
    # >100 clamps to 100
    assert _coerce_percent(150) == 100.0
    assert _coerce_percent(100.1) == 100.0


def test_coerce_percent_rejects_bools():
    # bools are ints in Python; guard against True/False being treated as 1/0
    assert _coerce_percent(True) is None
    assert _coerce_percent(False) is None


# ---------------------------------------------------------------------------
# _build_billing_url
# ---------------------------------------------------------------------------


def test_build_billing_url_default_base_passes_path_through():
    """Default base ``https://api.cline.bot/api/v1`` is the upstream's
    flat namespace — no path rewriting. The URL helper just joins the
    base with the caller-supplied path.

    Regression for the deployment on 2026-07-11: the previous
    implementation stripped a trailing ``/v1`` from the base, producing
    ``https://api.cline.bot/api/...`` paths that 404'd upstream and
    meant quota baselines never landed in the UsageManager (the WebUI
    quota card then had no data to render).
    """
    with patch.dict(
        "os.environ",
        {"CLINE_PASS_API_BASE": ""},
        clear=False,
    ):
        assert (
            _build_billing_url("/users/me/plan")
            == "https://api.cline.bot/api/v1/users/me/plan"
        )


def test_build_billing_url_preserves_v1_in_api_v1_base():
    """The Cline API is ``/api/v1``, NOT ``/v1``. We must not strip the
    trailing ``/v1`` from the base — it's part of the upstream namespace.
    """
    with patch.dict(
        "os.environ",
        {"CLINE_PASS_API_BASE": "https://api.cline.bot/api/v1"},
        clear=False,
    ):
        assert (
            _build_billing_url("/users/me/plan/usage-limits")
            == "https://api.cline.bot/api/v1/users/me/plan/usage-limits"
        )


def test_build_billing_url_no_leading_slash():
    with patch.dict(
        "os.environ",
        {"CLINE_PASS_API_BASE": "https://api.cline.bot/api/v1"},
        clear=False,
    ):
        assert (
            _build_billing_url("users/me/plan/usage-limits")
            == "https://api.cline.bot/api/v1/users/me/plan/usage-limits"
        )


def test_build_billing_url_trailing_slash_on_base_is_normalised():
    with patch.dict(
        "os.environ",
        {"CLINE_PASS_API_BASE": "https://api.cline.bot/api/v1/"},
        clear=False,
    ):
        # Base is rstripped so we don't get a doubled slash
        assert (
            _build_billing_url("/users/me/plan")
            == "https://api.cline.bot/api/v1/users/me/plan"
        )


# ---------------------------------------------------------------------------
# parse_usage_limits_payload
# ---------------------------------------------------------------------------


def test_parse_usage_limits_envelope():
    """Live `GET /v1/users/me/plan/usage-limits` shape (2026-07)."""
    data = {
        "success": True,
        "data": {
            "limits": [
                {"type": "five_hour", "percentUsed": 0},
                {"type": "weekly", "percentUsed": 12.5},
                {"type": "monthly", "percentUsed": 8.0},
            ]
        },
    }
    parsed = parse_usage_limits_payload(data)
    assert parsed == {
        "five_hour": 0.0,
        "weekly": 12.5,
        "monthly": 8.0,
    }


def test_parse_usage_limits_flat_shape():
    """Defensive: handle a flat `{"limits": [...]}` shape if the envelope changes."""
    data = {
        "limits": [
            {"type": "five_hour", "percentUsed": 50},
            {"type": "weekly", "percentUsed": 0},
            {"type": "monthly", "percentUsed": 0},
        ]
    }
    parsed = parse_usage_limits_payload(data)
    assert parsed["five_hour"] == 50.0
    assert parsed["weekly"] == 0.0
    assert parsed["monthly"] == 0.0


def test_parse_usage_limits_unknown_window_ignored():
    data = {
        "data": {
            "limits": [
                {"type": "five_hour", "percentUsed": 10},
                {"type": "yearly", "percentUsed": 50},  # unknown
            ]
        }
    }
    parsed = parse_usage_limits_payload(data)
    assert "yearly" not in parsed
    assert parsed["five_hour"] == 10.0


def test_parse_usage_limits_clamps_out_of_range():
    data = {
        "data": {
            "limits": [
                {"type": "weekly", "percentUsed": 150},  # clamps to 100
                {"type": "monthly", "percentUsed": -5},  # clamps to 0
            ]
        }
    }
    parsed = parse_usage_limits_payload(data)
    assert parsed["weekly"] == 100.0
    assert parsed["monthly"] == 0.0


def test_parse_usage_limits_non_dict_returns_empty():
    assert parse_usage_limits_payload(None) == {}
    assert parse_usage_limits_payload([]) == {}
    assert parse_usage_limits_payload({"data": "not-a-dict"}) == {}


# ---------------------------------------------------------------------------
# parse_plan_payload
# ---------------------------------------------------------------------------


def test_parse_plan_full_payload():
    data = {
        "displayName": "ClinePass",
        "currentPeriodStart": "2026-07-01T00:00:00Z",
        "currentPeriodEnd": "2026-08-01T00:00:00Z",
        "entitlements": {
            "cline_pass": {
                "inferenceCapThreshold": 50.0,
            }
        },
    }
    parsed = parse_plan_payload(data)
    assert parsed["display_name"] == "ClinePass"
    assert parsed["current_period_start_ts"] is not None
    assert parsed["current_period_end_ts"] is not None
    assert parsed["plan_hard_cap_usd"] == 50.0


def test_parse_plan_minimal_payload():
    """Plan endpoint may not include all fields; be tolerant."""
    data = {"displayName": "ClinePass"}
    parsed = parse_plan_payload(data)
    assert parsed["display_name"] == "ClinePass"
    assert parsed["current_period_start_ts"] is None
    assert parsed["plan_hard_cap_usd"] is None


def test_parse_plan_alternate_camel_case():
    """Some older API versions used snake_case; accept both."""
    data = {
        "displayName": "ClinePass",
        "current_period_start": "2026-07-01T00:00:00Z",
        "current_period_end": "2026-08-01T00:00:00Z",
    }
    parsed = parse_plan_payload(data)
    assert parsed["current_period_start_ts"] is not None
    assert parsed["current_period_end_ts"] is not None


def test_parse_plan_no_entitlements():
    data = {"displayName": "ClinePass"}
    parsed = parse_plan_payload(data)
    assert parsed["plan_hard_cap_usd"] is None


# ---------------------------------------------------------------------------
# _fetch_quota_for_credential
# ---------------------------------------------------------------------------


def test_fetch_quota_success():
    async def _run():
        host = _TrackerHost()

        # Mocked HTTP client
        limits_response = MagicMock()
        limits_response.status_code = 200
        limits_response.json.return_value = {
            "success": True,
            "data": {
                "limits": [
                    {"type": "five_hour", "percentUsed": 25},
                    {"type": "weekly", "percentUsed": 50},
                    {"type": "monthly", "percentUsed": 75},
                ]
            },
        }
        plan_response = MagicMock()
        plan_response.status_code = 200
        plan_response.json.return_value = {
            "displayName": "ClinePass",
            "currentPeriodEnd": "2026-08-01T00:00:00Z",
            "entitlements": {"cline_pass": {"inferenceCapThreshold": 100}},
        }

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        # Two awaits: limits then plan
        mock_client.get = AsyncMock(side_effect=[limits_response, plan_response])

        with patch(
            "rotator_library.providers.utilities.cline_pass_quota_tracker.httpx.AsyncClient",
            return_value=mock_client,
        ):
            snapshot = await host._fetch_quota_for_credential("cv-test-12345678")

        assert snapshot.status == "success"
        assert snapshot.five_hour_pct == 25.0
        assert snapshot.weekly_pct == 50.0
        assert snapshot.monthly_pct == 75.0
        assert snapshot.display_name == "ClinePass"
        assert snapshot.plan_hard_cap_usd == 100.0
        assert snapshot.current_period_end_ts is not None

    asyncio.run(_run())


def test_fetch_quota_auth_error_401():
    async def _run():
        host = _TrackerHost()
        response = MagicMock()
        response.status_code = 401
        response.text = "Unauthorized"

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=response)

        with patch(
            "rotator_library.providers.utilities.cline_pass_quota_tracker.httpx.AsyncClient",
            return_value=mock_client,
        ):
            snapshot = await host._fetch_quota_for_credential("cv-test-12345678")

        assert snapshot.status == "auth_error"
        assert "401" in (snapshot.error or "")

    asyncio.run(_run())


def test_fetch_quota_5xx_returns_error():
    async def _run():
        host = _TrackerHost()
        response = MagicMock()
        response.status_code = 503
        response.text = "Service Unavailable"
        # raise_for_status needs to be called and raise on 5xx
        response.raise_for_status.side_effect = Exception("HTTP 503")

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=response)

        with patch(
            "rotator_library.providers.utilities.cline_pass_quota_tracker.httpx.AsyncClient",
            return_value=mock_client,
        ):
            snapshot = await host._fetch_quota_for_credential("cv-test-12345678")

        assert snapshot.status == "error"
        assert "503" in (snapshot.error or "")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# _store_baselines_to_usage_manager
# ---------------------------------------------------------------------------


def test_store_baselines_writes_three_groups():
    async def _run():
        host = _TrackerHost()
        usage_manager = MagicMock()
        usage_manager.update_quota_baseline = AsyncMock()

        snapshot = ClinePassQuotaSnapshot(
            credential_path="/cred/c.json",
            identifier="cred",
            five_hour_pct=30.0,
            weekly_pct=50.0,
            monthly_pct=80.0,
            current_period_end_ts=time.time() + 86400,
            fetched_at=time.time(),
            status="success",
        )
        results = {"/cred/c.json": snapshot}

        stored = await host._store_baselines_to_usage_manager(
            results, usage_manager, force=True, is_initial_fetch=True
        )
        assert stored == 3

        calls = usage_manager.update_quota_baseline.await_args_list
        groups = {c.kwargs.get("quota_group") for c in calls}
        assert groups == {"5h", "weekly", "monthly"}

        # 5h window should NOT get a reset_ts (rolling, not period-bound)
        five_hour_call = next(c for c in calls if c.kwargs.get("quota_group") == "5h")
        assert five_hour_call.kwargs["quota_reset_ts"] is None

        # weekly/monthly should get the calendar reset timestamp
        weekly_call = next(
            c for c in calls if c.kwargs.get("quota_group") == "weekly"
        )
        assert weekly_call.kwargs["quota_reset_ts"] is not None

    asyncio.run(_run())


def test_store_baselines_marks_exhaustion_at_95():
    async def _run():
        host = _TrackerHost()
        usage_manager = MagicMock()
        usage_manager.update_quota_baseline = AsyncMock()

        snapshot = ClinePassQuotaSnapshot(
            credential_path="/cred/c.json",
            identifier="cred",
            five_hour_pct=98.0,  # exhausted
            weekly_pct=50.0,  # ok
            monthly_pct=99.0,  # exhausted
            current_period_end_ts=time.time() + 86400,
            fetched_at=time.time(),
            status="success",
        )
        await host._store_baselines_to_usage_manager(
            {"/cred/c.json": snapshot}, usage_manager, force=True, is_initial_fetch=True
        )

        calls = usage_manager.update_quota_baseline.await_args_list
        five_hour_call = next(c for c in calls if c.kwargs.get("quota_group") == "5h")
        assert five_hour_call.kwargs["apply_exhaustion"] is True

        weekly_call = next(
            c for c in calls if c.kwargs.get("quota_group") == "weekly"
        )
        assert weekly_call.kwargs["apply_exhaustion"] is False

    asyncio.run(_run())


def test_store_baselines_skips_non_success():
    async def _run():
        host = _TrackerHost()
        usage_manager = MagicMock()
        usage_manager.update_quota_baseline = AsyncMock()

        snapshot = ClinePassQuotaSnapshot(
            credential_path="/cred/c.json",
            identifier="cred",
            five_hour_pct=30.0,
            weekly_pct=50.0,
            monthly_pct=80.0,
            fetched_at=time.time(),
            status="auth_error",  # not "success" — should be skipped
        )
        stored = await host._store_baselines_to_usage_manager(
            {"/cred/c.json": snapshot}, usage_manager, force=True
        )
        assert stored == 0
        assert usage_manager.update_quota_baseline.await_count == 0

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# fetch_initial_baselines
# ---------------------------------------------------------------------------


def test_fetch_initial_baselines_concurrent():
    async def _run():
        host = _TrackerHost()

        ok_snapshot = ClinePassQuotaSnapshot(
            credential_path="/cred/a.json",
            identifier="a",
            five_hour_pct=10.0,
            weekly_pct=20.0,
            monthly_pct=30.0,
            status="success",
        )
        err_snapshot = ClinePassQuotaSnapshot(
            credential_path="/cred/b.json",
            identifier="b",
            status="auth_error",
            error="401",
        )

        with patch.object(
            host,
            "_fetch_quota_for_credential",
            side_effect=[ok_snapshot, err_snapshot],
        ):
            results = await host.fetch_initial_baselines(
                ["/cred/a.json", "/cred/b.json"]
            )

        assert results["/cred/a.json"].status == "success"
        assert results["/cred/b.json"].status == "auth_error"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Background job wiring
# ---------------------------------------------------------------------------


def test_get_background_job_config():
    host = _TrackerHost()
    config = host.get_background_job_config()
    assert config is not None
    assert config["name"] == "cline_pass_quota_refresh"
    assert config["run_on_start"] is True
    assert config["interval"] == 900  # default 15 min


def test_resolve_bearer_env_numbered():
    host = _TrackerHost()
    with patch.dict(
        "os.environ", {"CLINE_PASS_API_KEY_1": "cv-1-value"}, clear=False
    ):
        assert host._resolve_bearer("env://cline_pass/1") == "cv-1-value"


def test_resolve_bearer_env_unnumbered():
    host = _TrackerHost()
    with patch.dict(
        "os.environ", {"CLINE_PASS_API_KEY": "cv-default-value"}, clear=False
    ):
        assert host._resolve_bearer("env://cline_pass") == "cv-default-value"


def test_resolve_bearer_raw_value():
    """Non-env credentials (e.g. JSON file paths) are passed through as Bearer keys."""
    host = _TrackerHost()
    assert host._resolve_bearer("cv-direct-key-value") == "cv-direct-key-value"


# ---------------------------------------------------------------------------
# Model name round-trip (regression for PR #122 Kilo Code review)
# ---------------------------------------------------------------------------


def test_build_reverse_map_keys_are_raw_upstream_ids():
    """Regression: upstream ids in DEFAULT_CLINEPASS_MODELS already carry the
    ``cline-pass/`` prefix. The reverse map keys must NOT be wrapped with
    ``cline_pass/`` again (the previous implementation produced
    ``cline_pass/cline-pass/<bare>`` keys that never matched).
    """
    from rotator_library.providers.cline_pass_provider import (
        ClinePassProvider,
        DEFAULT_CLINEPASS_MODELS,
    )

    reverse = ClinePassProvider._build_reverse_map()
    # All upstream ids are already prefixed, so all reverse-map keys must
    # start with ``cline-pass/`` (NOT ``cline_pass/cline-pass/``).
    assert reverse, "reverse map should be populated"
    for upstream_key in reverse:
        assert upstream_key.startswith("cline-pass/"), (
            f"reverse-map key {upstream_key!r} should be a raw upstream id"
        )
        assert "/" not in upstream_key.removeprefix("cline-pass/"), (
            f"reverse-map key {upstream_key!r} should not have nested segments"
        )

    # Every shipped ClinePass model with a non-trivial id must have an entry.
    for bare, defn in DEFAULT_CLINEPASS_MODELS.items():
        upstream_id = defn.get("id") if isinstance(defn, dict) else defn
        if upstream_id != bare:
            assert upstream_id in reverse, (
                f"upstream id {upstream_id!r} missing from reverse map"
            )
            assert reverse[upstream_id] == f"cline_pass/{bare}"


def test_normalize_model_from_raw_upstream_id():
    """Caller passes a raw Cline upstream id (e.g. error message).
    Must map to the canonical display name.
    """
    from rotator_library.providers.cline_pass_provider import ClinePassProvider

    provider = ClinePassProvider()
    assert (
        provider.normalize_model_for_tracking("cline-pass/glm-5.2")
        == "cline_pass/glm-5.2"
    )
    assert (
        provider.normalize_model_for_tracking("cline-pass/qwen3.7-max")
        == "cline_pass/qwen3.7-max"
    )


def test_normalize_model_from_proxy_display_name():
    """Caller passes the proxy display name (``cline_pass/<bare>``).
    Should be returned unchanged (already canonical)."""
    from rotator_library.providers.cline_pass_provider import ClinePassProvider

    provider = ClinePassProvider()
    assert (
        provider.normalize_model_for_tracking("cline_pass/glm-5.2")
        == "cline_pass/glm-5.2"
    )
    assert (
        provider.normalize_model_for_tracking("cline_pass/deepseek-v4-flash")
        == "cline_pass/deepseek-v4-flash"
    )


def test_normalize_model_from_bare_name():
    """Caller passes a bare display name without provider prefix.
    Must map to the canonical proxy form."""
    from rotator_library.providers.cline_pass_provider import ClinePassProvider

    provider = ClinePassProvider()
    assert provider.normalize_model_for_tracking("glm-5.2") == "cline_pass/glm-5.2"
    assert (
        provider.normalize_model_for_tracking("kimi-k2.7-code")
        == "cline_pass/kimi-k2.7-code"
    )


def test_normalize_model_unknown_returns_input():
    """Unknown models (not in the catalog) pass through unchanged."""
    from rotator_library.providers.cline_pass_provider import ClinePassProvider

    provider = ClinePassProvider()
    # Unknown upstream id
    assert (
        provider.normalize_model_for_tracking("cline-pass/totally-fake")
        == "cline-pass/totally-fake"
    )
    # Unknown proxy form
    assert (
        provider.normalize_model_for_tracking("cline_pass/totally-fake")
        == "cline_pass/totally-fake"
    )
    # Unknown bare
    assert (
        provider.normalize_model_for_tracking("totally-fake") == "totally-fake"
    )
    # Empty
    assert provider.normalize_model_for_tracking("") == ""


# ---------------------------------------------------------------------------
# Routing URLs (regression for deployment 2026-07-11)
# ---------------------------------------------------------------------------


def test_provider_api_base_default_uses_documented_upstream():
    """Without an env override, the provider should default to the
    documented Cline API base ``https://api.cline.bot/api/v1``.

    Regression: the previous default introduced a separate
    ``/v1`` (no ``api/``) base for chat routing, which produced
    404s on every request — ``https://api.cline.bot/v1/chat/completions``
    is NOT the Cline API path; the path is ``/api/v1/chat/completions``.
    """
    from rotator_library.providers.cline_pass_provider import (
        ClinePassProvider,
        CLINE_PASS_DEFAULT_API_BASE,
    )

    assert CLINE_PASS_DEFAULT_API_BASE == "https://api.cline.bot/api/v1"
    provider = ClinePassProvider()
    assert provider.api_base == "https://api.cline.bot/api/v1"


def test_provider_uses_single_api_base_for_both_models_and_chat():
    """The provider must use the SAME base for model discovery and
    chat completions — the Cline API is rooted at ``/api/v1`` and
    every endpoint (models, chat completions, usage-limits, plan)
    lives under that prefix.
    """
    from rotator_library.providers.cline_pass_provider import ClinePassProvider

    provider = ClinePassProvider()
    # ``api_base`` drives both ``get_models`` (fetches ``{api_base}/models``)
    # and ``acompletion`` (sets litellm ``api_base`` so openai/ builds
    # ``{api_base}/chat/completions``). If we ever split them again, the
    # deployment 2026-07-11 failure will recur.
    api_base = provider.api_base
    assert api_base == "https://api.cline.bot/api/v1"
    # Sanity check the constructed URLs the proxy actually hits
    expected_chat_url = f"{api_base}/chat/completions"
    assert expected_chat_url == "https://api.cline.bot/api/v1/chat/completions"
    expected_models_url = f"{api_base.rstrip('/')}/models"
    assert expected_models_url == "https://api.cline.bot/api/v1/models"
