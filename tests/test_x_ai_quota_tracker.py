"""Unit tests for xAI CLI proxy billing quota tracker (no network)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from rotator_library.providers.utilities.x_ai_quota_tracker import (
    XAiQuotaTracker,
    _billing_val,
    _parse_period_end_ts,
    parse_billing_payload,
)


class _TrackerHost(XAiQuotaTracker):
    """Minimal host with methods the mixin expects."""

    provider_env_name = "x-ai"

    def __init__(self):
        self._init_quota_tracker()
        self.cli_proxy_base = "https://cli-chat-proxy.grok.com/v1"
        self._cli_version = "0.1.202"
        self.model_quota_groups = {
            "monthly-limit": ["_billing_monthly"],
            "on-demand($)": ["_billing_ondemand"],
        }

    async def get_auth_header(self, credential_path: str):
        return {"Authorization": "Bearer test-token"}

    async def _load_credentials(self, credential_path: str):
        return {"account_id": "user-abc", "access_token": "test-token"}

    def _build_proxy_client_kwargs(self, credential_path: str):
        return {}


def test_billing_val_unwraps_nested_val():
    assert _billing_val({"val": 42}) == 42
    assert _billing_val(7) == 7
    assert _billing_val(None) is None


def test_parse_period_end_ts_iso_z():
    ts = _parse_period_end_ts("2026-07-01T00:00:00.000Z")
    assert ts is not None
    assert ts > time.time()


def test_parse_billing_payload_nested_vals():
    data = {
        "monthlyLimit": {"val": 1000},
        "used": {"val": 800},
        "onDemandCap": {"val": 50},
        "billingPeriodEnd": {"val": "2026-07-01T00:00:00.000Z"},
        "billingPeriodStart": {"val": "2026-06-01T00:00:00.000Z"},
        "tier": {"val": 2},
    }
    parsed = parse_billing_payload(data)
    assert parsed["monthly_limit"] == 1000
    assert parsed["used"] == 800
    assert parsed["on_demand_cap"] == 50
    assert parsed["period_end"] == "2026-07-01T00:00:00.000Z"
    assert parsed["tier"] == 2
    assert parsed["period_end_ts"] is not None


def test_store_baselines_monthly_exhaustion():
    async def _run():
        host = _TrackerHost()
        usage_manager = MagicMock()
        usage_manager.update_quota_baseline = AsyncMock()

        quota_results = {
            "/cred/xai.json": {
                "status": "success",
                "monthly_limit": 100,
                "used": 100,
                "on_demand_cap": None,
                "period_end_ts": time.time() + 3600,
            }
        }

        stored = await host._store_baselines_to_usage_manager(
            quota_results, usage_manager, force=True, is_initial_fetch=True
        )
        assert stored >= 1
        calls = usage_manager.update_quota_baseline.await_args_list
        monthly_call = next(
            c for c in calls if c.kwargs.get("quota_group") == "monthly-limit"
        )
        assert monthly_call.kwargs["quota_used"] == 100
        assert monthly_call.kwargs["quota_max_requests"] == 100
        assert monthly_call.kwargs["apply_exhaustion"] is True

    asyncio.run(_run())


def test_fetch_initial_baselines_success_and_error():
    async def _run():
        host = _TrackerHost()
        ok_snapshot = MagicMock()
        ok_snapshot.status = "success"
        ok_snapshot.error = None
        ok_snapshot.monthly_limit = 500
        ok_snapshot.used = 100
        ok_snapshot.on_demand_cap = None
        ok_snapshot.period_end_ts = time.time() + 86400
        ok_snapshot.tier = 1

        err_snapshot = MagicMock()
        err_snapshot.status = "error"
        err_snapshot.error = "HTTP 401"

        with patch.object(
            host,
            "_fetch_billing_for_credential",
            side_effect=[ok_snapshot, err_snapshot],
        ):
            results = await host.fetch_initial_baselines(
                ["/cred/a.json", "env://X_AI_API_KEY"]
            )

        assert results["/cred/a.json"]["status"] == "success"
        assert results["/cred/a.json"]["monthly_limit"] == 500
        assert "env://X_AI_API_KEY" not in results

    asyncio.run(_run())


def test_xai_provider_get_model_quota_group():
    from rotator_library.providers.x_ai_provider import XAiProvider

    provider = object.__new__(XAiProvider)
    assert provider.get_model_quota_group("x-ai/grok-4.3") == "monthly-limit"
    assert provider.get_model_quota_group("grok-build") == "monthly-limit"
    assert provider.get_model_quota_group("x-ai/_billing_monthly") == "monthly-limit"
    assert provider.get_model_quota_group("x-ai/_billing_ondemand") == "on-demand($)"