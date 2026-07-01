import pytest
import time
from unittest.mock import Mock

from rotator_library.usage.limits.fair_cycle import FairCycleChecker
from rotator_library.usage.config import FairCycleConfig, WindowDefinition
from rotator_library.usage.tracking.windows import WindowManager
from rotator_library.usage.types import (
    CredentialState,
    FairCycleState,
    TrackingMode,
    LimitResult,
    ModelStats,
    WindowStats,
    ResetMode
)

@pytest.fixture
def base_config():
    return FairCycleConfig(
        enabled=True,
        tracking_mode=TrackingMode.MODEL_GROUP,
        cross_tier=True,
        duration=60,
        quota_threshold=0.9,
    )

@pytest.fixture
def window_manager():
    windows = [WindowDefinition(name="primary", duration_seconds=60, reset_mode=ResetMode.ROLLING, is_primary=True)]
    wm = WindowManager(windows)
    # Mock get_primary_definition since it's used
    wm.get_primary_definition = Mock(return_value=windows[0])

    # Mock get_active_window to return a WindowStats with limit 100
    wm.get_active_window = Mock(return_value=WindowStats(name="primary", limit=100))
    return wm

@pytest.fixture
def checker(base_config, window_manager):
    return FairCycleChecker(config=base_config, window_manager=window_manager)

@pytest.fixture
def mock_credential():
    state = CredentialState(
        stable_id="test_id",
        provider="test_provider",
        accessor="test_key",
        tier="basic",
        priority=1
    )
    # Add fake model stats
    state.model_usage["gpt-4"] = ModelStats()
    state.model_usage["gpt-4"].windows["primary"] = WindowStats(name="primary", limit=100)

    # Add fake fair cycle state
    state.fair_cycle["gpt-4"] = FairCycleState()
    return state

def test_check_disabled(checker, mock_credential):
    checker._config.enabled = False
    result = checker.check(mock_credential, "gpt-4")
    assert result.allowed

def test_check_not_exhausted(checker, mock_credential):
    result = checker.check(mock_credential, "gpt-4")
    assert result.allowed
    assert not mock_credential.fair_cycle["gpt-4"].exhausted

def test_check_quota_exhausted_threshold(checker, mock_credential):
    fc_state = mock_credential.fair_cycle["gpt-4"]
    # 100 limit * 0.9 threshold = 90
    fc_state.cycle_request_count = 90

    result = checker.check(mock_credential, "gpt-4")

    # Should be marked as exhausted now
    assert fc_state.exhausted
    assert fc_state.exhausted_reason == "quota_threshold"

    # But because cycle is just starting (or hasn't expired), it will be blocked
    # Let's check the result
    assert not result.allowed
    assert result.result == LimitResult.BLOCKED_FAIR_CYCLE

def test_check_exhausted_cycle_expired(checker, mock_credential):
    fc_state = mock_credential.fair_cycle["gpt-4"]
    fc_state.exhausted = True

    # Mock cycle start far in the past
    global_state = checker._get_global_state("test_provider", "gpt-4")
    global_state.cycle_start = time.time() - 100 # duration is 60

    result = checker.check(mock_credential, "gpt-4")
    # Cycle has expired, so it should be allowed (it will reset cycle eventually)
    assert result.allowed

def test_check_exhausted_cycle_not_expired(checker, mock_credential):
    fc_state = mock_credential.fair_cycle["gpt-4"]
    fc_state.exhausted = True

    # Mock cycle start recently
    global_state = checker._get_global_state("test_provider", "gpt-4")
    global_state.cycle_start = time.time() - 10 # duration is 60

    result = checker.check(mock_credential, "gpt-4")
    # Cycle not expired, so it should be blocked
    assert not result.allowed
    assert result.result == LimitResult.BLOCKED_FAIR_CYCLE


def test_reset_specific_model(checker, mock_credential):
    fc_state = mock_credential.fair_cycle["gpt-4"]
    fc_state.exhausted = True
    fc_state.exhausted_at = 12345.0
    fc_state.exhausted_reason = "quota_threshold"
    fc_state.cycle_request_count = 50

    # Also add another model to ensure it's not reset
    mock_credential.fair_cycle["claude"] = FairCycleState()
    mock_credential.fair_cycle["claude"].exhausted = True
    mock_credential.fair_cycle["claude"].cycle_request_count = 10

    checker.reset(mock_credential, model="gpt-4")

    assert not fc_state.exhausted
    assert fc_state.exhausted_at is None
    assert fc_state.exhausted_reason is None
    assert fc_state.cycle_request_count == 0

    # Claude should still be exhausted
    assert mock_credential.fair_cycle["claude"].exhausted
    assert mock_credential.fair_cycle["claude"].cycle_request_count == 10

def test_reset_all(checker, mock_credential):
    fc_state = mock_credential.fair_cycle["gpt-4"]
    fc_state.exhausted = True
    fc_state.cycle_request_count = 50

    mock_credential.fair_cycle["claude"] = FairCycleState()
    mock_credential.fair_cycle["claude"].exhausted = True
    mock_credential.fair_cycle["claude"].cycle_request_count = 10

    checker.reset(mock_credential)

    assert not fc_state.exhausted
    assert fc_state.cycle_request_count == 0

    assert not mock_credential.fair_cycle["claude"].exhausted
    assert mock_credential.fair_cycle["claude"].cycle_request_count == 0


def test_check_all_exhausted_empty(checker):
    assert checker.check_all_exhausted("test_provider", "gpt-4", [])

def test_check_all_exhausted_all_exhausted(checker, mock_credential):
    mock_credential.fair_cycle["gpt-4"].exhausted = True

    cred2 = CredentialState(
        stable_id="test_id2",
        provider="test_provider",
        accessor="test_key2",
        tier="basic",
        priority=1
    )
    cred2.fair_cycle["gpt-4"] = FairCycleState()
    cred2.fair_cycle["gpt-4"].exhausted = True

    assert checker.check_all_exhausted("test_provider", "gpt-4", [mock_credential, cred2])

def test_check_all_exhausted_some_not_exhausted(checker, mock_credential):
    mock_credential.fair_cycle["gpt-4"].exhausted = True

    cred2 = CredentialState(
        stable_id="test_id2",
        provider="test_provider",
        accessor="test_key2",
        tier="basic",
        priority=1
    )
    cred2.fair_cycle["gpt-4"] = FairCycleState()
    cred2.fair_cycle["gpt-4"].exhausted = False

    assert not checker.check_all_exhausted("test_provider", "gpt-4", [mock_credential, cred2])

def test_check_all_exhausted_priorities(checker, mock_credential):
    # Disable cross_tier to respect priorities
    checker._config.cross_tier = False

    mock_credential.fair_cycle["gpt-4"].exhausted = True

    cred2 = CredentialState(
        stable_id="test_id2",
        provider="test_provider",
        accessor="test_key2",
        tier="fallback",
        priority=2
    )
    cred2.fair_cycle["gpt-4"] = FairCycleState()
    # Priority 2 is not exhausted, but we group by priority
    cred2.fair_cycle["gpt-4"].exhausted = False

    # Priority 1 group (mock_credential) is fully exhausted.
    # Priority 2 group (cred2) is not fully exhausted.
    # `check_all_exhausted` checks if *every* priority group is fully exhausted.
    assert not checker.check_all_exhausted("test_provider", "gpt-4", [mock_credential, cred2], priorities={"test_id": 1, "test_id2": 2})

    # If both are exhausted:
    cred2.fair_cycle["gpt-4"].exhausted = True
    assert checker.check_all_exhausted("test_provider", "gpt-4", [mock_credential, cred2], priorities={"test_id": 1, "test_id2": 2})

    # If cross_tier is True, priorities are ignored
    checker._config.cross_tier = True
    cred2.fair_cycle["gpt-4"].exhausted = False
    assert not checker.check_all_exhausted("test_provider", "gpt-4", [mock_credential, cred2], priorities={"test_id": 1, "test_id2": 2})


def test_reset_cycle(checker, mock_credential):
    fc_state = mock_credential.fair_cycle["gpt-4"]
    fc_state.exhausted = True
    fc_state.exhausted_at = 12345.0
    fc_state.exhausted_reason = "quota"
    fc_state.cycle_request_count = 100

    cred2 = CredentialState(
        stable_id="test_id2",
        provider="test_provider",
        accessor="test_key2",
        tier="basic",
        priority=1
    )
    cred2.fair_cycle["gpt-4"] = FairCycleState()
    cred2.fair_cycle["gpt-4"].exhausted = True

    # Pre-setup global state
    global_state = checker._get_global_state("test_provider", "gpt-4")
    global_state.cycle_start = 1000.0
    global_state.all_exhausted_at = 2000.0
    global_state.cycle_count = 5

    checker.reset_cycle("test_provider", "gpt-4", [mock_credential, cred2])

    # Verify credential states reset
    assert not fc_state.exhausted
    assert fc_state.exhausted_at is None
    assert fc_state.exhausted_reason is None
    assert fc_state.cycle_request_count == 0

    assert not cred2.fair_cycle["gpt-4"].exhausted

    # Verify global state updated
    assert global_state.cycle_start > 1000.0
    assert global_state.all_exhausted_at is None
    assert global_state.cycle_count == 6

def test_mark_all_exhausted(checker):
    global_state = checker._get_global_state("test_provider", "gpt-4")
    assert global_state.all_exhausted_at is None

    checker.mark_all_exhausted("test_provider", "gpt-4")

    assert global_state.all_exhausted_at is not None
    assert global_state.all_exhausted_at <= time.time()

def test_get_tracking_key(checker):
    # Test MODEL_GROUP mode
    assert checker.get_tracking_key("gpt-4", None) == "gpt-4"
    assert checker.get_tracking_key("gpt-4", "group-1") == "group-1"

    # Test CREDENTIAL mode
    checker._config.tracking_mode = TrackingMode.CREDENTIAL
    from rotator_library.usage.types import FAIR_CYCLE_GLOBAL_KEY
    assert checker.get_tracking_key("gpt-4", None) == FAIR_CYCLE_GLOBAL_KEY

def test_serialization(checker):
    global_state = checker._get_global_state("test_provider", "gpt-4")
    global_state.cycle_start = 1000.0
    global_state.all_exhausted_at = 2000.0
    global_state.cycle_count = 5

    data = checker.get_global_state_dict()
    assert "test_provider" in data
    assert "gpt-4" in data["test_provider"]
    assert data["test_provider"]["gpt-4"]["cycle_start"] == 1000.0
    assert data["test_provider"]["gpt-4"]["all_exhausted_at"] == 2000.0
    assert data["test_provider"]["gpt-4"]["cycle_count"] == 5

    # Create new checker and load
    checker2 = FairCycleChecker(config=FairCycleConfig())
    checker2.load_global_state_dict(data)

    state2 = checker2._get_global_state("test_provider", "gpt-4")
    assert state2.cycle_start == 1000.0
    assert state2.all_exhausted_at == 2000.0
    assert state2.cycle_count == 5
