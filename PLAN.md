# Firmware.ai Quota Tracking Implementation Plan

## Overview

Implement quota tracking for the Firmware.ai provider based on their `/api/v1/quota` API endpoint. This follows the established mixin pattern used by Chutes and NanoGPT quota tracking.

## API Specification

**Endpoint:** `GET https://app.firmware.ai/api/v1/quota`
**Authentication:** `Authorization: Bearer <api_key>`

**Response:**
```json
{
  "used": 0.75,      // Ratio from 0 to 1 (quota utilization)
  "reset": "2024-01-15T14:30:00Z"  // ISO UTC timestamp, or null when no active window
}
```

**Key Characteristics:**
- **5-hour rolling window** (unlike Chutes daily / NanoGPT daily+monthly)
- `used` is already a ratio (no calculation needed, unlike Chutes which returns absolute values)
- `reset` can be `null` when no credits have been spent recently
- Simpler response than other providers

## Implementation Approach

### Pattern Analysis

| Aspect | Chutes | NanoGPT | Firmware.ai (Proposed) |
|--------|--------|---------|------------------------|
| Quota Window | Daily (00:00 UTC) | Daily + Monthly | 5-hour rolling |
| API Response | `{quota: int, used: float}` | `{daily: {...}, monthly: {...}}` | `{used: 0-1, reset: ISO\|null}` |
| Calculation | `remaining = quota - used` | `remaining / limit` | `remaining = 1 - used` |
| Reset Handling | Calculate next midnight | Parse from API | Parse ISO string |
| Tier Detection | From quota value | From state field | N/A (no tiers) |

### Files to Create

#### 1. `src/rotator_library/providers/utilities/firmware_quota_tracker.py`

Mixin class providing quota tracking functionality:

```python
class FirmwareQuotaTracker:
    """
    Mixin class providing quota tracking for Firmware.ai provider.

    Required provider attributes:
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300
    """

    async def fetch_quota_usage(api_key, client=None) -> Dict[str, Any]:
        """
        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "used": float,           # 0.0 to 1.0
                "remaining_fraction": float,  # 1.0 - used
                "reset_at": float | None,     # Unix timestamp
                "has_active_window": bool,    # True if reset is not null
                "fetched_at": float,
            }
        """
```

**Key Implementation Details:**
- Parse ISO 8601 timestamp to Unix timestamp
- Handle `null` reset gracefully (no active spend window)
- Since `used` is already a ratio, `remaining_fraction = 1.0 - used`
- No tier detection needed (Firmware.ai doesn't expose tier info)

#### 2. `src/rotator_library/providers/firmware_provider.py` (Modifications)

Integrate the mixin into the existing Firmware provider:

```python
from .utilities.firmware_quota_tracker import FirmwareQuotaTracker

class FirmwareProvider(FirmwareQuotaTracker, ProviderInterface):
    def __init__(self, ...):
        # Add quota tracking state
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval = int(
            os.getenv("FIRMWARE_QUOTA_REFRESH_INTERVAL", "300")
        )

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        return {
            "name": "firmware_quota_refresh",
            "interval": self._quota_refresh_interval,
        }

    async def run_background_job(self, job_name: str, usage_manager) -> None:
        # Refresh quota for all credentials
        # Update usage_manager baselines
```

### Files to Modify

#### 3. `src/rotator_library/usage_manager.py`

May need modifications if not already generic enough:
- Ensure `update_quota_baseline()` handles null reset timestamps
- Verify quota group handling works with rolling windows

#### 4. Configuration / Environment

Add environment variable:
- `FIRMWARE_QUOTA_REFRESH_INTERVAL`: Refresh interval in seconds (default: 300)

Given the 5-hour window, 5-minute refresh is reasonable (60 checks per window).

## Implementation Steps

### Phase 1: Core Quota Tracker
1. [ ] Create `firmware_quota_tracker.py` mixin
2. [ ] Implement `fetch_quota_usage()` with ISO timestamp parsing
3. [ ] Implement `get_remaining_fraction()` and `get_reset_timestamp()`
4. [ ] Handle edge cases (null reset, API errors)

### Phase 2: Provider Integration
5. [ ] Add `FirmwareQuotaTracker` mixin to `FirmwareProvider`
6. [ ] Initialize quota cache and refresh interval in `__init__`
7. [ ] Implement `get_background_job_config()`
8. [ ] Implement `run_background_job()` for quota refresh

### Phase 3: Usage Manager Integration
9. [ ] Verify `UsageManager` compatibility with rolling windows
10. [ ] Add virtual model `firmware/_quota` for tracking
11. [ ] Configure quota group `firmware_global` for shared credential tracking

### Phase 4: Testing & Validation
12. [ ] Unit tests for quota tracker (mock API responses)
13. [ ] Integration test with real API (if credentials available)
14. [ ] Verify background refresh works correctly
15. [ ] Test cooldown behavior when quota exhausted

## Edge Cases to Handle

### 1. Null Reset Timestamp
When `reset: null`, there's no active spending window:
```python
if reset is None:
    return {
        "remaining_fraction": 1.0,  # Full quota available
        "reset_at": None,
        "has_active_window": False,
    }
```

### 2. Rolling Window Behavior
Unlike daily resets, the 5-hour window starts when spending begins:
- Don't calculate "next reset" - use API-provided timestamp
- If `reset` is in the past, treat as full quota available

### 3. API Unavailability
Follow established pattern:
- Return `status: "error"` with error message
- Keep using cached data if available
- Log warning but don't crash background job

## Configuration Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `FIRMWARE_QUOTA_REFRESH_INTERVAL` | 300 | Seconds between quota API checks |
| `FIRMWARE_API_BASE` | `https://app.firmware.ai` | API base URL |

## Testing Checklist

- [ ] Quota fetch returns correct remaining_fraction
- [ ] ISO timestamp parsed correctly to Unix timestamp
- [ ] Null reset handled (remaining_fraction = 1.0)
- [ ] HTTP errors return structured error response
- [ ] Background job refreshes all credentials
- [ ] UsageManager baseline updated correctly
- [ ] Cooldown set when remaining_fraction = 0.0
- [ ] Cooldown cleared when reset timestamp passes

## Notes

### Differences from Chutes/NanoGPT

1. **Simpler API**: Already provides ratio, no tier info
2. **Rolling window**: 5 hours from first spend, not fixed daily/monthly
3. **No tier detection**: Firmware.ai doesn't expose subscription tiers
4. **ISO timestamps**: Need to parse ISO 8601, not epoch milliseconds

### API Base URL Assumption

The plan uses `https://app.firmware.ai` as the base URL with endpoint path `/api/v1/quota`, verified against official documentation at docs.firmware.ai.
