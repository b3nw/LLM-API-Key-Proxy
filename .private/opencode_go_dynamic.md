# Opencode Go — Dynamic Model Discovery via `/zen/go/v1/models`

## Current State

`get_models()` in `opencode_go_provider.py:121-133`:

```python
async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
    models = []
    static_models = self.model_definitions.get_all_provider_models("opencode_go")
    if static_models:
        models = static_models
    else:
        models = ["opencode_go/deepseek-v4-pro", "opencode_go/glm-5.1", "opencode_go/kimi-k2.6"]
    return models
```

When `OPENCODE_GO_MODELS` env var is unset (currently commented out in `.env`), the
provider falls back to 3 hardcoded models. The upstream go endpoint actually serves 15.

## Upstream go endpoint (`https://opencode.ai/zen/go/v1/models`)

Returns 200 with these model IDs:

```
minimax-m2.7       kimi-k2.5        deepseek-v4-pro      mimo-v2-pro
minimax-m2.5       glm-5.1          deepseek-v4-flash     mimo-v2-omni
kimi-k2.6          glm-5            qwen3.6-plus          mimo-v2.5-pro
                                     qwen3.5-plus          mimo-v2.5
                                                            hy3-preview
```

That's the old 14-model list plus `hy3-preview` (a new addition).

## Proposed Implementation

Replace the hardcoded fallback with a live query to the go models endpoint:

```python
async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
    # 1. Check env var override first (existing behavior)
    static_models = self.model_definitions.get_all_provider_models("opencode_go")
    if static_models:
        return static_models

    # 2. Query upstream go models endpoint
    try:
        # Derive go endpoint from the configured api_base
        go_base = self.api_base.replace("/zen/v1", "/zen/go/v1").rstrip("/")
        models_url = f"{go_base}/models"
        response = await client.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        discovered = [
            f"opencode_go/{m['id']}"
            for m in data.get("data", [])
            if m.get("id")
        ]
        if discovered:
            lib_logger.info(f"Discovered {len(discovered)} models from go endpoint")
            return discovered
    except Exception as e:
        lib_logger.warning(f"Failed to fetch go models: {e}")

    # 3. Graceful fallback
    return [
        "opencode_go/deepseek-v4-pro",
        "opencode_go/glm-5.1",
        "opencode_go/kimi-k2.6",
    ]
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cache | None — called once at startup via `_discover_provider_models` | Model list is stable; upstream rarely changes; restart is sufficient |
| Auth | Uses the caller's `api_key` (passed via `RotatingClient._discover_provider_models`) | Same pattern as zenmux, vertex, chutes providers |
| API base | Derives from `self.api_base` (`/zen/v1` → `/zen/go/v1`) | Consistent with existing routing in `acompletion()` |
| Error handling | Log warning + hardcoded fallback | Graceful degradation — proxy stays up even if upstream is unreachable |

## Testing

```bash
# Query the proxy model list — expect 15 opencode_go/ models
ssh docker-test 'curl -s -H "Authorization: Bearer sk-proxy-..." \
  http://localhost:9220/v1/models' | python3 -c "
import json, sys
data = json.load(sys.stdin)
ids = sorted([m['id'] for m in data.get('data', []) if m['id'].startswith('opencode_go/')])
print(f'opencode_go models ({len(ids)}):')
for m in ids:
    print(f'  {m}')
"
```
