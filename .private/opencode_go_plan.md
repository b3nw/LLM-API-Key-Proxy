# Opencode Go Provider — Model Discovery & Free Model Routing

## Current State

### `get_models()` (`opencode_go_provider.py:121-133`)

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

Three-model hardcoded fallback when `OPENCODE_GO_MODELS` env var is unset/empty.

### Routing in `acompletion()` (line 346-349)

```python
if "/zen/v1" in api_base and not "/zen/go/v1" in api_base:
    api_base = api_base.replace("/zen/v1", "/zen/go/v1")
```

All requests go through `/zen/go/v1/chat/completions` unconditionally.

### Existing `zenmux` provider

A separate provider (`zenmux_provider.py`) already queries `https://opencode.ai/zen/v1/models` and handles free models (`deepseek-v4-flash-free`, `minimax-m2.5-free`, `nemotron-3-super-free`, etc.) through the regular `/zen/v1` endpoint.

---

## Proposal (a): Dynamic Model Discovery from Go Endpoint

### What

Replace the hardcoded 3-model fallback with a live query to `https://opencode.ai/zen/go/v1/models`.

### Upstream go endpoint returns 15 models

```
minimax-m2.7       kimi-k2.5        deepseek-v4-pro      mimo-v2-pro
minimax-m2.5       glm-5.1          deepseek-v4-flash     mimo-v2-omni
kimi-k2.6          glm-5            qwen3.6-plus          mimo-v2.5-pro
                                     qwen3.5-plus          mimo-v2.5
                                                            hy3-preview
```

That's the old 14-model list plus `hy3-preview`.

### Implementation

```python
async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
    # 1. Check env var override first (existing behavior)
    static_models = self.model_definitions.get_all_provider_models("opencode_go")
    if static_models:
        return static_models

    # 2. Query upstream go models endpoint
    try:
        models_url = f"{self.api_base.replace('/zen/v1', '/zen/go/v1').rstrip('/')}/models"
        response = await client.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        discovered = [f"opencode_go/{m['id']}" for m in data.get("data", []) if m.get("id")]
        if discovered:
            lib_logger.info(f"Discovered {len(discovered)} models from go endpoint")
            return discovered
    except Exception as e:
        lib_logger.warning(f"Failed to fetch go models: {e}")

    # 3. Fallback
    return ["opencode_go/deepseek-v4-pro", "opencode_go/glm-5.1", "opencode_go/kimi-k2.6"]
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cache | None (called once per provider at startup via `_discover_provider_models`) | Model list is stable; upstream rarely changes; restart is fine for updates |
| Auth | Uses caller's `api_key` | Same pattern as other providers that query upstream endpoints (zenmux, vertex, chutes) |
| API base | Derives from `self.api_base` (replace `/zen/v1` → `/zen/go/v1`) | Consistent with existing routing logic in `acompletion()` |
| Error handling | Log warning + hardcoded fallback | Graceful degradation; proxy stays up even if upstream models endpoint is down |

---

## Proposal (b): Free Model Routing via `/zen/v1`

### Background

The `zenmux` provider already covers free Zen models via the regular `/zen/v1` endpoint. Under the proxy it currently exposes **0 models** (likely a credential/config issue rather than a code issue). Our options:

### Option 1: Fix the `zenmux` provider (Recommended)

The existing `zenmux_provider.py` already:
- Queries `https://opencode.ai/zen/v1/models` for discovery
- Routes to `https://opencode.ai/zen/v1/chat/completions`
- Uses a public API key
- Has its own prefix (`zenmux/`)

Fix the credential setup so `zenmux` provider actually shows up in the model list. No
routing changes needed to `opencode_go`.

### Option 2: Add free models to `opencode_go` with split routing

Add suffix-based routing in `acompletion()`:

```python
# Free models route to /zen/v1, paid models route to /zen/go/v1
if any(model.endswith(suffix) for suffix in ("-free", "big-pickle")):
    api_base = self.api_base  # keep /zen/v1
elif "/zen/v1" in api_base and not "/zen/go/v1" in api_base:
    api_base = api_base.replace("/zen/v1", "/zen/go/v1")
```

Models served via `/zen/v1` under `opencode_go/`:

| Model ID (from docs)        | Free? |
|-----------------------------|-------|
| `deepseek-v4-flash-free`    | Yes   |
| `minimax-m2.5-free`         | Yes   |
| `nemotron-3-super-free`     | Yes   |
| `big-pickle`                | Yes   |
| `qwen3.6-plus-free`         | Yes   |

### Drawback of Option 2

Duplicates the `zenmux` provider's responsibility. You'd end up with two providers
offering the same free models under different prefixes (`zenmux/` vs `opencode_go/`).

### Recommendation

**Fix the `zenmux` provider credentials** (Option 1). This keeps concerns separated:
- `opencode_go/` → paid go-tier models via `/zen/go/v1`
- `zenmux/` → free models via `/zen/v1`

If we want free models accessible under `opencode_go/` for convenience, we can add the
suffix-based routing (Option 2) AFTER fixing `zenmux`.

---

## Next Steps

1. **Proposal (a)**: Implement dynamic model discovery — update `get_models()` to query
   the go endpoint. This immediately expands from 3 → 15 models.

2. **Proposal (b)**: Investigate why `zenmux` shows 0 models, fix the credential config
   or provider initialization. Decide whether to also add free model routing to
   `opencode_go` as a convenience alias.
