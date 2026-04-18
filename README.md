# LLM API Key Proxy (Fork)

A personal fork of [Mirrowel/LLM-API-Key-Proxy](https://github.com/Mirrowel/LLM-API-Key-Proxy) with additional providers, fixes, and tooling.

> **For full documentation**, see the [upstream repository](https://github.com/Mirrowel/LLM-API-Key-Proxy).

---

## Fork-Specific Features

### Additional Providers

| Provider | Description |
|----------|-------------|
| **GitHub Copilot** | OAuth Device Flow with plan-based model filtering (free/pro/business/enterprise), premium interaction quota tracking |
| **NanoGPT** | Native Anthropic message routing, streaming fallback, embedding dispatch |
| **ZenMux** | OpenAI-compatible provider with custom header support for free models |
| **Kilocode** | OpenAI-compatible provider with frequent free model offerings |
| **Chutes** | Dollar credit quota tracking with sliding window, tool-calling support |
| **Firmware** | Credit balance tracking with dollar-denominated displays |
| **Lightning AI** | Dollar credit quotas with date-based parsing |

### Smart "Latest" Model Aliases

Resolve virtual `latest` model names to the current best-available model at request time:

```env
# Automatically resolves at request time based on available models
MODEL_LATEST_nanogpt=nanogpt/glm-5  # "latest" resolves to current best GLM-5
```

- Cost-based tiebreaking when multiple candidates match
- On-demand model cache warming for cold starts
- Configurable per-provider resolution rules

### Usage & Quota Stats

- **Current period** vs **global/lifetime** quota split — TUI toggle between windows
- **Cached token pricing** — correct discounted rates for cached input tokens in streaming cost calculations
- **Identity-based deduplication** — OAuth credential dedup handles GitHub login (not just email)

### Monitoring & Health Endpoints

Two new endpoints for remote observability, gated by `PROXY_API_KEY`:

#### `GET /v1/health`

```bash
# Summary (status, uptime, provider/credential counts)
curl -H "Authorization: Bearer $PROXY_API_KEY" http://localhost:8000/v1/health

# Full detail (+ per-model window stats + error summary)
curl -H "Authorization: Bearer $PROXY_API_KEY" "http://localhost:8000/v1/health?detail=full"
```

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "timestamp": "2026-04-18T00:00:00+00:00",
  "providers": {
    "total": 10,
    "active": ["antigravity", "copilot", "gemini_cli", "..."],
    "with_errors": ["modal"]
  },
  "credentials": { "total": 19, "active": 14, "on_cooldown": 5, "exhausted": 0 },
  // detail=full only:
  "models_current_window": [
    {
      "model": "antigravity/claude-opus-4.5",
      "provider": "antigravity",
      "window_name": "5h",
      "window_started_at": "2026-04-17T19:00:00+00:00",
      "requests": 211, "success_count": 211, "failure_count": 0,
      "tokens": { "prompt": 14934558, "completion": 73620, "total": 15008178 },
      "approx_cost": 0.0,
      "last_used": "2026-04-17T23:58:00+00:00"
    }
  ],
  "errors": { "total_errors": 3, "by_provider": { "modal": { "count": 3, "error_types": { "RateLimitError": 3 } } }, "by_model": { "..." } }
}
```

#### `GET /v1/health/errors`

```bash
# Recent errors (newest first), optionally filtered by provider and/or model
curl -H "Authorization: Bearer $PROXY_API_KEY" \
  "http://localhost:8000/v1/health/errors?provider=modal&limit=5"
```

```json
{
  "errors": [
    {
      "timestamp": "2026-04-18T00:20:05+00:00",
      "provider": "modal",
      "model": "modal/qwen3-coder-480b",
      "error_type": "RateLimitError",
      "status_code": 429,
      "error_message": "litellm.RateLimitError: ...",
      "credential": "...abc123",
      "attempt": 2
    }
  ],
  "total_matching": 8,
  "limit": 5
}
```

- Backed by an **in-memory ring buffer** (500 records, resets on restart — `logs/failures.log` is the durable audit trail)
- `models_current_window` uses each provider's configured primary window (`5h`, `daily`, etc.) so per-model stats reflect the correct quota period
- Unauthenticated `GET /` still returns `{"Status": "API Key Proxy is running"}` for load-balancer uptime checks

### Tooling

- **Transaction Log Viewer TUI** — Browse and inspect API request/response logs
- **Embedding Support** — Dispatch embeddings to appropriate providers

---

## Quick Start (Docker)

```bash
docker-compose up -d
```

Or use the Komodo stack for deployment.

### Environment Variables

See upstream documentation for base configuration. Fork-specific variables:

```bash
# GitHub Copilot (OAuth Device Flow — use credential tool to authenticate)
# Credentials stored in oauth_creds/copilot_oauth_*.json

# NanoGPT
NANOGPT_API_KEY_1=your-nanogpt-key

# Cursor provider
CURSOR_API_KEY_1=your-cursor-key

# ZenMux (free models)
ZENMUX_API_BASE=https://zenmux.example.com/v1
ZENMUX_API_KEY_1=your-zenmux-key

# Per-provider retry overrides
MAX_RETRIES_NANOGPT=2

# Log rotation (set in main.py automatically)
# scripts/cleanup-logs.sh for transaction directory cleanup
```

---

## Upstream Sync

This fork is regularly synced with upstream. See `.agent/skills/upstream-sync/` for the sync workflow.

---

## License

Same as upstream — see [LICENSE](LICENSE).
