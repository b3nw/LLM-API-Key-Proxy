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
| **Vertex AI** | Express Mode API key auth via `x-goog-api-key`, dynamic model discovery |

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

- `GET /v1/health` — status, uptime, provider/credential counts (add `?detail=full` for per-model window stats and error summary)
- `GET /v1/health/errors` — recent errors with optional `?provider=` and `?model=` filters
- Both endpoints are gated by `PROXY_API_KEY`

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

# Vertex AI (Express Mode API key)
VERTEX_PROJECT=your-default-project-id  # optional if keys embed project
VERTEX_LOCATION=global
VERTEX_API_KEY_1=your-vertex-express-key             # uses VERTEX_PROJECT
VERTEX_API_KEY_2=other-project:your-other-key        # project embedded in key

# Per-provider retry overrides
MAX_RETRIES_NANOGPT=2

# Log rotation (set in main.py automatically)
# scripts/cleanup-logs.sh for transaction directory cleanup
```

---

## Fork Strategy

This fork is maintained as a **linear commit stack** on top of `upstream/dev` — one squashed commit per feature area, no merge commits. Changes are folded into the correct commit using `fixup!` + `git rebase --autosquash`. See `AGENTS.md` for the full workflow.

---

## License

Same as upstream — see [LICENSE](LICENSE).
