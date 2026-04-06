# LLM API Key Proxy (Fork)

A personal fork of [Mirrowel/LLM-API-Key-Proxy](https://github.com/Mirrowel/LLM-API-Key-Proxy) with additional providers, advanced model routing, and operational tooling.

> **For full upstream documentation**, see the [upstream repository](https://github.com/Mirrowel/LLM-API-Key-Proxy).

---

## Fork-Specific Features

### Additional Providers

| Provider | Description |
|----------|-------------|
| **Anthropic** | Full Anthropic API provider with OAuth support, streaming null handling, and dedicated quota tracking |
| **Codex** | OpenAI Codex/Responses API with dynamic model discovery, OAuth exports, and quota tracking |
| **Cursor** | Cursor IDE API with quota monitoring integration and dedicated sidecar proxy |
| **Lightning AI** | Dollar-based credit quota tracking with date-aware billing cycle parsing |
| **NanoGPT** | Model source configuration, reasoning normalization, request tracking, and native Anthropic endpoint routing for Claude models |
| **ZenMux** | OpenAI-compatible provider with custom header support for free models |
| **Firmware** | Credit balance tracking with dollar-amount quota displays |
| **Gemini A2A** | Experimental Gemini Agent-to-Agent protocol provider with sidecar architecture |

### Model Routing & Aliases

- **MODEL_ALIASES** — Define custom model alias mappings via environment variables (e.g., `MODEL_ALIAS_my-model=provider/actual-model`)
- **Cross-Provider Rotation** — Route a single model alias across multiple providers for automatic failover
- **Smart "latest" Resolution** — Auto-resolve `provider/model-latest` to the best available version using cost-based tiebreaking
- **Pricing Inheritance** — Latest aliases inherit pricing and context window metadata from their resolved target

### Provider Transforms

| Transform | Description |
|-----------|-------------|
| **Kimi K2.5** | Auto-enforces `top_p=0.95` (mandatory for Kimi API) |
| **GLM-5/GLM-4** | Minimum `max_tokens` floor (4096) to prevent thinking models from exhausting output budget |
| **Chutes** | Injects `allowed_openai_params` so LiteLLM's `drop_params=True` doesn't strip tool calling parameters |
| **DedalusLabs** | Removes `tool_choice=auto` to avoid 422 errors |

### Fixes & Improvements

- **Streaming token counting** — Correct `input_tokens` in `message_start` for Claude Code statusline
- **Anthropic null response handling** — Defensive checks for empty/invalid streaming responses
- **Symlink atomic writes** — Resolve symlinks before writes for Docker volume mounts
- **Dynamic provider singleton fix** — Prevents `api_base` sharing between dynamic providers
- **Gemini CLI fast-fail** — Non-rotatable errors skip remaining credentials instead of exhausting all keys
- **Gemini CLI pro quota handling** — Correctly excludes permanently-exhausted free-tier credentials for pro models
- **Qwen Code WAF detection** — Detects Alibaba Cloud WAF HTML blocks during token refresh and retries
- **Qwen Code upstream alignment** — OAuth headers, URL normalization, and env:// credential path support
- **Per-provider retry counts** — Configure `MAX_RETRIES_{PROVIDER}` to override the global retry count
- **Quota group sync** — Dynamic model discovery for accurate quota tracking

### Tooling

- **Transaction Log Viewer TUI** — Browse and inspect API request/response logs with compact displays and detail views
- **Cursor Sidecar** — Standalone OpenAI-compatible proxy for Cursor API (`cursor-sidecar/`)
- **Embedding Support** — Dispatch embeddings to appropriate providers
- **Settings Tool** — TUI-based interactive settings management

---

## Quick Start (Docker)

```bash
docker-compose up -d
```

Or use the Komodo stack for deployment.

### Environment Variables

See upstream documentation for base configuration. Fork-specific variables:

```bash
# === Additional Providers ===

# Anthropic (OAuth — use credential_tool for setup)
# Credentials stored in oauth_creds/anthropic_oauth_1.json

# Codex (OAuth — use credential_tool for setup)
# Credentials stored in oauth_creds/codex_oauth_1.json

# Cursor
CURSOR_API_KEY_1=your-cursor-key

# Lightning AI
LIGHTNING_AI_API_KEY_1=your-lightning-key

# NanoGPT
NANOGPT_API_KEY_1=your-nanogpt-key
# Optional: configure model source (models endpoint)
NANOGPT_MODELS="nanogpt/model-1,nanogpt/model-2"

# ZenMux (free models)
ZENMUX_API_BASE=https://zenmux.example.com/v1
ZENMUX_API_KEY_1=your-zenmux-key

# Firmware
FIRMWARE_API_BASE=https://firmware.example.com/v1
FIRMWARE_API_KEY_1=your-firmware-key

# DedalusLabs
DEDALUSLABS_API_BASE=https://api.dedaluslabs.ai/v1
DEDALUSLABS_API_KEY_1=dsk-live-xxxxx

# Gemini A2A (experimental — requires sidecar)
# See gemini_a2a_provider.py for sidecar configuration

# === Model Routing ===

# Static model aliases
MODEL_ALIAS_my-claude=anthropic/claude-sonnet-4-20250514
MODEL_ALIAS_fast=gemini/gemini-2.5-flash

# Cross-provider rotation (comma-separated targets)
MODEL_ALIAS_best-code=codex/o3-pro,anthropic/claude-sonnet-4-20250514

# === Per-Provider Tuning ===

# Override retry count for specific providers
MAX_RETRIES_NANOGPT=5
MAX_RETRIES_CHUTES=3
```

---

## Upstream Sync

This fork is regularly synced with upstream. See `.agent/skills/upstream-sync/` for the sync workflow.

Each feature is maintained on its own branch (`feature/<provider>-all` or `feature/<feature>-all`) for clean rebasing.

---

## License

Same as upstream — see [LICENSE](LICENSE).
