# LLM API Key Proxy (Fork)

A personal fork of [Mirrowel/LLM-API-Key-Proxy](https://github.com/Mirrowel/LLM-API-Key-Proxy) with additional providers, fixes, and tooling.

> **For full documentation**, see the [upstream repository](https://github.com/Mirrowel/LLM-API-Key-Proxy).

---

## Fork-Specific Features

### Additional Providers

| Provider | Description |
|----------|-------------|
| **Cursor** | Cursor IDE API with quota monitoring integration |
| **Firmware.ai** | AI aggregator with shared quota tracking |
| **DedalusLabs** | OpenAI-compatible provider with tool_choice handling |

### Fixes & Improvements

- **Streaming token counting** — Correct `input_tokens` in `message_start` for Claude Code statusline
- **Anthropic null response handling** — Defensive checks for empty/invalid responses
- **Symlink atomic writes** — Resolve symlinks before writes for Docker volume mounts
- **Dynamic provider singleton fix** — Prevents api_base sharing between dynamic providers
- **Kimi K2.5 transform** — Auto-enforces `top_p=0.95` for Kimi models
- **Quota group sync** — Dynamic model discovery for accurate quota tracking

### Tooling

- **Transaction Log Viewer TUI** — Browse and inspect API request/response logs
- **Cursor Sidecar** — OpenAI-compatible proxy for Cursor API
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
# Cursor provider
CURSOR_API_KEY_1=your-cursor-key

# Firmware.ai
FIRMWARE_API_BASE=https://api.firmware.ai/v1
FIRMWARE_API_KEY_1=your-firmware-key

# DedalusLabs
DEDALUSLABS_API_BASE=https://api.dedaluslabs.ai/v1
DEDALUSLABS_API_KEY_1=dsk-live-xxxxx
```

---

## Upstream Sync

This fork is regularly synced with upstream. See `.agent/skills/upstream-sync/` for the sync workflow.

---

## License

Same as upstream — see [LICENSE](LICENSE).
