---
trigger: always_on
---

# LLM-API-Key-Proxy Development Notes

## Deployment Rules

**IMPORTANT: Stack redeployment requires explicit user confirmation.**

- Container restarts (`docker restart llm-proxy`) are OK without asking
- Stack redeployments via Komodo or docker-compose require user confirmation first
- This applies to any action that would recreate the container with new configuration

## Development Workflow

### Branching Strategy

**See `AGENTS.md` (repo root) for full commit stack rules.**

Key points:
- `dev` is a **linear stack** of squashed commits on `upstream/dev` — no merge commits
- Each commit has a topic prefix: `feat(codex):`, `fix(core):`, `feat(tui):`, etc.
- **Find the owning commit** before making changes: `git log --oneline upstream/dev..HEAD`
- To fix an existing feature: commit as `fixup! <exact target message>`, then `GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash upstream/dev`
- New features: commit at the tip with a new `feat(<area>):` prefix
- **Always push with `--force-with-lease`** (dev is a rewritten branch)

### Hot-Patching (Live Testing)

- The user may request **hot-patching the remote container** for live testing
- This means copying changed files directly into the running container via SSH
- Hot-patches are temporary — the canonical source of truth is always git
- **Still commit the fix properly on dev first** (using the fixup! workflow)
- Example hot-patch workflow:
  ```bash
  # Copy a file into the running container
  scp src/rotator_library/providers/codex_provider.py docker-test:/tmp/codex_provider.py
  ssh docker-test 'docker cp /tmp/codex_provider.py llm-proxy:/app/src/rotator_library/providers/codex_provider.py && docker restart llm-proxy'
  ```

### Local Testing

Test locally using `uv run` — **do not use docker-compose for local testing**:
```bash
cd /home/b3nw/projects/core/LLM-API-Key-Proxy
uv run python src/proxy_app/main.py
```

## SSH Access for Docker Testing

The llm-proxy stack runs on docker.local.ben.io. Use the `docker-test` SSH config for access:

```bash
# Direct access
ssh docker-test

# Run docker commands
ssh docker-test 'docker logs llm-proxy 2>&1 | tail -50'
ssh docker-test 'docker exec llm-proxy python3 -c "print(\"test\")"'

# Restart container
ssh docker-test 'docker restart llm-proxy'
```

### SSH Config Entry (~/.ssh/config)
```
Host docker-test
    HostName docker.local.ben.io
    User root
    IdentityFile ~/.ssh/docker-test
    IdentitiesOnly yes
```

### Test URLs
- Proxy endpoint: https://llm-proxy.ext.ben.io
- Test API key: `sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO`

### Common Debug Commands
```bash
# Check container status
ssh docker-test 'docker ps | grep llm-proxy'

# View recent logs
ssh docker-test 'docker logs --since 5m llm-proxy 2>&1'

# Check environment variables
ssh docker-test 'docker exec llm-proxy env | grep -i PROVIDER_NAME'

# Test Python in container
ssh docker-test 'docker exec llm-proxy python3 -c "from rotator_library.providers import PROVIDER_PLUGINS; print(list(PROVIDER_PLUGINS.keys()))"'

# Check quota stats
curl -s -H "Authorization: Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO" "https://llm-proxy.ext.ben.io/v1/quota-stats" | jq '.providers | keys'
```

## Deployment Pipeline

### Docker Image Build (Automated)

Pushing to the `dev` branch triggers the GitHub Actions workflow (`.github/workflows/docker-build.yml`):
- Builds `ghcr.io/b3nw/llm-api-key-proxy:dev-latest`
- Also tags with a versioned tag (`YYYYMMDD-HHMMSS-<sha>`)
- Only builds `linux/amd64`

### Container Update Process

The container is managed by Komodo via docker compose. The compose file is at
`/opt/llm-proxy/env/compose.yaml` with project name `llm-proxy-new`.

**IMPORTANT: Always use `docker compose` to recreate the container, never bare
`docker run`. Using `docker run` removes compose labels and breaks Komodo's
ability to view logs and redeploy.**

After the GitHub Actions build completes:

```bash
# 1. Pull the new image
ssh docker-test 'docker pull ghcr.io/b3nw/llm-api-key-proxy:dev-latest'

# 2. Recreate the container via compose (pulls new image and restarts)
ssh docker-test 'cd /opt/llm-proxy/env && docker compose -p llm-proxy-new up -d llm-proxy'
```

If you need to force a full recreate (e.g., after config changes):
```bash
ssh docker-test 'cd /opt/llm-proxy/env && docker compose -p llm-proxy-new up -d --force-recreate llm-proxy'
```

### Remote Container Layout

```
/opt/llm-proxy/
├── cache/          → /app/cache       (model cache, etc.)
├── data/           → /app/data        (persistent data)
├── env/.env        → /app/.env (ro)   (environment variables / API keys)
├── logs/           → /app/logs        (transaction logs)
├── oauth_creds/    → /app/oauth_creds (OAuth credential JSON files)
└── usage/          → /app/usage       (usage tracking)
```

### Syncing OAuth Credentials

To sync local OAuth credentials to the remote:
```bash
scp oauth_creds/<provider>_oauth_*.json docker-test:/opt/llm-proxy/oauth_creds/
```

### Environment Configuration

Environment variables and API keys live in `/opt/llm-proxy/env/.env` on the remote host.
This file is mounted read-only into the container.

## Project Structure

- `src/proxy_app/main.py` - FastAPI application entry point
- `src/rotator_library/` - Core library for credential rotation
  - `client/rotating_client.py` - Main client with model discovery and credential rotation
  - `providers/` - Provider plugins (OpenAI, Anthropic, custom providers)
  - `providers/__init__.py` - Dynamic provider registration via `_register_providers()`

## Dynamic Provider Registration

Custom OpenAI-compatible providers are auto-detected via environment variables:
- `<PROVIDER>_API_BASE` - API base URL (required)
- `<PROVIDER>_API_KEY_1`, `<PROVIDER>_API_KEY_2`, etc. - API credentials

Example for dedaluslabs:
```
DEDALUSLABS_API_BASE=https://api.dedaluslabs.ai/v1
DEDALUSLABS_API_KEY_1=dsk-live-xxxxx
```
