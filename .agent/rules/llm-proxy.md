---
trigger: always_on
---

llm-proxy is hosted on a docker container locally in our homelab:
https://llm-proxy.ext.ben.io/v1
key: sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO

Investigation into logs or configuration can be done via the docker-test ssh access.
ssh docker-test and review /opt/llm-proxy

No changes should be made to the llm-proxy-new stack, or its containers through komodo or otherwise without the explicit user confirmation, or direct instruction.

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