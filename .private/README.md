# Local Docker Testing — `docker-test` / `llm-proxy`

This folder contains reference information for the live Docker deployment
used for local integration testing.  It is **not** committed to the upstream
repo — it only exists in this fork.

---

## Container Overview

| Field | Value |
|-------|-------|
| Host | `docker-test` (SSH alias) |
| Container name | `llm-proxy` |
| Image | `ghcr.io/b3nw/llm-api-key-proxy:dev-latest` |
| Host port | `9220` → container `8000` |
| Restart policy | `unless-stopped` |

```bash
# Quick health check
ssh docker-test 'docker ps -f name=llm-proxy --format "{{.Names}}\t{{.Status}}"'
```

---

## Volume Mounts

All persistent state lives under `/opt/llm-proxy` on the host and is mounted
into the container at `/app`:

| Host path | Container path | Notes |
|-----------|---------------|-------|
| `/opt/llm-proxy/env/.env` | `/app/.env` | `ro` — all credentials & config |
| `/opt/llm-proxy/data` | `/app/data` | `rw` — key usage JSON, etc. |
| `/opt/llm-proxy/logs` | `/app/logs` | `rw` — proxy.log, failures.log, transactions/ |
| `/opt/llm-proxy/cache` | `/app/cache` | `rw` — model pricing & provider caches |
| `/opt/llm-proxy/usage` | `/app/usage` | `rw` — per-provider usage JSON files |
| `/opt/llm-proxy/oauth_creds` | `/app/oauth_creds` | `rw` — OAuth token stores |

The `custom_providers/` directory on the host **is not** volume-mounted
(it is baked into the image at build time).

---

## Remote Folder Structure

```
/opt/llm-proxy/
├── cache/                  # LiteLLM & provider caches
├── custom_providers/       # Host-side Python overrides (NOT mounted)
│   ├── chutes_quota_tracker.py
│   ├── error_handler.py
│   ├── executor.py
│   ├── model_info_service.py
│   ├── opencode_go_provider.py
│   ├── quota_viewer.py
│   ├── streaming.py
│   └── vertex_provider.py
├── data/
│   └── key_usage.json
├── env/
│   ├── .env                # LIVE credentials & whitelist config
│   └── compose.yaml        # Docker Compose definition
├── logs/
│   ├── failures.log        # Structured error log (RotatingFileHandler)
│   ├── proxy.log           # Main proxy log (RotatingFileHandler)
│   ├── proxy_debug.log     # Debug-level log (RotatingFileHandler)
│   └── transactions/       # Per-request dirs (cleaned by cron)
├── oauth_creds/            # OAuth token JSON files
├── scripts/
│   └── cleanup-logs.sh     # Cron script for transaction dir cleanup
├── usage/                  # Per-provider usage JSON files
└── test_opencode_quota.py  # Ad-hoc provider test script
```

---

## Hot-Patching a Running Container

> **⚠️ MANDATORY WORKFLOW — Test via hot-patch before pushing any code.**
>
> Every code change must be copied into the live container, verified against
> real traffic, and only then committed and pushed.  Do **not** push untested
> changes to `dev` — CI rebuilds the image automatically and a bad commit will
> break the live proxy.
>
> **MANDATORY:** After successful hot-patch verification, you MUST ask the
> user for explicit confirmation before pushing changes to the repository.

Because the container is built from `ghcr.io/b3nw/llm-api-key-proxy:dev-latest`,
the quickest way to test a code change without rebuilding the image is to copy
modified Python files directly into the running container.

### Safe hotpatch workflow (avoids crash-restart loops)

A bad patch causes `restart: unless-stopped` to spin the container in a tight
crash loop.  Follow these steps to gate on correctness **before** the container
ever tries to run the new code.

#### Step 1 — Lint locally first

```bash
# Syntax check (catches SyntaxError / bad imports at module level)
uv run python3 -m py_compile src/rotator_library/file_you_changed.py

# Undefined names, missing/unused imports
uv run ruff check src/rotator_library/file_you_changed.py --select F401,F811,F821,E9
```

Do **not** proceed if either command fails.

#### Step 2 — Copy into the container

```bash
scp src/rotator_library/file_you_changed.py docker-test:/tmp/patch.py
ssh docker-test 'docker cp /tmp/patch.py llm-proxy:/app/src/rotator_library/file_you_changed.py'
```

The container keeps serving traffic on the old in-memory code until it restarts,
so copying the file is safe even while it's running.

#### Step 3 — Dry-run the import INSIDE the container (before restart)

```bash
# Use the container's own Python/venv — catches dependency issues local checks miss
ssh docker-test 'docker exec llm-proxy python3 -c "from rotator_library import RotatingClient; print(\"OK\")"'
```

If this exits non-zero, fix the patch and repeat from Step 1.  The container is
still healthy and serving real traffic.

#### Step 4 — Restart and verify

```bash
ssh docker-test 'docker restart llm-proxy && sleep 5 && docker logs llm-proxy --tail 20'
```

---

### Iterative patching (disable auto-restart during a session)

If you expect to make several attempts, temporarily disable `restart: unless-stopped`
so a bad patch doesn't spin the container while you work:

```bash
# Disable auto-restart before your patch session
ssh docker-test 'docker update --restart=no llm-proxy'

# ... copy, test, iterate ...

# Re-enable when done
ssh docker-test 'docker update --restart=unless-stopped llm-proxy'
```

---

### Stop → patch → start (break an existing crash loop)

If the container is already crash-looping, `docker exec` won't work.  Stop it
first so you can patch safely:

```bash
ssh docker-test 'docker stop llm-proxy'
# Now copy the fixed file
ssh docker-test 'docker cp /tmp/fix.py llm-proxy:/app/src/rotator_library/__init__.py'
# Start (not restart) — gives you a clean single attempt
ssh docker-test 'docker start llm-proxy && sleep 5 && docker logs llm-proxy --tail 20'
```

### Important Notes

- **Ephemeral changes**: Files copied this way survive restarts but are lost
  when the image is updated (e.g. `docker compose pull && docker compose up -d`).
  Permanent fixes must be committed to this repo and pushed so CI rebuilds the
  image.
- **PYTHONPATH**: The container sets `PYTHONPATH=/app/src`, so modules are
  loaded from `/app/src/rotator_library/...` and `/app/src/proxy_app/...`.
- **No `custom_providers` mount**: The `custom_providers/` folder on the host
  is **not** mounted into the container.  If you want to hot-patch a custom
  provider, copy it to the container path manually or add a volume mount to
  `compose.yaml`.
- **Logs persist**: Because `/opt/llm-proxy/logs` is a volume mount, log data
  survives container restarts and image updates.

---

## Compose File

Location on host: `/opt/llm-proxy/env/compose.yaml`

```yaml
services:
  llm-proxy:
    container_name: llm-proxy
    image: ghcr.io/b3nw/llm-api-key-proxy:dev-latest
    restart: unless-stopped
    ports:
      - "9220:8000"
    volumes:
      - /opt/llm-proxy/env/.env:/app/.env:ro
      - /opt/llm-proxy/data:/app/data
      - /opt/llm-proxy/logs:/app/logs
      - /opt/llm-proxy/cache:/app/cache
      - /opt/llm-proxy/usage:/app/usage
      - /opt/llm-proxy/oauth_creds:/app/oauth_creds
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
      - SKIP_OAUTH_INIT_CHECK=true
    command: ["python", "src/proxy_app/main.py", "--host", "0.0.0.0", "--port", "8000", "--enable-request-logging"]
```

---

---

## Agent Skills & ATT Proxy Forward

Skills for managing this proxy are in `~/.pi/agent/skills/`:
- `cred-proxy-assign/` — Assign credentials to proxy routes
- `llm-proxy-investigate/` — Debug auth errors, test credentials, inspect logs
- `pi-skills/` — Pi agent utilities

The Codex OAuth credential builder skill is at `~/.config/opencode/skills/codex-oauth-creds/`.
It converts raw token data into `codex_oauth_N.json` files in `oauth_creds/`.

### ATT Proxy Forward

"ATT proxy forward" routes a credential's outbound traffic through a SOCKS5 proxy
that exits via an AT&T residential IP (`socks5h://att.exit.local.ben.io:1080`).
This makes LLM API requests appear to originate from an AT&T residential connection,
which prevents the upstream provider from detecting proxy/VPN IPs or associating
traffic with your server IP.

Per-credential proxy routing is configured via `PROXY_URL_CREDENTIAL_<SLUG>` env vars
in the remote `.env` file, where `<SLUG>` is an uppercase underscore-safe version
of the credential's stable ID (email or `email::account_id`).

For existing ATT proxy entries on the remote, see the `PROXY_URL_CREDENTIAL_*` lines
in `/opt/llm-proxy/env/.env` on `docker-test`.

---

## Model Inventory

### Querying `v1/models` on the live proxy

```bash
# Get full model list with pricing metadata (json)
ssh docker-test 'curl -s -H "Authorization: Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO" \
  http://localhost:9220/v1/models'

# Extract just model IDs (summary)
ssh docker-test 'curl -s -H "Authorization: Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO" \
  http://localhost:9220/v1/models' | python3 -c "
import json, sys
data = json.load(sys.stdin)
ids = sorted([m['id'] for m in data.get('data', [])])
print(f'Total: {len(ids)}')
for mid in ids:
    print(mid)
"

# Filter by provider prefix (e.g. opencode_go)
ssh docker-test 'curl -s -H "Authorization: Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO" \
  http://localhost:9220/v1/models' | python3 -c "
import json, sys
data = json.load(sys.stdin)
prefix = sys.argv[1] if len(sys.argv) > 1 else ''
ids = sorted([m['id'] for m in data.get('data', []) if m['id'].startswith(prefix)])
print(f'{prefix} models ({len(ids)}):')
for mid in ids:
    print(f'  {mid}')
" opencode_go

# Check what models a provider loads at startup
ssh docker-test 'docker logs llm-proxy 2>&1 | grep "Loaded.*models for provider" | sort | uniq -c'
```

### Upstream model discovery (opencode.ai)

The `opencode_go` provider routes to the **go tier** (`/zen/go/v1`). Query the upstream
endpoints directly to see what models are available:

```bash
# Go-tier models (15 models — the "go" workspace)
KEY="<opencode_go_api_key>"
curl -s -H "Authorization: Bearer $KEY" https://opencode.ai/zen/go/v1/models | \
  python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin).get('data',[])]"

# Full Zen catalog (includes GPT, Claude, Gemini, plus free models)
curl -s -H "Authorization: Bearer $KEY" https://opencode.ai/zen/v1/models | \
  python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin).get('data',[])]"
```

---

## Log Investigation Commands

```bash
# Tail live proxy log
ssh docker-test 'tail -f /opt/llm-proxy/logs/proxy.log'

# Recent failures (structured JSON)
ssh docker-test 'tail -20 /opt/llm-proxy/logs/failures.log'

# Find transaction logs for a specific request
ssh docker-test 'ls -d /opt/llm-proxy/logs/transactions/0430_* | tail -5'

# Inspect a specific transaction
ssh docker-test 'cat /opt/llm-proxy/logs/transactions/0430_002100_oai_nanogpt_deepseek_deepseek-v4-flash_7f054e72/request.json'
```
