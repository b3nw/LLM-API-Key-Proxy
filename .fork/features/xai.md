# xAI provider

Canonical feature ID: `xai`
Stack subjects:
- `feat(xai): add xAI Grok OAuth provider with PKCE and Device Code flows`
- `feat(xai): enable xAI Grok device-code OAuth in admin WebUI`
Manifest: `.fork/stack.yml`

This file is the shared, repo-tracked history for xAI feature changes.
Local workspace state may contain run logs and scratch notes, but this file is
canonical across contributors and developer workspaces.

## 2026-06-19 — Fix OAuth credential hot-load, replacement, and cleanup on delete

Target: `feat(xai): enable xAI Grok device-code OAuth in admin WebUI`
Files:
- `src/proxy_app/api/oauth.py`
- `src/proxy_app/api/config.py`
- `src/proxy_app/main.py`

Working commits before autosquash:
- `eab0cebe fixup! feat(xai): enable xAI Grok device-code OAuth in admin WebUI`

Final stack commit after autosquash:
- `e10f4229 feat(xai): enable xAI Grok device-code OAuth in admin WebUI`

Verification:
- `uv run python3 -m py_compile src/proxy_app/api/oauth.py src/proxy_app/api/config.py src/proxy_app/main.py` — passed
- `uv run ruff check src/proxy_app/api/oauth.py src/proxy_app/api/config.py src/proxy_app/main.py --select F401,F811,F821,E9` — passed
- Hot-patched to live `llm-proxy` container on `docker-test`, restarted via Komodo, verified healthy with x-ai provider active (10 models, 1 credential)
- Browser smoke-tested credentials page at `https://llm-proxy.ext.ben.io/ui/credentials`

Notes:
- Three issues fixed in this changeset:
  1. **Bug: no hot-load on OAuth add** — `_save_credential_file()` wrote the JSON
     but never updated the running `RotatingClient`. New credentials only became
     active after a full restart. Added `_hot_load_credential()` to register the
     credential in `all_credentials`, `oauth_credentials`, `oauth_providers`, and
     re-initialize the usage manager immediately.
  2. **Bug: empty provider key after last credential deleted** — The delete endpoint
     filtered credentials out but left `all_credentials["x-ai"] = []`, which caused
     `model_discovery` to return `[]` and `background_refresher` to skip the
     provider. Now `del`s the empty key and discards from `oauth_providers`.
  3. **Feature: credential replacement** — Previously, adding a credential for the
     same account always created `_N+1.json`. Now `_find_existing_credential()`
     matches by email/login/account_id and overwrites in-place (with timestamped
     `.bak` backup), preventing duplicate files for the same identity.
- `set_app_ref(app)` added to `main.py` lifespan so background OAuth poll tasks
  (which lack a `Request` object) can access the `RotatingClient` for hot-loading.

## 2026-07-09 — Fix xAI usage classification: Grok Build vs API

Branch: `fix/xai-grok-build-useragent`
Files:
- `src/rotator_library/providers/x_ai_provider.py`

Problem:
- xAI usage was reporting as "API" (pay-per-token) instead of "Grok Build"
  (subscription) on the xAI billing dashboard.
- Root cause: the `openai` Python SDK (used via LiteLLM) sets
  `User-Agent: OpenAI/Python ...` on all requests. xAI uses the User-Agent
  to classify traffic into billing tracks.
- 9router (decolua/9router PR #1286) and CLIProxyAPI both route OAuth chat
  completions to `api.x.ai/v1` with `User-Agent: grok/<version>`, which xAI
  recognizes as Grok Build subscription traffic.

Fix:
- Added `_get_grok_build_headers()` returning `{"User-Agent": "grok/<version>"}`.
- `acompletion()` now injects this header via `extra_headers` on **all** xAI
  requests (not just CLI proxy models). LiteLLM merges `extra_headers` into
  the outgoing HTTP request, overriding the openai SDK's default User-Agent.
- `aembedding()` similarly injects the Grok Build User-Agent.
- CLI proxy models continue to get the additional `x-xai-token-auth` and
  `x-grok-client-version` headers on top.

Verification:
- `uv run python3 -m py_compile` and `uv run ruff check` — passed
- Hot-patched to `llm-proxy-dev` on `docker-test`, restarted, verified healthy
- Dry-run import: `XAiProvider()._get_grok_build_headers()` returns correct UA
- Header capture test confirmed `user-agent: grok/0.1.202` reaches upstream
  (via openai SDK `extra_headers` → httpx request)
- Test completions to `grok-4.3` and `grok-build-0.1` both succeeded

## 2026-07-09 — Route all chat completions through CLI proxy

Branch: `feat/xai-cli-proxy-only`
Files:
- `src/rotator_library/providers/x_ai_provider.py`

Problem:
- The provider had dual-endpoint routing: models classified as "CLI-proxy-only"
  went through `cli-chat-proxy.grok.com/v1`, all others through `api.x.ai/v1`.
- Testing confirmed that **every chat-capable model** works through the CLI proxy
  endpoint, making the dual routing unnecessary complexity.
- Non-chat models (image gen `grok-imagine-*`, video gen, multi-agent
  `grok-4.20-multi-agent*`) correctly reject `/chat/completions` on both endpoints.

Changes:
- `acompletion()` and `aembedding()` now always route through `cli_proxy_base`
  (`cli-chat-proxy.grok.com/v1`) instead of conditionally selecting per model.
- CLI proxy headers (`User-Agent`, `x-xai-token-auth`, `x-grok-client-version`)
  injected on every request — merged `_get_grok_build_headers()` into
  `_get_cli_proxy_headers()`.
- Removed `_cli_proxy_models` set, `_is_cli_proxy_model()`, and the
  versionless-alias deduplication logic from `get_models()`.
- Added `_NON_CHAT_MODEL_PREFIXES` and `_is_chat_model()` to filter out
  non-chat models from discovery results.
- Model discovery still fetches from both `api.x.ai` (broadest catalog) and
  CLI proxy (context window metadata), but merges into a single filtered list.
- `XAI_API_BASE` env var retained for discovery; no longer used for routing.
- Net change: −101 / +60 lines (simplification).

Verification:
- `uv run python3 -m py_compile` — passed
- `uv run ruff check --select F401,F811,F821,E9` — passed
- Pre-commit hook passed
- Tested all models against CLI proxy before making the change:
  `grok-4.3`, `grok-4.5`, `grok-4.20-0309-non-reasoning`,
  `grok-4.20-0309-reasoning`, `grok-build-0.1`, `grok-composer-2.5-fast`
  — all returned 200 OK via CLI proxy
