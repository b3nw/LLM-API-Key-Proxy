# proxy — Outbound HTTP/SOCKS5 proxy support

## fix(proxy): resolve per-credential proxy routing and improve observability

**Branch:** `fix/proxy-routing-and-logging`
**Date:** 2026-07-09

### Problem

Per-credential proxy assignments (`PROXY_URL_CREDENTIAL_*`) were silently
ignored at runtime. Two root causes:

1. `ProxyConfig.resolve()` compared env-var slugs (e.g. `FALTERS_RIZON_NET`)
   against raw runtime `stable_id`s that include `email::account_id` suffixes,
   so the lookup never matched.
2. The streaming code path (`_execute_streaming`) never called
   `_resolve_litellm_client`, so streaming requests always bypassed the proxy.

### Changes

- **proxy_config.py** — `resolve()` now builds candidate slugs from the raw
  `stable_id`, its slugified form, and the email-only prefix (before `::`).
  Added `named_proxies` dict and `get_proxy_name()` for alias lookup.
  `load_proxy_config()` logs a warning on invalid `PROXY_NAME_*` URLs instead
  of silently discarding them.
- **executor.py** — Extracted `_resolve_proxy_spec()` to resolve `ProxySpec`
  and alias name once per request. Both non-streaming and streaming paths
  resolve once, log the proxy alias at INFO, store `proxy_name` in
  `TransactionLogger`, and pass the pre-resolved spec into
  `_resolve_litellm_client` to avoid redundant resolution.
- **transaction_logger.py** — Added `proxy_name` slot, included in
  `metadata.json` output.
- **api/logs.py** — `list_transactions` returns `proxy_name` from metadata.
- **webui** — `Logs.tsx` shows a blue globe badge on proxied transactions.
  `Settings.tsx` displays named proxy alias alongside the raw URL.
  TypeScript interfaces updated in `config.ts` and `logs.ts`.

### Verification

```bash
uv run python3 -m py_compile src/rotator_library/proxy_config.py
uv run python3 -m py_compile src/rotator_library/client/executor.py
uv run ruff check src/rotator_library/proxy_config.py --select F401,F811,F821,E9
uv run ruff check src/rotator_library/client/executor.py --select F401,F811,F821,E9
```

Hot-patched to `llm-proxy-dev` and confirmed:
- x-ai credential proxy resolves and routes through the named proxy
- Transaction metadata includes `proxy_name`
- Log Explorer shows proxy badge on proxied requests
- Settings page shows alias name alongside URL
