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

## fix(proxy): route provider plugins with custom logic through the configured proxy

**Branch:** `fix/proxy-support`
**Date:** 2026-09-03

### Problem

The previous proxy fix only helped providers that route through
`_resolve_litellm_client` (i.e. `has_custom_logic() == False`). The 4 providers
that build their own `openai.AsyncOpenAI(http_client=client)` - `opencode_go`,
`cline_pass`, `ollama_cloud`, `x_ai` - silently bypassed every configured
proxy because ``RequestExecutor._execute_non_streaming`` / ``_execute_streaming``
passed its unproxied shared client (``self._http_client``) into the plugin's
``acompletion(client=...)``. The plugin faithfully wired that unproxied client
into its own ``AsyncOpenAI``, so HTTPS traffic went straight to the upstream.
``_resolve_http_client`` was called and the proxy-aware ``httpx.AsyncClient``
was returned, but the executor then discarded it.

The streaming path made it worse: ``_resolve_http_client`` was never even
called there, so every streaming request bypassed the proxy regardless of
plugin.

The 12-hex ``stable_id`` format the user observed (``AF9907BBF840``) was
correct all along - SHA-256 of the API key, truncated to 12 chars. The bug
was purely in the executor, not in the lookup or the env-var naming.

### Changes

- **executor.py** (`_execute_non_streaming`) - pass the
  already-resolved ``request_client`` to ``plugin.acompletion`` /
  ``plugin.aembedding`` instead of ``self._http_client``.
- **executor.py** (`_execute_streaming`) - resolve ``request_client_s``
  via ``_resolve_http_client`` before the retry loop (it was missing
  entirely), and pass it to ``plugin.acompletion`` instead of
  ``self._http_client``.
- **tests/test_proxy_request_routing.py** (new) - E2E test that spins up
  a real local TCP socket as a fake CONNECT proxy (no mocking at the
  network seam) and asserts the executor's outbound CONNECT lands on it
  when ``PROXY_URL_CREDENTIAL_<id>`` is configured for the credential.
  Covers Pattern A (``opencode_go``, which wraps the client in
  ``openai.AsyncOpenAI(http_client=client)``) and Pattern B (``deepseek``,
  which calls ``client.post(...)`` and ``_stream_completion(client=client, ...)``
  directly). Both cases go RED on the unfixed executor and GREEN
  after this fix.

### Scope note

The fix benefits two distinct patterns of ``has_custom_logic() == True``
plugins, both verified by the new test:

- **Pattern A** - build own ``openai.AsyncOpenAI(http_client=client)``.
  4 providers: ``opencode_go``, ``cline_pass``, ``ollama_cloud``, ``x_ai``.
  The proxy-aware client flows directly into the OpenAI SDK's transport.
- **Pattern B** - call ``client.post/get/send(...)`` directly via
  internal helpers. ~10 providers including ``deepseek``, ``vertex``,
  ``codex``, ``anthropic_oauth_base``, ``gemini_cli``, ``nanogpt``,
  ``command``, ``chutes``, ``lightning_ai``. The proxy-aware client
  propagates through the helper call chain.

The remaining ``has_custom_logic() == True`` providers do not benefit
because they fall into one of these categories - both are provider-side
problems, not executor bugs:

- **Pattern C** - explicitly ignore the client and let LiteLLM's native
  handler manage HTTP. ``opencode_zen_provider.py`` has
  ``# client unused - LiteLLM manages its own`` in source. Fixing this
  requires a provider-specific refactor.
- **Pattern D** - receive the client but discard it before delegating
  to ``litellm.acompletion`` (e.g. ``anthropic_provider._apikey_completion``
  calls ``litellm.acompletion(**kwargs)`` without forwarding ``client``).
  One-line fix per affected provider (``kwargs["client"] = client``
  before the litellm call). Tracked separately.

### Verification

```bash
uv run python3 -m py_compile src/rotator_library/client/executor.py
uv run python3 -m py_compile tests/test_proxy_request_routing.py
uv run ruff check src/rotator_library/client/executor.py \
  tests/test_proxy_request_routing.py --select F401,F811,F821,E9
PYTHONPATH=src pytest tests/test_proxy_request_routing.py -v  # 2 passed
```

Full suite: 502 passed, 2 pre-existing failures unrelated to this change
(`tests/test_x_ai_quota_tracker.py::test_parse_period_end_ts_iso_z` uses a
hardcoded ``2026-07-01T00:00:00.000Z`` timestamp now in the past;
`tests/test_umans_quota_tracker.py::test_store_baselines_to_usage_manager`
HTTP 400 vs 200; `tests/test_failure_logger.py` and
`tests/utils/test_paths.py` have missing ``pytest-mock`` fixture / missing
``frozen`` attribute on the base branch - confirmed by stashing the patch
and re-running).
