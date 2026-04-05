# Feedback on Proxy Support Plan

The plan is directionally good, but it overstates what is missing and misses a few repo-specific constraints.

## What already exists

- `OpenAIOAuthBase` already supports proxy-aware credential traffic through `_proxy_config` and `_build_proxy_client_kwargs()` in `src/rotator_library/providers/openai_oauth_base.py`.
- `AnthropicOAuthBase` already supports the same pattern in `src/rotator_library/providers/anthropic_oauth_base.py`.
- Provider instances already receive proxy config injection in the runtime proxy path:
  - `src/rotator_library/client/rotating_client.py`
  - `src/rotator_library/client/executor.py`

Because of that, the plan should not describe Anthropic as missing proxy support.

## Real gaps in the current codebase

### 1. `credential_tool.py` does not inject proxy config

`setup_new_credential()` currently does this:

```python
auth_instance = auth_class()
```

It does not assign `_proxy_config`, unlike the runtime path in `RotatingClient`. This is likely the biggest missing piece for credential setup flows.

### 2. `GoogleOAuthBase` is not proxy-aware yet

There are still direct `httpx.AsyncClient()` calls in:

- token refresh
- token validation
- token exchange
- user info fallback

Those calls should be routed through the same proxy-resolution logic used elsewhere.

### 3. `QwenAuthBase` is not proxy-aware yet

There are still direct `httpx.AsyncClient()` calls in:

- token refresh
- device flow initiation/polling

These also need the same proxy handling.

## `ProxyConfig` is already doing most of the heavy lifting

`src/rotator_library/proxy_config.py` already provides:

- global default proxy support
- per-provider proxy support
- per-credential proxy support
- rotation pool support
- scheme validation via `ProxySpec`

So the proposal to add validation utility is mostly a UX wrapper concern, not a major library gap. If needed, add a small CLI-facing helper that wraps `ProxySpec(url=...)` and produces a friendlier error message.

## Constructor injection may not match the existing architecture

The plan suggests passing `ProxyConfig` into auth-class constructors. That can work, but the existing codebase primarily uses post-construction injection by setting `instance._proxy_config` after creating the object.

That means the safest implementation is:

- keep `_proxy_config` as the internal field
- optionally support constructor injection if desired
- continue supporting post-construction assignment for compatibility with provider factory usage

## Naming consistency

The plan proposes a helper named `_get_client_kwargs()`, but the existing proxy-aware auth bases already use `_build_proxy_client_kwargs(...)`.

To stay consistent, prefer reusing the current naming pattern instead of inventing a second one unless you intend to refactor all auth bases together.

## Proxy precedence needs clarification

The proposed precedence is:

1. CLI override
2. `PROXY_URL_CREDENTIAL_<ID>`
3. `PROXY_URL_<PROVIDER>`
4. `PROXY_URL_DEFAULT`

But current `ProxyConfig.resolve()` precedence is:

1. per-credential
2. per-provider
3. rotation pool
4. default

So the plan needs to define exactly how a session-specific CLI override interacts with rotation and existing config. Important questions:

- Does the CLI override replace only the default proxy?
- Does it bypass provider and credential overrides?
- Is it intended only for the setup session, without persisting anything?

Without that definition, implementation details may become inconsistent.

## Connectivity validation should be treated as advisory

Validating proxy URL format is straightforward. Validating connectivity is more complicated:

- OAuth flows may touch multiple endpoints
- a generic connectivity check can produce false confidence
- some environments may allow provider endpoints but block a generic test target

Recommendation:

- do local URL validation with `ProxySpec`
- optionally do a lightweight provider-specific test
- treat connectivity-test failures as warnings, not definitive blockers

## Session-only proxy choice should stay in memory

For new credential setup, per-credential stable IDs may not be known until after auth or metadata discovery. Because of that, a session-level proxy override should probably use an in-memory `ProxyConfig(default=ProxySpec(url=...))` for the setup flow rather than trying to persist `PROXY_URL_CREDENTIAL_<ID>` immediately.

## Scope may be incomplete

If the goal is truly “credential management system” proxy support, the plan should also audit other auth bases used by setup flows, such as:

- `CopilotAuthBase`
- `IFlowAuthBase`

If the scope is intentionally limited to Google, Anthropic, and Qwen, that should be stated explicitly.

## Documentation scope

- `README.md`: yes, this makes sense
- `AGENTS.md`: probably not necessary for user-facing proxy behavior unless contributor workflow changes are being documented

## Recommended revised plan

### Phase 1: inject proxy config into credential setup

- Update `credential_tool.py` to load or construct a `ProxyConfig` for setup sessions
- Assign it to `auth_instance._proxy_config`
- Optionally allow a temporary session override

### Phase 2: add proxy-aware HTTP clients to missing auth bases

- `GoogleOAuthBase`
  - refresh
  - token validation
  - token exchange
  - user info lookup
- `QwenAuthBase`
  - refresh
  - device flow initiation/polling

### Phase 3: optional CLI UX improvements

- prompt to use configured default proxy when present
- allow temporary override
- validate with `ProxySpec`
- warn when user enters `socks5://` instead of `socks5h://`

### Phase 4: audit the remaining credential auth flows

- `CopilotAuthBase`
- `IFlowAuthBase`
- any other auth flows reachable from credential setup

### Phase 5: documentation

- update `README.md`

## Bottom line

This plan is partially accurate, but it is over-scoped in some areas and under-specific in the most important ones. The highest-value work is:

1. injecting proxy config in `credential_tool.py`
2. adding proxy-aware `httpx` client creation to `GoogleOAuthBase`
3. adding proxy-aware `httpx` client creation to `QwenAuthBase`
4. clarifying CLI override semantics relative to existing proxy resolution and rotation
