"""Admin API for proxy configuration and credential management."""

import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from rotator_library.proxy_config import ProxySpec, _slugify_stable_id, load_proxy_config
from rotator_library.utils.paths import get_data_file

_credential_lock = asyncio.Lock()

router = APIRouter(prefix="/v1/admin", tags=["admin-config"])


def _read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)

logger = logging.getLogger(__name__)

# Matches a .env KEY (exported or not), capturing the key name.
# Handles: KEY=..., export KEY=...,  KEY =...
_ENV_KEY_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _env_path() -> Path:
    return get_data_file(".env")


def _inplace_set_key(dotenv_path: str, key: str, value: str) -> None:
    """Write-in-place replacement for dotenv.set_key.

    python-dotenv's set_key uses os.replace() under the hood, which fails
    with EBUSY when the .env file is a Docker bind-mount. This helper
    reads, modifies, and writes back in-place (truncate mode) instead.
    """
    path = Path(dotenv_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines(keepends=True)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    new_line = f'{key}="{escaped}"\n'

    found = False
    for i, line in enumerate(lines):
        m = _ENV_KEY_RE.match(line)
        if m and m.group(1) == key:
            lines[i] = new_line
            found = True
            break

    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _inplace_unset_key(dotenv_path: str, key: str) -> None:
    """Write-in-place replacement for dotenv.unset_key.

    Same motivation as _inplace_set_key — avoids os.replace().
    """
    path = Path(dotenv_path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    for line in lines:
        m = _ENV_KEY_RE.match(line)
        if m and m.group(1) == key:
            continue
        new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _oauth_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path.cwd()
    d = base / "oauth_creds"
    d.mkdir(exist_ok=True)
    return d


def _get_env_vars() -> dict[str, str]:
    """Read all env vars from the .env file."""
    from dotenv import dotenv_values
    vals = dotenv_values(_env_path())
    return {k: v for k, v in vals.items() if v is not None}


def _parse_named_proxies(env_vars: dict[str, str]) -> dict[str, str]:
    """Parse PROXY_LIST and PROXY_NAME_* env vars into a {name: url} dict.

    PROXY_LIST is a comma-separated list of proxy names (e.g. "us-east,eu-west").
    For each name, the URL is read from PROXY_NAME_<UPPERCASE_NAME_WITH_UNDERSCORES>.
    Each URL is validated against supported proxy schemes via ProxySpec.
    """
    named: dict[str, str] = {}
    raw_list = env_vars.get("PROXY_LIST", "")
    if not raw_list.strip():
        return named
    for name in raw_list.split(","):
        name = name.strip()
        if not name:
            continue
        env_key = "PROXY_NAME_" + re.sub(r"[^A-Z0-9]", "_", name.upper())
        url = env_vars.get(env_key)
        if not url:
            logger.warning(f"PROXY_LIST contains '{name}' but {env_key} is not set")
            continue
        try:
            ProxySpec(url=url)  # validates scheme
        except ValueError as exc:
            logger.warning(f"Named proxy '{name}' has invalid URL '{url}': {exc}")
            continue
        named[name] = url
    return named


def _resolve_credential_proxy_name(
    slug: str, env_vars: dict[str, str], named_proxies: dict[str, str]
) -> Optional[str]:
    """Given a credential slug, find which named proxy (if any) is assigned.

    Looks up PROXY_URL_CREDENTIAL_<slug> in env_vars, then matches the URL
    against the named_proxies dict to find the proxy name.
    """
    key = f"PROXY_URL_CREDENTIAL_{slug.upper()}"
    url = env_vars.get(key)
    if not url:
        return None
    for name, named_url in named_proxies.items():
        if named_url == url:
            return name
    # Proxy URL is set but doesn't match any named proxy (e.g. the proxy was
    # removed from PROXY_LIST after assignment, or it was set manually via env).
    # Return the raw URL so the frontend can display it and the user knows
    # traffic is still being routed through a proxy they may not expect.
    return url


def _mask_key(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


@router.get("/config")
async def get_config():
    env_vars = _get_env_vars()
    oauth_dir = _oauth_dir()

    try:
        from proxy_app.provider_urls import PROVIDER_URL_MAP
    except ImportError:
        PROVIDER_URL_MAP = {}

    providers: dict = {}
    custom_providers: dict = {}
    concurrency: dict = {}
    rotation_modes: dict = {}
    model_filters: dict = {}
    latest_aliases: dict = {}
    strip_suffixes: list = []

    for key, value in env_vars.items():
        if key == "PROXY_API_KEY":
            continue

        api_key_match = re.match(r"^(.+?)_API_KEY(?:_\d+)?$", key)
        if api_key_match and not key.startswith("PROXY_"):
            provider_name = api_key_match.group(1).lower()
            if provider_name not in providers:
                providers[provider_name] = {"api_key_count": 0, "oauth_count": 0, "has_custom_base": False}
            providers[provider_name]["api_key_count"] += 1

        elif key.endswith("_API_BASE"):
            provider_name = key.replace("_API_BASE", "").lower()
            if provider_name not in PROVIDER_URL_MAP:
                custom_providers[provider_name] = value
            if provider_name not in providers:
                providers[provider_name] = {"api_key_count": 0, "oauth_count": 0, "has_custom_base": True}
            else:
                providers[provider_name]["has_custom_base"] = True

        elif key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
            provider_name = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
            if provider_name not in concurrency:
                concurrency[provider_name] = {"max": -1, "optimal": -1}
            try:
                concurrency[provider_name]["max"] = int(value)
            except ValueError:
                pass

        elif key.startswith("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_"):
            provider_name = key.replace("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
            if provider_name not in concurrency:
                concurrency[provider_name] = {"max": -1, "optimal": -1}
            try:
                concurrency[provider_name]["optimal"] = int(value)
            except ValueError:
                pass

        elif key.startswith("ROTATION_MODE_"):
            provider_name = key.replace("ROTATION_MODE_", "").lower()
            rotation_modes[provider_name] = value

        elif key.startswith("IGNORE_MODELS_"):
            provider_name = key.replace("IGNORE_MODELS_", "").lower()
            if provider_name not in model_filters:
                model_filters[provider_name] = {"ignore": [], "whitelist": []}
            model_filters[provider_name]["ignore"] = [p.strip() for p in value.split(",") if p.strip()]

        elif key.startswith("WHITELIST_MODELS_"):
            provider_name = key.replace("WHITELIST_MODELS_", "").lower()
            if provider_name not in model_filters:
                model_filters[provider_name] = {"ignore": [], "whitelist": []}
            model_filters[provider_name]["whitelist"] = [p.strip() for p in value.split(",") if p.strip()]

        elif key.startswith("MODEL_LATEST_") and key != "MODEL_LATEST_STRIP_SUFFIXES":
            alias_name = key.replace("MODEL_LATEST_", "").lower()
            latest_aliases[alias_name] = value

        elif key == "MODEL_LATEST_STRIP_SUFFIXES":
            strip_suffixes = [s.strip() for s in value.split(",") if s.strip()]

    # Collect PROXY_URL_* settings
    proxy_urls: dict = {}
    for key, value in env_vars.items():
        if key == "PROXY_URL_DEFAULT":
            proxy_urls["default"] = value
        elif key.startswith("PROXY_URL_CREDENTIAL_"):
            slug = key[len("PROXY_URL_CREDENTIAL_"):].lower()
            proxy_urls.setdefault("credentials", {})[slug] = value
        elif key.startswith("PROXY_URL_") and not key.startswith("PROXY_URL_CREDENTIAL_"):
            provider = key[len("PROXY_URL_"):].lower()
            proxy_urls.setdefault("providers", {})[provider] = value

    # Parse named proxies from PROXY_LIST + PROXY_NAME_*
    named_proxies = _parse_named_proxies(env_vars)

    # Resolve which named proxy each credential is using
    if "credentials" in proxy_urls and named_proxies:
        cred_proxy_names: dict[str, str] = {}
        url_to_name = {url: name for name, url in named_proxies.items()}
        for slug, url in proxy_urls["credentials"].items():
            proxy_name = url_to_name.get(url)
            if proxy_name:
                cred_proxy_names[slug] = proxy_name
        if cred_proxy_names:
            proxy_urls["credential_proxy_names"] = cred_proxy_names

    # Count OAuth credentials from files
    if oauth_dir.exists():
        for f in oauth_dir.iterdir():
            if f.is_file() and f.suffix == ".json" and "_oauth_" in f.name:
                provider_name = f.name.split("_oauth_")[0].lower()
                if provider_name not in providers:
                    providers[provider_name] = {"api_key_count": 0, "oauth_count": 0, "has_custom_base": False}
                providers[provider_name]["oauth_count"] += 1

    result: dict = {
        "proxy_api_key_set": bool(env_vars.get("PROXY_API_KEY")),
        "providers": providers,
        "custom_providers": custom_providers,
        "concurrency": concurrency,
        "rotation_modes": rotation_modes,
        "model_filters": model_filters,
        "latest_aliases": latest_aliases,
        "strip_suffixes": strip_suffixes,
    }
    if proxy_urls:
        result["proxy_urls"] = proxy_urls
    if named_proxies:
        result["available_proxies"] = [{"name": n, "url": u} for n, u in named_proxies.items()]
    return result


class ConfigUpdate(BaseModel):
    changes: dict[str, Optional[str]]


_CONFIG_BLOCKED_KEYS = {"PROXY_API_KEY", "PATH", "HOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"}
_CONFIG_ALLOWED_PREFIXES = (
    "ROTATION_MODE_", "MAX_CONCURRENT_REQUESTS_PER_KEY_", "OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_",
    "IGNORE_MODELS_", "WHITELIST_MODELS_", "MODEL_LATEST_",
)


@router.patch("/config")
async def update_config(update: ConfigUpdate):
    env_file = str(_env_path())
    updated = []
    rejected = []
    for key, value in update.changes.items():
        if key in _CONFIG_BLOCKED_KEYS:
            rejected.append(key)
            continue
        if not any(key.startswith(p) for p in _CONFIG_ALLOWED_PREFIXES) and not key.endswith(("_API_BASE",)):
            rejected.append(key)
            continue
        if value is None:
            _inplace_unset_key(env_file, key)
            os.environ.pop(key, None)
        else:
            _inplace_set_key(env_file, key, value)
            os.environ[key] = value
        updated.append(key)

    load_dotenv(env_file, override=True)
    result: dict = {"updated": updated}
    if rejected:
        result["rejected"] = rejected
    return result


@router.get("/credentials")
async def get_credentials(request: Request):
    env_vars = _get_env_vars()
    oauth_dir = _oauth_dir()

    # Build lookups of runtime credential status and tier from quota stats
    runtime_status: dict[str, str] = {}
    runtime_tiers: dict[str, str] = {}
    loaded_providers: set[str] = set()
    try:
        client = request.app.state.rotating_client
        loaded_providers = {p.lower() for p in client.all_credentials}
        quota_stats = await client.get_quota_stats()
        for pstats in quota_stats.get("providers", {}).values():
            for cred_data in pstats.get("credentials", {}).values():
                full_path = cred_data.get("full_path", "")
                if full_path:
                    fname = Path(full_path).name
                    runtime_status[fname] = cred_data.get("status", "unknown")
                    tier = cred_data.get("tier")
                    if tier:
                        runtime_tiers[fname] = tier
    except Exception:
        pass

    # Cross-reference ErrorTracker for credentials with token refresh errors
    errored_creds: set[str] = set()
    try:
        from rotator_library.error_tracker import get_error_tracker
        tracker = get_error_tracker()
        records, _ = tracker.get_recent_errors(limit=50)
        for rec in records:
            if rec.error_type in (
                "CredentialNeedsReauth",
                "TokenRefreshFailed",
                "QuotaAuthFailed",
                "BillingAuthFailed",
            ):
                cred_id = rec.credential_masked
                errored_creds.add(cred_id)
    except Exception:
        pass

    api_keys: dict[str, list] = {}
    named_proxies = _parse_named_proxies(env_vars)
    for key, value in env_vars.items():
        api_key_match = re.match(r"^(.+?)_API_KEY(?:_\d+)?$", key)
        if api_key_match and not key.startswith("PROXY_"):
            provider_name = api_key_match.group(1).lower()
            if provider_name not in api_keys:
                api_keys[provider_name] = []
            stable_id = hashlib.sha256(value.encode()).hexdigest()[:12]
            slug = _slugify_stable_id(stable_id)
            proxy_name = _resolve_credential_proxy_name(slug, env_vars, named_proxies)
            api_keys[provider_name].append({
                "key_name": key,
                "masked_value": _mask_key(value),
                "provider": provider_name,
                "stable_id": slug,
                "proxy": proxy_name,
            })

    oauth: dict[str, list] = {}
    if oauth_dir.exists():
        for f in sorted(oauth_dir.iterdir()):
            if f.is_file() and f.suffix == ".json" and "_oauth_" in f.name:
                provider_name = f.name.split("_oauth_")[0].lower()
                if provider_name not in oauth:
                    oauth[provider_name] = []
                # Extract number from filename (e.g. codex_oauth_2.json -> 2)
                num_match = re.search(r"_oauth_(\d+)\.json$", f.name)
                cred_number = int(num_match.group(1)) if num_match else None
                info: dict = {
                    "filename": f.name,
                    "provider": provider_name,
                    "number": cred_number,
                }
                try:
                    data = await asyncio.to_thread(_read_json, f)
                    meta = data.get("_proxy_metadata", {})
                    info["email"] = meta.get("email") or meta.get("login") or data.get("email")
                    info["tier"] = (
                        meta.get("tier")
                        or meta.get("plan_type")
                        or meta.get("sku")
                        or runtime_tiers.get(f.name)
                    )
                    file_status = meta.get("status", "unknown")
                    # Runtime status takes precedence, then file metadata,
                    # then infer "active" if the provider is loaded in the proxy
                    resolved = runtime_status.get(f.name)
                    if not resolved:
                        if file_status and file_status != "unknown":
                            resolved = file_status
                        elif provider_name in loaded_providers:
                            resolved = "active"
                        else:
                            resolved = "unknown"
                    # Override to needs_reauth if ErrorTracker has recent refresh errors
                    if resolved == "active" and f.name in errored_creds:
                        resolved = "needs_reauth"
                    info["status"] = resolved
                    # stable_id and proxy assignment
                    stable_id = info.get("email") or meta.get("login") or data.get("login")
                    if stable_id:
                        slug = _slugify_stable_id(stable_id)
                        info["stable_id"] = slug
                        info["proxy"] = _resolve_credential_proxy_name(slug, env_vars, named_proxies)
                    else:
                        info["stable_id"] = None
                        info["proxy"] = None
                except Exception:
                    info["status"] = runtime_status.get(f.name, "error")
                oauth[provider_name].append(info)

    return {"api_keys": api_keys, "oauth": oauth}


@router.get("/proxies")
async def get_proxies():
    """Return the list of named proxies available for per-credential assignment,
    plus the global default proxy URL (if any).
    """
    env_vars = _get_env_vars()
    named = _parse_named_proxies(env_vars)
    default = env_vars.get("PROXY_URL_DEFAULT")
    return {
        "proxies": [{"name": n, "url": u} for n, u in named.items()],
        "default": default,
    }


class SetCredentialProxyRequest(BaseModel):
    credential_slug: str  # The env-var-safe slug (e.g. USER_GMAIL_COM)
    proxy_name: Optional[str] = None  # Named proxy from PROXY_LIST, or None to clear


@router.put("/credentials/proxy")
async def set_credential_proxy(req: SetCredentialProxyRequest, request: Request):
    """Assign or clear a named proxy for a specific credential.

    Writes PROXY_URL_CREDENTIAL_<slug>=<url> to .env, then hot-reloads
    the running ProxyConfig if one exists on app.state.
    """
    env_file = str(_env_path())
    env_vars = _get_env_vars()
    named_proxies = _parse_named_proxies(env_vars)

    if req.proxy_name is not None:
        if req.proxy_name not in named_proxies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown proxy name '{req.proxy_name}'. "
                       f"Available: {list(named_proxies.keys())}",
            )
        proxy_url = named_proxies[req.proxy_name]
        env_key = f"PROXY_URL_CREDENTIAL_{req.credential_slug}"
        _inplace_set_key(env_file, env_key, proxy_url)
        os.environ[env_key] = proxy_url
    else:
        env_key = f"PROXY_URL_CREDENTIAL_{req.credential_slug}"
        _inplace_unset_key(env_file, env_key)
        os.environ.pop(env_key, None)

    load_dotenv(env_file, override=True)

    # Hot-reload the ProxyConfig on the running RotatingClient if present.
    # The ProxyConfig instance lives at client._proxy_config (set during
    # RotatingClient.__init__), not at app.state.proxy_config.
    try:
        client = getattr(request.app.state, "rotating_client", None)
        if client is not None:
            proxy_config = getattr(client, "_proxy_config", None)
            if proxy_config is not None:
                new_config = load_proxy_config(env=dict(os.environ))
                # Copy fields into the existing config object to preserve identity
                proxy_config.default = new_config.default
                proxy_config.provider_proxies = new_config.provider_proxies
                proxy_config.credential_proxies = new_config.credential_proxies
                proxy_config.rotation_pool = new_config.rotation_pool
                proxy_config.rotation_strategy = new_config.rotation_strategy
                proxy_config.rotation_scope = new_config.rotation_scope
                logger.info(
                    f"Hot-reloaded proxy config after credential proxy update "
                    f"(slug={req.credential_slug}, name={req.proxy_name})"
                )
    except Exception:
        logger.warning("Could not hot-reload ProxyConfig", exc_info=True)

    return {
        "ok": True,
        "credential_slug": req.credential_slug,
        "proxy_name": req.proxy_name,
    }


class AddApiKeyRequest(BaseModel):
    provider: str = Field(pattern=r"^[a-zA-Z0-9_]+$", min_length=1, max_length=50)
    key: str = Field(min_length=1, max_length=500)


async def _ensure_usage_manager(client, provider: str, credentials: list) -> None:
    """Create a usage manager if missing, then (re-)initialize with current credentials."""
    from rotator_library.usage.config import load_provider_usage_config
    from rotator_library.usage import UsageManager as NewUsageManager

    usage_manager = client.get_usage_manager(provider)
    if usage_manager is None:
        reg = client._usage_registry
        config = load_provider_usage_config(provider, client._provider_plugins)
        config.rotation_tolerance = reg._rotation_tolerance
        reg.apply_usage_reset_config(provider, credentials, config)
        mode = config.rotation_mode.value
        max_c, opt_c = reg.get_concurrency_settings(provider, mode)
        usage_manager = NewUsageManager(
            provider=provider,
            file_path=client._usage_base_path / f"usage_{provider}.json",
            provider_plugins=client._provider_plugins,
            config=config,
            max_concurrent_per_key=max_c,
            optimal_concurrent_per_key=opt_c,
        )
        reg.managers[provider] = usage_manager

    priorities, tiers = client._usage_registry.get_credential_metadata(
        provider, credentials
    )
    await usage_manager.initialize(
        credentials, priorities=priorities, tiers=tiers
    )

    plugin = client._get_provider_instance(provider)
    if plugin and hasattr(plugin, "set_usage_manager"):
        plugin.set_usage_manager(usage_manager)


async def _hot_load_api_key(client, provider: str, api_key: str) -> bool:
    """Hot-load a new API key into the running client's credential maps.

    Returns True if the credential was newly added, False if it was already present.
    """
    provider = provider.lower()
    added = False

    client.api_keys.setdefault(provider, [])
    if api_key not in client.api_keys[provider]:
        client.api_keys[provider].append(api_key)
        added = True

    client.all_credentials.setdefault(provider, [])
    if api_key not in client.all_credentials[provider]:
        client.all_credentials[provider].append(api_key)
        added = True

    if added:
        await _ensure_usage_manager(client, provider, client.all_credentials[provider])

    return added


@router.post("/credentials/api-key")
async def add_api_key(req: AddApiKeyRequest, request: Request):
    async with _credential_lock:
        env_file = str(_env_path())
        env_vars = _get_env_vars()

        provider_upper = req.provider.upper()
        existing = [k for k in env_vars if k.startswith(f"{provider_upper}_API_KEY")]
        if existing:
            nums = []
            for k in existing:
                suffix = k.replace(f"{provider_upper}_API_KEY", "")
                if suffix.startswith("_") and suffix[1:].isdigit():
                    nums.append(int(suffix[1:]))
                elif not suffix:
                    nums.append(0)
            next_num = max(nums) + 1 if nums else 1
            key_name = f"{provider_upper}_API_KEY_{next_num}"
        else:
            key_name = f"{provider_upper}_API_KEY"

        _inplace_set_key(env_file, key_name, req.key)
        os.environ[key_name] = req.key
        load_dotenv(env_file, override=True)

        hot_loaded = False
        try:
            client = request.app.state.rotating_client
            hot_loaded = await _hot_load_api_key(client, req.provider, req.key)
        except Exception as exc:
            logger.warning(
                f"Could not hot-load API key for {req.provider}", exc_info=True
            )

    return {"key_name": key_name, "hot_loaded": hot_loaded}


@router.delete("/credentials/api-key/{provider}/{key_name}")
async def delete_api_key(provider: str, key_name: str, request: Request):
    async with _credential_lock:
        env_file = str(_env_path())
        env_vars = _get_env_vars()
        if key_name not in env_vars:
            raise HTTPException(status_code=404, detail=f"Key {key_name} not found")

        key_value = env_vars[key_name]
        _inplace_unset_key(env_file, key_name)
        os.environ.pop(key_name, None)
        load_dotenv(env_file, override=True)

        try:
            client = request.app.state.rotating_client
            provider_lower = provider.lower()
            if provider_lower in client.all_credentials:
                client.all_credentials[provider_lower] = [
                    c for c in client.all_credentials[provider_lower]
                    if c != key_value
                ]
            if provider_lower in client.api_keys:
                client.api_keys[provider_lower] = [
                    c for c in client.api_keys[provider_lower]
                    if c != key_value
                ]
        except Exception as e:
            logger.warning(f"Could not remove API key from running proxy: {e}")

    return {"deleted": key_name}


@router.delete("/credentials/oauth/{provider}/{filename}")
async def delete_oauth_credential(provider: str, filename: str, request: Request):
    async with _credential_lock:
        oauth_dir = _oauth_dir()
        target = oauth_dir / filename
        if not target.exists():
            raise HTTPException(status_code=404, detail="OAuth credential not found")
        if not target.resolve().is_relative_to(oauth_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")

        removed_accessor = str(target.resolve())
        target.unlink()

        removed_from_proxy = False
        try:
            client = request.app.state.rotating_client
            provider_lower = provider.lower()
            if provider_lower in client.all_credentials:
                before = len(client.all_credentials[provider_lower])
                client.all_credentials[provider_lower] = [
                    c for c in client.all_credentials[provider_lower]
                    if not c.endswith(filename)
                ]
                removed_from_proxy = len(client.all_credentials[provider_lower]) < before
                if not client.all_credentials[provider_lower]:
                    del client.all_credentials[provider_lower]
            if provider_lower in client.oauth_credentials:
                client.oauth_credentials[provider_lower] = [
                    c for c in client.oauth_credentials[provider_lower]
                    if not c.endswith(filename)
                ]
                if not client.oauth_credentials[provider_lower]:
                    del client.oauth_credentials[provider_lower]
                    if hasattr(client, "oauth_providers"):
                        client.oauth_providers.discard(provider_lower)

            # Remove from usage manager so stale state isn't persisted on shutdown
            usage_manager = client.get_usage_manager(provider_lower)
            if usage_manager:
                await usage_manager.remove_credential(removed_accessor)
        except Exception as e:
            logger.warning(f"Could not remove credential from running proxy: {e}")

    return {"deleted": filename, "removed_from_proxy": removed_from_proxy}


class AddCustomProviderRequest(BaseModel):
    name: str
    base_url: str
    api_key: str


async def _hot_load_custom_provider(client, provider_name: str, api_key: str) -> dict:
    """Register a new custom OpenAI-compatible provider at runtime."""
    from rotator_library.providers import PROVIDER_PLUGINS, DynamicOpenAICompatibleProvider
    from rotator_library.provider_config import KNOWN_PROVIDERS

    provider = provider_name.lower()
    result = {"plugin_registered": False, "models_discovered": 0}

    if provider not in KNOWN_PROVIDERS and provider not in PROVIDER_PLUGINS:
        def _make_plugin(name):
            class _Plug(DynamicOpenAICompatibleProvider):
                def __init__(self):
                    super().__init__(name)
            return _Plug

        PROVIDER_PLUGINS[provider] = _make_plugin(provider)
        result["plugin_registered"] = True

    client.provider_config._load_api_bases()

    client.api_keys.setdefault(provider, [])
    if api_key not in client.api_keys[provider]:
        client.api_keys[provider].append(api_key)

    client.all_credentials.setdefault(provider, [])
    if api_key not in client.all_credentials[provider]:
        client.all_credentials[provider].append(api_key)

    await _ensure_usage_manager(client, provider, client.all_credentials[provider])

    try:
        models = await client.get_available_models(provider, force_refresh=True)
        result["models_discovered"] = len(models)
    except Exception as exc:
        logger.warning(f"Model discovery failed for {provider}: {exc}")
        result["model_discovery_error"] = str(exc)

    return result


@router.post("/credentials/custom-provider")
async def add_custom_provider(req: AddCustomProviderRequest, request: Request):
    async with _credential_lock:
        env_file = str(_env_path())
        provider_upper = req.name.upper()

        _inplace_set_key(env_file, f"{provider_upper}_API_BASE", req.base_url)
        _inplace_set_key(env_file, f"{provider_upper}_API_KEY", req.api_key)
        os.environ[f"{provider_upper}_API_BASE"] = req.base_url
        os.environ[f"{provider_upper}_API_KEY"] = req.api_key
        load_dotenv(env_file, override=True)

        hot_load_info = {}
        try:
            client = request.app.state.rotating_client
            hot_load_info = await _hot_load_custom_provider(
                client, req.name, req.api_key
            )
            logger.info(
                f"Hot-loaded custom provider {req.name}: "
                f"plugin={'new' if hot_load_info.get('plugin_registered') else 'existing'}, "
                f"models={hot_load_info.get('models_discovered', 0)}"
            )
        except Exception:
            logger.warning(f"Hot-load failed for {req.name}", exc_info=True)
            hot_load_info["error"] = "hot_load_failed"

    return {"provider": req.name, **hot_load_info}


@router.get("/config/model-filters/{provider}")
async def get_model_filters(provider: str):
    env_vars = _get_env_vars()
    provider_upper = provider.upper()

    ignore_key = f"IGNORE_MODELS_{provider_upper}"
    whitelist_key = f"WHITELIST_MODELS_{provider_upper}"

    ignore = [p.strip() for p in env_vars.get(ignore_key, "").split(",") if p.strip()]
    whitelist = [p.strip() for p in env_vars.get(whitelist_key, "").split(",") if p.strip()]

    return {"ignore": ignore, "whitelist": whitelist}


class ModelFilterUpdate(BaseModel):
    ignore: list[str]
    whitelist: list[str]


@router.put("/config/model-filters/{provider}")
async def update_model_filters(provider: str, filters: ModelFilterUpdate):
    env_file = str(_env_path())
    provider_upper = provider.upper()

    ignore_key = f"IGNORE_MODELS_{provider_upper}"
    whitelist_key = f"WHITELIST_MODELS_{provider_upper}"

    if filters.ignore:
        _inplace_set_key(env_file, ignore_key, ",".join(filters.ignore))
    else:
        _inplace_unset_key(env_file, ignore_key)

    if filters.whitelist:
        _inplace_set_key(env_file, whitelist_key, ",".join(filters.whitelist))
    else:
        _inplace_unset_key(env_file, whitelist_key)

    load_dotenv(env_file, override=True)
    return {"provider": provider, "updated": True}


@router.post("/reload")
async def reload_proxy():
    try:
        env_file = _env_path()
        load_dotenv(str(env_file), override=True)
        logger.info("Proxy configuration reloaded via admin API")
        return {"status": "ok", "message": "Configuration reloaded from .env"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
