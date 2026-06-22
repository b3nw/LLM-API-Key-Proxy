# Retired Providers

This folder archives provider implementations that are intentionally no longer
registered, discovered, or exposed by the proxy.

Archived modules are kept for historical reference only. Do not import them from
active code or add them back to provider factory/discovery wiring unless the
provider is explicitly un-retired.

## Archived Providers

### Upstream retirements (from Mirrowel/LLM-API-Key-Proxy)

| Provider | Files | Notes |
|----------|-------|-------|
| Antigravity | `antigravity_provider.py`, `antigravity_auth_base.py`, `antigravity_quota_tracker.py`, `device_profile.py` | Google Gemini via reverse-engineered Antigravity API |
| Qwen Code | `qwen_code_provider.py`, `qwen_auth_base.py` | Qwen Code (Alibaba) provider |
| iFlow | `iflow_provider.py`, `iflow_auth_base.py` | iFlow provider with cookie auth |

### Fork-only retirements (recovered from git history)

| Provider | Files | Notes |
|----------|-------|-------|
| Cursor | `cursor_provider.py`, `cursor_quota_tracker.py` | Cursor AI via sidecar proxy with quota tracking |
| Gemini A2A | `gemini_a2a_provider.py`, `a2a_client.py`, `a2a_session_manager.py`, `a2a_sidecar_manager.py`, `a2a_translator.py` | Experimental Gemini CLI A2A sidecar architecture |
| Bedrock | `bedrock_provider.py` | Minimal AWS Bedrock stub (hardcoded models, never fully implemented) |
| Gemini file logger | `gemini_file_logger.py` | Superseded by unified `transaction_logger` module |
