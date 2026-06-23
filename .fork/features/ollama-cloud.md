# Ollama Cloud Provider

Canonical feature ID: `ollama-cloud`
Stack subject: `feat(ollama-cloud): add Ollama Cloud provider with session cookie quota tracking`
Manifest: `.fork/stack.yml`

## 2026-06-22 — Initial provider implementation

Target: `feat(ollama-cloud): add Ollama Cloud provider with session cookie quota tracking`
Files:
- `src/rotator_library/providers/ollama_cloud_provider.py`
- `src/rotator_library/providers/utilities/ollama_cloud_quota_tracker.py`
- `.fork/features/ollama-cloud.md`

Working commits before autosquash:
- pending

Final stack commit after autosquash:
- pending

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/ollama_cloud_provider.py` — passed
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/ollama_cloud_quota_tracker.py` — passed
- `uv run ruff check ... --select F401,F811,F821,E9` — passed (both files)

Notes:
- Ollama Cloud has NO public JSON API for quota tracking (GitHub issues #15663,
  #15132, #16448 all closed without implementation).
- Quota tracking requires scraping https://ollama.com/settings HTML page using a
  `__Secure-session` browser cookie. This is the same approach used by Guanaco
  (evangit2/guanaco) — the only known working method.
- Provider uses OpenAI-compatible endpoint: POST https://ollama.com/v1/chat/completions
  with `Authorization: Bearer <api_key>`.
- Model discovery via GET https://ollama.com/api/tags (no auth required).
- Credential format: `api_key` (simple) or `api_key:session_cookie` (with quota tracking).
- Session cookie can also be set via `OLLAMA_CLOUD_SESSION_COOKIE` env var.
- Plans: free, pro, max — session resets ~6h, weekly resets 7d.
- 429 responses are parsed for quota exhaustion (reactive fallback when no cookie).

Design references:
- Guanaco (evangit2/guanaco) client.py get_usage() method
- Umans provider pattern (UmansQuotaTracker mixin + ProviderInterface)
- OpenCode Go provider pattern (custom logic + litellm routing)
