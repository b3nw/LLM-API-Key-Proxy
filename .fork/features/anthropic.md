## 2026-06-30 — Remove unused imports in anthropic_provider.py

Target: fixup! feat(anthropic): add OAuth support and handle streaming nulls
Files: src/rotator_library/providers/anthropic_provider.py

Verification:
- uv run python3 -m py_compile src/rotator_library/providers/anthropic_provider.py — passed
- uv run ruff check src/rotator_library/providers/anthropic_provider.py --select F401 — passed

Notes: Removed unused imports (asyncio, re, Path, UsageManager and TYPE_CHECKING).
