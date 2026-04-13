## 16:37 24.June.2026 — Retry Codex stream overloads

Target: `feat(core): infrastructure improvements - latest aliases, error standardization, and utilities`
Files:
- `src/rotator_library/error_handler.py`
- `src/rotator_library/client/executor.py`

Working commits before autosquash:
- `a36d9ed fixup! feat(core): ...`

Final stack commit after autosquash:
- `6ece37e feat(core): ...` before ledger metadata refresh

Verification:
- `uv run --with pytest --with pytest-mock pytest tests/test_error_handler.py tests/test_failure_logger.py -q` — passed
- `uv run python3 -m py_compile src/rotator_library/error_handler.py src/rotator_library/client/executor.py tests/test_error_handler.py` — passed
- `uv run ruff check src/rotator_library/error_handler.py src/rotator_library/client/executor.py tests/test_error_handler.py --select F401,F811,F821,E9` — passed

Notes:
- Codex stream `server_is_overloaded` is transient provider capacity, not a bad request.
- Classify overload-shaped `StreamedAPIError` values as `server_error`/503 so same-key retry policy can handle them.
- Preserve wrapped stream errors when `.data` is absent; classifying `None` masked the real overload text.
