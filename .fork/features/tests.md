## 16:37 24.June.2026 — Cover Codex stream overload classification

Target: `feat(tests): add local test suite (153 tests, zero-cost, no network)`
Files:
- `tests/test_error_handler.py`

Working commits before autosquash:
- `ddd269f fixup! feat(tests): ...`

Final stack commit after autosquash:
- `e54a134 feat(tests): ...` before ledger metadata refresh

Verification:
- `uv run --with pytest --with pytest-mock pytest tests/test_error_handler.py tests/test_failure_logger.py -q` — passed
- `uv run python3 -m py_compile src/rotator_library/error_handler.py src/rotator_library/client/executor.py tests/test_error_handler.py` — passed
- `uv run ruff check src/rotator_library/error_handler.py src/rotator_library/client/executor.py tests/test_error_handler.py --select F401,F811,F821,E9` — passed

Notes:
- Added a regression test proving Codex overload stream errors classify as retryable `server_error`/503.
- Removed stale unused imports from `tests/test_error_handler.py` so touched-file lint remains clean.
