# Fork tooling and workflow

Canonical feature ID: `tooling`
Stack subject: `feat(tooling): add AGENTS.md and .agent/ config for linear stack workflow`
Manifest: `.fork/stack.yml`

This file is the shared, repo-tracked history for fork workflow/tooling changes.
Local workspace state under `/opt/data/workspace/developer/state/llm-api-key-proxy/`
may contain run logs and scratch notes, but it is not canonical.

## 2026-06-19 — Add shared fork workflow metadata

Target: `feat(tooling): add AGENTS.md and .agent/ config for linear stack workflow`
Files:
- `AGENTS.md`
- `.fork/stack.yml`
- `.fork/features/tooling.md`
- `.fork/features/gemini-cli.md`
- `.fork/check-stack.py`

Final stack commit:
- `5f7cbf99 feat(tooling): add AGENTS.md and .agent/ config for linear stack workflow`

Verification:
- `uv run python .fork/check-stack.py` — passed at the time this feature landed.

Notes:
- Replaces local-only workspace ledgers as the canonical feature history with repo-tracked `.fork/` metadata.
- Adds `.fork/stack.yml` as the shared source of truth for feature IDs, stack order, file ownership, and allowed historical exceptions.
- Adds `.fork/check-stack.py` to catch duplicate release-note sections, executable `git add -A` examples, missing feature metadata, and unexpected duplicate feature commits.

## 2026-07-03 — Scope fork-stack gate to canonical branch; tolerate transient commits

Target: `feat(tooling): add AGENTS.md and .agent/ config for linear stack workflow`
Files:
- `.githooks/pre-commit`
- `.fork/check-stack.py`
- `.fork/stack.yml`

Working commits before autosquash:
- `fix(tooling): scope fork-stack gate to canonical branch and tolerate transient commits`

Final stack commit after autosquash:
- pending

Verification:
- `python3 .fork/check-stack.py` → `fork stack validation passed` (previously failed on `docs(fork):` and legacy `feat: Jules …` commits).
- Pre-commit on a feature/worktree branch skips stack validation; simulated `dev` branch runs it and passes.
- Unit-checked that `fixup!`/`squash!`/`amend!` markers are filtered from the stack and `allowed_subjects` parses.

Notes:
- The pre-commit hook previously ran the full `upstream/dev..HEAD` stack validation on every commit in every worktree, blocking the documented `fixup!` workflow and any PR branch under review. It now enforces the stack invariant only on the canonical branch (`branch:` in `stack.yml`, default `dev`).
- `check-stack.py` now ignores transient autosquash markers (`fixup!`/`squash!`/`amend!`), accepts `docs(fork):` bookkeeping commits, and honors a new top-of-manifest `allowed_subjects` list for exact-subject exceptions.
- `stack.yml` registers the two legacy bare `feat: Jules …` batch commits under `allowed_subjects` so a clean `dev` validates green.
- Reminder: the canonical hook is `.githooks/pre-commit`; install/activate it with `git config core.hooksPath .githooks` so the tracked hook (not a stale `.git/hooks/` copy) is authoritative.
