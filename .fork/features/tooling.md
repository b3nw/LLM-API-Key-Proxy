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

## 2026-06-26 — Migrate workflow docs from fixup!/autosquash to feature-branch PRs

Target: `fix(tooling): update AGENTS.md to feature-branch PR workflow`
Files:
- `AGENTS.md`

Changes:
- Replaced the fixup!/autosquash/force-push-to-dev documentation with the
  feature-branch + PR + squash-merge flow that had already become the actual
  practice for landing changes on `dev` (see the merged-PR history: #87, #98,
  #99–#105 all landed via individual feature-branch PRs, not stack rewrites).
- Added an explicit rule that agents push branches and report the URL; the
  user creates the PR and chooses the squash-merge commit message.
- Moved maintainer-only upstream-sync instructions to the gitignored
  `.private/README.md` (local-only, not repo-tracked).

Rationale:
- `AGENTS.md` and `.fork/check-stack.py` still described the old linear
  fixup!/autosquash/force-push model, but the fork had already moved to
  per-feature branches merged individually. The mismatch caused
  `.fork/check-stack.py` to flag legitimately-merged commits (e.g. bare
  `feat: Jules ...` batches, `docs(fork):` ledger-update commits) as stack
  violations, because the tooling was validating against a workflow nobody
  was following anymore.

Verification:
- `uv run python .fork/check-stack.py` — this change is documentation-only and
  does not alter validator behavior. At the time this was first authored,
  the validator still failed against `dev` for the pre-existing reasons this
  audit documents; those were fixed separately by `fix(tooling): scope
  fork-stack gate to canonical branch and tolerate transient commits` (#106,
  see the following ledger entry), which has since merged.

Notes:
- This commit was originally authored and pushed to `fix/agents-docs` before
  `dev` advanced further; it is re-applied here as a clean cherry-pick onto
  current `dev` (same content, new hash) rather than reusing the stale branch.

## 2026-07-03 — Scope fork-stack gate to canonical branch; tolerate transient commits

Target: `feat(tooling): add AGENTS.md and .agent/ config for linear stack workflow`
Files:
- `.githooks/pre-commit`
- `.fork/check-stack.py`
- `.fork/stack.yml`

Working commits before autosquash:
- `fix(tooling): scope fork-stack gate to canonical branch and tolerate transient commits`

Final stack commit:
- `d7d22654 fix(tooling): scope fork-stack gate to canonical branch and tolerate transient commits`

Verification:
- `python3 .fork/check-stack.py` → `fork stack validation passed` (previously failed on `docs(fork):` and legacy `feat: Jules …` commits).
- Pre-commit on a feature/worktree branch skips stack validation; simulated `dev` branch runs it and passes.
- Unit-checked that `fixup!`/`squash!`/`amend!` markers are filtered from the stack and `allowed_subjects` parses.

Notes:
- The pre-commit hook previously ran the full `upstream/dev..HEAD` stack validation on every commit in every worktree, blocking the documented `fixup!` workflow and any PR branch under review. It now enforces the stack invariant only on the canonical branch (`branch:` in `stack.yml`, default `dev`).
- `check-stack.py` now ignores transient autosquash markers (`fixup!`/`squash!`/`amend!`), accepts `docs(fork):` bookkeeping commits, and honors a new top-of-manifest `allowed_subjects` list for exact-subject exceptions.
- `stack.yml` registers the two legacy bare `feat: Jules …` batch commits under `allowed_subjects` so a clean `dev` validates green.
- Reminder: the canonical hook is `.githooks/pre-commit`; install/activate it with `git config core.hooksPath .githooks` so the tracked hook (not a stale `.git/hooks/` copy) is authoritative.
