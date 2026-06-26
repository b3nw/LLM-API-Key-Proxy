---
description:
alwaysApply: true
---

# LLM-API-Key-Proxy — Agent Instructions

## ⚠️ MANDATORY: Read Before Any Code Change

This repository is a **fork** maintained as a linear commit stack on top of `upstream/dev`.
**You MUST follow the workflow below for every change you make, no exceptions.**

---

## ⚠️ Worktrees Must Branch From `b3nw/dev`

The active development branch is **`dev`** (pushed to `b3nw/dev`), **not** `main`.
`main` lags behind `dev` and does not contain in-flight feature work, so a
worktree created from `main` will be missing files and will not merge cleanly.

**Every git worktree for this repo MUST be based on `dev`, never `main`.**

- When creating a worktree, base it on `dev`:
  ```bash
  git worktree add -b <branch-name> <path> dev
  ```
- If a tool created the worktree from `main` (the repo default branch), reset
  the worktree branch onto `dev` before making any changes:
  ```bash
  git reset --hard dev
  ```
- Open PRs against **`dev`** (`--base dev`), never `main`.

---

## How the Fork Works

```
upstream/dev
  ├── feat(anthropic): ...        ← one clean commit per feature area
  ├── feat(chutes): ...
  ├── feat(codex): ...
  ├── ... (15 more) ...
  └── feat: add health endpoints  ← HEAD (dev)
```

- `dev` is a **linear stack** of squashed, self-contained commits on `upstream/dev`
- Each commit has a **topic prefix**: `feat(codex):`, `fix(core):`, `feat(tui):`, etc.
- There are **no merge commits** — the history is always flat and linear
- Per-feature change history is tracked in repo-tracked `.fork/` metadata so
  every contributor and developer workspace sees the same ledger (see
  **Feature Tracking Ledger** below)

### Release Notes

The automated build workflow (`build.yml`) generates release changelogs from
commit messages. It works by comparing topic prefixes between builds — each
topic prefix is treated as a stable feature identifier.

- **New topics** appear in the "What's New" section of the release
- **Renamed topics** show as both "removed" (old name) and "new" (new name) — avoid unless intentional
- **Upstream syncs** are detected and reported when `upstream/dev` advances

### Feature Tracking Ledger

Because PRs are squash-merged into `dev`, git history on `dev` only shows the
final squashed commits. It does **not** preserve the incremental commits,
rationale, verification notes, or deployment observations that happened while a
feature evolved on its branch.

To preserve that history across contributors and developer workspaces, the
canonical feature ledger is committed to this repository under:

```text
.fork/
  stack.yml                  # shared feature IDs, stack order, file ownership
  check-stack.py             # stack validation (CI and local)
  features/
    <feature-key>.md         # append-only shared feature history
```

Local workspace state under paths such as
`/opt/data/workspace/developer/state/llm-api-key-proxy/` is useful for scratch
notes, run logs, reviews, and temporary artifacts, but it is **not canonical**.
Do not rely on local state as the only record of a durable feature change.

`<feature-key>` matches the topic prefix area (e.g. `feat(codex):` → `codex`).
See `.fork/stack.yml` for the full registry of feature IDs and file ownership.
See existing entries in `.fork/features/` for the ledger format.

---

## Making a Change

### Step 1: Identify which commit owns the files you're changing

```bash
git log --oneline upstream/dev..HEAD
```

Match files to commits:

| File Pattern | Owning Commit Prefix |
|-------------|---------------------|
| `providers/<name>_provider.py` | `feat(<name>):` |
| `providers/utilities/<name>_*` | `feat(<name>):` |
| `providers/copilot_*` | `feat(copilot):` |
| `client/rotating_client.py` | `feat(core):` |
| `client/executor.py`, `streaming.py`, `errors.py` | `feat(core):` |
| `client/transforms.py` | `feat(core):` |
| `proxy_app/main.py` | `feat(core):` |
| `proxy_app/quota_viewer.py` | `feat(tui):` |
| `proxy_app/log_viewer.py` | `feat(tui):` |
| `model_alias_registry.py`, `cross_provider_executor.py` | `feat(model-routing):` |
| `error_handler.py`, `error_tracker.py` | `feat(core):` |
| `credential_manager.py`, `credential_tool.py` | `feat(core):` |
| `tests/*` | `feat(tests):` |

### Step 2: Create a feature branch in a worktree

All development happens on **feature branches**, never directly on `dev`.
Use a git worktree so you can work on multiple branches without switching:

```bash
# Branch naming: <type>/<area>-<short-description>
git worktree add worktrees/fix-codex-credits -b fix/codex-credits dev
cd worktrees/fix-codex-credits
```

Branch name conventions:
- `fix/<area>-<description>` — bug fixes to existing features
- `feat/<area>-<description>` — new features or enhancements

### Step 3: Lint all changed Python files before committing

**MANDATORY — do not skip this step.** Run the following on every `.py` file you touched:

```bash
# Syntax check (stdlib — zero deps)
uv run python3 -m py_compile src/path/to/file.py

# Undefined names / missing imports / unused imports
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9
```

> This project uses `uv` for environment management. Always prefix `python3` and
> `ruff` commands with `uv run` rather than relying on system-level installations.

The pre-commit hook runs these automatically when you `git commit`, plus
validates symlink rejection and `.fork/check-stack.py`. Install it with:

```bash
git config core.hooksPath .githooks
```

Running checks manually first gives faster feedback.

Common things to verify after a change:
- Every name used in the file is either defined locally or imported.
- No import statements were accidentally deleted while editing.
- `py_compile` exits 0.

### Step 4: Commit with the proper topic prefix

Stage only the files that belong to the change. Do **not** use `git add -A` in
this repository: `worktrees/` is intentionally untracked for local git worktrees,
and `.dev` symlinks or other workspace artifacts may also exist locally.

Use the standard topic prefix form for the commit message. This is the message
that will appear on `dev` after the PR is squash-merged:

```bash
git add src/path/to/file.py tests/path/to/test_file.py
git commit -m "fix(codex): don't block routing when paid credits bypass window exhaustion"
```

For new features:

```bash
git commit -m "feat(newprovider): add SomeProvider with quota tracking"
```

> **CRITICAL:** The commit message must use a recognized topic prefix:
> `feat(<area>):`, `fix(<area>):`. This becomes the squash-merge commit
> message on `dev` and feeds the automated release changelog.

You may have multiple commits on the feature branch during development. They
will all be squashed into a single commit when the PR is merged.

### Step 5: Update the feature ledger

Before pushing, update the repo-tracked per-feature ledger for the owning
feature under:

```text
.fork/features/<feature-key>.md
```

For small documentation-only changes, the ledger entry may be brief. For code,
behavior, release, quota, auth, provider, or WebUI changes, include the files
changed, verification commands, and the branch name.

If this is a new feature area:

1. Add the feature to `.fork/stack.yml` with its stable ID, commit subject,
   stack order, and file ownership globs.
2. Create `.fork/features/<feature-key>.md` with the shared change history.
3. Keep bulky logs/reviews in local workspace state if useful, but summarize the
   durable outcome in `.fork/features/<feature-key>.md`.

### Step 6: Push the branch and hand off to the user

```bash
git push -u origin fix/codex-credits
```

> **IMPORTANT:** Agents do NOT create PRs. Push the branch and report the URL.
> The user creates the PR on GitHub, sets the squash-merge commit message to the
> topic-prefix form, and merges.

The user creates the PR targeting `dev`, then uses **Squash and merge** with
the topic-prefix commit message to preserve the linear stack.

---

## Rules

1. **NEVER commit directly to dev.** All changes go through feature branches
   and PR squash-merge. The only exception is upstream syncs by the maintainer.

2. **Every commit message must have a topic prefix.** Use `feat(<area>):` or
   `fix(<area>):`. This is the squash-merge message that lands on `dev`.

3. **NEVER merge branches into dev.** Dev is a linear branch. PRs must use
   **Squash and merge** (not merge commit, not rebase-merge).

4. **One commit per feature area on dev.** The PR squash-merge produces a single
   commit. If fixing something in an existing area, the squash commit replaces
   the existing one via the linear stack convention.

5. **Keep the stack ordered.** Independent providers come first, shared
   infrastructure (`core`) in the middle, cross-cutting features (`tui`,
   `model-routing`, `copilot`) at the end.

6. **Always lint Python files before pushing.** Run `uv run python3 -m py_compile
   <file>` and `uv run ruff check <file> --select F401,F811,F821,E9` on every file
   you changed. The pre-commit hook enforces this automatically, but treat it
   as a manual checklist item too — catching errors before `git add` is faster
   than fixing a broken deployment.

7. **Keep topic prefixes stable.** The automated release changelog uses commit
   messages as feature identifiers. Renaming a topic prefix (e.g.
   `feat(codex):` → `feat(openai-codex):`) causes the release notes to show
   both a "removed" entry and a "new" entry. If a rename is intentional, do it
   in a single rebase so the changelog shows both sides cleanly.

8. **Update the repo-tracked feature ledger for every durable change.** The
   `.fork/features/<feature>.md` files are the durable shared history of how
   each feature evolved. Update before pushing the feature branch.

9. **Treat local workspace state as non-canonical.** Local state directories are
   useful for bulky logs, reviews, and scratch notes, but `.fork/stack.yml` and
   `.fork/features/*.md` are the shared records that must travel with the repo.

10. **Agents do NOT create PRs.** Push the feature branch and report the URL
    to the user. The user handles PR creation and merge.

---

## Quick Reference

```bash
# See the full fork stack
git log --oneline upstream/dev..HEAD

# Find which commit owns a file
git log --oneline upstream/dev..HEAD -- path/to/file.py

# Create a feature branch in a worktree
git worktree add worktrees/<branch-name> -b <type>/<area>-<desc> dev

# Lint changed Python files (run BEFORE committing)
uv run python3 -m py_compile src/path/to/file.py
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9

# Stage and commit with topic prefix
git add src/path/to/file.py
git commit -m "fix(codex): description of the fix"

# Update shared feature ledger
$EDITOR .fork/features/<feature-key>.md

# Push the feature branch (agent stops here)
git push -u origin <type>/<area>-<desc>

# User creates PR on GitHub → Squash and merge into dev
# Squash commit message: "fix(codex): description of the fix"
```

## Additional References

- **Local Docker testing** (container info, hot-patching, remote folder structure): `.private/README.md`
