# LLM-API-Key-Proxy — Agent Instructions

## ⚠️ MANDATORY: Read Before Any Code Change

This repository is a **fork** maintained as a linear commit stack on top of `upstream/dev`.
**You MUST follow the workflow below for every change you make, no exceptions.**

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
| `tests/*` | `feat: add local test suite` |

### Step 2: Lint all changed Python files before staging

**MANDATORY — do not skip this step.** Run the following on every `.py` file you touched:

```bash
# Syntax check (stdlib — zero deps)
uv run python3 -m py_compile src/path/to/file.py

# Undefined names / missing imports / unused imports
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9
```

> This project uses `uv` for environment management. Always prefix `python3` and
> `ruff` commands with `uv run` rather than relying on system-level installations.

The pre-commit hook (`.git/hooks/pre-commit`) also runs these automatically when
you `git commit`, but running them manually first gives faster feedback.

Common things to verify after a change:
- Every name used in the file is either defined locally or imported.
- No import statements were accidentally deleted while editing.
- `py_compile` exits 0.

### Step 3: Commit with the `fixup!` prefix

```bash
# Edit files...
git add -A
git commit -m "fixup! feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports"
```

> **CRITICAL:** The text after `fixup!` must **exactly match** the first line of the
> target commit. Copy it from `git log --oneline`.

### Step 4: Fold it into the correct commit

```bash
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash upstream/dev
```

This automatically moves your fixup commit next to its target and squashes them.

### Step 5: Push

```bash
git push origin dev --force-with-lease
```

---

## Adding an Entirely New Feature

```bash
# Just commit at the tip with a new prefix:
git add -A
git commit -m "feat(newprovider): add SomeProvider with quota tracking"

# Push
git push origin dev --force-with-lease
```

No fixup needed — new features go at the end of the stack naturally.

---

## Upstream Sync

When the upstream repository updates:

```bash
git fetch upstream
git rebase upstream/dev
# Resolve any conflicts in the specific commit that breaks
git push origin dev --force-with-lease
```

Each commit is replayed one at a time. Conflicts are localized to the specific
commit that touched the affected lines — resolve it there and continue.

---

## Rules

1. **NEVER add raw commits** without a topic prefix. Every commit must be
   `feat(<area>):`, `fix(<area>):`, or `fixup! <exact target commit message>`.

2. **NEVER merge branches into dev.** Dev is a linear rebase-only branch.

3. **Always use `--force-with-lease`** when pushing dev (it's a rewritten branch).

4. **One commit per feature area.** If you're fixing something in an existing
   area, use `fixup!` + autosquash to fold it back in.

5. **Keep the stack ordered.** Independent providers come first, shared
   infrastructure (`core`) in the middle, cross-cutting features (`tui`,
   `model-routing`, `copilot`) at the end.

6. **When a rebase conflict occurs during autosquash**, stop and resolve it
   carefully. You can always compare with the current file content using
   `git stash` to save your work and inspect.

7. **Always lint Python files before committing.** Run `uv run python3 -m py_compile
   <file>` and `uv run ruff check <file> --select F401,F811,F821,E9` on every file
   you changed. The pre-commit hook enforces this automatically, but treat it
   as a manual checklist item too — catching errors before `git add` is faster
   than fixing a broken deployment.

---

## Quick Reference

```bash
# See the full fork stack
git log --oneline upstream/dev..HEAD

# Find which commit owns a file
git log --oneline upstream/dev..HEAD -- path/to/file.py

# Lint changed Python files (run BEFORE git add)
uv run python3 -m py_compile src/path/to/file.py
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9

# Make a fix and fold it in
git commit -m "fixup! <exact commit message from git log>"
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash upstream/dev

# Sync with upstream
git fetch upstream && git rebase upstream/dev

# Push
git push origin dev --force-with-lease
```

## Additional References

- **Deployment & hot-patching**: `.agent/rules/llm-proxy.md`
- **Development environment**: `.agent/rules/claude.md`
- **Local Docker testing** (container info, hot-patching, remote folder structure): `.private/README.md`
