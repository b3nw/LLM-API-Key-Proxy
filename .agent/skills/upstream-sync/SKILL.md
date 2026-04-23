---
name: upstream-sync
description: Synchronize forked repository with upstream using a linear commit stack
---

# Upstream Sync Skill

Synchronize the LLM-API-Key-Proxy fork with upstream changes.

## Core Concepts

- `dev` is a **linear stack** of commits on `upstream/dev` — no merge commits
- Each commit has a topic prefix: `feat(codex):`, `fix(core):`, etc.
- Upstream sync is a single `git rebase upstream/dev`

## Prerequisites

- Remote `upstream` must be configured: `git remote add upstream <url>`
- Working directory must be clean

---

## Upstream Sync

```bash
# Fetch latest upstream
git fetch upstream

# Rebase our stack on top of new upstream/dev
git rebase upstream/dev

# If conflicts arise, resolve in the specific commit that broke:
#   - Edit the conflicting files
#   - git add <resolved files>
#   - git rebase --continue

# Push (force-with-lease because we rewrote history)
git push origin dev --force-with-lease
```

Each commit replays one at a time. Conflicts are localized to just the
commit that touches the affected lines — much simpler than resolving
multi-branch merge conflicts.

---

## Making Changes

### Fix an existing feature

```bash
# 1. Find the owning commit
git log --oneline upstream/dev..HEAD

# 2. Make the fix and commit with fixup! prefix
git add -A
git commit -m "fixup! feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports"

# 3. Fold it into the right commit (no editor needed)
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash upstream/dev

# 4. Push
git push origin dev --force-with-lease
```

### Add a new feature

```bash
# Just commit at the tip with a descriptive prefix
git add -A
git commit -m "feat(newprovider): add SomeProvider with quota tracking"
git push origin dev --force-with-lease
```

---

## Commit Ownership Reference

| File Pattern | Owning Commit |
|-------------|---------------|
| `providers/<name>_provider.py` | `feat(<name>):` |
| `providers/utilities/<name>_*` | `feat(<name>):` |
| `client/rotating_client.py` | `feat(core):` |
| `client/executor.py`, `streaming.py` | `feat(core):` |
| `proxy_app/main.py` | `feat(core):` |
| `proxy_app/quota_viewer.py` | `feat(tui):` |
| `proxy_app/log_viewer.py` | `feat(tui):` |
| `model_alias_registry.py` | `feat(model-routing):` |
| `tests/*` | `feat: add local test suite` |

When in doubt, use `git log --oneline upstream/dev..HEAD -- path/to/file.py`
to find which commit last touched the file.

---

## Legacy

The old manifest-driven multi-branch approach (branch-manifest.yml, replay mode)
has been replaced by this simpler linear stack. Individual feature branches
(`feature/codex-all`, etc.) may still exist as refs but are no longer used
for the dev workflow.
