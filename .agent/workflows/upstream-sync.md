---
description: Sync local dev branch with upstream/dev while preserving local fix branches with open PRs
---

# Upstream Sync Workflow

This workflow uses a **two-phase approach** to avoid hanging on interactive prompts.

## Phase 1: Analysis (non-interactive)

Run the script in `--analyze` mode. This fetches upstream, classifies branches,
and outputs a structured report — then exits immediately with no prompts.

// turbo
```bash
./.agent/skills/upstream-sync/scripts/upstream-sync.sh --analyze
```

After reading the output, **present the following to the user for confirmation**:

1. **Divergence**: How many commits behind/ahead of upstream
2. **Branches to PRESERVE**: List each branch, its unique commit count, and whether it's already merged upstream
3. **Branches to DELETE**: Local-only branches that will be removed
4. **Worktrees**: Any worktrees that may need rebasing

**⚠️ STOP HERE and wait for explicit user confirmation before proceeding to Phase 2.**

Ask the user:
- "Do you want to proceed with this sync plan?"
- "Any branches you want to override (keep something marked for deletion, or skip a preserved branch)?"

## Phase 2: Execute Sync (agent-driven, with `--force`)

Once the user confirms, run the sync in `--force` mode (no interactive prompts).
Use `--skip-cherry-pick` since cherry-picking requires manual agent-driven intervention.

```bash
./.agent/skills/upstream-sync/scripts/upstream-sync.sh --force --skip-cherry-pick --skip-push
```

Then follow the SKILL.md cherry-pick strategy manually:

### Cherry-Pick Ordering

**⚠️ IMPORTANT: `feature/core-all` MUST be cherry-picked LAST.**

It contains the fork's README and fork-wide infrastructure changes. Merging it last
ensures our README overwrites any upstream README changes, and all other branches'
code is already in place for conflict resolution.

Branches NOT currently merged into dev (kept on their own branches only):
- `feature/gemini-a2a` — experimental, not ready
- `feature/cursor-all` — not actively maintained

Recommended order:
1. Provider branches (anthropic, chutes, codex, firmware, lightning-ai, nanogpt, zenmux, etc.)
2. Feature branches (tui-all, model-routing-all, gemini-cli-all, qwen-code-all)
3. **`feature/core-all` — ALWAYS LAST** (README, dynamic routing, export fixes)

### Cherry-Pick Steps

1. For each preserved branch, identify key commits:
   ```bash
   git log --oneline --no-merges upstream/dev..<branch> | head -10
   ```

2. Cherry-pick and squash into one commit per branch:
   ```bash
   git cherry-pick --no-commit <sha1> [sha2] [sha3]
   git commit -m "<prefix>: <description>"
   ```

3. Verify the result:
   ```bash
   git log --oneline upstream/dev..dev
   ```

4. **Ask user for confirmation** before force pushing:
   ```bash
   git push origin dev --force-with-lease
   ```

## Skill Reference

For detailed step-by-step instructions (useful if conflicts occur or manual intervention is needed):

**Read the skill:** `.agent/skills/upstream-sync/SKILL.md`

The skill includes:
- Prerequisite verification
- Branch preservation logic (checks origin/<branch> existence)
- **Cherry-pick strategy** for clean one-commit-per-branch history
- Conflict resolution guidance
- Rollback procedures
- Troubleshooting tips

## Strategy Overview

This workflow uses **cherry-pick** instead of merge to avoid commit duplication:

1. Reset `dev` to `upstream/dev`
2. For each preserved branch, cherry-pick its key commits
3. Squash related commits into one commit per branch
4. Force push the clean history

**Result**: ~10-15 clean commits instead of 80+ duplicated merge commits.
