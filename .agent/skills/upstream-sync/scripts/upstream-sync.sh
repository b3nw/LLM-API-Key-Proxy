#!/usr/bin/env bash
#
# upstream-sync.sh - Sync fork's dev branch with upstream/dev
#
# This script automates the upstream sync workflow using a branch manifest
# that defines merge order, dependencies, and branch metadata.
#
# Usage: ./upstream-sync.sh [MODE] [OPTIONS]
#
# Modes:
#   --analyze         Analysis mode: fetch, report divergence and branch
#                     classification, then EXIT. No prompts, no mutations.
#                     Designed for AI agent consumption.
#   --replay          Replay mode: reset dev to upstream/dev, then merge all
#                     active branches from the manifest in order.
#   --validate        Validate mode: check that all manifest branches exist
#                     and have exactly 1 commit ahead of upstream/dev.
#   --absorb          Absorb mode: fold fix/* branches into their parent
#                     branches as defined in the manifest.
#   (default)         Legacy mode: original upstream-sync behavior.
#
# Options:
#   --dry-run         Show what would be done without making changes
#   --no-backup       Skip creating backup branch (not recommended)
#   --force           Skip ALL confirmation prompts (agent-driven mode)
#   --skip-push       Skip force-push to origin
#   --skip-worktrees  Skip worktree rebase step
#   --stop-on-conflict  Stop replay at first merge conflict (default: skip and continue)
#   --manifest PATH   Path to branch manifest (default: .agent/branch-manifest.yml)
#
set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Configuration ───────────────────────────────────────────────────────────
MODE="legacy"
DRY_RUN=false
CREATE_BACKUP=true
FORCE=false
SKIP_PUSH=false
SKIP_WORKTREES=false
STOP_ON_CONFLICT=false
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
MANIFEST="${REPO_ROOT}/.agent/branch-manifest.yml"

# ─── Parse arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --analyze)      MODE="analyze"; shift ;;
    --replay)       MODE="replay"; shift ;;
    --validate)     MODE="validate"; shift ;;
    --absorb)       MODE="absorb"; shift ;;
    --dry-run)      DRY_RUN=true; shift ;;
    --no-backup)    CREATE_BACKUP=false; shift ;;
    --force)        FORCE=true; shift ;;
    --skip-push)    SKIP_PUSH=true; shift ;;
    --skip-worktrees) SKIP_WORKTREES=true; shift ;;
    --stop-on-conflict) STOP_ON_CONFLICT=true; shift ;;
    --manifest)     MANIFEST="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: upstream-sync.sh [--analyze|--replay|--validate|--absorb] [OPTIONS]"
      exit 1
      ;;
  esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_header()  { echo -e "\n${BOLD}${CYAN}═══ $1 ═══${NC}\n"; }

confirm() {
  if [[ "$FORCE" == "true" ]]; then return 0; fi
  read -p "$1 [y/N] " response
  case "$response" in
    [yY][eE][sS]|[yY]) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY-RUN]${NC} Would run: $*"
  else
    "$@"
  fi
}

# ─── Manifest parser ────────────────────────────────────────────────────────
# Lightweight YAML parser — extracts branch entries from the manifest.
# No dependency on yq/python; uses grep/sed/awk.

parse_manifest_branches() {
  # Returns: name|status|type|description|absorb_into|squash_commit (one line per branch)
  local manifest="$1"
  local filter_status="${2:-active}"  # default: only active branches

  if [[ ! -f "$manifest" ]]; then
    log_error "Manifest not found: $manifest"
    exit 1
  fi

  awk -v filter="$filter_status" '
    function emit() {
      if (cur_name != "" && (filter == "all" || cur_status == filter)) {
        print cur_name "|" cur_status "|" cur_type "|" cur_desc "|" cur_absorb "|" cur_squash
      }
    }
    /^  - name:/ {
      emit()
      cur_name = $3; cur_status = ""; cur_type = ""; cur_desc = ""; cur_absorb = ""; cur_squash = ""
    }
    /^    status:/ { cur_status = $2 }
    /^    type:/ { cur_type = $2 }
    /^    absorb_into:/ { cur_absorb = $2 }
    /^    squash_commit:/ { gsub(/"/, "", $2); cur_squash = $2 }
    /^    description:/ {
      cur_desc = $0
      sub(/^    description: "/, "", cur_desc)
      sub(/"$/, "", cur_desc)
    }
    END { emit() }
  ' "$manifest"
}

get_manifest_base() {
  grep "^base:" "$1" 2>/dev/null | awk '{print $2}' || echo "upstream/dev"
}

parse_manifest_cherry_picks() {
  # Returns: sha|description (one line per cherry-pick)
  local manifest="$1"
  local in_section=false

  awk '
    /^cherry_picks:/ { in_section = 1; next }
    /^[a-z]/ && !/^  / { in_section = 0 }
    in_section && /^  - sha:/ {
      gsub(/"/, "", $3)
      sha = $3
    }
    in_section && /^    description:/ {
      desc = $0
      sub(/^    description: "/, "", desc)
      sub(/"$/, "", desc)
      if (sha != "") {
        print sha "|" desc
        sha = ""
      }
    }
  ' "$manifest"
}

# ─── Preflight checks ───────────────────────────────────────────────────────
log_info "Running preflight checks..."

if ! git rev-parse --git-dir > /dev/null 2>&1; then
  log_error "Not in a git repository"
  exit 1
fi

if ! git remote get-url upstream > /dev/null 2>&1; then
  log_error "Remote 'upstream' not configured"
  echo "Add it with: git remote add upstream <upstream-url>"
  exit 1
fi

if [[ "$MODE" != "analyze" && "$MODE" != "validate" && "$DRY_RUN" != "true" ]]; then
  if [[ -n $(git status --porcelain -uno) ]]; then
    log_error "Working directory not clean. Commit or stash changes first."
    git status --short
    exit 1
  fi
fi

log_success "Preflight checks passed"

# ─── Fetch ───────────────────────────────────────────────────────────────────
log_info "Fetching upstream and origin..."
git fetch upstream
git fetch origin --prune

UPSTREAM_BASE=$(get_manifest_base "$MANIFEST")
BEHIND=$(git log --oneline dev.."$UPSTREAM_BASE" 2>/dev/null | wc -l | tr -d ' ')
AHEAD=$(git log --oneline "$UPSTREAM_BASE"..dev 2>/dev/null | wc -l | tr -d ' ')

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE MODE
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "analyze" ]]; then
  echo ""
  echo "===== UPSTREAM SYNC ANALYSIS ====="
  echo ""
  echo "DIVERGENCE:"
  echo "  behind_upstream: $BEHIND"
  echo "  ahead_of_upstream: $AHEAD"
  echo ""

  if [[ "$BEHIND" == "0" ]]; then
    echo "STATUS: UP_TO_DATE"
  else
    echo "STATUS: SYNC_NEEDED"
  fi
  echo ""

  echo "RECENT_UPSTREAM_COMMITS:"
  git log --oneline -10 "$UPSTREAM_BASE" | sed 's/^/  /'
  echo ""

  echo "MANIFEST_BRANCHES:"
  while IFS='|' read -r name status type desc absorb squash; do
    unique_count=$(git log --oneline --no-merges "$UPSTREAM_BASE".."$name" 2>/dev/null | wc -l | tr -d ' ')

    if git merge-base --is-ancestor "$name" "$UPSTREAM_BASE" 2>/dev/null; then
      merge_status="ABSORBED_BY_UPSTREAM"
    elif [[ "$unique_count" == "0" ]]; then
      merge_status="EMPTY (needs branch pointer fix)"
    elif [[ "$unique_count" == "1" ]]; then
      merge_status="CLEAN"
    else
      merge_status="MULTI_COMMIT ($unique_count commits — needs squash)"
    fi

    echo "  - branch: $name"
    echo "    status: $status"
    echo "    type: $type"
    echo "    unique_commits: $unique_count"
    echo "    merge_status: $merge_status"
    [[ -n "$absorb" ]] && echo "    absorb_into: $absorb"
    [[ -n "$squash" ]] && echo "    squash_commit: $squash"
    echo ""
  done < <(parse_manifest_branches "$MANIFEST" "all")

  # Fix branches not in manifest
  echo "UNTRACKED_BRANCHES:"
  for branch in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -v "^main$" | grep -v "^dev$" | grep -v "^archive/"); do
    if ! grep -q "name: $branch" "$MANIFEST" 2>/dev/null; then
      unique=$(git log --oneline "$UPSTREAM_BASE".."$branch" 2>/dev/null | wc -l | tr -d ' ')
      echo "  - branch: $branch"
      echo "    unique_commits: $unique"
      echo "    note: NOT in manifest"
      echo ""
    fi
  done

  # Worktree info
  MAIN_WORKTREE=$(pwd)
  OTHER_WORKTREES=$(git worktree list --porcelain | grep "worktree " | cut -d' ' -f2- | grep -v "^$MAIN_WORKTREE$" || true)
  echo "WORKTREES:"
  if [[ -n "$OTHER_WORKTREES" ]]; then
    while IFS= read -r wt_path; do
      wt_branch=$(git -C "$wt_path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "DETACHED")
      echo "  - path: $wt_path"
      echo "    branch: $wt_branch"
    done <<< "$OTHER_WORKTREES"
  else
    echo "  (none)"
  fi
  echo ""
  echo "===== END ANALYSIS ====="
  exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATE MODE
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "validate" ]]; then
  log_header "Validating manifest branches"

  errors=0
  warnings=0

  while IFS='|' read -r name status type desc absorb squash; do
    # Check branch exists
    if ! git show-ref --verify --quiet "refs/heads/$name" 2>/dev/null; then
      if ! git show-ref --verify --quiet "refs/remotes/origin/$name" 2>/dev/null; then
        log_error "$name: branch does not exist locally or on origin"
        errors=$((errors + 1))
        continue
      fi
    fi

    unique_count=$(git log --oneline --no-merges "$UPSTREAM_BASE".."$name" 2>/dev/null | wc -l | tr -d ' ')

    if [[ "$status" == "active" ]]; then
      if [[ "$unique_count" == "0" ]]; then
        if [[ -n "$squash" ]]; then
          log_warn "$name: branch is empty but has squash_commit=$squash (needs pointer fix)"
          warnings=$((warnings + 1))
        else
          log_error "$name: active branch has 0 unique commits"
          errors=$((errors + 1))
        fi
      elif [[ "$unique_count" -gt 1 && "$type" != "fix" ]]; then
        log_warn "$name: has $unique_count commits (expected 1 for clean branch)"
        warnings=$((warnings + 1))
      else
        log_success "$name: OK ($unique_count commit(s), $type)"
      fi
    elif [[ "$status" == "wip" ]]; then
      log_info "$name: WIP (skipped, $unique_count commits)"
    fi
  done < <(parse_manifest_branches "$MANIFEST" "all")

  echo ""
  if [[ $errors -gt 0 ]]; then
    log_error "Validation failed: $errors error(s), $warnings warning(s)"
    exit 1
  else
    log_success "Validation passed: 0 errors, $warnings warning(s)"
  fi
  exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY MODE
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "replay" ]]; then
  log_header "Manifest-driven replay"

  echo "  Upstream base: $UPSTREAM_BASE"
  echo "  Dev behind upstream: $BEHIND"
  echo "  Dev ahead of upstream: $AHEAD"
  echo ""

  # Read active branches from manifest
  mapfile -t BRANCHES < <(parse_manifest_branches "$MANIFEST" "active")

  echo "Branches to merge (in order):"
  for entry in "${BRANCHES[@]}"; do
    IFS='|' read -r name status type desc absorb squash <<< "$entry"
    echo "  $(printf '%-35s' "$name") [$type] $desc"
  done
  echo ""

  if ! confirm "Proceed with replay? This will reset dev to $UPSTREAM_BASE."; then
    log_warn "Aborted by user"
    exit 0
  fi

  # Backup
  if [[ "$CREATE_BACKUP" == "true" ]]; then
    BACKUP_BRANCH="archive/dev-pre-sync-$(date +%Y%m%d-%H%M%S)"
    log_info "Creating backup: $BACKUP_BRANCH"
    run_cmd git branch "$BACKUP_BRANCH" dev
  fi

  # Reset dev
  log_info "Resetting dev to $UPSTREAM_BASE..."
  run_cmd git checkout dev
  run_cmd git reset --hard "$UPSTREAM_BASE"

  # Merge each branch
  merged=0
  skipped=0
  failed=0
  failed_branches=()

  for entry in "${BRANCHES[@]}"; do
    IFS='|' read -r name status type desc absorb squash <<< "$entry"

    # Skip fix branches — they get applied after all feature/core branches
    if [[ "$type" == "fix" ]]; then
      continue
    fi

    echo ""
    log_info "Merging: $name [$type]"

    # Check if branch has content
    unique_count=$(git log --oneline --no-merges "$UPSTREAM_BASE".."$name" 2>/dev/null | wc -l | tr -d ' ')

    if [[ "$unique_count" == "0" ]]; then
      if [[ -n "$squash" ]]; then
        log_warn "$name: branch pointer needs fix (squash_commit=$squash available on dev)"
        log_warn "Skipping — run with --absorb or manually: git checkout -B $name $squash"
      else
        log_warn "$name: 0 unique commits, skipping"
      fi
      skipped=$((skipped + 1))
      continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
      echo -e "${YELLOW}[DRY-RUN]${NC} Would merge: git merge $name --no-ff -m \"merge: $name\""
      merged=$((merged + 1))
      continue
    fi

    # Attempt the merge
    if git merge "$name" --no-ff -m "merge: $name" 2>/dev/null; then
      log_success "Merged $name ($unique_count commit(s))"
      merged=$((merged + 1))
    else
      log_error "Merge conflict with $name!"
      git merge --abort 2>/dev/null || true
      failed_branches+=("$name")
      failed=$((failed + 1))

      if [[ "$STOP_ON_CONFLICT" == "true" ]]; then
        log_error "Stopping at first conflict (--stop-on-conflict)"
        echo ""
        echo "To resolve manually:"
        echo "  git merge $name --no-ff -m \"merge: $name\""
        echo "  # resolve conflicts"
        echo "  git merge --continue"
        break
      else
        log_warn "Skipping $name, continuing with remaining branches"
      fi
    fi
  done

  # Now apply post-merge cherry-picks from the manifest
  echo ""
  log_header "Applying post-merge cherry-picks"

  while IFS='|' read -r cp_sha cp_desc; do
    [[ -z "$cp_sha" ]] && continue

    log_info "Cherry-pick: $cp_sha — $cp_desc"

    if [[ "$DRY_RUN" == "true" ]]; then
      echo -e "${YELLOW}[DRY-RUN]${NC} Would cherry-pick $cp_sha"
      merged=$((merged + 1))
      continue
    fi

    if git cherry-pick "$cp_sha" 2>/dev/null; then
      log_success "Applied: $(git log --oneline -1 "$cp_sha")"
      merged=$((merged + 1))
    else
      log_error "Cherry-pick failed for $cp_sha ($cp_desc)"
      git cherry-pick --abort 2>/dev/null || true
      failed_branches+=("cherry-pick:$cp_sha")
      failed=$((failed + 1))

      if [[ "$STOP_ON_CONFLICT" == "true" ]]; then
        log_error "Stopping at first conflict (--stop-on-conflict)"
        break
      fi
    fi
  done < <(parse_manifest_cherry_picks "$MANIFEST")

  # Summary
  echo ""
  log_header "Replay summary"
  echo "  Merged:  $merged"
  echo "  Skipped: $skipped"
  echo "  Failed:  $failed"

  if [[ ${#failed_branches[@]} -gt 0 ]]; then
    echo ""
    log_warn "Failed branches:"
    for fb in "${failed_branches[@]}"; do
      echo "    - $fb"
    done
  fi

  echo ""
  echo "Commits ahead of upstream:"
  git log --oneline "$UPSTREAM_BASE"..dev | head -30
  echo ""

  # Push
  if [[ "$SKIP_PUSH" == "false" && "$failed" == "0" ]]; then
    if confirm "Force push dev to origin?"; then
      run_cmd git push origin dev --force-with-lease
      log_success "Pushed to origin/dev"
    else
      log_warn "Skipped push. Run manually: git push origin dev --force-with-lease"
    fi
  elif [[ "$failed" -gt 0 ]]; then
    log_warn "Skipping push due to $failed failed merge(s)"
    echo "Resolve conflicts and push manually: git push origin dev --force-with-lease"
  fi

  # Worktree rebase
  if [[ "$SKIP_WORKTREES" == "false" ]]; then
    MAIN_WORKTREE=$(pwd)
    WORKTREES=$(git worktree list --porcelain | grep "worktree " | cut -d' ' -f2-)
    OTHER_WORKTREES=$(echo "$WORKTREES" | grep -v "^$" | grep -v "^$MAIN_WORKTREE$" || true)

    if [[ -n "$OTHER_WORKTREES" ]]; then
      log_header "Worktree update"
      for worktree_path in $OTHER_WORKTREES; do
        branch=$(git -C "$worktree_path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
        if [[ -n "$branch" && "$branch" != "dev" && "$branch" != "main" && "$branch" != "HEAD" ]]; then
          echo "  $worktree_path → $branch"
        fi
      done
      echo ""
      if confirm "Rebase worktree branches onto updated dev?"; then
        for worktree_path in $OTHER_WORKTREES; do
          branch=$(git -C "$worktree_path" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
          if [[ -n "$branch" && "$branch" != "dev" && "$branch" != "main" && "$branch" != "HEAD" ]]; then
            log_info "Rebasing $branch in $worktree_path..."
            if [[ "$DRY_RUN" == "true" ]]; then
              echo -e "${YELLOW}[DRY-RUN]${NC} Would rebase $branch onto dev"
            else
              git -C "$worktree_path" fetch origin
              if git -C "$worktree_path" rebase dev; then
                log_success "Rebased $branch"
              else
                log_error "Rebase conflict in $worktree_path — resolve manually"
                git -C "$worktree_path" rebase --abort 2>/dev/null || true
              fi
            fi
          fi
        done
      fi
    fi
  fi

  log_success "Replay complete!"
  exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# ABSORB MODE
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "absorb" ]]; then
  log_header "Absorbing fix branches into parent branches"

  while IFS='|' read -r name status type desc absorb squash; do
    if [[ "$type" != "fix" || -z "$absorb" ]]; then
      continue
    fi

    # Get unique commits from the fix branch beyond the parent
    fix_commits=$(git log --oneline --no-merges "$absorb".."$name" 2>/dev/null)
    fix_count=$(echo "$fix_commits" | grep -c . 2>/dev/null || echo "0")

    if [[ "$fix_count" == "0" || -z "$fix_commits" ]]; then
      log_info "$name → $absorb: nothing to absorb"
      continue
    fi

    echo ""
    log_info "$name → $absorb ($fix_count commit(s) to absorb):"
    echo "$fix_commits" | sed 's/^/    /'
    echo ""

    if ! confirm "Absorb $name into $absorb? (will checkout, cherry-pick, squash)"; then
      log_warn "Skipped $name"
      continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
      echo -e "${YELLOW}[DRY-RUN]${NC} Would absorb $name into $absorb"
      continue
    fi

    # Save current branch
    original_branch=$(git rev-parse --abbrev-ref HEAD)

    # Checkout parent branch
    git checkout "$absorb"

    # Cherry-pick fix commits
    fix_shas=$(git log --reverse --format='%H' --no-merges "$absorb".."$name" 2>/dev/null)
    for sha in $fix_shas; do
      if ! git cherry-pick --no-commit "$sha" 2>/dev/null; then
        log_error "Cherry-pick conflict absorbing $name into $absorb"
        git cherry-pick --abort 2>/dev/null || true
        git checkout "$original_branch"
        continue 2
      fi
    done

    # Amend the existing squash commit to include the fix
    git commit --amend --no-edit
    log_success "Absorbed $name into $absorb"

    # Return to original branch
    git checkout "$original_branch"

    if confirm "Delete fix branch $name?"; then
      git branch -D "$name" 2>/dev/null || true
      log_success "Deleted $name"
    fi
  done < <(parse_manifest_branches "$MANIFEST" "active")

  log_success "Absorb complete!"
  exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY MODE (original behavior)
# ═══════════════════════════════════════════════════════════════════════════════
log_header "Legacy upstream sync"

echo "  Commits behind upstream: $BEHIND"
echo "  Commits ahead of upstream: $AHEAD"
echo ""

if [[ "$BEHIND" == "0" ]]; then
  log_success "Already up-to-date with upstream/dev!"
  exit 0
fi

# Classify branches
PRESERVE_BRANCHES=""
DELETE_BRANCHES=""

for branch in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -v "^main$" | grep -v "^dev$"); do
  if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    PRESERVE_BRANCHES="${PRESERVE_BRANCHES}${branch}"$'\n'
  else
    DELETE_BRANCHES="${DELETE_BRANCHES}${branch}"$'\n'
  fi
done

PRESERVE_BRANCHES=$(echo -n "$PRESERVE_BRANCHES" | sed '/^$/d')
DELETE_BRANCHES=$(echo -n "$DELETE_BRANCHES" | sed '/^$/d')

echo "=== Branches to PRESERVE (exist on origin/) ==="
if [[ -n "$PRESERVE_BRANCHES" ]]; then
  echo "$PRESERVE_BRANCHES"
else
  echo "  (none)"
fi
echo ""
echo "=== Branches to DELETE (local-only) ==="
if [[ -n "$DELETE_BRANCHES" ]]; then
  echo "$DELETE_BRANCHES"
else
  echo "  (none)"
fi
echo ""

log_warn "Legacy mode does not use the manifest. Consider using --replay instead."
echo ""

if ! confirm "Proceed with legacy sync?"; then
  log_warn "Aborted by user"
  exit 0
fi

# Backup
if [[ "$CREATE_BACKUP" == "true" ]]; then
  BACKUP_BRANCH="archive/dev-pre-sync-$(date +%Y%m%d-%H%M%S)"
  log_info "Creating backup: $BACKUP_BRANCH"
  run_cmd git branch "$BACKUP_BRANCH" dev
fi

# Reset dev
log_info "Resetting dev to $UPSTREAM_BASE..."
run_cmd git checkout dev
run_cmd git reset --hard "$UPSTREAM_BASE"

# Delete local-only branches
if [[ -n "$DELETE_BRANCHES" ]]; then
  if confirm "Delete local-only branches?"; then
    for branch in $DELETE_BRANCHES; do
      log_info "Deleting $branch..."
      run_cmd git branch -D "$branch" 2>&1 || true
    done
  fi
fi

# Cherry-pick guidance
if [[ -n "$PRESERVE_BRANCHES" ]]; then
  log_warn "⚠️  MANUAL STEP: Cherry-pick preserved branches onto dev"
  echo ""
  echo "Consider using --replay mode for automated branch merging."
  echo ""
  for branch in $PRESERVE_BRANCHES; do
    unique_count=$(git log --oneline --no-merges "$UPSTREAM_BASE"..$branch 2>/dev/null | wc -l | tr -d ' ')
    echo "  $branch ($unique_count commits)"
    echo "    git merge $branch --no-ff -m \"merge: $branch\""
  done
fi

echo ""
log_success "Legacy sync complete. Use --replay for full automated workflow."
