# CI / release workflow

Canonical feature ID: `ci`
Stack subject: `feat(ci): fork-aware release notes with incremental topic diff`
Manifest: `.fork/stack.yml`

## 2026-06-24 — Re-pin short SHA after stack rebase dropped PR #60 fix

Target: `feat(ci): fork-aware release notes with incremental topic diff`

Files:
- `.github/workflows/build.yml`
- `.fork/features/ci.md`

Working commit before autosquash:
- (on branch `fix/ci-executable-build`, one commit atop `origin/dev`)

### Why

Run [28050361475](https://github.com/b3nw/LLM-API-Key-Proxy/actions/runs/28050361475) failed in **Generate Build Metadata** with `find: 'release-assets': No such file or directory`. Build matrix jobs uploaded `proxy-app-build-*-ef8d7e9` (7-char SHA); release job filtered with `proxy-app-build-*-ef8d7e9e` (8-char). `download-artifact@v4` matched 0 artifacts and exited 0.

PR #60 (merged 2026-06-21) had pinned `--short=7` and added **Verify downloaded artifacts**; a later linear stack rewrite on `dev` dropped that change from `build.yml`.

### Fix

1. Pin both **Get short SHA** steps to `git rev-parse --short=7 HEAD`.
2. Add **Verify downloaded artifacts** immediately after download.
3. Grant **`actions: read`** on the `release` job so the verify step can list run artifacts via `gh api` (Kilo review PR #75).

### Verification

- `uv run --no-project python3 .fork/check-stack.py`
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))"`

### Notes

- Land via `fixup! feat(ci): ...` then autosquash into `feat(ci)` on merge (rebase-and-merge preserves stack intent).