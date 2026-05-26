# CI — Fork-Aware Release Notes

Canonical feature ID: `ci`
Stack subject: `feat(ci): fork-aware release notes with incremental topic diff`
Manifest: `.fork/stack.yml`

## 2026-06-26 — Feature-ledger enriched incremental release notes

Target: `feat(ci): fork-aware release notes with incremental topic diff`
Files:
- `.github/workflows/build.yml`
- `scripts/create_release.sh`

Notes:
- Problem: Release notes dumped the full 30-commit fork stack on every build,
  making it impossible to see what actually changed between builds.
- Fix: Incremental diff (tree-based file diff) is now the primary "What's Changed"
  section. Each modified commit is enriched with a human-readable description
  from `.fork/features/<id>.md` (the latest `## ` heading).
- The full git-cliff changelog is moved to a collapsed `<details>` block titled
  "Full Fork Stack (N commits on upstream/dev)".
- Topic prefix → feature ID mapping uses underscore-to-hyphen conversion
  (e.g. `lightning_ai` → `lightning-ai.md`).
- Section header in create_release.sh changed to "What's Changed Since Last Build".
