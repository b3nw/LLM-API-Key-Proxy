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

## 2026-07-11 — Add AI PR Review workflow (Nikita-Filonov/ai-review)

Branch: `feat/ci-ai-pr-review`
Files:
- `.github/workflows/ai-review.yml` (new)

Notes:
- Adds a second AI code reviewer alongside the existing minimax/kilocode
  `pr-review.yml` workflow. Uses Nikita-Filonov/ai-review@v0.69.0 (latest).
- Routes LLM requests through the local llm-proxy via OpenAI-compatible
  endpoint (`LLM__PROVIDER: "OPENAI"` with custom `API_URL`).
- Triggers on PR opened/synchronize/reopened against `dev` and `main`.
- Fixes from review of the initial draft:
  - Added `LLM__META__MAX_TOKENS: "15000"` — the default (1200) would
    silently truncate review output.
  - Removed `REVIEW__MAX_CONTEXT_COMMENTS` (no-op — `run` executes inline +
    summary only, not context review).
  - Corrected `REVIEW__MAX_INLINE_COMMENTS` comment to "per file".
  - Added `LLM__HTTP_CLIENT__TIMEOUT: "180"` (proxy adds a hop).
  - Added `REVIEW__IGNORE_CHANGES` for markdown, lockfiles, docs, and fork
    metadata to reduce noise.

Required GitHub Secrets/Variables:
- Secret `LLM_PROXY_URL` — proxy endpoint, no trailing slash
- Secret `LLM_PROXY_API_KEY` — proxy token
- Variable `LLM_PROXY_MODEL` — model alias (e.g. `kilo/minimax/minimax-m3`)

Verification:
- Env var names cross-checked against `action.yml`, `docs/configs/.ai-review.yaml`,
  and `docs/ci/github.yaml` from the upstream repo.
- v0.69.0 confirmed as latest release (July 8, 2026).

## 2026-09-03 — Support fork PRs via pull_request_target in AI PR Review workflow

Branch: `fix/ci-ai-review-pull-request-target`
Files:
- `.github/workflows/ai-review.yml`

Notes:
- Replaces `on: pull_request` with `on: pull_request_target` so external contributor
  fork PRs receive repository secrets/variables and no longer crash with Pydantic
  validation errors.
- Pins checkout ref to `github.event.pull_request.head.sha` with `actions/checkout@v4`.
- Adds fallback between Secrets and Variables for `LLM_PROXY_URL`, `LLM_PROXY_API_KEY`,
  and `LLM_PROXY_MODEL`.
- Uses sensible fallback model `codex/gpt-5.6-sol` if `LLM_PROXY_MODEL` is empty.
