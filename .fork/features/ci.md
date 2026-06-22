# CI Feature History

Canonical stack subject: `feat(ci): fork-aware release notes with incremental topic diff`

---

## 2026-06-22 — Add Build PR ref workflow for fork PR testing

Target: `feat(ci): add Build PR ref workflow for fork PR testing`
Files:
- `.github/workflows/docker-build-pr.yml`

Working commits before autosquash:
- `fd957cd ci: add Build PR ref workflow for fork PR testing` (original, pre-rebase)
- `83862b1 feat(ci): add Build PR ref workflow for fork PR testing` (post-rebase onto dev d4bb75b)

Verification:
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docker-build-pr.yml'))"` — passed
- `uv run --no-project python3 .fork/check-stack.py` — passed
- Dry run `pr_ref=62` succeeded: https://github.com/b3nw/LLM-API-Key-Proxy/actions/runs/27972476769
- Dry run `pr_ref=pull/62/head` succeeded: https://github.com/b3nw/LLM-API-Key-Proxy/actions/runs/27975749173
- Existing `docker-build.yml` regression check passed: https://github.com/b3nw/LLM-API-Key-Proxy/actions/runs/27973080735

Notes:
- Adds a reusable workflow (`workflow_call`) that builds a Docker image from a PR
  ref (numeric PR number or `pull/N/head`), without pushing to registry.
- Designed for fork contributors to validate Docker builds before merge.
- `push-to-registry: false` matches the existing `docker-build.yml` pattern.
- GHCR image name is lowercased (`b3nw/llm-api-key-proxy`) to avoid Docker registry rejection.
- SHA checkout uses `fetch-depth: 0` and accepts uppercase hex in the regex.
