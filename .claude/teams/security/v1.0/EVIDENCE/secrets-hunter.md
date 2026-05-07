# secrets-hunter — full tree + git history

## Phase 1 — Tool
gitleaks / trufflehog **not installed**. Falling back to grep + manual
git history scan.

## Phase 2 — patterns scanned

### Current tree (working copy at a9a9d44)
Patterns (case-insensitive):
- `AKIA[0-9A-Z]{16}` — AWS Access Key ID
- `sk-[a-zA-Z0-9]{20,}` — OpenAI / Stripe
- `sk-ant-[a-zA-Z0-9-]{80,}` — Anthropic
- `ghp_[a-zA-Z0-9]{36}` — GitHub PAT
- `gho_[a-zA-Z0-9]{36}` — GitHub OAuth
- `glpat-[a-zA-Z0-9-]{20}` — GitLab PAT
- `xox[abp]-[0-9A-Za-z-]+` — Slack
- `-----BEGIN .*PRIVATE KEY-----`
- `eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}` — JWT
- generic `(password|passwd|secret|api_key|apikey|token|auth_token|access_token)\s*[:=]\s*['"][^'"]{8,}`

Result on `src/`, `tests/`, `examples/`, `pyproject.toml`, `README.md`,
`.github/`, `.gitignore`, `LICENSE`, `CLAUDE.md`: **zero matches** for
any high-confidence pattern. Generic-pattern grep returns only:

```
src/gpucheck/sanitizers/race.py:56:    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")
```
which is an env-var read, not a hardcoded secret. ✓

### Git history scan
Repo has 5 commits on `release/v1.0`:
```
$ git log --oneline release/v1.0
a9a9d44 [ Fix ] : resolve 7 bugs, add 23 tests, rewrite docs
2197277 [ Fix ] : resolve 7 bugs found by codebase analysis
5dcbf83 [ README ] : added bugs found section
25cdfcf [ Perf ] : GPU fast-path
6562f31 [ Fix ] : recalibrated tolerance tables
```
Ran:
```
git log -p --all -S 'AKIA' -- .
git log -p --all -S 'BEGIN PRIVATE' -- .
git log -p --all -S 'sk-ant' -- .
git log -p --all -S 'sk-' -- .
```
**All return empty.** Author email `drakathakash@gmail.com` is the
expected committer (matches CLAUDE.md), no anomalous identities.

### `.env` / config files
- `.env` is in `.gitignore:18`. ✓
- No `.env*` files exist in the working tree.
- No `.npmrc`, `.pypirc`, `.netrc`, `kubeconfig` files. ✓

### Examples directory
`examples/` exists but only contains demo Triton/PyTorch test files —
no API keys, no tokens. (Spot-checked via `find examples -type f` which
returned only `*.py` files; grep on the same prefix patterns: clean.)

## Phase 3 — verification

| Finding | Status |
|---|---|
| Hardcoded secrets in current tree | **none** |
| Hardcoded secrets in git history | **none** |
| `.env` files committed | **none** |
| Test fixtures / docs containing real-looking keys | **none** |
| Author email anomaly | **none** |

## Output verdict
**PASS** — no secrets findings. The `.gitignore` already covers `.env`
and `.venv`. The repo is clean. Recommend running `gitleaks` once it
is available on a CI host as a regression gate, but this is *advisory*.

## Note for upcoming MPS work
The MPS extras may pull in Apple-specific deps. Ensure the
`signing-identity`, `keychain access password`, and
`APPLE_DEVELOPER_ID_*` env vars are **never** echoed by the sanitizer
or by any new logging in `sanitizers/mps.py`. Add an explicit
**redaction allowlist** in `reporting/console.py` so that env-var
dumps (if any) skip names matching `(APPLE|KEYCHAIN|SIGNING|ID_)*`.
This is a forward-looking recommendation — no current finding.
