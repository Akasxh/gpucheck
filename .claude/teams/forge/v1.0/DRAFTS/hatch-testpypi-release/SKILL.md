---
name: hatch-testpypi-release
description: Build, test-publish, and sign a Python package release using Hatch (PyPA-official build frontend) + sigstore for keyless signing + the `pypa/gh-action-pypi-publish` GitHub Action with Trusted Publishing (OIDC, no API tokens). Covers `hatch build`, `hatch publish` against TestPyPI, sigstore-signed wheels and sdists, version bumping via `hatch version`, and the GitHub-side OIDC trust configuration. Use this skill IMMEDIATELY when the user asks to "publish to PyPI", "do a release", "tag and ship a Python package", "set up TestPyPI", "sign with sigstore", or "configure trusted publishing". Anchored to projects whose `pyproject.toml` declares `[build-system] requires = ["hatchling"]`.
when-to-use: Python package release flow; `pyproject.toml` already uses `hatchling` build backend; user wants TestPyPI dry-run before real PyPI; sigstore-signed artifacts; GitHub Actions Trusted Publisher (OIDC) instead of long-lived `PYPI_API_TOKEN`.
disable-model-invocation: false
---

# hatch-testpypi-release

You are walking a Python project from a clean working tree to a signed release on PyPI. This skill assumes the project already uses Hatchling (gpucheck does — see its `pyproject.toml`). Do not re-architect the build backend; only run the release.

## When to apply this skill

Apply when the user says: `hatch publish`, `release`, `tag`, `cut a version`, `push to PyPI`, `TestPyPI`, `trusted publishing`, `sigstore`, or `gh-action-pypi-publish`. Apply when `pyproject.toml` contains `requires = ["hatchling"]`.

Do NOT apply when:
- Build backend is `setuptools`, `poetry`, `flit`, or `pdm` — different toolchain. Defer or redirect.
- The project is private/internal-only and never goes to public PyPI — different release flow.
- The user wants to publish a npm/cargo/Maven package — wrong ecosystem.

## Procedure

### Step 1: Verify environment

```bash
hatch --version              # >=1.9 for Trusted Publishing support
python -c "import sigstore; print(sigstore.__version__)"   # >=3.0
gh auth status               # logged into the GitHub repo's org
```

If `hatch` is missing: `pipx install hatch` (preferred) or `pip install --user hatch`.

### Step 2: Pre-flight checks on the working tree

```bash
git status                    # must be clean
git log -n 1 --format=%H     # commit being released
hatch version                 # current version in pyproject.toml or _version.py
```

If working tree is dirty: stop. Tell the user to commit or stash before tagging.

### Step 3: Bump version

```bash
hatch version patch    # or: minor / major / 1.2.3 (explicit)
```

This rewrites the `version` field in the path declared by `[tool.hatch.version]` in `pyproject.toml`. Commit the bump:

```bash
git add pyproject.toml src/<pkg>/__about__.py
git commit -m "chore: bump version to $(hatch version)"
```

### Step 4: Build wheels and sdist

```bash
hatch build --clean
```

Output lands in `dist/`. Verify:

```bash
ls -la dist/                  # should contain *.whl and *.tar.gz
twine check dist/*           # PyPA's metadata sanity-checker; install via `pipx install twine`
```

`twine check` catches the most common metadata bugs (missing long_description, invalid URLs).

### Step 5: Publish to TestPyPI (dry-run)

TestPyPI requires a separate account from PyPI. The user must have a TestPyPI account configured. Two paths:

**Path A — local with token (legacy):**

```bash
hatch publish --repo test --user __token__ --auth pypi-<test-token>
```

**Path B — Trusted Publisher via GitHub Actions (recommended):**

Skip the local upload and let CI do it. See Step 7.

After publish, verify:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            <pkg>==<new-version>
python -c "import <pkg>; print(<pkg>.__version__)"
```

If install fails or version mismatches: stop. Do not proceed to real PyPI.

### Step 6: Configure Trusted Publishing on PyPI

One-time setup, in the PyPI web UI (https://pypi.org/manage/project/<pkg>/settings/publishing/):
- Owner: `Akasxh` (or org)
- Repository: `gpucheck` (or relevant repo)
- Workflow filename: `release.yml`
- Environment name: `pypi`

Repeat on TestPyPI (https://test.pypi.org/manage/project/...). After this, no `PYPI_API_TOKEN` secret is needed; the GitHub OIDC token is exchanged for a short-lived PyPI token at publish time.

### Step 7: GitHub Actions release workflow

Author or update `.github/workflows/release.yml`:

```yaml
name: release
on:
  push:
    tags: ['v*']
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install hatch
      - run: hatch build --clean
      - uses: actions/upload-artifact@v4
        with: { name: dist, path: dist/ }

  publish-testpypi:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write   # required for OIDC
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    needs: publish-testpypi
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write   # OIDC for trusted publishing
      contents: write   # for sigstore release attachments
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1     # uses Trusted Publisher (no token)
      - uses: sigstore/gh-action-sigstore-python@v3.0.0
        with:
          inputs: ./dist/*.tar.gz ./dist/*.whl
```

Notes:
- `id-token: write` is mandatory for OIDC. Without it, `pypa/gh-action-pypi-publish` falls back to looking for `PYPI_API_TOKEN` and fails clearly.
- `sigstore/gh-action-sigstore-python` produces `.sigstore` bundle files alongside each artifact. They are attached to the GitHub Release automatically.
- Pin the action version (`@release/v1` or a SHA), not `@main`.

### Step 8: Cut the tag

```bash
git tag -a v$(hatch version) -m "Release v$(hatch version)"
git push origin v$(hatch version)
```

The push triggers `release.yml`. Watch:

```bash
gh run watch
```

### Step 9: Post-release verification

```bash
pip install <pkg>==<new-version>
python -c "import <pkg>; print(<pkg>.__version__)"
```

And manually verify the sigstore bundle on the GitHub Release page — each artifact should have a sibling `.sigstore` file.

## Failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `400 File already exists` from PyPI | Duplicate version | Bump version with `hatch version patch` and re-tag |
| `403 invalid-publisher` | OIDC not configured on PyPI side | Re-do Step 6 with exact workflow filename + environment name |
| `twine check` warns about long_description | Missing or wrong-format `README.md` reference in `pyproject.toml` | Add `readme = "README.md"` and `[project]` content-type |
| sigstore upload fails with `oidc.identity-token`-not-found | Missing `permissions: id-token: write` on the job | Add to the job, not to the workflow root |

## Out-of-scope

- Conda / conda-forge release: separate flow.
- Yanking a release: use `pip install yank` or PyPI web UI manually.
- Internal/private PyPI mirrors (Artifactory, Nexus): deviates from PyPA-trusted-publishing path.

## References (STRONG-PRIMARY only)

- Hatch publishing docs: https://hatch.pypa.io/latest/publish/
- `pypa/gh-action-pypi-publish` README: https://github.com/pypa/gh-action-pypi-publish
- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- sigstore-python action: https://github.com/sigstore/gh-action-sigstore-python
- PyPA project metadata spec (PEP 621): https://peps.python.org/pep-0621/
