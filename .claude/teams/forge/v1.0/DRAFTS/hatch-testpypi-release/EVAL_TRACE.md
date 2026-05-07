# EVAL_TRACE — hatch-testpypi-release

## Eval 1: trigger on release request

User prompt: "Cut a v0.1.1 release of gpucheck and push to PyPI."

Expected: skill triggers (keywords "release", "PyPI"), `pyproject.toml` has hatchling -> match, walks Steps 2-9 (preflight, version bump, build, TestPyPI, GitHub Actions OIDC release, sigstore signing, tag).
Verdict: **PASS** — description anchored on hatchling + release verbs; gpucheck pyproject.toml uses hatchling so the anchor matches.

## Eval 2: behavior — Trusted Publisher misconfiguration

User prompt: "My CI fails with `403 invalid-publisher` after I tagged v0.1.0."

Expected: skill triggers, routes to **Failure modes table**, prescribes re-doing Step 6 with exact workflow filename + environment name on PyPI's settings page.
Verdict: **PASS** — exact symptom is in the failure table with concrete remediation.

## Eval 3: anti-trigger on different build backend

User prompt: "I'm using poetry for my package — how do I publish?"

Expected: skill does NOT trigger; description requires hatchling. The `when-to-use` field excludes setuptools/poetry/flit/pdm.
Verdict: **PASS** — Step 0's "do NOT apply" list explicitly handles this and would defer.

Trigger eval: 3/3 PASS. Promotable.
