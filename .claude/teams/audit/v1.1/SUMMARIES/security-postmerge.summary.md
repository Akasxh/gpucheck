# security-postmerge summary (W1)

**Verdict:** ADVISORY (clean to ship; 2 NEW LOW deferred to v1.1)
**Tally:** 0 BLOCKER, 0 HIGH, 0 MEDIUM, 2 LOW, 3 PASS

**v1.0 baseline 3 MEDIUMs all addressed in merge:** CFG-2 permissions ✓, TM-E1 CUDA_HOME allowlist ✓, DEP-1 lockfile ✓.

**Top-3 NEW post-merge findings:**
1. **PM-5 — HTML reporter defense-in-depth gap.** `reporting/html.py` escapes attacker-tainted strings but interpolates `class="{klass}"` (lines 171/218/243) and `style="background:{bg}"` (line 93) raw. Not exploitable today (whitelisted constants); one wrong line in v1.1 → stored-XSS in CI artifacts. Fix: wrap klass/bg in _esc + hostile-name regression test + CSP meta tag.
2. **PM-4 — assert_close slow-path on stride-fuzzed tensors.** Track-B's stride helpers produce non-contiguous and stride-0 tensors. `assertions/close.py:_to_numpy` calls `.numpy()` without `.contiguous()`. Availability bug on torch <2.1 (RuntimeError). One-line fix.
3. **PM-2 — `_load_pyproject_config` swallows all exceptions.** `plugin.py:86-88` bare `except Exception: pass` masks misconfiguration. Narrow to `(OSError, tomllib.TOMLDecodeError)` and warn.

**Cleared:** PM-1 (MPS sysctl uses hardened argv, untouched by CUDA_HOME path), PM-3 (no env/config knob downgrades backend selection — hard-coded CUDA→MPS allowlist).
