# verifier-C.md

**Branch**: `feat/track-c-thread-safety` @ `5ddd26e`

```
$ pytest --tb=short -q
126 passed, 3 skipped, 6 warnings in 0.46s
```

```
$ ruff check src/ tests/
All checks passed!
```

```
$ mypy src/
Success: no issues found in 34 source files
```

| Acceptance | Status |
|---|---|
| `_tolerance_overrides` is a `ContextVar` | PASS |
| 4-thread race test passes | PASS |
| Exception inside `tolerance_context` restores prior state | PASS |
| Nested contexts observe LIFO order | PASS |
| TM-E1: symlink outside allowlist rejected | PASS |
| TM-E1: lookalike `/usr/local/cuda-evil` rejected | PASS |
| TM-E1: warning emitted on rejection | PASS |
| TM-E1: allowlisted path returns the binary | PASS |

**Verdict: PASS.**
