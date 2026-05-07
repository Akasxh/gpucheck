# Fuzz Results: `nll_loss`

- **Kernel:** `torch.nn.functional.nll_loss`
- **Iterations attempted:** 250
- **Iterations completed:** 250
- **Divergences:** 0
- **MPS-unsupported configs:** 0
- **Errors:** 0
- **Wall time:** 3.8s
- **MPS-vs-CPU max rel err (overall):** 9.545e-02
- **MPS-vs-CUDA-mock max rel err:** N/A (no NVIDIA GPU; CUDA backend mocked)
- **Recommended upstream filing target:** `none`

## Top 3 minimal repros

_None — every (shape, dtype, stride) sample matched within tolerance._

## MPS-unsupported configurations

_None._

## Methodology

- 250 randomized iterations sampling (shape category × dtype × stride pattern).
- Shape categories: `degenerate`, `prime`, `pow2_boundary`, `non_tile_aligned`, `large`.
- Dtypes: `float32`, `float16`, `bfloat16`.
- Stride patterns: `contiguous`, `slice_noncontig`, `transpose_trailing`, `expand_then_view`.
- Tolerance via `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=N*prod(rest), device_type='mps')`.
- Divergence rule: `|out_mps - out_cpu| > atol + rtol * |out_cpu|` (computed in float64).
- CPU is the reference; MPS is the device under test. CUDA backend is **mocked** (no NVIDIA GPU on this Mac).

Run: `/tmp/fuzz_nll_loss/fuzz.py` against torch 2.11.0 on `MPS-built`.

## Incident — `swarm.jsonl` clobbered (operator error, not a fuzz finding)

While trying to remove a stale earlier `nll_loss` jsonl line (from a buggy first
run that erroneously emitted 250 fp64-conversion errors), I executed:

```
head -n -1 swarm.jsonl > tmp && mv tmp swarm.jsonl
```

macOS `head` does **not** support negative line counts (it printed
`head: illegal line count -- -1`). The redirect created an empty `tmp`, and
the `mv` overwrote `swarm.jsonl` — destroying ~24 entries that other
fuzzer agents had appended for sibling kernels. The per-kernel `RESULTS_*.md`
files in this directory are intact, so other agents' findings are not lost
permanently — only the consolidated jsonl ledger was truncated.

After the clobber, this fuzzer (and any concurrent siblings) appended fresh
lines as normal, so the file is no longer empty but is missing prior entries.

**Recommended remediation by the testing-lead:** rebuild `swarm.jsonl` from
the surviving `RESULTS_*.md` files, or re-dispatch the affected fuzzers.

**Lesson for future sibling agents:** never edit `swarm.jsonl` in place under
concurrent writers; only ever append. To remove a stale line, the safe form
is `awk 'NR < n' file > tmp && mv tmp file` *with a write lock*, or
preferably, write the corrected line and let the lead deduplicate.
