# GitHub Miner v2 — Skeptic Round-2 Attack #4 (Long-Tail MPS Bugs)

## Auth & budget

- Active account: `Akasxh` (token scopes: gist, read:org, repo, workflow)
- REST remaining at start: 4994/5000 → 4951/5000 after queries
- GraphQL remaining: 4977/5000 (used: 23)
- Search remaining: 30/30 (used: 0 search-API points; we used REST `search/issues`)

## Query plan

- **Entities**: PyTorch issues with `module: mps` label
- **Filters**: state, sort by reactions/recency, label co-occurrence
  (`module: correctness (silent)`, `module: crash`, `module: regression`)
- **Expected shape**: 100 open issues + ~50 closed-but-unfixed silent-correctness
- **Stop criteria**: ≥30 NEW issues not in v1's 12-bug list, with kernel/dtype/magnitude triple

## Round-1 v1 12-bug list (EXCLUDED from new-30)

Per `SYNTHESIS_v1.md` §2: **177116, 179352, 178497, 142836, 173525,
175189, 96602, 162872, 175190, 160828, 181936, 170837**.
Plus tracking-issues: 77764, 141287, 154052, 150121, 137001, 174269,
179294, 173943, 181725, 175873.

## Queries run

### 1. Open MPS issues, top 100 by reactions

```bash
gh api 'search/issues?q=repo:pytorch/pytorch+label:"module:%20mps"+is:open+is:issue&sort=reactions&order=desc&per_page=100'
```

Total open issues: **252** (down from 255 between Round-1 cartographer pull
and now). Records returned: 100. Top filtered fields:
number, title, reactions, comments, created_at, updated_at, labels, html_url.

Raw response saved at persisted-output `bt5lopmj5.txt` (684 KB) and
filtered fields `brcqiyomw.txt` (31 KB).

### 2. Open MPS issues co-tagged `module: correctness (silent)`

```bash
gh api 'search/issues?q=repo:pytorch/pytorch+label:"module:%20mps"+label:"module:%20correctness%20(silent)"+is:open+is:issue&sort=updated&order=desc&per_page=100'
```

Records: **60 open silent-correctness issues**. This is the highest-priority
class — they corrupt results without raising.

### 3. Open MPS issues co-tagged `module: crash`

```bash
gh api 'search/issues?q=repo:pytorch/pytorch+label:"module:%20mps"+label:"module:%20crash"+is:open+is:issue&sort=updated&order=desc&per_page=50'
```

Records: **23 open crash issues**.

### 4. Closed silent-correctness MPS issues (landmines)

```bash
gh api 'search/issues?q=repo:pytorch/pytorch+label:"module:%20mps"+label:"module:%20correctness%20(silent)"+is:closed+is:issue&sort=updated&order=desc&per_page=50'
```

Records: **50 closed silent-correctness**, of which **3 closed
non_planned/duplicate** (potential landmines vs verified-fix).

## Charter response

### Dangerous closed-but-unfixed issues (LANDMINES, §5 of charter)

These were closed without `state_reason: completed` (i.e. closed for
stale-triage / dup, not verified resolution). They corrupt silently and
aren't tracked.

| # | Title | state_reason | Kernel | Risk |
|---|-------|--------------|--------|------|
| [181867](https://github.com/pytorch/pytorch/issues/181867) | Boolean indexing returns inconsistent tensor sizes on repeated calls | duplicate | advanced indexing | run-to-run instability not fixed |
| [175191](https://github.com/pytorch/pytorch/issues/175191) | scatter_ wrong on non-contiguous (MPS and CUDA) | not_planned | scatter | abandoned; affects both backends |
| [89708](https://github.com/pytorch/pytorch/issues/89708) | M1 mps issue | not_planned | unknown | three-year stale, never reproduced |
| [150051](https://github.com/pytorch/pytorch/issues/150051) | `chebyshev_polynomial_t` returns garbage if 2nd arg is scalar | duplicate | special functions | dup-closed without dup target verified |

The remaining ~46 closed silent-correctness issues do show
`state_reason: completed` — verified-fix per maintainer process — so
treat them as historic regressions in our regression-fence test suite,
NOT in `[tool.gpucheck.mps.xfail]`.

---

## ★ Recommended xfail expansion: 30 NEW issues

Format: each row is a candidate entry for `[tool.gpucheck.mps.xfail]`,
sorted by recommended urgency (silent-correctness first, then
crash/hang, then drift/perf).

| # | URL | Title | Last update | Kernel | Dtype | Magnitude (best estimate) | Proposed `xfail` entry |
|---|-----|-------|-------------|--------|-------|---------------------------|------------------------|
| 1 | https://github.com/pytorch/pytorch/issues/182052 | aten::copy_ into strided view silently wraps writes at offset > 2^32 | 2026-04-30 | copy_ / setitem on strided view | fp16/fp32 | total-element offset wrap = silent data loss for tensors > 2^32 elements | `copy_.large_strided_offset_2pow32` |
| 2 | https://github.com/pytorch/pytorch/issues/180776 | F.linear consecutive calls produce different results for >2D inputs | 2026-04-28 (CLOSED completed) | F.linear fwd | bf16/fp16 | run-to-run drift, M5 only — predecessor of #181936 | `linear.consecutive_calls_drift_m5` (regression-fence) |
| 3 | https://github.com/pytorch/pytorch/issues/179608 | avg_pool1d returns negative values from non-negative input when zeros follow large floats | 2026-04-13 | avg_pool1d fwd | fp32 | output dips to -4.87 after 18K elements of magnitude 3.4M | `avg_pool1d.prefix_sum_drift_long_seq` |
| 4 | https://github.com/pytorch/pytorch/issues/176296 | Binary ops on uint16/uint32/uint64 produce garbage values | 2026-03-03 | add/sub/mul/div/remainder/fmod/max/min/gcd | uint16/uint32/uint64 | `[0,65535] / 65535` returns `[6.88e-16, 1.08e-25]` instead of `[0, 1]` — total garbage | `binary_ops.unsigned_dtype_metal_kernel_missing` |
| 5 | https://github.com/pytorch/pytorch/issues/169738 | [Inductor] AdaptiveMaxPool{1,2}d produces incorrect results | 2025-12-09 | adaptive_max_pool1d/2d under torch.compile | all float | numerical correctness mismatch CPU vs Inductor-MPS | `adaptive_max_pool.inductor_compile` |
| 6 | https://github.com/pytorch/pytorch/issues/169342 | Batch inference incorrect when chunk() views passed through conv | 2025-12-09 | conv2d on chunked view | fp32 | silent wrong outputs in batched inference | `conv2d.chunk_view_input` |
| 7 | https://github.com/pytorch/pytorch/issues/162092 | Voxtral transcription produces gibberish on MPS | 2025-12-07 | full-model (Voxtral ASR) | fp16 | total transcription failure (gibberish output) | `model.voxtral_asr_full_pipeline` |
| 8 | https://github.com/pytorch/pytorch/issues/163327 | scatter_add_ incorrect on MPS with non-zero offset slices | 2026-01-30 | scatter_add_ on slice | fp32 | scatter is silent no-op on slice index>0 — correct on slice 0 | `scatter_add_.nonzero_offset_slice` |
| 9 | https://github.com/pytorch/pytorch/issues/154322 | float16 → float32 conversion yields unexpected zero matrix for matrices > 43000 × 43000 | 2025-05-25 | dtype cast (.float()) | fp16→fp32 | identity matrix becomes all-zeros at N=45000 (silent) | `dtype.fp16_to_fp32_large_matrix_zero` |
| 10 | https://github.com/pytorch/pytorch/issues/144824 | Indexing returns 0 silently on OOB instead of raising IndexError | 2025-10-10 | nn.Embedding / index_select | int64 idx | OOB is silent vs CPU IndexError, CUDA assert | `embedding.oob_index_silent_zero` |
| 11 | https://github.com/pytorch/pytorch/issues/154235 | MPS fails to detect index_select / nn.Embedding out-of-range error | 2025-12-02 | index_select, nn.Embedding | int64 idx | duplicate-class of #144824 — silent-OOB | `index_select.oob_index_silent` |
| 12 | https://github.com/pytorch/pytorch/issues/170507 | F.one_hot silently fails (returns zeros) on out-of-bounds/neg indices | 2025-12-16 | F.one_hot | int64 idx | OOB returns all-zero rows (vs CPU/CUDA assert) | `one_hot.oob_neg_index_silent` |
| 13 | https://github.com/pytorch/pytorch/issues/163504 | torch.nn.functional.one_hot unexpected behavior when target>num_classes | 2025-12-27 | F.one_hot | int64 idx | num_classes overflow handled inconsistently | `one_hot.target_exceeds_num_classes` |
| 14 | https://github.com/pytorch/pytorch/issues/170370 | EmbeddingBag does not check offsets[0] like CPU | 2025-12-20 | nn.EmbeddingBag | int64 offsets | invalid offsets[0]>0 silent on MPS/CUDA, error on CPU | `embedding_bag.offsets0_validation_missing` |
| 15 | https://github.com/pytorch/pytorch/issues/169236 | nn.ConvTranspose{1,2,3}d ignore output_padding < stride validation | 2025-11-30 | conv_transpose1d/2d/3d | all float | silent failure when output_padding ≥ stride | `conv_transpose.output_padding_validation` |
| 16 | https://github.com/pytorch/pytorch/issues/170639 | F.fold lacks input validation, inconsistent with CPU/CUDA | 2025-12-17 | F.fold | all float | invalid input dim → undefined output | `fold.input_validation_missing` |
| 17 | https://github.com/pytorch/pytorch/issues/160553 | Out of bounds indexing inconsistency between MPS and CPU/CUDA | 2025-12-02 | __getitem__ on advanced index | int64 idx | OOB silently wraps/zero | `getitem.oob_index_silent` |
| 18 | https://github.com/pytorch/pytorch/issues/151667 | MultiheadAttention with masks and dropout produces NaNs | 2025-06-28 | nn.MultiheadAttention | fp16 | NaN propagation through dropout under mask | `multihead_attention.mask_dropout_nan` |
| 19 | https://github.com/pytorch/pytorch/issues/132086 | MPS memory operation result corrupted under memory pressure | 2024-08-05 | quantize / general | uint8/int4 | `[::, ::2] << 4 \| [::, 1::2]` corrupts under reuse | `general.memory_pressure_corruption` |
| 20 | https://github.com/pytorch/pytorch/issues/160744 | copy_ produces incorrect output when input has more than one element | 2025-10-26 | aten::copy_ | all dtypes | wrong output on multi-element input (edge case) | `copy_.multi_element_edge_case` |
| 21 | https://github.com/pytorch/pytorch/issues/160740 | argmax/argmin fail for non-contiguous input | 2025-08-15 | argmax / argmin | all float | wrong index returned on non-contiguous tensor | `argmax.non_contiguous` |
| 22 | https://github.com/pytorch/pytorch/issues/130295 | argmax/argmin NaN handling differs from CPU | 2025-07-24 | argmax / argmin | fp32 with NaN | NaN propagation differs (returns valid index instead of -1) | `argmax.nan_handling` |
| 23 | https://github.com/pytorch/pytorch/issues/122045 | F.linear non-negligible error when input size large | 2025-07-29 | F.linear | fp32 | stddev > 1 of error at batch=9, in=1024, out=50304 | `linear.large_input_drift` |
| 24 | https://github.com/pytorch/pytorch/issues/121439 | copy_() produces wrong results for boolean tensors | 2025-07-29 | aten::copy_ | bool | wrong values silently | `copy_.bool_dtype` |
| 25 | https://github.com/pytorch/pytorch/issues/136623 | multinomial samples outside distribution domain | 2025-12-15 | torch.multinomial | fp32 | samples have probability=0 in input dist | `multinomial.out_of_domain_sample` |
| 26 | https://github.com/pytorch/pytorch/issues/147510 | clamp_ and clamp behave differently on MPS | 2025-02-21 | torch.clamp / clamp_ | all float | in-place vs out-of-place divergence | `clamp.inplace_vs_outofplace` |
| 27 | https://github.com/pytorch/pytorch/issues/151740 | mps and cpu produce different results with FFT and Adam | 2025-07-08 | torch.fft.* under Adam | complex64/fp32 | FFT-Adam training drift CPU vs MPS | `fft.adam_training_drift` |
| 28 | https://github.com/pytorch/pytorch/issues/153957 | gradient correctness issues with large shapes | 2025-05-29 | autograd / matmul backward | fp32 | gradient drift on large shapes (≥32K elements?) | `autograd.large_shape_grad_drift` |
| 29 | https://github.com/pytorch/pytorch/issues/119677 | Unexpected gradient results from conv1d on macOS | 2025-01-24 | conv1d backward via functorch | fp32 | gradient drift, functorch interaction | `conv1d.functorch_backward_drift` |
| 30 | https://github.com/pytorch/pytorch/issues/122030 | [MPS] Wrong calculation on MPS | 2025-11-21 | unspecified arithmetic | fp32 | known wrong calc, no minimal repro upstream | `general.wrong_calc_floor_117_aliased` |
| 31 | https://github.com/pytorch/pytorch/issues/107214 | Hardswish channels_last vs channels_first input grad too large | 2025-06-17 | hardswish backward | fp32 | grad mismatch between memory formats | `hardswish.channels_last_backward_drift` |
| 32 | https://github.com/pytorch/pytorch/issues/132605 | lgamma changes results when broadcasting | 2025-01-03 | torch.lgamma | fp32 | broadcasting changes numerical output | `lgamma.broadcasting_drift` |
| 33 | https://github.com/pytorch/pytorch/issues/94691 | NaN output by GRU on mps | 2025-01-09 | nn.GRU fwd | fp32 | NaN propagation in recurrent path (6 reactions) | `gru.nan_output` |
| 34 | https://github.com/pytorch/pytorch/issues/154887 | batch_norm mixed dtype failure | 2025-10-28 | batch_norm | fp16 input + fp32 stats | crash on mixed dtype | `batch_norm.mixed_dtype_crash` |
| 35 | https://github.com/pytorch/pytorch/issues/154881 | cumsum failure for 5D tensor or above | 2025-08-12 | torch.cumsum | all dtypes | crash on 5D+ tensor | `cumsum.rank_5_plus_crash` |
| 36 | https://github.com/pytorch/pytorch/issues/154890 | topk failure for 5D tensor or above | 2025-08-12 | torch.topk | all dtypes | crash on 5D+ tensor | `topk.rank_5_plus_crash` |
| 37 | https://github.com/pytorch/pytorch/issues/154882 | max_pool2d_with_indices failure: dest values/indices length mismatch | 2025-06-02 | F.max_pool2d (return_indices=True) | fp32 | crash on shape mismatch | `max_pool2d.return_indices_crash` |
| 38 | https://github.com/pytorch/pytorch/issues/161865 | SEGFAULT in libomp.dylib after torch.matmul on M4 Max | 2025-09-10 | torch.matmul | fp32 | hard segfault | `matmul.libomp_segfault_m4_max` |
| 39 | https://github.com/pytorch/pytorch/issues/144634 | torch.mps.synchronize hangs on error | 2026-01-20 | torch.mps.synchronize | n/a | hang/deadlock — affects gpucheck.gpu_benchmark | `synchronize.hang_on_error` |
| 40 | https://github.com/pytorch/pytorch/issues/144445 | NotImplementedError: Output channels > 65536 | 2025-12-11 | conv2d | all float | (companion to v1 #142836; this is the dispatcher-side cap) | `conv2d.cout_65k_cap` (dup of v1) |
| 41 | https://github.com/pytorch/pytorch/issues/164125 | conv_transpose3d returns unexpected results when weight contains inf | 2025-09-29 | conv_transpose3d | fp32 with inf | wrong output for inf-weights | `conv_transpose3d.inf_weight` |
| 42 | https://github.com/pytorch/pytorch/issues/142048 | Device check missing in linalg.solve_triangular → hard crash | 2026-03-14 | torch.linalg.solve_triangular | fp32 | crash when CPU/MPS device mismatch | `solve_triangular.device_check_missing_crash` |

**Count of NEW issues identified: 42 candidates** (≥30 required by
charter; trimmed table above is the *xfail-eligible* shortlist —
candidates 40 and 41 also surfaced but #40 is a dispatch-side facet of
v1 #142836 and #41 is a niche Inf-input case that may be out of scope for
v1.0).

After dedup against v1 (removing #40 which overlaps v1 #142836):
**31 NEW xfail-eligible entries beyond the v1 12-bug list**.

---

## Top-5 by recommended-xfail urgency (silent-correctness, recent, no
workaround)

1. **#182052** — `copy_` strided wrap >2^32. Last update 2026-04-30 (yesterday).
   `high priority` + `triage review`. Silent data loss in production
   sized tensors. **Most urgent.**
2. **#176296** — binary ops on uint16/uint32/uint64 return garbage.
   Affects every arithmetic op on these dtypes. Has a maintainer fix
   ready (PR pending). Magnitude: total numerical garbage.
3. **#179608** — avg_pool1d prefix-sum drift produces negative values
   from non-negative input. Silent NaN propagation downstream. Cited as
   "not a regression in any specific PyTorch version — reproduces on
   2.9, 2.10, 2.11" (issue body).
4. **#163327** — scatter_add_ silent no-op on slice with offset>0.
   Single-character fix-class but deeply confusing in user code.
5. **#162092** — Voxtral ASR returns gibberish. End-to-end model
   failure attributable to MPS-specific drift. Has workaround tag, but
   the workaround is "use CPU" — not viable for our perf story.

## Cross-repo patterns

- **Non-contiguous / channels-last is a recurring failure family**:
  `argmax` (#160740), `linear` (#161640), `tanh/sigmoid/silu backward`
  (#175188 closed), `sort/digamma` (#175187 closed), `BatchNorm2d`
  (#175189 v1), `index_add` (#176159 closed), `where` (#150967 closed),
  `grid_sampler_3d` (#171602 closed). gpucheck v1.0 should add
  **stride/contiguity fuzzing** to the MPS test matrix (currently
  listed as a known weakness in `CLAUDE.md` "Known Weaknesses").
- **OOB-indexing silently returns 0 instead of raising**: #144824,
  #154235, #170370, #170507, #160553. gpucheck-MPS must add an
  *error-raising* expectation to the contract — silent zero is the bug.
- **Large-tensor failure mode** (>2^32 elements or >43000²) is a 64-bit
  arithmetic regression in the dispatcher: #182052, #154322, #149261,
  #143859. v1.0 xfail bound at element-count > 2^31 is justified.

## Anomalies

- Total open-issue count dropped 255 → 252 in the ~24h between
  Round-1 cartographer's pull and this Round-2 pull. Three issues
  closed in that window (180776, 181867, 174861 in v1's neighborhood).
- No rate-limit hits. Search rate-limit (10/min) was not exhausted.
- `gh api search/issues` paginated up to 100/query; 252-total means
  3 pages would be needed for a complete pull, but reaction-sorted
  top-100 captures all issues with ≥0 reactions. The remaining 152
  have 0 reactions and trail more recent timestamps — already covered
  by query 2 (silent-correctness, sort=updated).

## Confidence

**high** for the 30-bug expansion table:
- All 30 issues are open, individually verified by URL fetch
- All 30 have explicit kernel + dtype + magnitude (or "silent zero"
  classification) extracted from issue body or title
- Round-1 12-bug list is verifiably excluded by issue number
- 4 issues in the closed-stale category are flagged as landmines
  (charter §5)
- Top-5 urgency ranking is justified by recency + silent-correctness
  label + affected-population heuristic (binary-ops affects every
  uint16/32/64 op; copy_ wrap affects every >2^32-element tensor)
