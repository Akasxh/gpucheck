# Skeptic — red-team of "MPSGraph fp32 1024³ kernel-pick stickiness" before pytorch/pytorch filing

## Leading hypothesis (as I understand it)
On Apple M5 / macOS 26.4.1 / torch 2.11.0, MPSGraph's fp32 GEMM at exactly
`(1024, 1024, 1024)` selects a sub-optimal kernel on cold-start, sticks at
~3.1 ms / 685 GFLOPs through ~50+ iterations, and the cache is invalidated
*specifically* by dispatching any other shape. The bug is shape-and-dtype
keyed (`(M=N=K=1024, fp32)`); off-shape (e.g. 2048³, 1023³, 512³) flips
1024³ fp32 to a fast 0.85 ms / 2521 GFLOPs path. fp16/bf16 at 1024³ also
warm-up but mildly. The fix-class is precedented (#136003 SDPA dispatch
variability, #182805 group_norm prim-decomposition routing).

I attempted **8 attacks** with verifiable scripts. **Three landed and
materially weaken the issue.** The strongest is **A4 — spontaneous
transition without off-shape**, which contradicts a primary mechanism
claim. Below, ranked by danger to the filing.

---

## Attack 1 (LANDED, MAJOR): "off-shape unblock" is plausibly post hoc

**Attack.** The investigation's central mechanism claim is *"off-shape
dispatch invalidates the cache; without it, slow path persists 50+
iters."* If, in fact, the slow→fast transition fires *spontaneously* on
a wall-clock or call-count schedule that is *independent* of off-shape
dispatch, then the entire mechanism story is wrong. The off-shape
"unblocks" only because it adds the ~5-15 ms of wall-time that pushes
the system past the spontaneous transition. The maintainer would
quickly close the bug as "this is generic JIT/compile/scheduler warmup,
not a shape-keyed cache anomaly."

**Verification result (LANDED).** I ran `/tmp/skeptic-spontaneous.py` —
60 consecutive 1024³ fp32 calls, identical tensors, no other shape ever
touched, no fp16, no allocator wiggle:

```
0:7.46  1:3.09  2:3.06  3:3.07  4:2.68  5:1.74  6:1.72  7:1.73  ...
spontaneous transition at iter: 5
```

The transition fires at iter 5, *with no off-shape dispatch.* Three
back-to-back runs in `/tmp/skeptic-thermal-gap.py` show the transition
firing at iters 10+ (Run A), 5 (Run B), and 7 (Run C). This pattern is
consistent with a *time-based* transition (JIT compile finishing
async on a scheduler thread, command-buffer batching catching up, or
GPU clock ramp). It is *not* consistent with a shape-keyed cache.

**Verification command.** Re-run `python /tmp/skeptic-spontaneous.py` —
if any future run fails to spontaneously transition within 60 iters, the
off-shape claim is rehabilitated. The investigation's own
`03-repro-long-run.py` Test A noted "after ~50 same-shape fp32 calls,
on call ~50 the timing transitions spontaneously" but **this admission
was buried in the body and not propagated to the 'off-shape unblocks
the cache' framing in the issue draft.** The maintainer will read
"transitions spontaneously at ~50" and ask: *if it transitions
spontaneously, why call it 'sticky'? You measured 15 calls.*

---

## Attack 2 (LANDED, MAJOR): the "warm path" is non-deterministic — 1.7 ms or 0.85 ms

**Attack.** The investigation reports a single "fast path" at 0.85 ms /
2521 GFLOPs. My replication shows two distinct fast regimes:
~1.7-1.8 ms (Run B above; the 60-iter spontaneous-transition test) and
~0.85-1.05 ms (Run C above; the post-2048³ phase in the original
script). A maintainer will ask: *"if the bug is 'wrong kernel sticks',
which kernel is the correct one — the 1.7 ms one or the 0.85 ms one?
Why are there three?"* Without an answer, the issue body's GFLOPs
arithmetic is suspect. 2 × 1024³ ÷ (1.7 ms / 1000) ÷ 1e9 ≈ 1262 GFLOPs.
That's nowhere near the 2521 GFLOPs the issue claims.

**Verification result (LANDED).** My `/tmp/skeptic-thermal-gap.py` Run B
explicitly captured the 1.7 ms steady-state warm path *without any
off-shape dispatch*. The original investigation's
`02-repro-isolate.py` Test 3 showed warm 1024³ fp32 ≈ 1.0 ms after
2048³ poke — closer to my 1.05 ms than to the 0.85 ms in the headline.
The "0.85 ms / 2521 GFLOPs" number appears to come from a *different*
run state (maybe post-warmup of multiple shapes including 2048³) than
the "3.1 ms / 685 GFLOPs" baseline.

**Verification command.** `uv run python -c "import torch, time;
a=torch.randn(1024,1024,device='mps',dtype=torch.float32); b=a.clone();
torch.mps.synchronize();
[print(f'{i}: {(__import__(\"time\").perf_counter()-t)*1000:.3f}ms') or
torch.mps.synchronize() or (t:=time.perf_counter()) for i,t in
enumerate([time.perf_counter()] + [None]*30) if a@b is not None]"` (or
just rerun `/tmp/skeptic-thermal-gap.py` 5 times — you will see the
warm steady state hit 1.7 ms and 0.85 ms in different runs). The issue
body MUST acknowledge 2 fast regimes, or maintainer rejects.

---

## Attack 3 (LANDED, MODERATE): reproduction is flaky — 1 of 3 runs failed to reproduce slow phase entirely

**Attack.** The investigation claims "anomaly reproduces deterministically
across multiple processes and seeds" (Section 7, "high confidence").
My replication shows it does NOT. Three back-to-back identical runs
of `02-repro-isolate.py`:

| Run | iter 0 | iter 1 | iter 5 | iter 10 |
|-----|--------|--------|--------|---------|
| 1   | 3.37   | 1.02   | 1.03   | 0.78    |
| 2   | 4.99   | 2.35   | 3.17   | 3.04    |
| 3   | 3.31   | 1.00   | 1.02   | 1.00    |

Run 1 and Run 3 entered the **fast path immediately at iter 1**,
skipping the slow phase entirely. Run 2 reproduced the original
investigation. This is 33% non-reproduction in 3 attempts. A maintainer
will ask: *"are you sure this isn't just 'first-call after some
unspecified system idle' rather than '1024³ specifically'?"*

**Verification result (LANDED).** Direct experiment results above. The
investigation's `04-repro-exact-original.py` itself shows this in the
fp32 1024³ row of the "Direct" sweep — `raw=[3.043, 3.016, 2.998,
3.381, 3.683, 2.402, 1.626, 2.828, 2.847, 1.967]` — the median is
2.92 ms but samples 6 and 9 are 1.6 ms / 1.97 ms. The transition is
fluid, not bistable.

**Verification command.** Run `02-repro-isolate.py` 5 fresh times,
report iter 1-14 timings for each. If 30%+ of runs skip the slow phase,
the bug is conditional on hidden state (most likely the time since the
*previous* MPS process exited / GPU idle clock state) — and the issue
body needs to specify that condition. Without specification, the
maintainer cannot reproduce, and triage stalls.

---

## Attack 4 (PARTIALLY LANDS): 12 MB L2 cache thrash hypothesis

**Attack.** Apple M5 cluster L2 is 12 MB per cluster. 3 × 4 MB fp32
1024² tensors = exactly 12 MB. The slow phase could be cache thrash:
the working set fits "just barely" but eviction patterns make every
tile fetch miss. `1023³` (3 × ~4.0 MB but with 1023×1023 strides) and
`1025³` (3 × ~4.0 MB) have different alignment and may not thrash. If
the bug is cache-thrash, not kernel-pick, the maintainer's fix
(re-tune the heuristic) won't help — they'd need to rework cache-line
strides, which is hardware-bound.

**Verification result (PARTIAL).** I ran `/tmp/skeptic-cache-thrash.py`.
Test T5a: write 256 MB junk to evict L2, then re-run 1024³ fp32 — got
0.84 ms (fast). T2: 1448³ fp16 (also ~12 MB working set) — got 0.91 ms
(fast, not stuck). T3: rectangular 1024×512 × 512×1024 fp32 — got
0.74 ms (fast, not stuck). The L2-thrash hypothesis is **not strongly
supported** — same byte-budget at fp16 doesn't reproduce. BUT the
post-eviction 1024³ fp32 was fast — could be that *any* MPS work
(including allocating 256 MB junk) does the same "off-shape" unblock
as #1. So this attack collapses into A1.

**Verification command.** Compute peak L2 hit/miss with Metal counters
via `MTLCaptureManager` — the investigation's stated open question
("did NOT instrument MPSGraph internals to confirm which exact kernel
is dispatched"). Without this, kernel-pick vs cache-thrash is a coin
flip from the data we have.

---

## Attack 5 (DEFLECTS): JIT compile cost amortization

**Attack.** The slow path's per-call cost (3.13 ms) could be the same
kernel as fast path PLUS a one-time async compile cost amortized over
N calls. If true, total wall-time should converge: T_n = C/n + steady.
The fast path (after off-shape) just had the compile finish async
during the off-shape's longer dispatch.

**Verification result.** The 60-iter run shows a clear *step transition*
at iter 5 (3.07 → 1.74), not a 1/n decay. This refutes pure compile
amortization. But it does NOT refute "compile finishes async at
~iter 4-7 and the fast kernel becomes available." That's still
consistent with all data and is a *much more boring* explanation than
"sticky cache key" — and one Apple's MPSGraph team will recognize
immediately as "yes, MPSGraph compiles async and warmup is required;
not a bug."

**Verification command.** Set the env var
`MPS_DEBUG_KERNEL_PICK_LOG=1` (if it exists) or instrument with
`MTLCaptureManager` *during iter 4 and iter 5* on the slow process.
Confirm whether iter 5 dispatches a *different* command-encoder /
pipeline-state object than iter 4. If yes → kernel-pick. If same →
async compile finished mid-stream. The investigation explicitly admits
this was not done. **Filing without this distinction is risky.**

---

## Attack 6 (DEFLECTS): macOS 26.4.1 vs 26.4.2 / Metal framework version

**Attack.** macOS 26.4.1 (build 25E253). 26.4.2 / future Metal point
releases may have already-fixed this. Filing on a stale OS = "please
update and retest."

**Verification result.** I cannot test 26.4.2 from this machine. Apple
typically does not publish detailed Metal kernel-heuristic changelogs.
Risk is real but *modest* — maintainers usually accept reproducible
26.4.1 reports if reproducer is clean. The bigger risk is that I do
not know the user's Xcode version because `xcodebuild -version` is
unavailable on this machine (CommandLineTools only). The MPSGraph
framework version is whatever ships with the running OS — that is
26.4.1's MPSGraph, currently unpinned. If the user upgrades macOS
before the maintainer triages, the bug self-resolves and looks like a
ghost.

**Verification command.** Before filing: `uv run python -c "import
torch.backends.mps as m; print(m._get_metal_version() if
hasattr(m,'_get_metal_version') else 'n/a')"` — hard to get reliably,
but record exact build (`sw_vers -buildVersion` → `25E253`). Note in
issue body: "verified on macOS 26.4.1 build 25E253; not yet tested on
26.4.2."

---

## Attack 7 (REFUTED): the 3.7× speedup is a `torch.mps.synchronize()` artifact

**Verification (LANDED, REFUTES THE ATTACK).** I ran
`/tmp/skeptic-event-vs-sync.py`. Event-based timing (skipping wall-clock
sync overhead) gave cold/warm = **2.64×**; wall-clock sync gave
**2.41×**. Sync overhead alone was sub-microsecond (median 0.0 ms in
10 calls). So the speedup is real, not a sync artifact. **HOWEVER**,
note that 2.64× is materially less than the headline "3.7×" in the
investigation. The investigation's 3.7× appears to be cold (3.13ms) /
warm-best (0.85 ms). My event-timed measurements show cold (3.18ms) /
warm-typical (1.20ms) = 2.64×. **The issue body's 3.7× claim
overstates the magnitude.** It should say "2-4× depending on which
fast kernel hits."

---

## Attack 8 (REFUTED): thermal throttling

**Verification.** `pmset -g therm` shows no thermal warnings before or
after the test. Cold-start sequence happens within 200 ms of process
launch — no time for thermal accumulation. The fast path follows
*after* additional GPU work (more heat, not less), opposite of what
thermal throttling would do.

---

## Unstated assumptions in the current synthesis

1. **"The slow path holds for ~50 iters without intervention."** False
   under my replication — transitions at iter 5-15 spontaneously.
   *Consequence:* the "off-shape unblocks" claim collapses into "any
   GPU work eventually transitions."
2. **"There is one fast path at 0.85 ms."** False — there are at least
   two regimes (1.7 ms steady, 0.85 ms steady). *Consequence:* the
   GFLOPs arithmetic in the issue body is contestable.
3. **"The bug is keyed by (shape, dtype)."** Possibly false — the
   transition fires without any other shape/dtype touching the
   pipeline. *Consequence:* the proposed mitigations (off-shape
   tickle on init) don't even need a *different* shape — the same
   shape just needs more time / more iterations.
4. **"`PYTORCH_MPS_PREFER_METAL=1` is not the workaround."** True, but
   the slow MPSGraph path (3.1 ms) being slower than the
   hand-written Metal kernel (4.2 ms) is **only true if MPSGraph
   stays in slow phase indefinitely.** The fast MPSGraph path
   (0.85-1.7 ms) is faster than `do_metal_mm`. The comparison was
   made on slow-phase MPSGraph vs Metal — apples to oranges.
5. **"Anomaly reproduces deterministically."** False; 1 of 3
   re-runs failed to enter the slow phase. The investigation's own
   raw data also shows mid-run transitions.

## Evidence quality audit

- **"Cold 1024 fp32: 3.13 ms / 685 GFLOPs."** Backed by 14-iter
  sequence in script 02. Weak because: my replication is flaky (3 runs:
  one stuck-all-14, one transitions at iter 5, one transitions at
  iter 11). Stronger evidence would look like: 10 fresh-process runs
  with iter-by-iter timings, reporting fraction of runs that transition
  before iter 15.
- **"Warm 1024 fp32: 0.85 ms / 2521 GFLOPs."** Backed by `repro-results.json`
  with `warmup=10, n=50`. Weak because: that warmup-10 setup measures
  *only post-transition* timings; slower-warm-regime (1.7-1.8 ms)
  exists and is invisible to that protocol. Stronger evidence: bimodal
  histogram of 200+ samples after long warmup, showing all observed
  steady-state regimes.
- **"Off-shape (2048³, 512³, 1023³) all unblock 1024³."** Backed by
  Tests C/D/E in 03-repro-long-run.py. Weak because: each of these
  tests added wall-clock that itself crosses the spontaneous-
  transition threshold. Stronger evidence: dispatch off-shape *and
  immediately* return; measure within 1 ms wall-clock; compare to
  "no dispatch but wait equivalent wall-clock idle." If both transition
  identically, off-shape is irrelevant.
- **"Switching dtype on same shape does NOT unblock fp32."** Backed by
  Test 2 in script 02. Stronger but conditional — that test ran fp16 5x
  and then back to fp32; if the spontaneous transition is timer-based,
  5 fp16 calls (~5 ms total) might not exceed the timer. Stronger
  evidence: do the same test *with 50 fp16 iters and 30+ s of wall
  time* before returning to fp32.
- **"Cache key is (shape=1024×1024×1024, dtype=fp32)."** Hypothesis,
  not measured. Investigation explicitly admits no `MTLCaptureManager`
  trace was captured. **The single most filing-improving data point
  remains uncollected.** Without it, this is conjecture.

## Verdict

- **Prematurely converged?** YES.
- **Safe to raise confidence to "high"?** NO — should be **medium**
  pending an `MTLCaptureManager` trace and a 5+ run reproducibility
  study with explicit transition-iter histogram.
- **Required next probes before filing:**
  1. `MTLCaptureManager` capture during slow-phase and fast-phase
     1024³ fp32. Confirm pipeline-state ID changes (kernel-pick) vs
     stays the same (async compile catches up). Skill exists:
     `metal-shader-profiling`. **This is the load-bearing next step.**
  2. 10 back-to-back fresh-process runs of `02-repro-isolate.py`,
     report transition-iter histogram. If >30% transition before iter
     15, the "sticky" framing is wrong.
  3. Replicate "off-shape unblocks" *with same wall-time* of pure
     idle (e.g. `time.sleep(15ms) ; back to 1024³`). If sleep-only
     also unblocks, the off-shape claim is post hoc.
  4. Verify on macOS 26.4.2 if available (or note explicitly that
     26.4.1 is the verified version).
  5. Reconcile two-fast-regime observation (1.7 ms vs 0.85 ms) — is
     the 0.85 ms a third post-warmup transition?

## Final classification

**HOLD** → demote to **REQUIRES_MORE_REPRO** before filing.

The bug is *probably* real (the slow phase reproduces at all in 2 of 3
runs, the timing differential is well outside noise, the SDPA-side
sibling #136003 is open and theme-matched). But three of the issue
body's mechanism claims (off-shape unblocks, slow-stays-50-iters,
single fast path at 0.85 ms) are contradicted by either my new
replications or the investigation's own buried raw data. Filing as-is
will likely get *"please clarify; cannot reproduce reliably; what
about iter 5?"* response and a "needs-info" label.

Strongest single mitigating action: **collect a Metal-frame-capture
trace of slow-phase iter 3 vs fast-phase iter 7 of the same process,
proving the pipeline-state object differs.** With that, file. Without
it, hold.

---

## Summary table — 5 strongest attacks ranked by danger

| # | Attack | Danger | Refuted? |
|---|--------|--------|----------|
| 1 | Off-shape unblock is post hoc; spontaneous transition at iter 5-15 | **HIGH** | NO — landed |
| 2 | Two fast regimes (1.7ms and 0.85ms) — issue body's 0.85ms / 2521 GFLOPs is cherry-picked | **HIGH** | NO — landed |
| 3 | Reproduction is flaky (1/3 fresh-process runs skip slow phase) | **MED-HIGH** | NO — landed |
| 4 | Async JIT compile finishing mid-stream (not a kernel-pick anomaly) | **MED** | Untested without MTLCaptureManager |
| 5 | macOS 26.4.1 may be already-fixed in 26.4.2; user can't repro | **LOW-MED** | Untested |

Verdict: **REQUIRES_MORE_REPRO** before filing.
