#!/usr/bin/env bash
# Tier-3 kernel-fuzzer swarm launcher.
# Spawns 26 headless `claude -p` processes, one per kernel target, in waves
# of 6 (so we don't blow rate limits or kill the Mac).
#
# Each kernel runs in its own worktree at ~/Code/gpucheck-worktrees/fuzz-<kernel>/
# and writes RESULTS_<kernel>.md + a JSONL line into:
#   ~/Code/gpucheck/.claude/teams/testing/v1.0/swarm/
#
# Usage:
#   bash launch-swarm.sh                # full 26-kernel run
#   bash launch-swarm.sh relu softmax   # subset
#   DRYRUN=1 bash launch-swarm.sh       # print, don't execute
#
# Required: claude CLI in PATH, gpucheck installed in each worktree's venv (or
# global), Python with torch.mps available.

set -euo pipefail

KERNELS_ALL=(
  relu softmax layernorm
  matmul-fp32 matmul-fp16 matmul-bf16
  attention cross_entropy gelu silu
  rmsnorm rope conv2d batchnorm groupnorm
  gemm-3d flash-attn-v1 flash-attn-v2
  scatter gather index_select
  topk argmax nll_loss kl_div cosine_sim
)

WAVE_SIZE="${WAVE_SIZE:-6}"
WAVE_DELAY_S="${WAVE_DELAY_S:-15}"
DRYRUN="${DRYRUN:-0}"

if [ $# -gt 0 ]; then
  KERNELS=("$@")
else
  KERNELS=("${KERNELS_ALL[@]}")
fi

SWARM_DIR="$HOME/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
WORKTREE_BASE="$HOME/Code/gpucheck-worktrees"

mkdir -p "$SWARM_DIR/logs"

prompt_for() {
  local kernel="$1"
  cat <<EOF
You are kernel-fuzzer-${kernel}.

CWD: ${WORKTREE_BASE}/fuzz-${kernel}/
Output dir: ${SWARM_DIR}/

Task: Run gpucheck's stride/contiguity + shape + dtype fuzzers against the ${kernel}
kernel. Use the new MPS backend (real, on this Mac) and the existing CUDA backend
via mocked detection (since no NVIDIA GPU is present). Run 250 iterations.

Method:
1. Confirm torch.mps.is_available() is True; if False, log SKIPPED and exit 0.
2. For each iteration: sample shape (degenerate, prime, power-of-2 boundary,
   non-tile-aligned, large), dtype (fp32, fp16, bf16 as supported), and stride
   pattern (contiguous, non-contiguous via slice, transpose, broadcast).
3. Run the kernel on MPS and on CPU (reference). Compute max relative error.
4. Mark a divergence if max-rel-err > tolerance for the dtype (use gpucheck's
   default per-dtype tolerance, scaled by sqrt(k/128) for matmul-class ops).
5. For each divergence, capture the minimal repro: shape, dtype, stride spec.

Deliverables:
- Markdown summary at ${SWARM_DIR}/RESULTS_${kernel}.md with:
  * kernel name
  * total iterations attempted, completed
  * divergences found
  * top 3 minimal repros (shape + dtype + stride)
  * MPS-vs-CPU max relative error
  * MPS-vs-CUDA-mock max relative error (or N/A if mocked)
  * recommended upstream filing target (pytorch/pytorch | triton-lang/triton | none)
- One JSON line appended to ${SWARM_DIR}/swarm.jsonl with the same fields.

Hard rules:
- Total runtime budget: 8 minutes. If you exceed budget, write what you have
  and exit cleanly.
- Do not invent divergences. If torch.mps cannot run the op, mark UNSUPPORTED.
- Halt and report any process error immediately.

When done, write the markdown and JSON line, then exit.
EOF
}

run_kernel() {
  local kernel="$1"
  local logfile="${SWARM_DIR}/logs/${kernel}.log"
  echo "[$(date -u +%H:%M:%S)] launching kernel-fuzzer-${kernel}"
  if [ "$DRYRUN" = "1" ]; then
    echo "(dryrun) prompt for ${kernel}:"
    prompt_for "$kernel" | head -5
    return 0
  fi
  (
    cd "${WORKTREE_BASE}/fuzz-${kernel}"
    prompt_for "$kernel" | claude -p --model opus > "$logfile" 2>&1 &
    echo $! > "${SWARM_DIR}/logs/${kernel}.pid"
  )
}

wait_wave() {
  echo "[$(date -u +%H:%M:%S)] waiting ${WAVE_DELAY_S}s before next wave..."
  sleep "$WAVE_DELAY_S"
}

i=0
for kernel in "${KERNELS[@]}"; do
  run_kernel "$kernel"
  i=$((i + 1))
  if [ $((i % WAVE_SIZE)) -eq 0 ] && [ "$i" -lt "${#KERNELS[@]}" ]; then
    wait_wave
  fi
done

echo "[$(date -u +%H:%M:%S)] all ${#KERNELS[@]} swarm processes spawned. PIDs in ${SWARM_DIR}/logs/"
echo "monitor with: tail -f ${SWARM_DIR}/logs/*.log"
