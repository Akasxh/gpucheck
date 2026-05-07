#!/usr/bin/env bash
# v2 swarm launcher: 98 kernels × 1000 iters × 4 dtypes
set -euo pipefail

# All 98 kernels from v2 spec
KERNELS_ALL=(
  relu gelu silu mish swish softmax log_softmax softplus sigmoid tanh
  hardtanh hardsigmoid hardswish elu selu celu prelu rrelu
  layernorm rmsnorm batchnorm groupnorm instancenorm
  matmul-fp32 matmul-fp16 matmul-bf16 matmul-fp64
  bmm-fp32 bmm-fp16 baddbmm einsum-2d einsum-3d einsum-4d
  conv1d conv2d conv3d conv-transpose-2d conv-depthwise
  avg-pool max-pool adaptive-avg-pool adaptive-max-pool
  attention attention-causal attention-padded flash-attn-v1 flash-attn-v2
  cross_entropy nll_loss bce bce_with_logits mse l1 huber smooth_l1 kl_div cosine_sim
  scatter scatter_add gather index_select index_put masked_select masked_fill
  topk argmax argmin sort argsort cumsum cumprod cummax cummin
  rope alibi-bias
  embedding embedding_bag one_hot
  split chunk cat stack repeat_interleave tile expand broadcast_to
  transpose permute view reshape flatten unflatten
  normalize l2_normalize standardize
  dropout dropout-2d alpha-dropout
)

WAVE_SIZE="${WAVE_SIZE:-10}"
WAVE_DELAY_S="${WAVE_DELAY_S:-30}"
ITERS="${ITERS:-1000}"
SEEDS="${SEEDS:-0,1,2,3,4}"
DRYRUN="${DRYRUN:-0}"

if [ $# -gt 0 ]; then
  KERNELS=("$@")
else
  KERNELS=("${KERNELS_ALL[@]}")
fi

SWARM_DIR="$HOME/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
WORKTREE_BASE="$HOME/Code/gpucheck-worktrees"

mkdir -p "$SWARM_DIR/v2_logs"

prompt_for() {
  local kernel="$1"
  cat <<PROMPT
You are kernel-fuzzer-${kernel} (v2). Run gpucheck's stride/contiguity + shape + dtype fuzzers against the ${kernel} kernel using the new MPSBackend. ${ITERS} iterations across seeds ${SEEDS}, dtypes fp32/fp16/bf16 (skip bf16 for kernels where unsupported).

CRITICAL DIVERGENCE FILTERING:
- max_rel_err > 10× tolerance ONLY counts as divergence if denom_magnitude >= 1e-6 (i.e., NOT a near-zero-denominator artifact)
- max_abs_err > 10× tolerance always counts
- Both must be reproducible across ≥3 seeds to be FILABLE
- 1-5× tolerance = TOLERANCE_RECALIBRATION bucket (recommend xfail entry)
- Below 1× tolerance = OK

Output:
- ${SWARM_DIR}/RESULTS_${kernel}.md with: kernel, iters_attempted, iters_completed, divergences_filable, divergences_recalibration, max_abs_err, max_rel_err, top_3_repros
- One JSON line appended to ${SWARM_DIR}/swarm.jsonl

Use the merged release/v1.0 (current worktree at ${WORKTREE_BASE}/fuzz-${kernel}). torch==2.11.0, MPS available.

Hard rules:
- 12 minute wall budget. If exceeded, write what you have and exit cleanly.
- Do not invent divergences. UNSUPPORTED if torch.mps lacks the op.
- Halt and report any process error immediately.
PROMPT
}

run_kernel() {
  local kernel="$1"
  local logfile="${SWARM_DIR}/v2_logs/${kernel}.log"
  echo "[$(date -u +%H:%M:%S)] launching kernel-fuzzer-${kernel}-v2"
  if [ "$DRYRUN" = "1" ]; then
    echo "(dryrun)"
    return 0
  fi
  (
    cd "${WORKTREE_BASE}/fuzz-${kernel}" 2>/dev/null || cd ~/Code/gpucheck
    prompt_for "$kernel" | claude -p --model opus > "$logfile" 2>&1 &
    echo $! > "${SWARM_DIR}/v2_logs/${kernel}.pid"
  )
}

i=0
for kernel in "${KERNELS[@]}"; do
  run_kernel "$kernel"
  i=$((i + 1))
  if [ $((i % WAVE_SIZE)) -eq 0 ] && [ "$i" -lt "${#KERNELS[@]}" ]; then
    echo "[$(date -u +%H:%M:%S)] wave $((i/WAVE_SIZE)) done; sleeping ${WAVE_DELAY_S}s"
    sleep "$WAVE_DELAY_S"
  fi
done

echo "[$(date -u +%H:%M:%S)] all ${#KERNELS[@]} v2 swarm processes spawned."
