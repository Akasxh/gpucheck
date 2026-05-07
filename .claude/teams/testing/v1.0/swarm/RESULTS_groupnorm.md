# groupnorm fuzz — torch.nn.functional.group_norm
## Setup
- torch: 2.11.0
- mps available: True
- seed: 0xc0ffee
- requested iterations: 250
- attempted: 250
- completed (status=ok): 243
- unsupported: 0
- errors: 7
- divergences found: 11
- max rel err (ok runs): 9.618e-02

## Results table — counts by category
- shape categories: {'degenerate': 46, 'large': 57, 'non_tile_aligned': 44, 'pow2_boundary': 31, 'prime': 65}
- stride categories: {'broadcast': 54, 'contiguous': 52, 'slice': 72, 'transpose': 65}
- dtypes: {'bfloat16': 83, 'float16': 80, 'float32': 80}

## Top divergences (minimal repro)
1. shape=(1, 18, 13, 35) num_groups=6 dtype=float32 stride=transpose (applied=transpose_HW) max_abs_err=4.768e-07 max_rel_err=1.511e-03 atol=6.531e-04 rtol=2.000e-04
2. shape=(1, 32, 17, 16) num_groups=1 dtype=float32 stride=transpose (applied=transpose_HW) max_abs_err=4.768e-07 max_rel_err=3.279e-02 atol=1.649e-03 rtol=2.000e-04
3. shape=(1, 40, 21, 21) num_groups=10 dtype=float32 stride=transpose (applied=transpose_HW) max_abs_err=4.768e-07 max_rel_err=1.054e-03 atol=7.425e-04 rtol=2.000e-04

## Cross-backend max relative error
- MPS-vs-CPU max rel err (across ok runs): 9.618e-02
- MPS-vs-CUDA-mock max rel err: N/A (no NVIDIA GPU; CUDA backend mocked at detection level only — no kernel execution)

## Unsupported / errors (sample up to 3)
- iter=19 status=ERROR shape=[1, 2, 1, 1] dtype=bfloat16 stride=slice err=ValueError: Expected more than 1 value per channel when training, got input size [1, 2, 1, 1]
- iter=24 status=ERROR shape=[1, 1, 1, 1] dtype=float16 stride=contiguous err=ValueError: Expected more than 1 value per channel when training, got input size [1, 1, 1, 1]
- iter=98 status=ERROR shape=[1, 2, 1, 1] dtype=float32 stride=broadcast err=ValueError: Expected more than 1 value per channel when training, got input size [1, 2, 1, 1]

## Recommended upstream filing target
- pytorch/pytorch — divergences are between PyTorch's CPU and MPS group_norm implementations; both are PyTorch native.
