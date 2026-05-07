# Swarm fuzz results — sigmoid

- kernel: `sigmoid`
- iters_attempted: 990
- iters_completed: 990
- divergences_filable: 0
- divergences_recalibration: 0
- max_abs_err: 2.441406e-04
- max_rel_err (denom>=1e-6): 4.909180e-04
- max_rel_err (raw): 4.909180e-04
- unsupported_dtypes: none
- errors: 0
- elapsed_sec: 40.75
- torch: 2.11.0

## Top 3 repros

### #1
```json
{
  "seed": 4,
  "dtype": "fp16",
  "shape": [
    31,
    31
  ],
  "category": "slice",
  "max_abs": 0.000244140625,
  "max_rel_filtered": 0.0004909179988317192,
  "max_rel_raw": 0.0004909179988317192,
  "denom_at_max_rel_raw": 0.497314453125,
  "atol": 0.02,
  "rtol": 0.02,
  "abs_ratio": 0.01220703125,
  "rel_ratio_filtered": 0.024545899941585958
}
```

### #2
```json
{
  "seed": 3,
  "dtype": "fp16",
  "shape": [
    257,
    257
  ],
  "category": "column_major",
  "max_abs": 0.000244140625,
  "max_rel_filtered": 0.0004909179988317192,
  "max_rel_raw": 0.0004909179988317192,
  "denom_at_max_rel_raw": 0.497314453125,
  "atol": 0.02,
  "rtol": 0.02,
  "abs_ratio": 0.01220703125,
  "rel_ratio_filtered": 0.024545899941585958
}
```

### #3
```json
{
  "seed": 0,
  "dtype": "fp16",
  "shape": [
    511,
    511
  ],
  "category": "non_contig",
  "max_abs": 0.000244140625,
  "max_rel_filtered": 0.0004909179988317192,
  "max_rel_raw": 0.0004909179988317192,
  "denom_at_max_rel_raw": 0.497314453125,
  "atol": 0.02,
  "rtol": 0.02,
  "abs_ratio": 0.01220703125,
  "rel_ratio_filtered": 0.024545899941585958
}
```

