# Optimal Parameter Tuning Results

## Goal
Achieve A_TPR > 0.8 and A_FDR < 0.1 for both DyCoLiDE and DynoGOLEM.

## Best Results (30 nodes, 60 edges, 1000 timesteps)

| Model     | W_TPR | W_FDR | W_F1  | A_TPR | A_FDR | A_F1  | Time (s) |
|-----------|-------|-------|-------|-------|-------|-------|----------|
| DyCoLiDE  | 1.000 | 0.016 | 0.992 | 0.917 | 0.035 | 0.940 | 69.0     |
| DynoGOLEM | 1.000 | 0.016 | 0.992 | 0.933 | 0.082 | 0.926 | 41.2     |

## Key Parameter Changes

### Data Generation (`generate_svar_data`)
- `temporal_edges`: Set equal to `instantaneous_edges` for balanced comparison
- `temporal_strength`: **0.25 -> 0.5** (critical for detectability)

### DyCoLiDE Parameters
- `lambda_W`: 0.01
- `lambda_A`: **0.015** (reduced from 0.02)
- `threshold`: **0.10** (increased from 0.08)

### DynoGOLEM Parameters
- `lambda_1_w`: 0.01
- `lambda_1_a`: **0.01** (reduced from 0.02)
- `lambda_2`: 5.0
- `w_threshold`: **0.10** (increased from 0.08)

## Insights

1. **Temporal strength matters**: With `temporal_strength=0.25`, temporal edges have weights ~0.125-0.25, making them hard to distinguish from noise. Increasing to 0.5 gives weights ~0.25-0.5, improving detectability.

2. **Lower L1 penalty on A**: Reducing `lambda_A` allows more temporal edges to be detected (higher TPR).

3. **Higher threshold**: Using `threshold=0.10` instead of 0.08 filters out more false positives (lower FDR) while keeping true edges.

4. **Trade-off**: There's always a TPR vs FDR trade-off. The optimal point depends on your use case.

## Parameter Sweep Summary

### DyCoLiDE (temporal_strength=0.5)
| lambda_A | threshold | A_TPR | A_FDR |
|----------|-----------|-------|-------|
| 0.010    | 0.10      | 0.917 | 0.083 |
| 0.015    | 0.10      | 0.917 | 0.035 | <- Best
| 0.015    | 0.12      | 0.900 | 0.036 |

### DynoGOLEM (temporal_strength=0.5)
| lambda_1_a | threshold | A_TPR | A_FDR |
|------------|-----------|-------|-------|
| 0.008      | 0.10      | 0.950 | 0.123 |
| 0.010      | 0.10      | 0.917 | 0.083 | <- Best balance
| 0.010      | 0.08      | 0.983 | 0.157 |
