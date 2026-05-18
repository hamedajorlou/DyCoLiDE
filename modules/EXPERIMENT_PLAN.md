# DyCoLiDE Experiment Plan

## Paper Goal
Show that DyCoLiDE is capable of:
1. Operating in **batch and online settings** for SEM data
2. Performing well on **SVAR data** in full batch and streaming cases

---

## Part 1: SEM Data (Static Causal Discovery)

### Experiments

| Experiment | Setting | What to Show |
|------------|---------|--------------|
| **1.1 Full Batch SEM** | All data available | Compare DyCoLiDE vs baselines on accuracy (SHD, TPR, FDR) and runtime |
| **1.2 Mini-Batch SEM** | Data in fixed batches | Show convergence with mini-batch SGD, compare to full batch |
| **1.3 Online SEM** | Data arrives sequentially | Show DyCoLiDE updates incrementally without retraining from scratch |

### Variables to Sweep
- **Nodes**: 20, 50, 100
- **Edges**: 1×d, 2×d, 4×d (sparsity levels)
- **Samples**: 500, 1000, 2000
- **Noise**: Equal variance (EV), Non-equal variance (NV)

### Baselines

| Category | Method | Package |
|----------|--------|---------|
| **Continuous Optimization** | | |
| | NOTEARS | `notears` |
| | GOLEM | `golem` |
| | DAGMA | `dagma` |
| | DAG-GNN | custom |
| | GraN-DAG | custom |
| **Constraint-Based** | | |
| | PC | `causal-learn` |
| | FCI | `causal-learn` |
| | GES | `causal-learn` |
| | FGES | `causal-learn` |
| **LiNGAM Family** | | |
| | ICA-LiNGAM | `lingam` |
| | DirectLiNGAM | `lingam` |
| | CAM | `cdt` |
| **Neural/Deep** | | |
| | DAG-GNN | custom |
| | DCDI | `dcdi` |

**Recommended (most cited, easy to run)**:
- NOTEARS, GOLEM, DAGMA, PC, DirectLiNGAM

---

## Part 2: SVAR Data (Dynamic Causal Discovery)

### Experiments

| Experiment | Setting | What to Show |
|------------|---------|--------------|
| **2.1 Full Batch SVAR** | All time-series available | Compare DyCoLiDE vs baselines on W and A recovery |
| **2.2 Streaming SVAR** | Time-series arrives in chunks | Show identical results to full batch, demonstrate memory efficiency |
| **2.3 Lag Order Study** | Vary p=1,2,3 | Show DyCoLiDE handles multiple lags |

### Variables to Sweep
- **Nodes**: 10, 20, 30, 50
- **Timesteps**: 500, 1000, 2000
- **Lag order**: 1, 2, 3
- **Edge density**: sparse vs dense

### Baselines

| Category | Method | Package |
|----------|--------|---------|
| **VAR-Based** | | |
| | Dynotears | `causalnex` |
| | VARLiNGAM | `lingam` |
| | SVAR-GFCI | `causal-learn` |
| **Granger Causality** | | |
| | Linear Granger | `statsmodels` |
| | TCDF (Nonlinear) | `tcdf` |
| | Neural GC (NGC) | custom |
| | cMLP/cLSTM | custom |
| **Constraint-Based** | | |
| | PCMCI | `tigramite` |
| | PCMCI+ | `tigramite` |
| | tsFCI | `causal-learn` |
| | LPCMCI | `tigramite` |
| **Score-Based** | | |
| | NTS-NOTEARS | custom |
| **Deep Learning** | | |
| | CUTS | custom |
| | Rhino | custom |
| | IDYNO | custom |

**Recommended**:
- Dynotears, PCMCI/PCMCI+, VARLiNGAM, Linear Granger, tsFCI

---

## Results Table Templates

### Table 1: SEM Comparison
```
Method          | Type          | d=20       | d=50       | d=100
                |               | SHD TPR FDR| SHD TPR FDR| SHD TPR FDR
────────────────|───────────────|────────────|────────────|────────────
PC              | Constraint    |            |            |
DirectLiNGAM    | LiNGAM        |            |            |
NOTEARS         | Continuous    |            |            |
GOLEM           | Continuous    |            |            |
DAGMA           | Continuous    |            |            |
DyCoLiDE (ours) | Continuous    |            |            |
```

### Table 2: SVAR Comparison
```
Method          | Type          | W (instant.)    | A (temporal)
                |               | TPR  FDR  F1    | TPR  FDR  F1
────────────────|───────────────|─────────────────|─────────────────
Granger         | Statistical   |  -    -    -    |
PCMCI+          | Constraint    |                 |
VARLiNGAM       | LiNGAM        |                 |
Dynotears       | Continuous    |                 |
DyCoLiDE (ours) | Continuous    |                 |
```

### Table 3: Batch vs Streaming (DyCoLiDE only)
```
Setting     | W_TPR | W_FDR | A_TPR | A_FDR | Time | Memory
────────────|────---|────---|────---|-------|------|--------
Full Batch  |       |       |       |       |      |
Streaming   |       |       |       |       |      |
```

---

## Ablation Studies

1. **Effect of regularization**: Sweep λ_W and λ_A
2. **Effect of threshold**: Sweep 0.05 to 0.15
3. **Batch size sensitivity**: For streaming, test batch_size = 1, 10, 50, 100
4. **Convergence speed**: Iterations needed vs accuracy

---

## Installation Commands

```bash
# Core baselines
pip install causal-learn    # PC, FCI, GES, LiNGAM
pip install tigramite       # PCMCI, PCMCI+
pip install lingam          # VARLiNGAM, DirectLiNGAM
pip install causalnex       # Dynotears
pip install dagma           # DAGMA
pip install statsmodels     # Granger causality
```

---

## Tuned Parameters (from TUNING_NOTES.md)

### DyCoLiDE for 30 nodes SVAR
- `lambda_W`: 0.01
- `lambda_A`: 0.015
- `threshold`: 0.10
- `temporal_strength`: 0.5 (for data generation)

Best single-seed results (seed=42):
- W_TPR: 1.000, W_FDR: 0.016
- A_TPR: 0.917, A_FDR: 0.035
