# Online DAG Learning from Observational Data

Reference implementation and experiments for **DyCoLiDE** (Dynamic CoLiDE) — an extension of [CoLiDE](https://arxiv.org/abs/2310.02894) that learns linear DAGs from observational data in three regimes:

- **Static SEM** — the classic `X = X W + noise` setting, solved in full batch or mini-batch / true-online SGD.
- **Streaming SEM** — data arrives sequentially; the model is updated incrementally without retraining from scratch.
- **SVAR time series** — `X_t = X_t W + Y_t A + noise`, where `Y_t` stacks lagged observations. Acyclicity is enforced on the contemporaneous DAG `W`; the temporal matrix `A` is L1-regularized only. Supports full batch and a streaming-accumulation variant.

The same optimizer skeleton is used everywhere: log-det acyclicity penalty `h(W) = -log det(sI − W⊙W) + d·log s`, ℓ1 sparsity, augmented Lagrangian with a decaying `mu` schedule, Adam inner loop.

## Repository layout

### Core algorithms
| File | Contents |
|---|---|
| [Colide.py](Colide.py) | `colide_ev`, `colide_nv` (static), plus `colide_ev_batch`, `colide_nv_batch` (EMA σ), `colide_nv_batch_cov` (covariance-based σ). The `_batch` classes support `batch_size=1` for true online learning via Welford's update. |
| [DyCoLiDE.py](DyCoLiDE.py) | `DyCoLiDE_EV`, `DyCoLiDE_NV` for SVAR, plus `DyCoLiDE_BatchStreaming` which buffers the last `p` observations across batches and runs full optimization at `finalize()`. |
| [SVAR_data_generator.py](SVAR_data_generator.py) | `generate_svar_data` — random DAG `B`, lag matrices `A_k`, companion-matrix stability check, EV/NV noise. |
| [Utils.py](Utils.py) | `create_dag`, `simulate_sem`, `count_accuracy` (SHD/TPR/FDR), graph helpers. |

### Baselines for comparison
| File | Method |
|---|---|
| [dagma_svar.py](dagma_svar.py) | DAGMA extended to SVAR (acyclicity on `W` only). |
| [dyno_golem.py](dyno_golem.py) | GOLEM extended to SVAR with the trace-exp DAG penalty. |
| [golem_sem.py](golem_sem.py) | Static GOLEM-EV / GOLEM-NV in PyTorch. |
| [transformers.py](transformers.py), [wrappers.py](wrappers.py) | Vendored Dynotears / causalnex utilities (Apache-2.0). |

### Experiment drivers
| File | Purpose |
|---|---|
| [compare_three_methods.py](compare_three_methods.py) | DyCoLiDE-EV vs Dynotears vs DynoDAGMA vs DynoGOLEM on SVAR data. Ships the tuned hyperparameters from [modules/TUNING_NOTES.md](modules/TUNING_NOTES.md). |
| [run_comparison.py](run_comparison.py) | GOLEM / DAGMA / CoLiDE / DyCoLiDE on static SEM (200 nodes, ER4). |
| [evolving_dag_experiment.py](evolving_dag_experiment.py) | Online tracking demo: the true DAG changes at `T/2`; measures how quickly DyCoLiDE adapts. |

### Tests
| File | Purpose |
|---|---|
| [test.py](test.py) | Single-config DyCoLiDE-EV smoke test on SVAR data. |
| [test_colide_batch.py](test_colide_batch.py) | Coverage for the `colide_*_batch*` variants in [Colide.py](Colide.py). |
| [test_EV_batch.py](test_EV_batch.py) | EV-side comparison across GOLEM / DAGMA / CoLiDE / CoLiDE-Batch. |

### Auxiliary code and notes
| Path | Contents |
|---|---|
| [modules/](modules/) | Experimental and ablation scripts, additional baselines (PCMCI+, Meta-D2AG, GOLEM in pure torch), hyperparameter search and tuning runs, and notebooks. |
| [modules/EXPERIMENT_PLAN.md](modules/EXPERIMENT_PLAN.md) | The full experimental plan — SEM and SVAR variants, baselines, ablations. |
| [modules/TUNING_NOTES.md](modules/TUNING_NOTES.md) | Best-known hyperparameters for 30-node SVAR. |

## Setup

```bash
pip install numpy scipy networkx pandas scikit-learn matplotlib tqdm torch
```

For the comparison baselines, install separately:

```bash
pip install causal-learn lingam causalnex statsmodels
```

A few of the comparison scripts ([run_comparison.py](run_comparison.py), [test_EV_batch.py](test_EV_batch.py), [modules/compare_dagma_*.py](modules/)) expect upstream `dagma/` and `golem/` source checkouts and currently reference an absolute path from the author's machine — update the `sys.path.insert(...)` line at the top of each before running.

## Reproducing the headline results

### SVAR comparison (30 nodes, 60 edges, 1000 timesteps)

```bash
python compare_three_methods.py --n_nodes 30 --n_edges 60 --n_timesteps 1000 --threshold 0.10 --methods dycolide dynogolem
```

Best-known numbers from [modules/TUNING_NOTES.md](modules/TUNING_NOTES.md) at seed=42:

| Method     | W_TPR | W_FDR | A_TPR | A_FDR | Time |
|------------|-------|-------|-------|-------|------|
| DyCoLiDE   | 1.000 | 0.016 | 0.917 | 0.035 | 69 s |
| DynoGOLEM  | 1.000 | 0.016 | 0.933 | 0.082 | 41 s |

Tuned parameters: `lambda_W=0.01`, `lambda_A=0.015`, `threshold=0.10`, with `temporal_strength=0.5` in the data generator.

### Online tracking under a structural change

```bash
python evolving_dag_experiment.py --d 20 --T 10000 --edges 40 --change 0.5
```

Generates non-stationary data where the true DAG flips ~50% of edges at `T/2`, then plots TPR / FDR / SHD over time as the sliding-window estimator adapts.

### Static SEM benchmark (200 nodes, ER4)

```bash
python run_comparison.py
```

Compares GOLEM-EV, DAGMA, CoLiDE-EV, and DyCoLiDE-EV at two noise levels (σ=1, σ=5) and reports mean ± std over multiple seeds. Requires DAGMA upstream; see Setup.

## Citation

If you use this code, please cite the paper *Online DAG Learning from Observational Data*.
