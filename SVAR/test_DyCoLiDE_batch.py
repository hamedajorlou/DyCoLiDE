"""
SVAR Model Comparison: DyCoLiDE-EV static (full batch) vs mini-batch SGD
(DyCoLiDE_EV_batch) as graph size and density scale.

Mirrors SEM/test_EV_batch.py for SVAR data:
  - DyCoLiDE_EV         — static full-batch reference (no "vanilla static
                          colide" exists for SVAR; this plays that role)
  - DyCoLiDE_EV_batch   at batch_size in {100, 50, 1}

Sweeps n_nodes × ER density × seed (multi-seed, mean ± std). Both W
(intra-slice DAG) and A (inter-slice / temporal effects) metrics are
reported. The same data X is generated once per (er, d, seed) and passed
to all 4 methods (with .copy() for safety).
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# Make repo root and SVAR/ importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from Utils import generate_svar_data
from dycolide import DyCoLiDE_EV, DyCoLiDE_EV_batch
from Utils import count_accuracy_svar


# =============================================================================
# DATA / MODEL CONFIG
# =============================================================================

# SVAR generation defaults from TUNING_NOTES.md (the 30-node config the paper
# tuned on). edges = ER_level × d, temporal_edges matched, strength=0.5.
DATA_CONFIG = {
    'n_timesteps':       1000,
    'lag_order':         1,
    'temporal_strength': 0.5,
    'noise_scale':       1.0,
    'noise_type':        'ev',     # this file is the EV variant
}

# Tuned hyperparameters from TUNING_NOTES.md
COLIDE_CONFIG = {
    'lambda_W':  0.01,
    'lambda_A':  0.015,
}

THRESHOLD = 0.10                    # 0.10 from TUNING_NOTES.md


# =============================================================================
# SWEEP CONFIG
# =============================================================================

NODES_LIST  = [10, 20, 30, 40, 50]
BATCH_SIZES = [100, 50, 1]
ER_LEVELS   = [2, 3, 4]
SEEDS       = [42, 43, 44, 45, 46]

# Same iteration budget across all methods (matches the SEM sweeps).
SWEEP_STATIC_KW = dict(T=4, mu_init=1.0, mu_factor=0.1,
                       s=[1.0, 0.9, 0.8, 0.7],
                       warm_iter=20000, max_iter=70000, lr=0.0003,
                       checkpoint=5000)

SWEEP_BATCH_KW  = dict(T=4, mu_init=1.0, mu_factor=0.1,
                       s=[1.0, 0.9, 0.8, 0.7],
                       n_batches_warm=20000, n_batches_final=70000, lr=0.0003,
                       checkpoint=5000)


# =============================================================================
# METRICS / BOOKKEEPING
# =============================================================================

def nmse(W_est, W_true):
    denom = np.sum(W_true ** 2)
    if denom <= 1e-12:
        return float(np.sum(W_est ** 2))
    return float(np.sum((W_est - W_true) ** 2) / denom)


def _empty_cells():
    cells = {}
    for er in ER_LEVELS:
        for d in NODES_LIST:
            for m in ['Static'] + [f'bs={bs}' for bs in BATCH_SIZES]:
                cells[(er, d, m)] = []
    return cells


def _agg(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'values': [float(x) for x in a]}


def _save_partial(cells, path):
    methods = ['Static'] + [f'bs={bs}' for bs in BATCH_SIZES]
    results = {f'ER{er}': {m: {} for m in methods} for er in ER_LEVELS}
    for (er, d, m), seed_rows in cells.items():
        if not seed_rows:
            continue
        results[f'ER{er}'][m][f'd={d}'] = {
            # W (intra-slice DAG)
            'W_tpr':  _agg([r['W_tpr']  for r in seed_rows]),
            'W_fdr':  _agg([r['W_fdr']  for r in seed_rows]),
            'W_shd':  _agg([r['W_shd']  for r in seed_rows]),
            'W_nmse': _agg([r['W_nmse'] for r in seed_rows]),
            # A (inter-slice / temporal)
            'A_tpr':  _agg([r['A_tpr']  for r in seed_rows]),
            'A_fdr':  _agg([r['A_fdr']  for r in seed_rows]),
            'A_shd':  _agg([r['A_shd']  for r in seed_rows]),
            'A_nmse': _agg([r['A_nmse'] for r in seed_rows]),
            'time_s': _agg([r['time_s'] for r in seed_rows]),
            'seeds_done': [r['seed'] for r in seed_rows],
        }

    payload = {
        'experiment': 'dycolide_ev_svar_scaling_sweep_multiseed',
        'description': (
            'DyCoLiDE-EV static (full batch) vs mini-batch SGD '
            '(DyCoLiDE_EV_batch) at batch_size in {100, 50, 1}, across '
            'n_nodes in {20,40,60,80,100} and ER densities {2,3,4} on SVAR '
            'data (lag_order=1), averaged over 5 seeds. Reports both W '
            '(intra-slice DAG) and A (inter-slice / temporal) metrics. '
            "Equal-variance noise (noise_type='ev'). JSON rewrites after "
            'every fit so progress is never lost if the run is interrupted.'
        ),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': {
            'nodes_list':        NODES_LIST,
            'batch_sizes':       BATCH_SIZES,
            'er_levels':         ER_LEVELS,
            'seeds':             SEEDS,
            'n_timesteps':       DATA_CONFIG['n_timesteps'],
            'lag_order':         DATA_CONFIG['lag_order'],
            'temporal_strength': DATA_CONFIG['temporal_strength'],
            'noise_scale':       DATA_CONFIG['noise_scale'],
            'noise_type':        DATA_CONFIG['noise_type'],
            'threshold':         THRESHOLD,
            'lambda_W':          COLIDE_CONFIG['lambda_W'],
            'lambda_A':          COLIDE_CONFIG['lambda_A'],
            'static_kw':         SWEEP_STATIC_KW,
            'batch_kw':          SWEEP_BATCH_KW,
            'static_class':      'DyCoLiDE_EV',
            'batch_class':       'DyCoLiDE_EV_batch',
        },
        'results': results,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# SWEEP
# =============================================================================

def run_scaling(json_path=None):
    """Multi-seed sweep: ER → d → seed → method."""
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__),
                                 'dycolide_ev_scaling_5seeds.json')
    cells = _empty_cells()
    p = DATA_CONFIG['lag_order']

    for er in ER_LEVELS:
        print(f"\n{'#' * 80}\n#  ER{er} (edges = {er} * d, SVAR)\n{'#' * 80}",
              flush=True)

        for d in NODES_LIST:
            edges = er * d

            for seed in SEEDS:
                # Generate SVAR data once per (er, d, seed); pass same X to all methods.
                # Wrap in try/except: rare seed×config combos can hit SVAR stability
                # failures (B's (I-B)^-1 too ill-conditioned). Skip those cells.
                try:
                    X, B_true, A_list_true, _ = generate_svar_data(
                        n_nodes=d,
                        n_timesteps=DATA_CONFIG['n_timesteps'],
                        lag_order=p,
                        instantaneous_edges=edges,
                        temporal_edges=edges,
                        temporal_strength=DATA_CONFIG['temporal_strength'],
                        noise_scale=DATA_CONFIG['noise_scale'],
                        noise_type=DATA_CONFIG['noise_type'],
                        seed=seed,
                        max_stability_attempts=50,
                    )
                except ValueError as e:
                    print(f"\n--- ER{er}, d={d}, seed={seed}: SKIPPED "
                          f"(SVAR generator: {e}) ---", flush=True)
                    continue
                # DyCoLiDE convention: row-vector form
                W_true = B_true.T
                A_true = np.vstack([A.T for A in A_list_true])
                n_W = int((np.abs(W_true) > 0).sum())
                n_A = int((np.abs(A_true) > 0).sum())
                print(f"\n--- ER{er}, d={d}, seed={seed}, "
                      f"W edges={n_W}, A edges={n_A} ---", flush=True)

                # Static DyCoLiDE-EV (full batch)
                t0 = time.time()
                model = DyCoLiDE_EV(seed=seed)
                W_est, A_est, _ = model.fit(
                    X.copy(), p=p,
                    lambda_W=COLIDE_CONFIG['lambda_W'],
                    lambda_A=COLIDE_CONFIG['lambda_A'],
                    **SWEEP_STATIC_KW)
                dt = time.time() - t0
                m = count_accuracy_svar(W_true, W_est, A_true, A_est, threshold=THRESHOLD)
                row = {
                    'seed': seed,
                    'W_tpr': float(m['W_tpr']), 'W_fdr': float(m['W_fdr']),
                    'W_shd': int(m['W_shd']),  'W_nmse': nmse(W_est, W_true),
                    'A_tpr': float(m['A_tpr']), 'A_fdr': float(m['A_fdr']),
                    'A_shd': int(m['A_shd']),  'A_nmse': nmse(A_est, A_true),
                    'time_s': dt,
                }
                cells[(er, d, 'Static')].append(row)
                print(f"  Static     W: SHD={row['W_shd']:>3} NMSE={row['W_nmse']:.4f}   "
                      f"A: SHD={row['A_shd']:>3} NMSE={row['A_nmse']:.4f}   "
                      f"t={dt:6.1f}s", flush=True)
                _save_partial(cells, json_path)

                # Mini-batch DyCoLiDE-EV-batch at each batch_size
                for bs in BATCH_SIZES:
                    t0 = time.time()
                    model = DyCoLiDE_EV_batch(seed=seed)
                    W_est, A_est, _ = model.fit(
                        X.copy(), p=p,
                        lambda_W=COLIDE_CONFIG['lambda_W'],
                        lambda_A=COLIDE_CONFIG['lambda_A'],
                        batch_size=bs, **SWEEP_BATCH_KW)
                    dt = time.time() - t0
                    m = count_accuracy_svar(W_true, W_est, A_true, A_est, threshold=THRESHOLD)
                    row = {
                        'seed': seed,
                        'W_tpr': float(m['W_tpr']), 'W_fdr': float(m['W_fdr']),
                        'W_shd': int(m['W_shd']),  'W_nmse': nmse(W_est, W_true),
                        'A_tpr': float(m['A_tpr']), 'A_fdr': float(m['A_fdr']),
                        'A_shd': int(m['A_shd']),  'A_nmse': nmse(A_est, A_true),
                        'time_s': dt,
                    }
                    cells[(er, d, f'bs={bs}')].append(row)
                    print(f"  bs={bs:<5}  W: SHD={row['W_shd']:>3} NMSE={row['W_nmse']:.4f}   "
                          f"A: SHD={row['A_shd']:>3} NMSE={row['A_nmse']:.4f}   "
                          f"t={dt:6.1f}s", flush=True)
                    _save_partial(cells, json_path)

    return cells, json_path


def print_summary(cells):
    methods = ['Static'] + [f'bs={bs}' for bs in BATCH_SIZES]

    def stat(er, m, d, field):
        seeds = cells.get((er, d, m), [])
        if not seeds:
            return None
        vals = np.asarray([r[field] for r in seeds], dtype=float)
        return vals.mean(), vals.std()

    for er in ER_LEVELS:
        for side in ['W', 'A']:
            print("\n" + "=" * 110)
            print(f"ER{er}  —  {side} SHD (mean ± std over {len(SEEDS)} seeds)")
            print("=" * 110)
            print(f"{'Method':<10}" + "".join(f"{'d='+str(d):>20}" for d in NODES_LIST))
            print("-" * 110)
            for m in methods:
                line = f"{m:<10}"
                for d in NODES_LIST:
                    s = stat(er, m, d, f'{side}_shd')
                    line += f"   SHD={s[0]:5.1f}±{s[1]:4.1f}" if s else f"{'-':>20}"
                print(line)

            print(f"\nER{er}  —  {side} NMSE (mean ± std)")
            print("-" * 110)
            print(f"{'Method':<10}" + "".join(f"{'d='+str(d):>20}" for d in NODES_LIST))
            for m in methods:
                line = f"{m:<10}"
                for d in NODES_LIST:
                    s = stat(er, m, d, f'{side}_nmse')
                    line += f" NMSE={s[0]:.4f}±{s[1]:.4f}" if s else f"{'-':>20}"
                print(line)


if __name__ == "__main__":
    cells, json_path = run_scaling()
    print_summary(cells)
    print(f"\nFinal results saved to {json_path}")
