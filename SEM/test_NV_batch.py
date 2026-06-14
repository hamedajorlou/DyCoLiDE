"""
NV Model Comparison: CoLiDE-NV static vs mini-batch SGD (covariance-based
sigma update — colide_nv_batch_cov) as graph size and density scale.

Mirrors test_EV_batch.py but for the heteroscedastic case:
  - colide_nv         — static full-batch reference
  - colide_nv_batch_cov at batch_size in {100, 50, 1}

Sweeps n_nodes × ER density × seed (multi-seed, mean ± std). The same data
X is generated once per (er, d, seed) and passed to all 4 methods.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# Make repo root importable when running this file from the SEM/ subfolder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils import simulate_sem, to_bin, count_accuracy
from SEM.dycolide import colide_nv, colide_nv_batch_cov


# =============================================================================
# DATA / MODEL CONFIG
# =============================================================================

DATA_CONFIG = {
    'n_samples':  1000,
    'graph_type': 'er',
    'edge_type':  'weighted',
    'noise':      'normal',
    'var':        1.0,
    'w_range':    ((-2.0, -0.5), (0.5, 2.0)),
}

# Default lambda1 (same as EV). Lambda tuning per d can override later.
COLIDE_CONFIG       = {'lambda1': 0.05}
COLIDE_BATCH_CONFIG = {'lambda1': 0.05}

THRESHOLD = 0.3


# =============================================================================
# SWEEP CONFIG
# =============================================================================

NODES_LIST  = [20, 40, 60, 80, 100]
BATCH_SIZES = [100, 50, 1]                       # bs=1 → true online
ER_LEVELS   = [2, 3, 4]                          # edges per node → ER2, ER3, ER4
SEEDS       = [42, 43, 44, 45, 46]               # 5 seeds

# Same iteration budget across all methods (matches the EV sweep).
SWEEP_STATIC_KW = dict(T=4, mu_init=1.0, mu_factor=0.1,
                       s=[1.0, 0.9, 0.8, 0.7],
                       warm_iter=20000, max_iter=70000, lr=0.0003)

SWEEP_BATCH_KW  = dict(T=4, mu_init=1.0, mu_factor=0.1,
                       s=[1.0, 0.9, 0.8, 0.7],
                       n_batches_warm=20000, n_batches_final=70000, lr=0.0003)


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
            'tpr':    _agg([r['tpr']    for r in seed_rows]),
            'fdr':    _agg([r['fdr']    for r in seed_rows]),
            'shd':    _agg([r['shd']    for r in seed_rows]),
            'nmse':   _agg([r['nmse']   for r in seed_rows]),
            'time_s': _agg([r['time_s'] for r in seed_rows]),
            'seeds_done': [r['seed'] for r in seed_rows],
        }

    payload = {
        'experiment': 'colide_nv_scaling_sweep_multiseed',
        'description': (
            'CoLiDE-NV static (colide_nv) vs mini-batch SGD with covariance-'
            'based per-node sigma update (colide_nv_batch_cov) at batch_size '
            'in {100, 50, 1}, across n_nodes in {20,40,60,80,100} and ER '
            'densities {2,3,4}, averaged over 5 seeds. Each metric stores '
            'mean, std, and per-seed values. Heteroscedastic noise '
            "(var_type='nv'). JSON rewrites after every fit so progress is "
            'never lost if the run is interrupted.'
        ),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': {
            'nodes_list':  NODES_LIST,
            'batch_sizes': BATCH_SIZES,
            'er_levels':   ER_LEVELS,
            'seeds':       SEEDS,
            'graph_type':  DATA_CONFIG['graph_type'],
            'edge_type':   DATA_CONFIG['edge_type'],
            'n_samples':   DATA_CONFIG['n_samples'],
            'noise':       DATA_CONFIG['noise'],
            'var':         DATA_CONFIG['var'],
            'w_range':     [list(r) for r in DATA_CONFIG['w_range']],
            'threshold':   THRESHOLD,
            'lambda1':     COLIDE_CONFIG['lambda1'],
            'static_kw':   SWEEP_STATIC_KW,
            'batch_kw':    SWEEP_BATCH_KW,
            'var_type':    'nv',
            'static_class': 'colide_nv',
            'batch_class':  'colide_nv_batch_cov',
        },
        'results': results,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# SWEEP
# =============================================================================

def run_scaling(json_path=None):
    """Multi-seed sweep: ER → d → seed → method (Static, then each bs)."""
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__),
                                 'colide_nv_scaling_5seeds.json')
    cells = _empty_cells()

    for er in ER_LEVELS:
        print(f"\n{'#' * 80}\n#  ER{er} (edges = {er} * d)\n{'#' * 80}", flush=True)

        for d in NODES_LIST:
            edges = er * d

            for seed in SEEDS:
                # Same X for all 4 methods in this (er, d, seed) cell
                X, W_true, _ = simulate_sem(
                    n_nodes=d, n_samples=DATA_CONFIG['n_samples'], edges=edges,
                    graph_type=DATA_CONFIG['graph_type'],
                    edge_type=DATA_CONFIG['edge_type'],
                    var_type='nv', noise=DATA_CONFIG['noise'],
                    var=DATA_CONFIG['var'], w_range=DATA_CONFIG['w_range'],
                    seed=seed,
                )
                W_true_bin = to_bin(W_true, thr=0.0)
                n_true = int(W_true_bin.sum())
                print(f"\n--- ER{er}, d={d}, seed={seed}, true edges={n_true} ---",
                      flush=True)

                # Static CoLiDE-NV
                t0 = time.time()
                W_est, _ = colide_nv(seed=seed).fit(
                    X=X.copy(), lambda1=COLIDE_CONFIG['lambda1'], **SWEEP_STATIC_KW)
                dt = time.time() - t0
                shd, tpr, fdr = count_accuracy(W_true_bin, to_bin(W_est, thr=THRESHOLD))
                err = nmse(W_est, W_true)
                cells[(er, d, 'Static')].append({
                    'seed': seed, 'tpr': float(tpr), 'fdr': float(fdr),
                    'shd': int(shd), 'nmse': err, 'time_s': dt})
                print(f"  Static     SHD={shd:>3}  NMSE={err:.4f}  "
                      f"TPR={tpr:.3f}  FDR={fdr:.3f}   t={dt:6.1f}s", flush=True)
                _save_partial(cells, json_path)

                # Mini-batch CoLiDE-NV (covariance-based sigma)
                for bs in BATCH_SIZES:
                    t0 = time.time()
                    W_est, _ = colide_nv_batch_cov(seed=seed).fit(
                        X=X.copy(), lambda1=COLIDE_BATCH_CONFIG['lambda1'],
                        batch_size=bs, **SWEEP_BATCH_KW)
                    dt = time.time() - t0
                    shd, tpr, fdr = count_accuracy(W_true_bin, to_bin(W_est, thr=THRESHOLD))
                    err = nmse(W_est, W_true)
                    cells[(er, d, f'bs={bs}')].append({
                        'seed': seed, 'tpr': float(tpr), 'fdr': float(fdr),
                        'shd': int(shd), 'nmse': err, 'time_s': dt})
                    print(f"  bs={bs:<5}  SHD={shd:>3}  NMSE={err:.4f}  "
                          f"TPR={tpr:.3f}  FDR={fdr:.3f}   t={dt:6.1f}s", flush=True)
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
        print("\n" + "=" * 110)
        print(f"ER{er}  —  SHD (mean ± std over {len(SEEDS)} seeds)")
        print("=" * 110)
        print(f"{'Method':<10}" + "".join(f"{'d='+str(d):>20}" for d in NODES_LIST))
        print("-" * 110)
        for m in methods:
            line = f"{m:<10}"
            for d in NODES_LIST:
                s = stat(er, m, d, 'shd')
                line += f"   SHD={s[0]:5.1f}±{s[1]:4.1f}" if s else f"{'-':>20}"
            print(line)

        print(f"\nER{er}  —  NMSE (mean ± std)")
        print("-" * 110)
        print(f"{'Method':<10}" + "".join(f"{'d='+str(d):>20}" for d in NODES_LIST))
        for m in methods:
            line = f"{m:<10}"
            for d in NODES_LIST:
                s = stat(er, m, d, 'nmse')
                line += f" NMSE={s[0]:.4f}±{s[1]:.4f}" if s else f"{'-':>20}"
            print(line)


if __name__ == "__main__":
    cells, json_path = run_scaling()
    print_summary(cells)
    print(f"\nFinal results saved to {json_path}")
