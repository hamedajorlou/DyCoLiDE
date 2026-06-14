"""
Noise-robustness sweep on pure ER2 SVAR (d=20): DyCoLiDE vs DyDAGMA vs DYNOTEARS.

Sweeps the exogenous noise scale from 1 to 5 (true per-node sigma = noise_scale)
on ER2 for both W (40 instantaneous edges) and A (40 temporal edges), lag=1, EV.
For each noise level and method we run 5 trials (seeds) and record W/A SHD, TPR,
FDR and NMSE.

Rationale: DyCoLiDE estimates the noise scale (CoLiDE sigma) and so should hold
up as noise grows; DyDAGMA (DAGMA log-det, no sigma) and DYNOTEARS (NOTEARS
matrix-exp, no sigma) have a fixed L1 vs a loss that scales with the noise, so
their regularization mis-scales and they should degrade.

Results stream to noise_sweep_results.json after every fit. The final figure
(median line + 25-75 percentile band over the 5 trials) is saved to
noise_sweep.png. Re-plot from the JSON without recomputing via:
    python noise_sweep.py plot
"""

import os
import sys
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, _HERE)

from Utils import generate_svar_data, count_accuracy_svar
from dycolide import DyCoLiDE_EV
from baselines import run_dagma_svar, run_dynotears

# ---- setting -----------------------------------------------------------------
D, ER, EDGES, P, N = 20, 2, 40, 1, 1000           # pure ER2 on both W and A
TEMP_STR = 0.5
NOISE_SCALES = [1, 2, 3, 4, 5]
SEEDS = [42, 43, 44, 45, 46]                        # 5 trials
LAMBDA_W, LAMBDA_A = 0.01, 0.015
W_THR, A_THR = 0.10, 0.08

# continuous-opt schedule shared by DyCoLiDE and DyDAGMA (lr=1e-3 converges)
OPT_KW = dict(T=4, mu_init=1.0, mu_factor=0.1, s=[1.0, 0.9, 0.8, 0.7],
              warm_iter=20000, max_iter=70000, lr=1e-3, checkpoint=5000)
DYNO_KW = dict(max_iter=100, h_tol=1e-8)

JSON_PATH = os.path.join(_HERE, 'noise_sweep_results.json')
FIG_PATH = os.path.join(_HERE, 'noise_sweep.png')

METHOD_ORDER = ['DyCoLiDE', 'DyDAGMA', 'DYNOTEARS']
COLORS = {'DyCoLiDE': '#1f77b4', 'DyDAGMA': '#2ca02c', 'DYNOTEARS': '#d62728'}


# ---- fitters -----------------------------------------------------------------
def fit_dycolide(X, seed):
    W, A, _ = DyCoLiDE_EV(seed=seed).fit(
        X.copy(), p=P, lambda_W=LAMBDA_W, lambda_A=LAMBDA_A, **OPT_KW)
    return W, A


def fit_dydagma(X, seed):
    return run_dagma_svar(X.copy(), p=P, lambda_w=LAMBDA_W, lambda_a=LAMBDA_A,
                          seed=seed, **OPT_KW)


def fit_dynotears(X, seed):
    return run_dynotears(X.copy(), p=P, lambda_w=LAMBDA_W, lambda_a=LAMBDA_A,
                         **DYNO_KW)


FITTERS = {'DyCoLiDE': fit_dycolide, 'DyDAGMA': fit_dydagma,
           'DYNOTEARS': fit_dynotears}


def nmse(est, true):
    d = np.sum(true ** 2)
    return float(np.sum((est - true) ** 2) / d) if d > 1e-12 else float(np.sum(est ** 2))


def score(W_true, W_e, A_true, A_e):
    mW = count_accuracy_svar(W_true, W_e, threshold=W_THR)
    mA = count_accuracy_svar(A_true, A_e, threshold=A_THR)
    return {'W_shd': int(mW['W_shd']), 'W_tpr': float(mW['W_tpr']),
            'W_fdr': float(mW['W_fdr']), 'W_nmse': nmse(W_e, W_true),
            'A_shd': int(mA['W_shd']), 'A_tpr': float(mA['W_tpr']),
            'A_fdr': float(mA['W_fdr']), 'A_nmse': nmse(A_e, A_true)}


# results[method][str(noise)] = list of per-seed metric dicts
def _empty_results():
    return {m: {str(ns): [] for ns in NOISE_SCALES} for m in METHOD_ORDER}


def _save(results):
    payload = {
        'experiment': 'noise_sweep_ER2_dycolide_dydagma_dynotears',
        'setting': {'d': D, 'er': ER, 'edges_W': EDGES, 'edges_A': EDGES,
                    'lag': P, 'n_timesteps': N, 'noise_scales': NOISE_SCALES,
                    'seeds': SEEDS, 'lambda_W': LAMBDA_W, 'lambda_A': LAMBDA_A,
                    'W_threshold': W_THR, 'A_threshold': A_THR,
                    'opt_kw': OPT_KW, 'dyno_kw': DYNO_KW},
        'results': results,
    }
    with open(JSON_PATH, 'w') as f:
        json.dump(payload, f, indent=2)


def run_sweep():
    results = _empty_results()
    for ns in NOISE_SCALES:
        print(f"\n{'#'*70}\n#  noise_scale = {ns}  (true sigma = {ns})\n{'#'*70}",
              flush=True)
        for seed in SEEDS:
            X, B_true, A_list_true, _ = generate_svar_data(
                n_nodes=D, n_timesteps=N, lag_order=P, instantaneous_edges=EDGES,
                temporal_edges=EDGES, temporal_strength=TEMP_STR,
                noise_scale=float(ns), noise_type='ev', seed=seed,
                max_stability_attempts=50)
            W_true = B_true.T
            A_true = np.vstack([A.T for A in A_list_true])
            print(f"\n-- noise={ns} seed={seed} --", flush=True)
            for m in METHOD_ORDER:
                t0 = time.time()
                W_e, A_e = FITTERS[m](X, seed)
                dt = time.time() - t0
                row = score(W_true, W_e, A_true, A_e)
                row['time_s'] = dt
                row['seed'] = seed
                results[m][str(ns)].append(row)
                print(f"   {m:<10} W SHD={row['W_shd']:>3} A SHD={row['A_shd']:>3} "
                      f"A TPR={row['A_tpr']:.2f}  t={dt:6.1f}s", flush=True)
                _save(results)
    return results


# ---- plotting ----------------------------------------------------------------
def plot(results=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if results is None:
        with open(JSON_PATH) as f:
            results = json.load(f)['results']

    panels = [('W_shd', 'W SHD (lower better)'),
              ('A_shd', 'A SHD (lower better)'),
              ('W_tpr', 'W TPR (higher better)'),
              ('A_tpr', 'A TPR (higher better)')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = NOISE_SCALES

    for ax, (key, title) in zip(axes.ravel(), panels):
        for m in METHOD_ORDER:
            med, lo, hi = [], [], []
            for ns in NOISE_SCALES:
                vals = np.array([r[key] for r in results[m][str(ns)]], dtype=float)
                if vals.size == 0:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    continue
                med.append(np.median(vals))
                lo.append(np.percentile(vals, 25))
                hi.append(np.percentile(vals, 75))
            ax.plot(x, med, '-o', color=COLORS[m], label=m, lw=2, ms=5)
            ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.20, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel('noise scale (true $\\sigma$)')
        ax.set_xticks(NOISE_SCALES)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc='best', framealpha=0.9)
    fig.suptitle('Noise robustness on pure ER2 SVAR (d=20, lag=1) — '
                 'median & 25–75% band over 5 trials', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved figure -> {FIG_PATH}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot()
    else:
        res = run_sweep()
        plot(res)
        print(f"\nDone. Results -> {JSON_PATH}\nFigure -> {FIG_PATH}")
