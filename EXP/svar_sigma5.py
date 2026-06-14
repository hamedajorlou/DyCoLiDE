"""
SVAR experiment at EV noise scale sigma=5, standard Gaussian generator
(Utils.generate_svar_data, ER2, mixed-sign [0.5,2] weights).

Three subplots side by side:
  (a) SHD  vs d   (sigma=5)        -- W solid, A dashed
  (b) NMSE vs d   (sigma=5)        -- W solid, A dashed
  (c) SHD  vs noise sigma (1..5)   (d=20)  -- W solid, A dashed

Methods: DyCoLiDE(EV), DyDAGMA, DYNOTEARS (best threshold for SHD), VARLiNGAM
(threshold 0.0). NMSE is on the raw weighted estimates.

Re-plot from JSON: python svar_sigma5.py plot
"""

import os
import sys
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'SVAR'))
sys.path.insert(0, os.path.dirname(__file__))

from Utils import generate_svar_data, count_accuracy_svar
from baselines import run_dagma_svar, run_dynotears, run_varlingam, DyCoLiDE_EV

D_LIST = [10, 20, 30, 40, 50]
SIG_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEEDS = [42, 43, 44, 45, 46]
N = 1000
DENSITY = 2
SIGMA_MAIN = 5.0                         # noise scale for the d-sweep panels (a),(b)
SIG_D = 40                               # node count for the sigma-sweep panel (c)
THR_GRID = [0.05, 0.1, 0.2, 0.3]
METHODS = ['DyCoLiDE', 'DyDAGMA', 'DYNOTEARS', 'VARLiNGAM']
COLORS = {'DyCoLiDE': '#1f77b4', 'DyDAGMA': '#2ca02c',
          'DYNOTEARS': '#d62728', 'VARLiNGAM': '#ff7f0e'}
OPT = dict(T=4, mu_init=1.0, mu_factor=0.1, s=[1.0, 0.9, 0.8, 0.7],
           warm_iter=20000, max_iter=70000, lr=1e-3, checkpoint=5000)
JSON_PATH = os.path.join(os.path.dirname(__file__), 'svar_sigma5.json')
FIG_PATH = os.path.join(os.path.dirname(__file__), 'svar_sigma5.png')


def gen_gauss(d, n, seed, noise_scale, density=DENSITY):
    X, B, A_list, _ = generate_svar_data(
        n_nodes=d, n_timesteps=n, lag_order=1,
        instantaneous_edges=density * d, temporal_edges=density * d,
        temporal_strength=0.5, noise_type='ev', noise_scale=noise_scale, seed=seed)
    return X, B.T, np.vstack([A.T for A in A_list])


def _shd(Wt, We, grid):
    return min(int(count_accuracy_svar(Wt, We, threshold=t)['W_shd']) for t in grid)


def _nmse(Et, Ee):
    return float(np.linalg.norm(Ee - Et) ** 2 / (np.linalg.norm(Et) ** 2 + 1e-12))


def fit_methods(X, Wt, At, seed):
    """Returns {method: (W_shd, A_shd, W_nmse, A_nmse, time)}."""
    out = {}

    def rec(name, W, A, t0, shd_grid):
        out[name] = (_shd(Wt, W, shd_grid), _shd(At, A, shd_grid),
                     _nmse(Wt, W), _nmse(At, A), time.time() - t0)

    t0 = time.time(); W, A, _ = DyCoLiDE_EV(seed=seed).fit(X.copy(), p=1, lambda_W=0.01, lambda_A=0.01, **OPT)
    rec('DyCoLiDE', W, A, t0, THR_GRID)
    t0 = time.time(); W, A = run_dagma_svar(X.copy(), p=1, lambda_w=0.01, lambda_a=0.01, seed=seed, **OPT)
    rec('DyDAGMA', W, A, t0, THR_GRID)
    t0 = time.time(); W, A = run_dynotears(X.copy(), p=1, lambda_w=0.01, lambda_a=0.01, max_iter=100)
    rec('DYNOTEARS', W, A, t0, THR_GRID)
    t0 = time.time()
    try:
        W, A = run_varlingam(X.copy(), p=1, w_threshold=0.0)
        rec('VARLiNGAM', W, A, t0, [0.0])
    except Exception as e:
        print(f"    VARLiNGAM failed: {e}", flush=True)
        out['VARLiNGAM'] = (np.nan, np.nan, np.nan, np.nan, time.time() - t0)
    return out


def run():
    res = {'d': {m: {str(d): [] for d in D_LIST} for m in METHODS},
           'sig': {m: {str(s): [] for s in SIG_LIST} for m in METHODS}}

    def save():
        json.dump({'D_LIST': D_LIST, 'SIG_LIST': SIG_LIST, 'SEEDS': SEEDS,
                   'SIGMA_MAIN': SIGMA_MAIN, 'results': res}, open(JSON_PATH, 'w'), indent=2)

    print(f"### SHD/NMSE vs d  (Gaussian ER{DENSITY}, sigma={SIGMA_MAIN}) ###", flush=True)
    for d in D_LIST:
        for seed in SEEDS:
            X, Wt, At = gen_gauss(d, N, seed, SIGMA_MAIN)
            o = fit_methods(X, Wt, At, seed)
            for m in METHODS:
                res['d'][m][str(d)].append(o[m])
            save()
        print(f"  d={d} done", flush=True)

    print(f"### SHD vs noise sigma (1..5), d={SIG_D} ###", flush=True)
    for s in SIG_LIST:
        for seed in SEEDS:
            X, Wt, At = gen_gauss(SIG_D, N, seed, float(s))
            o = fit_methods(X, Wt, At, seed)
            for m in METHODS:
                res['sig'][m][str(s)].append(o[m])
            save()
        print(f"  sigma={s} done", flush=True)
    return res


def _stats(rows, i):
    v = np.array([r[i] for r in rows], float); v = v[np.isfinite(v)]
    if not v.size:
        return np.nan, np.nan, np.nan
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def table(res):
    L = ["=" * 78,
         f"HEADLINE (Gaussian ER{DENSITY}, EV sigma={SIGMA_MAIN}, d={SIG_D}, "
         f"{len(SEEDS)} seeds, median)", "=" * 78,
         f"{'Method':<12}{'W SHD':>9}{'A SHD':>9}{'W NMSE':>10}{'A NMSE':>10}{'time(s)':>10}"]
    for m in METHODS:
        r = res['d'][m][str(SIG_D)]
        L.append(f"{m:<12}{_stats(r,0)[0]:>9.1f}{_stats(r,1)[0]:>9.1f}"
                 f"{_stats(r,2)[0]:>10.3f}{_stats(r,3)[0]:>10.3f}{_stats(r,4)[0]:>10.1f}")
    t = "\n".join(L); print("\n" + t)
    open(os.path.join(os.path.dirname(__file__), 'svar_sigma5_table.txt'), 'w').write(t + "\n")


def plot(res=None):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if res is None:
        res = json.load(open(JSON_PATH))['results']
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    def panel(a, sweep, xs, wi, ai, xlabel, ylabel, title, yscale='linear'):
        for m in METHODS:
            for idx, ls, mk in [(wi, '-', 'o'), (ai, '--', '^')]:
                med = [_stats(res[sweep][m][str(x)], idx)[0] for x in xs]
                lo = [_stats(res[sweep][m][str(x)], idx)[1] for x in xs]
                hi = [_stats(res[sweep][m][str(x)], idx)[2] for x in xs]
                a.plot(xs, med, ls, color=COLORS[m], lw=2, marker=mk, ms=4)
                if ls == '-':
                    a.fill_between(xs, lo, hi, color=COLORS[m], alpha=0.12, lw=0)
        if yscale == 'log':
            a.set_yscale('log')
        elif yscale == 'symlog':
            a.set_yscale('symlog', linthresh=1)   # linear near 0 (handles SHD=0)
        a.set_xlabel(xlabel); a.set_ylabel(ylabel); a.set_title(title)
        a.set_xticks(xs); a.grid(alpha=0.3, which='both')

    panel(ax[0], 'd', D_LIST, 0, 1, 'number of nodes $d$', 'SHD (lower better)',
          f'(a) SHD vs scale ($\\sigma{{=}}{int(SIGMA_MAIN)}$)', yscale='symlog')
    panel(ax[1], 'd', D_LIST, 2, 3, 'number of nodes $d$', 'NMSE (lower better)',
          f'(b) NMSE vs scale ($\\sigma{{=}}{int(SIGMA_MAIN)}$)', yscale='log')
    panel(ax[2], 'sig', SIG_LIST, 0, 1, r'noise scale $\sigma$', 'SHD (lower better)',
          f'(c) SHD vs noise ($d{{=}}{SIG_D}$)', yscale='symlog')

    mh = [Line2D([0], [0], color=COLORS[m], lw=2, label=m) for m in METHODS]
    sh = [Line2D([0], [0], color='0.3', lw=2, ls='-', marker='o', ms=4, label='W (intra)'),
          Line2D([0], [0], color='0.3', lw=2, ls='--', marker='^', ms=4, label='A (inter)')]
    ax[0].legend(handles=sh, fontsize=9, loc='upper left')
    ax[2].legend(handles=mh, fontsize=8, loc='upper left', ncol=2)
    fig.suptitle(f'SVAR on Gaussian ER{DENSITY} data, EV noise — '
                 'DyCoLiDE vs DyDAGMA, DYNOTEARS, VARLiNGAM (median, IQR band on W)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved figure -> {FIG_PATH}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        plot(); table(json.load(open(JSON_PATH))['results'])
    else:
        r = run(); table(r); plot(r)
