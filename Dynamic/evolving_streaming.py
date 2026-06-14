"""
Streaming recovery under a mid-stream DAG change, for both SEM and SVAR data.

Setup:
- d = 20 nodes, ER2 density (40 edges).
- Total length T = 2000 samples (or timesteps for SVAR), change at t = T/2.
- The first half is generated from (W_1[, A_1] for SVAR); the second half
  from (W_2[, A_2]), where ~50% of the edges flip.
- The learner ingests data in mini-batches of `BATCH_SIZE` with a sliding
  window of the last `WINDOW` samples. At each batch arrival it refits
  CoLiDE / DyCoLiDE on the window and records SHD and TPR against the
  active (pre- or post-change) ground truth.
- The whole experiment is repeated for `N_SEEDS` independent data
  realizations (same DAG, different noise / lag-driver seed) and the plot
  shows the per-batch median with a 25--75% inter-quantile band.

Outputs (written next to this script):
- streaming_evolving_sem_svar.png   — 2x2 plot: rows SEM/SVAR, cols SHD/TPR(SVAR:A SHD)
- streaming_evolving_sem_svar.json  — per-seed metrics and config

Run from anywhere (the script self-adjusts sys.path):
    python Dynamic/evolving_streaming.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# All data-generation / structure / stability helpers live in repo-root
# Utils.py. The SEM / SVAR streaming models live in their own packages.
from Utils import (
    create_dag, generate_temporal_structure,
    check_stability, simulate_svar,
)
from SEM.dycolide import colide_ev_batch
from SVAR.dycolide import DyCoLiDE_EV_batch


# =============================================================================
# Experiment configuration
# =============================================================================
SEED         = 42                   # DAG / structure seed (fixed across realizations)
N_SEEDS      = 3                    # data realizations for the shaded band
D            = 20
EDGES        = 2 * D                # ER2 -> 40 edges
T_TOTAL      = 2000
BATCH_SIZE   = 100
WINDOW       = 500
CHANGEPOINT  = T_TOTAL // 2
N_BATCHES    = T_TOTAL // BATCH_SIZE
THRESHOLD    = 0.10                # binarization threshold for W
A_THRESHOLD  = 0.05                # separate threshold for A (smaller weights
                                   # after the stability-decay loop, so a
                                   # looser cut catches true entries that
                                   # the W threshold would discard)
SKIP_SEM     = False               # set True to reuse last JSON SEM block

# Full iteration budget — same as the original cold-start runs. When
# WARM_START is True, the optimizer early-exits via the tol=1e-6
# convergence check once the warm-started iterate stops moving, so the
# *effective* per-batch cost is much smaller than the budget on
# steady-state batches; only the first batch and the post-change
# transition consume the full budget.
LIGHT_BATCH_KW = dict(
    T=4, mu_init=1.0, mu_factor=0.1, s=[1.0, 0.9, 0.8, 0.7],
    n_batches_warm=4000, n_batches_final=10000,
    lr=0.0003, checkpoint=500,
)
HEAVY_BATCH_KW = dict(
    T=5, mu_init=1.0, mu_factor=0.1, s=[1.0, 0.9, 0.8, 0.7, 0.6],
    n_batches_warm=10000, n_batches_final=30000,
    lr=0.0003, checkpoint=1000,
)
# Whether to warm-start each batch's fit from the previous batch's
# estimate. Set to False to recover the original cold-start behavior.
WARM_START = True

LAMBDA_W_SVAR = 0.10    # bumped from 0.05 to clear warm-start FP edges
LAMBDA_A_SVAR = 0.05    # pushed further from 0.03 to suppress more A-side FPs
LAMBDA1_SEM   = 0.05
TEMP_STRENGTH = 0.30


# =============================================================================
# Metrics
# =============================================================================
def shd(W_true, W_est, thr):
    Wtb = (np.abs(W_true) > 0).astype(int)
    Web = (np.abs(W_est) > thr).astype(int)
    tp = int(np.sum(Wtb * Web))
    fp = int(np.sum(Web * (1 - Wtb)))
    fn = int(np.sum((1 - Web) * Wtb))
    tpr = tp / max(tp + fn, 1)
    fdr = fp / max(tp + fp, 1)
    return int(fp + fn), float(tpr), float(fdr)


# =============================================================================
# Structure builders (shared across realizations)
# =============================================================================
def _perturb_dag(W, frac, rng):
    """Flip ~frac of edges (remove some, add some) preserving the
    lower-triangular DAG ordering."""
    d = W.shape[0]
    W_new = W.copy()
    existing = np.argwhere(np.abs(W_new) > 0)
    n_change = max(1, int(len(existing) * frac))
    if len(existing) >= n_change:
        idx = rng.choice(len(existing), n_change, replace=False)
        for k in idx:
            i, j = existing[k]
            W_new[i, j] = 0
    empty = [(i, j) for i in range(1, d) for j in range(i) if W_new[i, j] == 0]
    rng.shuffle(empty)
    w_ranges = [(-2.0, -0.5), (0.5, 2.0)]
    for (i, j) in empty[:n_change]:
        lo, hi = w_ranges[rng.integers(0, 2)]
        W_new[i, j] = rng.uniform(lo, hi)
    return W_new


def _build_phase_svar(d, edges, seed):
    """Random ER-style DAG (via the canonical ``create_dag``) + lag
    matrix that satisfies the repo's SVAR stability check. Decay A
    iteratively until stable.

    Both Python's ``random`` and numpy's ``np.random`` are seeded
    because `create_dag` calls `nx.erdos_renyi_graph` without a seed,
    so the graph topology depends on Python's ``random`` state.

    `permute=False` (lower-triangular B) is required: `_perturb_dag`
    only adds new edges in lower-triangular positions, so applying it
    to a permuted (non-lower-tri) B introduces cycles in B_2.
    """
    random.seed(seed)
    np.random.seed(seed)
    B, _ = create_dag(d, 'er', edges, permute=False, edge_type='weighted',
                      w_range=((-2.0, -0.5), (0.5, 2.0)))
    A_list = generate_temporal_structure(d, lag_order=1, n_edges=edges,
                                         strength=TEMP_STRENGTH, seed=seed + 100)
    while not check_stability(A_list, d, B=B):
        A_list = [a * 0.8 for a in A_list]
    return B, A_list[0]


def _build_sem_phases():
    """Two SEM DAGs sharing ~50% of edges. Deterministic in SEED.

    `create_dag` calls `nx.erdos_renyi_graph` without a seed kwarg, so it
    samples from Python's built-in ``random`` module rather than numpy's
    — both need seeding for the DAG to be reproducible.
    """
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    W1, _ = create_dag(D, 'er', EDGES, permute=False, edge_type='weighted',
                       w_range=((-2.0, -0.5), (0.5, 2.0)))
    W2 = _perturb_dag(W1, 0.5, rng)
    return W1, W2


def _build_svar_phases():
    """Two SVAR (B, A) phases sharing ~50% of edges. Deterministic in SEED."""
    B1, A1 = _build_phase_svar(D, EDGES, SEED)
    rng = np.random.default_rng(SEED + 1)
    B2 = _perturb_dag(B1, 0.5, rng)
    A2 = _perturb_dag(A1, 0.5, rng)
    while not check_stability([A2], D, B=B2):
        A2 = A2 * 0.8
    return B1, A1, B2, A2


# =============================================================================
# SEM evolving experiment (one realization)
# =============================================================================
def _gen_sem(d, n, W, rng, sigma=1.0):
    """Topologically-ordered linear SEM sampler: X = X W + noise."""
    G = nx.DiGraph(W)
    order = list(nx.topological_sort(G))
    X = np.zeros((n, d))
    for t in range(n):
        for j in order:
            parents = list(G.predecessors(j))
            eta = X[t, parents].dot(W[parents, j]) if parents else 0.0
            X[t, j] = eta + rng.normal(scale=sigma)
    return X


def run_evolving_sem_one(realization_seed, W1, W2):
    rng = np.random.default_rng(realization_seed)
    X = np.empty((T_TOTAL, D))
    X[:CHANGEPOINT] = _gen_sem(D, CHANGEPOINT, W1, rng)
    X[CHANGEPOINT:] = _gen_sem(D, T_TOTAL - CHANGEPOINT, W2, rng)

    times, shd1, shd2, tpr1, tpr2 = [], [], [], [], []
    prev_W = None
    window = None
    for b in range(N_BATCHES):
        s0, s1 = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        new_batch = X[s0:s1]
        window = new_batch if window is None else np.vstack([window, new_batch])
        if len(window) > WINDOW:
            window = window[-WINDOW:]
        model = colide_ev_batch(seed=realization_seed)
        W_init_kw = {'W_init': prev_W} if (WARM_START and prev_W is not None) else {}
        W_est, _ = model.fit(X=window.copy(), lambda1=LAMBDA1_SEM,
                             batch_size=BATCH_SIZE,
                             **LIGHT_BATCH_KW, **W_init_kw)
        prev_W = W_est.copy()
        sh1, tp1, _ = shd(W1, W_est, THRESHOLD)
        sh2, tp2, _ = shd(W2, W_est, THRESHOLD)
        times.append(s1)
        shd1.append(sh1); shd2.append(sh2)
        tpr1.append(tp1); tpr2.append(tp2)
        print(f'    SEM seed={realization_seed} batch {b+1:2}/{N_BATCHES}  '
              f'SHD1={sh1:2} SHD2={sh2:2}', flush=True)
    return dict(times=times, shd1=shd1, shd2=shd2, tpr1=tpr1, tpr2=tpr2)


# =============================================================================
# SVAR evolving experiment (one realization)
# =============================================================================
def run_evolving_svar_one(realization_seed, B1, A1, B2, A2):
    p = 1
    X = np.empty((T_TOTAL, D))
    X[:CHANGEPOINT], _ = simulate_svar(B1, [A1], CHANGEPOINT,
                                       noise_scale=1.0, noise_type='ev',
                                       burnin=500, seed=realization_seed)
    X[CHANGEPOINT:], _ = simulate_svar(B2, [A2], T_TOTAL - CHANGEPOINT,
                                       noise_scale=1.0, noise_type='ev',
                                       burnin=500, seed=realization_seed + 5000)

    W1_true, W2_true = B1.T, B2.T
    A1_true, A2_true = A1.T, A2.T

    times = []
    shdW1, shdW2, shdA1, shdA2 = [], [], [], []
    tprW1, tprW2, tprA1, tprA2 = [], [], [], []
    MIN_WINDOW_SVAR = max(4 * BATCH_SIZE, BATCH_SIZE + p + 1)

    prev_W, prev_A = None, None
    window = None
    for b in range(N_BATCHES):
        s0, s1 = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        new_batch = X[s0:s1]
        window = new_batch if window is None else np.vstack([window, new_batch])
        if len(window) > WINDOW:
            window = window[-WINDOW:]
        if len(window) < MIN_WINDOW_SVAR:
            continue
        try:
            model = DyCoLiDE_EV_batch(seed=realization_seed)
            init_kw = {}
            if WARM_START and prev_W is not None:
                init_kw = {'W_init': prev_W.copy(), 'A_init': prev_A.copy()}
            W_est, A_est, _ = model.fit(window.copy(), p=p,
                                        lambda_W=LAMBDA_W_SVAR,
                                        lambda_A=LAMBDA_A_SVAR,
                                        batch_size=BATCH_SIZE,
                                        **HEAVY_BATCH_KW, **init_kw)
            prev_W, prev_A = W_est, A_est
        except (ValueError, np.linalg.LinAlgError) as e:
            if prev_W is None:
                print(f'    SVAR seed={realization_seed} batch {b+1:2}: '
                      f'fit failed ({e}), skipping', flush=True)
                continue
            W_est, A_est = prev_W, prev_A
        sw1, tw1, _ = shd(W1_true, W_est, THRESHOLD)
        sw2, tw2, _ = shd(W2_true, W_est, THRESHOLD)
        sa1, ta1, _ = shd(A1_true, A_est, A_THRESHOLD)
        sa2, ta2, _ = shd(A2_true, A_est, A_THRESHOLD)
        times.append(s1)
        shdW1.append(sw1); shdW2.append(sw2)
        shdA1.append(sa1); shdA2.append(sa2)
        tprW1.append(tw1); tprW2.append(tw2)
        tprA1.append(ta1); tprA2.append(ta2)
        print(f'    SVAR seed={realization_seed} batch {b+1:2}/{N_BATCHES}  '
              f'W SHD1={sw1:3} SHD2={sw2:3}  A SHD1={sa1:3} SHD2={sa2:3}',
              flush=True)
    return dict(times=times,
                shdW1=shdW1, shdW2=shdW2, shdA1=shdA1, shdA2=shdA2,
                tprW1=tprW1, tprW2=tprW2, tprA1=tprA1, tprA2=tprA2)


# =============================================================================
# Multi-realization aggregation
# =============================================================================
def _stack(runs, key):
    return np.array([r[key] for r in runs])


def aggregate_sem(runs):
    """Stack per-seed runs and produce median + 25/75 quantile arrays."""
    keys = ('shd1', 'shd2', 'tpr1', 'tpr2')
    out = {'times': runs[0]['times'], 'changepoint': CHANGEPOINT,
           'per_seed': [{k: r[k] for k in keys} for r in runs]}
    for k in keys:
        arr = _stack(runs, k)              # shape (n_seeds, n_batches)
        out[f'{k}_med'] = np.median(arr, axis=0).tolist()
        out[f'{k}_lo']  = np.quantile(arr, 0.25, axis=0).tolist()
        out[f'{k}_hi']  = np.quantile(arr, 0.75, axis=0).tolist()
    return out


def aggregate_svar(runs):
    keys = ('shdW1', 'shdW2', 'shdA1', 'shdA2',
            'tprW1', 'tprW2', 'tprA1', 'tprA2')
    times = runs[0]['times']  # all realizations skip the same first batches
    out = {'times': times, 'changepoint': CHANGEPOINT,
           'per_seed': [{k: r[k] for k in keys} for r in runs]}
    for k in keys:
        arr = _stack(runs, k)
        out[f'{k}_med'] = np.median(arr, axis=0).tolist()
        out[f'{k}_lo']  = np.quantile(arr, 0.25, axis=0).tolist()
        out[f'{k}_hi']  = np.quantile(arr, 0.75, axis=0).tolist()
    return out


# =============================================================================
# Plotting
# =============================================================================
def plot_active_band(ax, t, y_pre, y_post, lo_pre, hi_pre, lo_post, hi_post,
                     cp, ylabel, title):
    """Median curve vs the active ground truth with a 25/75% IQR band."""
    t = np.asarray(t)
    y_active  = np.where(t <= cp, np.asarray(y_pre),  np.asarray(y_post))
    lo_active = np.where(t <= cp, np.asarray(lo_pre), np.asarray(lo_post))
    hi_active = np.where(t <= cp, np.asarray(hi_pre), np.asarray(hi_post))
    cp_idx = int(np.searchsorted(t, cp))

    ax.fill_between(t[:cp_idx + 1], lo_active[:cp_idx + 1], hi_active[:cp_idx + 1],
                    color='#1f77b4', alpha=0.20)
    ax.plot(t[:cp_idx + 1], y_active[:cp_idx + 1],
            color='#1f77b4', linewidth=2.0, marker='o', markersize=4)
    ax.fill_between(t[cp_idx:], lo_active[cp_idx:], hi_active[cp_idx:],
                    color='#d62728', alpha=0.20)
    ax.plot(t[cp_idx:], y_active[cp_idx:],
            color='#d62728', linewidth=2.0, marker='s', markersize=4)
    ax.axvline(cp, color='#2ca02c', linestyle='--', linewidth=1.5)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def make_plot(sem_res, svar_res, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    plot_active_band(axes[0, 0], sem_res['times'],
                     sem_res['shd1_med'], sem_res['shd2_med'],
                     sem_res['shd1_lo'],  sem_res['shd1_hi'],
                     sem_res['shd2_lo'],  sem_res['shd2_hi'],
                     sem_res['changepoint'], 'SHD', 'SEM: W')
    plot_active_band(axes[0, 1], sem_res['times'],
                     sem_res['tpr1_med'], sem_res['tpr2_med'],
                     sem_res['tpr1_lo'],  sem_res['tpr1_hi'],
                     sem_res['tpr2_lo'],  sem_res['tpr2_hi'],
                     sem_res['changepoint'], 'TPR', 'SEM: W')
    axes[0, 1].set_ylim([0, 1.05])

    plot_active_band(axes[1, 0], svar_res['times'],
                     svar_res['shdW1_med'], svar_res['shdW2_med'],
                     svar_res['shdW1_lo'],  svar_res['shdW1_hi'],
                     svar_res['shdW2_lo'],  svar_res['shdW2_hi'],
                     svar_res['changepoint'], 'SHD', 'SVAR: W')
    plot_active_band(axes[1, 1], svar_res['times'],
                     svar_res['shdA1_med'], svar_res['shdA2_med'],
                     svar_res['shdA1_lo'],  svar_res['shdA1_hi'],
                     svar_res['shdA2_lo'],  svar_res['shdA2_hi'],
                     svar_res['changepoint'], 'SHD', 'SVAR: A')

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')


# =============================================================================
# Main
# =============================================================================
def main():
    # Build the two phase structures once — shared across all realizations.
    W1_sem, W2_sem = _build_sem_phases()
    B1, A1, B2, A2 = _build_svar_phases()
    print(f'SEM:  W1 edges = {int((np.abs(W1_sem) > 0).sum())}, '
          f'W2 edges = {int((np.abs(W2_sem) > 0).sum())}', flush=True)
    print(f'SVAR: B1 edges = {int((np.abs(B1) > 0).sum())}, '
          f'B2 edges = {int((np.abs(B2) > 0).sum())} ; '
          f'A1 edges = {int((np.abs(A1) > 0).sum())}, '
          f'A2 edges = {int((np.abs(A2) > 0).sum())}', flush=True)
    print(f'Running {N_SEEDS} realizations.\n', flush=True)

    # Realization seeds are the conventional {SEED, SEED+1, SEED+2, ...}
    # set used elsewhere in this repo so the per-seed behavior is
    # comparable to the scaling sweeps.
    json_path = os.path.join(_THIS_DIR, 'streaming_evolving_sem_svar.json')

    sem_res = None
    if SKIP_SEM and os.path.exists(json_path):
        try:
            with open(json_path) as f:
                cached = json.load(f)
            sem_res = cached.get('sem')
            print(f'SKIP_SEM=True: reusing SEM block from {json_path}',
                  flush=True)
        except (json.JSONDecodeError, KeyError):
            sem_res = None

    if sem_res is None:
        t0 = time.time()
        sem_runs = []
        for k in range(N_SEEDS):
            rs = SEED + k
            print(f'--- SEM realization {k+1}/{N_SEEDS} (seed={rs}) ---',
                  flush=True)
            sem_runs.append(run_evolving_sem_one(rs, W1_sem, W2_sem))
        print(f'SEM done in {time.time() - t0:.0f}s', flush=True)
        sem_res = aggregate_sem(sem_runs)

    t1 = time.time()
    svar_runs = []
    for k in range(N_SEEDS):
        rs = SEED + k
        print(f'\n--- SVAR realization {k+1}/{N_SEEDS} (seed={rs}) ---',
              flush=True)
        svar_runs.append(run_evolving_svar_one(rs, B1, A1, B2, A2))
    print(f'SVAR done in {time.time() - t1:.0f}s', flush=True)

    svar_res = aggregate_svar(svar_runs)

    plot_path = os.path.join(_THIS_DIR, 'streaming_evolving_sem_svar.png')
    make_plot(sem_res, svar_res, plot_path)
    print(f'\nPlot saved to {plot_path}', flush=True)

    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'd': D, 'edges': EDGES, 'T_total': T_TOTAL,
                'batch_size': BATCH_SIZE, 'window': WINDOW,
                'changepoint': CHANGEPOINT, 'threshold': THRESHOLD,
                'seed': SEED, 'n_seeds': N_SEEDS,
                'lambda_W_svar': LAMBDA_W_SVAR,
                'lambda_A_svar': LAMBDA_A_SVAR,
                'lambda1_sem': LAMBDA1_SEM,
                'temporal_strength': TEMP_STRENGTH,
            },
            'sem': sem_res, 'svar': svar_res,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }, f, indent=2)
    print(f'Data saved to {json_path}', flush=True)


if __name__ == '__main__':
    main()
