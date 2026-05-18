"""
POC: anomaly detection on top of DyCoLiDE.

Two detectors, run side by side, both scoring against an EMA-smoothed
reference Ŵ_smooth rather than DyCoLiDE's raw per-batch fit. The smoothing
filters DyCoLiDE's SGD jitter (which otherwise creates false positives
every few batches), while still tracking real evolution with a tunable lag.

  1) Residual detector (fast): score each new batch by how well Ŵ_smooth
     explains it. Large residuals = data no longer fits the reference model.
     Because Ŵ_smooth moves slowly, it can't absorb a real anomaly in one
     step → the anomaly shows up immediately in residuals.
  2) Structural detector (slow, interpretable): ||ΔŴ_smooth||_F between
     consecutive smoothed estimates. Smoothing suppresses jitter while
     preserving real-change signal.

Normal DAG evolution is a 1-edge flip every few batches. One batch injects
a much larger simultaneous change — the anomaly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from evolving_dag_experiment import generate_dag, OnlineDyCoLiDE


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# DAG evolution and data generation
# =============================================================================

def perturb_dag(W, n_flips, w_ranges=((-2.0, -0.5), (0.5, 2.0)), rng=None):
    rng = rng or np.random.default_rng()
    d = W.shape[0]
    W_new = W.copy()

    n_remove = n_flips // 2
    n_add = n_flips - n_remove

    existing = np.argwhere(np.abs(W_new) > 0)
    if len(existing) > 0 and n_remove > 0:
        idx = rng.choice(len(existing), min(n_remove, len(existing)), replace=False)
        for k in idx:
            i, j = existing[k]
            W_new[i, j] = 0

    empty = [(i, j) for i in range(1, d) for j in range(i) if W_new[i, j] == 0]
    rng.shuffle(empty)
    for (i, j) in empty[:n_add]:
        lo, hi = w_ranges[rng.integers(0, 2)]
        W_new[i, j] = rng.uniform(lo, hi)

    return W_new


def build_dag_schedule(W0, n_batches, burn_in, normal_flip_every, anomaly_batch, anomaly_flips, seed=0):
    """Phase 1 (batches 0..burn_in-1): DAG fixed at W0 so DyCoLiDE can converge.
    Phase 2 (burn_in..): normal evolution with occasional 1-edge flips.
    One batch in Phase 2 injects a multi-edge anomaly.
    """
    rng = np.random.default_rng(seed)
    schedule = [W0.copy()]
    for b in range(1, n_batches):
        W_prev = schedule[-1]
        if b < burn_in:
            schedule.append(W_prev.copy())
        elif b == anomaly_batch:
            schedule.append(perturb_dag(W_prev, anomaly_flips, rng=rng))
        elif (b - burn_in) % normal_flip_every == 0:
            schedule.append(perturb_dag(W_prev, 1, rng=rng))
        else:
            schedule.append(W_prev.copy())
    return schedule


def generate_batch(W, n, sigma=1.0, rng=None):
    rng = rng or np.random.default_rng()
    d = W.shape[0]
    G = nx.DiGraph(W)
    order = list(nx.topological_sort(G))
    X = np.zeros((n, d))
    for t in range(n):
        for j in order:
            parents = list(G.predecessors(j))
            eta = X[t, parents].dot(W[parents, j]) if parents else 0.0
            X[t, j] = eta + rng.normal(scale=sigma)
    return X


# =============================================================================
# Detectors
# =============================================================================

def residual_score(X_batch, W_est, threshold=0.3):
    """Mean squared residual of X_batch under the linear SEM implied by Ŵ.

    Under the model X_j = X_pa(j) W + ε, a well-fit Ŵ yields residuals close
    to the noise floor. A structural change breaks the fit immediately —
    residuals spike before the tracker has time to adapt.
    """
    W_thr = W_est * (np.abs(W_est) > threshold)
    residuals = X_batch - X_batch @ W_thr
    return float((residuals ** 2).mean())


def structural_score(W_new, W_old):
    return float(np.linalg.norm(W_new - W_old))


def zscore(series, calib_slice):
    calib = np.asarray(series)[calib_slice]
    mu, sd = calib.mean(), calib.std() + 1e-8
    return (np.asarray(series) - mu) / sd


# =============================================================================
# Main experiment
# =============================================================================

def run(
    d=10,
    expected_edges=20,
    n_batches=45,
    batch_size=50,
    burn_in=15,            # phase 1: DAG fixed so DyCoLiDE converges
    normal_flip_every=3,   # phase 2: flip 1 edge every N batches
    anomaly_batch=35,
    anomaly_flips=8,
    calib_batches=(18, 34),  # baseline from stable normal-evolution region
    window_size=400,
    lambda1=0.05,
    n_iter_warm=5000,
    n_iter_final=25000,
    ema_alpha=0.25,        # Ŵ_smooth tracking speed; smaller = more stable reference
    seed=42,
    save_name='anomaly_detection.png',
):
    rng = np.random.default_rng(seed)

    W0 = generate_dag(d, expected_edges, seed=seed)
    dag_schedule = build_dag_schedule(
        W0, n_batches, burn_in, normal_flip_every, anomaly_batch, anomaly_flips, seed=seed + 1
    )

    tracker = OnlineDyCoLiDE(d, window_size=window_size, seed=seed)
    W_smooth_hist = []
    true_delta = [0.0]
    res_raw = []        # residual score per batch (fast detector)
    struct_raw = [0.0]  # ||ΔŴ_smooth|| per batch (slow detector)

    for b in range(n_batches):
        W_true = dag_schedule[b]
        X_batch = generate_batch(W_true, batch_size, sigma=1.0, rng=rng)

        # Fast detector: score the fresh batch against the CURRENT smoothed
        # reference BEFORE updating. Ŵ_smooth moves slowly, so a real anomaly
        # blows up the residual immediately instead of being absorbed.
        if len(W_smooth_hist) > 0:
            res_raw.append(residual_score(X_batch, W_smooth_hist[-1]))
        else:
            res_raw.append(np.nan)  # no model yet on batch 0

        W_smooth_prev = W_smooth_hist[-1] if W_smooth_hist else None
        W_raw = tracker.fit_incremental(
            X_batch,
            lambda1=lambda1,
            sgd_batch_size=min(100, window_size),
            n_iter_warm=n_iter_warm,
            n_iter_final=n_iter_final,
        )

        # EMA smoothing on the estimate
        if W_smooth_prev is None:
            W_smooth = W_raw.copy()
        else:
            W_smooth = (1 - ema_alpha) * W_smooth_prev + ema_alpha * W_raw
        W_smooth_hist.append(W_smooth)

        if W_smooth_prev is not None:
            struct_raw.append(structural_score(W_smooth, W_smooth_prev))

        if b > 0:
            true_delta.append(np.linalg.norm(dag_schedule[b] - dag_schedule[b-1]))

        print(f"  batch {b+1}/{n_batches} done  "
              f"(residual={res_raw[-1]:.3f}, struct={struct_raw[-1]:.3f})")

    res_raw = np.array(res_raw)
    struct_raw = np.array(struct_raw)
    true_delta = np.array(true_delta)

    lo, hi = calib_batches
    # Residual score may be NaN at batch 0 — skip it in calibration.
    res_calib_mask = np.arange(len(res_raw))
    res_calib_mask = res_calib_mask[(res_calib_mask >= lo) & (res_calib_mask < hi)]
    res_calib_mask = res_calib_mask[~np.isnan(res_raw[res_calib_mask])]

    res_mu = res_raw[res_calib_mask].mean()
    res_sd = res_raw[res_calib_mask].std() + 1e-8
    z_res = (res_raw - res_mu) / res_sd

    struct_mu = struct_raw[lo:hi].mean()
    struct_sd = struct_raw[lo:hi].std() + 1e-8
    z_struct = (struct_raw - struct_mu) / struct_sd

    threshold = 3.0
    res_flags = np.where(z_res > threshold)[0]
    struct_flags = np.where(z_struct > threshold)[0]

    print("\n=== results ===")
    print(f"anomaly injected at batch {anomaly_batch} (true |ΔW|={true_delta[anomaly_batch]:.3f})")
    print(f"residual baseline  mean={res_mu:.4f} std={res_sd:.4f}")
    print(f"structural baseline mean={struct_mu:.4f} std={struct_sd:.4f}")
    print(f"flagged by residual   (z>{threshold}): {res_flags.tolist()}")
    print(f"flagged by structural (z>{threshold}): {struct_flags.tolist()}")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Row 1: raw signals
    ax = axes[0]
    ax.plot(true_delta, 'k--', alpha=0.5, label='true ||ΔW||_F')
    ax.plot(struct_raw, 'b-o', markersize=4, label='estimated ||ΔŴ||_F (slow)')
    ax2 = ax.twinx()
    ax2.plot(res_raw, 'g-s', markersize=4, label='residual MSE (fast)')
    ax2.set_ylabel('residual MSE', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax.axvspan(0, burn_in, alpha=0.1, color='grey', label='burn-in (fixed DAG)')
    ax.axvspan(lo, hi, alpha=0.1, color='green', label='calibration window')
    ax.axvline(anomaly_batch, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('change magnitude')
    ax.set_title('Raw signals')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Row 2: fast detector (residual z-score)
    ax = axes[1]
    ax.plot(z_res, 'g-o', markersize=5, label='residual z-score')
    ax.axhline(threshold, color='orange', linestyle=':', label=f'threshold (z={threshold})')
    ax.axvline(anomaly_batch, color='red', linestyle='--', label=f'anomaly injected (batch {anomaly_batch})')
    for f in res_flags:
        ax.plot(f, z_res[f], 'r*', markersize=16)
    ax.set_ylabel('z-score')
    ax.set_title('FAST detector: residual-based (detects event immediately)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # Row 3: slow detector (structural z-score)
    ax = axes[2]
    ax.plot(z_struct, 'b-o', markersize=5, label='structural z-score')
    ax.axhline(threshold, color='orange', linestyle=':', label=f'threshold (z={threshold})')
    ax.axvline(anomaly_batch, color='red', linestyle='--')
    for f in struct_flags:
        ax.plot(f, z_struct[f], 'r*', markersize=16)
    ax.set_xlabel('batch index')
    ax.set_ylabel('z-score')
    ax.set_title('SLOW detector: structural ||ΔŴ|| (lags by window size, localizes edges)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    save_path = RESULTS_DIR / save_name
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"plot saved to {save_path}")
    plt.show()

    return {
        'residual_raw': res_raw,
        'residual_z': z_res,
        'residual_flags': res_flags,
        'structural_raw': struct_raw,
        'structural_z': z_struct,
        'structural_flags': struct_flags,
        'true_delta': true_delta,
        'anomaly_batch': anomaly_batch,
    }


if __name__ == '__main__':
    run()
