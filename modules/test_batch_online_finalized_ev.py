"""
Test script comparing Batch vs Online CoLiDE-EV on SEM data.
"""

import numpy as np
import time
from Colide import colide_ev_batch
from Utils import  count_accuracy
from Utils import simulate_sem

def to_bin(W, thr=0.3):
    """Threshold matrix to binary."""
    W_bin = np.zeros_like(W)
    W_bin[np.abs(W) > thr] = 1  # Use > instead of >= to handle thr=0 correctly
    return W_bin


def run_experiment(n_nodes=20, n_samples=1000, n_edges=40, seed=42, threshold=0.2):
    """Run comparison between batch and online modes."""

    print("=" * 70)
    print(f"Experiment: {n_nodes} nodes, {n_samples} samples, {n_edges} edges")
    print("=" * 70)

    # Generate data (simulate_sem handles seeding internally)
    X, W_true, var_ev = simulate_sem(
        n_nodes=n_nodes,
        n_samples=n_samples,
        edges=n_edges,
        graph_type='er',
        edge_type='weighted',
        var_type='ev',
        noise='normal',
        var=1.0,
        w_range=((-2.0, -0.5), (0.5, 2.0)),
        seed=seed
        )
    print(f"Data shape: {X.shape}")
    print(f"True edges: {np.sum(W_true != 0)}")

    results = {}

    # ========================================
    # 1. Batch mode (batch_size=100)
    # ========================================
    print(f"\n1. Running Batch CoLiDE-EV (batch_size=100)")
    print("-" * 50)

    model_batch = colide_ev_batch(seed=seed)
    t_start = time.time()
    W_batch, sigma_batch = model_batch.fit(
        X,
        lambda1=0.05,
        batch_size=100,
        n_batches_warm=5000,  # More iterations for convergence
        n_batches_final=10000,
        T=5,
        lr=0.0003
    )
    t_batch = time.time() - t_start

    W_batch_bin = to_bin(W_batch, thr=threshold)
    W_true_bin = to_bin(W_true, thr=0.0)
    shd_batch, tpr_batch, fdr_batch = count_accuracy(W_true_bin, W_batch_bin)

    results['batch'] = {
        'time': t_batch,
        'shd': shd_batch,
        'tpr': tpr_batch,
        'fdr': fdr_batch,
        'edges': np.sum(W_batch_bin)
    }

    print(f"   Time: {t_batch:.2f}s")
    print(f"   SHD: {shd_batch}, TPR: {tpr_batch:.4f}, FDR: {fdr_batch:.4f}")
    print(f"   Edges found: {results['batch']['edges']}")

    # ========================================
    # 2. Online mode (batch_size=1)
    # ========================================
    print(f"\n2. Running Online CoLiDE-EV (batch_size=1)")
    print("-" * 50)

    model_online = colide_ev_batch(seed=seed)
    t_start = time.time()
    W_online, sigma_online = model_online.fit(
        X,
        lambda1=0.05,
        batch_size=1,            # Online: one sample at a time
        n_batches_warm=5000,     # More iterations for convergence
        n_batches_final=10000,
        T=5,
        lr=0.0003
    )
    t_online = time.time() - t_start

    W_online_bin = to_bin(W_online, thr=threshold)
    shd_online, tpr_online, fdr_online = count_accuracy(W_true_bin, W_online_bin)

    results['online'] = {
        'time': t_online,
        'shd': shd_online,
        'tpr': tpr_online,
        'fdr': fdr_online,
        'edges': np.sum(W_online_bin)
    }

    print(f"   Time: {t_online:.2f}s")
    print(f"   SHD: {shd_online}, TPR: {tpr_online:.4f}, FDR: {fdr_online:.4f}")
    print(f"   Edges found: {results['online']['edges']}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Time':>10} {'SHD':>8} {'TPR':>10} {'FDR':>10} {'Edges':>8}")
    print("-" * 70)
    print(f"{'Batch (bs=100)':<20} {results['batch']['time']:>10.2f}s {results['batch']['shd']:>8} "
          f"{results['batch']['tpr']:>10.4f} {results['batch']['fdr']:>10.4f} {results['batch']['edges']:>8}")
    print(f"{'Online (bs=1)':<20} {results['online']['time']:>10.2f}s {results['online']['shd']:>8} "
          f"{results['online']['tpr']:>10.4f} {results['online']['fdr']:>10.4f} {results['online']['edges']:>8}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Run with default settings

    n_nodes=100
    results = run_experiment(
        n_nodes=n_nodes,
        n_samples=1000,
        n_edges= 4*n_nodes,
        seed=42,
        threshold=0.3
    )
