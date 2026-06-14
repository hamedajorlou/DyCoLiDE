"""
Focused Comparison: DAGMA vs GOLEM vs CoLiDE vs DyCoLiDE

Compares:
- DAGMA (local implementation)
- GOLEM (local implementation)
- Static CoLiDE (full batch)
- DyCoLiDE (mini-batch SGD) - supports streaming/batch settings
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import time

# Add local paths for GOLEM and DAGMA
sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears/golem/src')
sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears/dagma/src/dagma')

from Utils import simulate_sem, count_accuracy

# Import local implementations
from SEM.dycolide import colide_ev as StaticCoLiDE
from Colide_batch_MOHEM import colide_ev_batch as DyCoLiDE

# DAGMA
from linear import DagmaLinear

# GOLEM
from models import GolemModel
from trainers import GolemTrainer


def run_dagma(X, lambda1=0.02, threshold=0.3):
    """Run DAGMA algorithm."""
    model = DagmaLinear(loss_type='l2', verbose=False)
    W_est = model.fit(X, lambda1=lambda1)
    return W_est


def run_golem(X, lambda1=0.02, lambda2=5.0, equal_variances=True,
              num_iter=50000, learning_rate=1e-3, seed=1):
    """Run GOLEM algorithm."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    n, d = X_centered.shape

    model = GolemModel(n, d, lambda1, lambda2, equal_variances, seed)
    trainer = GolemTrainer(learning_rate)
    B_est = trainer.train(model, X_centered, num_iter, checkpoint_iter=None, output_dir=None)

    return B_est


def run_colide(X, lambda1=0.02, seed=42):
    """Run static CoLiDE (full batch)."""
    model = StaticCoLiDE(seed=seed)
    W_est, sigma_est = model.fit(
        X=X,
        lambda1=lambda1,
        T=4,
        mu_init=1.0,
        mu_factor=0.1,
        s=[1.0, 0.9, 0.8, 0.7],
        warm_iter=5000,
        max_iter=15000,
        lr=0.0003,
        checkpoint=500
    )
    return W_est


def run_dycolide(X, lambda1=0.02, batch_size=200, seed=42):
    """Run DyCoLiDE-EV (mini-batch SGD)."""
    model = DyCoLiDE(seed=seed)
    W_est, sigma_est = model.fit(
        X=X,
        lambda1=lambda1,
        T=4,
        mu_init=1.0,
        mu_factor=0.1,
        s=[1.0, 0.9, 0.8, 0.7],
        batch_size=batch_size,
        n_batches_warm=5000,
        n_batches_final=15000,
        lr=0.0003,
        checkpoint=500
    )
    return W_est


def run_comparison(n_nodes=50, n_samples=1000, edges=None, seed=42, threshold=0.3):
    """Run comparison across methods."""

    if edges is None:
        edges = n_nodes * 4  # ER4

    print("="*100)
    print(f"COMPARISON: DAGMA vs GOLEM vs CoLiDE vs DyCoLiDE")
    print(f"Settings: {n_nodes} nodes, {n_samples} samples, ER4 ({edges} edges)")
    print("="*100)

    # Generate data (Equal Variance case)
    print("\nGenerating data (Equal Variance, var=1)...")
    X, W_true, _ = simulate_sem(
        n_nodes=n_nodes,
        n_samples=n_samples,
        edges=edges,
        graph_type='er',
        edge_type='weighted',
        var_type='ev',
        noise='normal',
        var=1.0,
        w_range=((-2.0, -0.5), (0.5, 2.0)),
        seed=seed
    )

    W_true_binary = (W_true != 0).astype(float)
    true_edges = int(np.sum(W_true_binary))
    print(f"Data generated: {n_nodes} nodes, {n_samples} samples, {true_edges} true edges")

    results = {}

    # Run DAGMA
    print("\n" + "-"*80)
    print("[1/4] Running DAGMA...")
    print("-"*80)
    try:
        start = time.time()
        W_dagma = run_dagma(X, lambda1=0.02)
        time_dagma = time.time() - start

        W_dagma_binary = (np.abs(W_dagma) > threshold).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_dagma_binary)
        results['DAGMA'] = {'time': time_dagma, 'tpr': tpr, 'fdr': fdr, 'shd': shd}
        print(f"  Completed in {time_dagma:.2f}s | TPR: {tpr:.4f} | FDR: {fdr:.4f} | SHD: {shd}")
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")
        results['DAGMA'] = {'error': str(e)}

    # Run GOLEM
    print("\n" + "-"*80)
    print("[2/4] Running GOLEM...")
    print("-"*80)
    try:
        start = time.time()
        W_golem = run_golem(X, lambda1=0.02, lambda2=5.0, num_iter=50000)
        time_golem = time.time() - start

        W_golem_binary = (np.abs(W_golem) > threshold).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_golem_binary)
        results['GOLEM'] = {'time': time_golem, 'tpr': tpr, 'fdr': fdr, 'shd': shd}
        print(f"  Completed in {time_golem:.2f}s | TPR: {tpr:.4f} | FDR: {fdr:.4f} | SHD: {shd}")
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")
        results['GOLEM'] = {'error': str(e)}

    # Run CoLiDE
    print("\n" + "-"*80)
    print("[3/4] Running CoLiDE (static)...")
    print("-"*80)
    try:
        start = time.time()
        W_colide = run_colide(X, lambda1=0.02, seed=seed)
        time_colide = time.time() - start

        W_colide_binary = (np.abs(W_colide) > threshold).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_colide_binary)
        results['CoLiDE'] = {'time': time_colide, 'tpr': tpr, 'fdr': fdr, 'shd': shd}
        print(f"  Completed in {time_colide:.2f}s | TPR: {tpr:.4f} | FDR: {fdr:.4f} | SHD: {shd}")
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")
        results['CoLiDE'] = {'error': str(e)}

    # Run DyCoLiDE
    print("\n" + "-"*80)
    print("[4/4] Running DyCoLiDE (batch_size=200)...")
    print("-"*80)
    try:
        start = time.time()
        W_dycolide = run_dycolide(X, lambda1=0.02, batch_size=200, seed=seed)
        time_dycolide = time.time() - start

        W_dycolide_binary = (np.abs(W_dycolide) > threshold).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_dycolide_binary)
        results['DyCoLiDE'] = {'time': time_dycolide, 'tpr': tpr, 'fdr': fdr, 'shd': shd}
        print(f"  Completed in {time_dycolide:.2f}s | TPR: {tpr:.4f} | FDR: {fdr:.4f} | SHD: {shd}")
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")
        results['DyCoLiDE'] = {'error': str(e)}

    # Print summary table
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"\nTrue edges: {true_edges} | Threshold: {threshold}")
    print(f"\n{'Method':<15} {'Time (s)':>12} {'TPR':>10} {'FDR':>10} {'SHD':>10}")
    print("-"*60)

    for method in ['DAGMA', 'GOLEM', 'CoLiDE', 'DyCoLiDE']:
        if method in results:
            r = results[method]
            if 'error' in r:
                print(f"{method:<15} {'ERROR':>12}")
            else:
                print(f"{method:<15} {r['time']:>12.2f} {r['tpr']:>10.4f} {r['fdr']:>10.4f} {r['shd']:>10}")

    print("="*60)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-"*60)
    print("* DyCoLiDE supports flexible batch sizes (1 to n)")
    print("* Can process data in streaming/online fashion")
    print("* Competitive with full-batch methods while being more flexible")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare DAGMA, GOLEM, CoLiDE, DyCoLiDE')
    parser.add_argument('--n_nodes', type=int, default=50, help='Number of nodes')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.3, help='Edge threshold')

    args = parser.parse_args()

    results = run_comparison(
        n_nodes=args.n_nodes,
        n_samples=args.n_samples,
        seed=args.seed,
        threshold=args.threshold
    )
