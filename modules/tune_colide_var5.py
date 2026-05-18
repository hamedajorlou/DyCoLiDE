"""
Hyperparameter tuning for CoLiDE and DyCoLiDE with var=5
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import time

sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears/dagma/src/dagma')

from Utils import simulate_sem, count_accuracy
from Colide import colide_ev as StaticCoLiDE
from Colide_batch_MOHEM import colide_ev_batch as DyCoLiDE

# Generate data once
np.random.seed(42)
n_nodes, n_samples, seed = 100, 1000, 42
edges = n_nodes * 4

print("Generating data (var=5)...")
X, W_true, _ = simulate_sem(
    n_nodes=n_nodes,
    n_samples=n_samples,
    edges=edges,
    graph_type='er',
    edge_type='weighted',
    var_type='ev',
    noise='normal',
    var=5.0,
    w_range=((-2.0, -0.5), (0.5, 2.0)),
    seed=seed
)

W_true_binary = (W_true != 0).astype(float)
true_edges = int(np.sum(W_true_binary))
print(f"Data: {n_nodes} nodes, {n_samples} samples, {true_edges} true edges\n")

# Hyperparameter grid
lambda1_values = [0.03, 0.05, 0.07, 0.10]
threshold_values = [0.3, 0.35, 0.4, 0.45]

print("="*80)
print("TUNING CoLiDE (var=5)")
print("="*80)

best_colide = {'fdr': 1.0, 'tpr': 0.0, 'shd': float('inf')}
colide_results = []

for lambda1 in lambda1_values:
    print(f"\n--- lambda1 = {lambda1} ---")

    # Run CoLiDE once per lambda1
    model = StaticCoLiDE(seed=seed)
    start = time.time()
    W_est, _ = model.fit(
        X=X.copy(),
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
    elapsed = time.time() - start

    # Test different thresholds
    for thresh in threshold_values:
        W_binary = (np.abs(W_est) > thresh).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_binary)

        result = {'lambda1': lambda1, 'threshold': thresh,
                  'tpr': tpr, 'fdr': fdr, 'shd': shd, 'time': elapsed}
        colide_results.append(result)

        print(f"  thresh={thresh:.2f}: TPR={tpr:.4f}, FDR={fdr:.4f}, SHD={shd}")

        # Update best (prioritize low FDR with reasonable TPR)
        if fdr < best_colide['fdr'] or (fdr == best_colide['fdr'] and tpr > best_colide['tpr']):
            if tpr >= 0.7:  # Minimum TPR threshold
                best_colide = result.copy()

print("\n" + "="*80)
print("TUNING DyCoLiDE (var=5)")
print("="*80)

best_dycolide = {'fdr': 1.0, 'tpr': 0.0, 'shd': float('inf')}
dycolide_results = []

for lambda1 in lambda1_values:
    print(f"\n--- lambda1 = {lambda1} ---")

    # Run DyCoLiDE once per lambda1
    model = DyCoLiDE(seed=seed)
    start = time.time()
    W_est, _ = model.fit(
        X=X.copy(),
        lambda1=lambda1,
        T=4,
        mu_init=1.0,
        mu_factor=0.1,
        s=[1.0, 0.9, 0.8, 0.7],
        batch_size=200,
        n_batches_warm=5000,
        n_batches_final=15000,
        lr=0.0003,
        checkpoint=500
    )
    elapsed = time.time() - start

    # Test different thresholds
    for thresh in threshold_values:
        W_binary = (np.abs(W_est) > thresh).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_binary)

        result = {'lambda1': lambda1, 'threshold': thresh,
                  'tpr': tpr, 'fdr': fdr, 'shd': shd, 'time': elapsed}
        dycolide_results.append(result)

        print(f"  thresh={thresh:.2f}: TPR={tpr:.4f}, FDR={fdr:.4f}, SHD={shd}")

        # Update best
        if fdr < best_dycolide['fdr'] or (fdr == best_dycolide['fdr'] and tpr > best_dycolide['tpr']):
            if tpr >= 0.7:
                best_dycolide = result.copy()

# Summary
print("\n" + "="*80)
print("BEST RESULTS (var=5, 100 nodes)")
print("="*80)

print(f"\nCoLiDE Best:")
print(f"  lambda1={best_colide['lambda1']}, threshold={best_colide['threshold']}")
print(f"  TPR={best_colide['tpr']:.4f}, FDR={best_colide['fdr']:.4f}, SHD={best_colide['shd']}")

print(f"\nDyCoLiDE Best:")
print(f"  lambda1={best_dycolide['lambda1']}, threshold={best_dycolide['threshold']}")
print(f"  TPR={best_dycolide['tpr']:.4f}, FDR={best_dycolide['fdr']:.4f}, SHD={best_dycolide['shd']}")

# Full results table sorted by FDR
print("\n" + "="*80)
print("ALL CoLiDE RESULTS (sorted by FDR)")
print("="*80)
print(f"{'lambda1':>8} {'thresh':>8} {'TPR':>8} {'FDR':>8} {'SHD':>8}")
print("-"*45)
for r in sorted(colide_results, key=lambda x: (x['fdr'], -x['tpr'])):
    print(f"{r['lambda1']:>8.2f} {r['threshold']:>8.2f} {r['tpr']:>8.4f} {r['fdr']:>8.4f} {r['shd']:>8}")

print("\n" + "="*80)
print("ALL DyCoLiDE RESULTS (sorted by FDR)")
print("="*80)
print(f"{'lambda1':>8} {'thresh':>8} {'TPR':>8} {'FDR':>8} {'SHD':>8}")
print("-"*45)
for r in sorted(dycolide_results, key=lambda x: (x['fdr'], -x['tpr'])):
    print(f"{r['lambda1']:>8.2f} {r['threshold']:>8.2f} {r['tpr']:>8.4f} {r['fdr']:>8.4f} {r['shd']:>8}")
