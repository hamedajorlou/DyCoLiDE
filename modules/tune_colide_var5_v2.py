"""
Fine-grained tuning for CoLiDE and DyCoLiDE with var=5
Goal: Higher TPR while keeping FDR low
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

# Fine-grained hyperparameter grid
lambda1_values = [0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
threshold_values = [0.32, 0.35, 0.38, 0.40, 0.42]

print("="*80)
print("FINE-GRAINED TUNING CoLiDE (var=5)")
print("="*80)

colide_results = []

for lambda1 in lambda1_values:
    print(f"\n--- lambda1 = {lambda1} ---")
    model = StaticCoLiDE(seed=seed)
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

    for thresh in threshold_values:
        W_binary = (np.abs(W_est) > thresh).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_binary)

        # Calculate F1 score
        precision = 1 - fdr if (1 - fdr) > 0 else 0
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

        result = {'lambda1': lambda1, 'threshold': thresh,
                  'tpr': tpr, 'fdr': fdr, 'shd': shd, 'f1': f1}
        colide_results.append(result)
        print(f"  thresh={thresh:.2f}: TPR={tpr:.4f}, FDR={fdr:.4f}, F1={f1:.4f}, SHD={shd}")

print("\n" + "="*80)
print("FINE-GRAINED TUNING DyCoLiDE (var=5)")
print("="*80)

dycolide_results = []

for lambda1 in lambda1_values:
    print(f"\n--- lambda1 = {lambda1} ---")
    model = DyCoLiDE(seed=seed)
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

    for thresh in threshold_values:
        W_binary = (np.abs(W_est) > thresh).astype(float)
        shd, tpr, fdr = count_accuracy(W_true_binary, W_binary)

        precision = 1 - fdr if (1 - fdr) > 0 else 0
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

        result = {'lambda1': lambda1, 'threshold': thresh,
                  'tpr': tpr, 'fdr': fdr, 'shd': shd, 'f1': f1}
        dycolide_results.append(result)
        print(f"  thresh={thresh:.2f}: TPR={tpr:.4f}, FDR={fdr:.4f}, F1={f1:.4f}, SHD={shd}")

# Find best configurations by different criteria
print("\n" + "="*80)
print("BEST CONFIGURATIONS")
print("="*80)

# Best by F1 score (balance of precision and recall)
best_colide_f1 = max(colide_results, key=lambda x: x['f1'])
best_dycolide_f1 = max(dycolide_results, key=lambda x: x['f1'])

# Best with FDR < 0.10 and highest TPR
colide_low_fdr = [r for r in colide_results if r['fdr'] < 0.10]
dycolide_low_fdr = [r for r in dycolide_results if r['fdr'] < 0.10]

best_colide_low_fdr = max(colide_low_fdr, key=lambda x: x['tpr']) if colide_low_fdr else None
best_dycolide_low_fdr = max(dycolide_low_fdr, key=lambda x: x['tpr']) if dycolide_low_fdr else None

print("\n** CoLiDE **")
print(f"Best by F1: lambda1={best_colide_f1['lambda1']}, thresh={best_colide_f1['threshold']}")
print(f"  TPR={best_colide_f1['tpr']:.4f}, FDR={best_colide_f1['fdr']:.4f}, F1={best_colide_f1['f1']:.4f}, SHD={best_colide_f1['shd']}")
if best_colide_low_fdr:
    print(f"Best (FDR<0.10): lambda1={best_colide_low_fdr['lambda1']}, thresh={best_colide_low_fdr['threshold']}")
    print(f"  TPR={best_colide_low_fdr['tpr']:.4f}, FDR={best_colide_low_fdr['fdr']:.4f}, SHD={best_colide_low_fdr['shd']}")

print("\n** DyCoLiDE **")
print(f"Best by F1: lambda1={best_dycolide_f1['lambda1']}, thresh={best_dycolide_f1['threshold']}")
print(f"  TPR={best_dycolide_f1['tpr']:.4f}, FDR={best_dycolide_f1['fdr']:.4f}, F1={best_dycolide_f1['f1']:.4f}, SHD={best_dycolide_f1['shd']}")
if best_dycolide_low_fdr:
    print(f"Best (FDR<0.10): lambda1={best_dycolide_low_fdr['lambda1']}, thresh={best_dycolide_low_fdr['threshold']}")
    print(f"  TPR={best_dycolide_low_fdr['tpr']:.4f}, FDR={best_dycolide_low_fdr['fdr']:.4f}, SHD={best_dycolide_low_fdr['shd']}")

# Top 5 by F1 for each
print("\n" + "="*80)
print("TOP 5 CoLiDE by F1")
print("="*80)
print(f"{'lambda1':>8} {'thresh':>8} {'TPR':>8} {'FDR':>8} {'F1':>8} {'SHD':>8}")
print("-"*55)
for r in sorted(colide_results, key=lambda x: -x['f1'])[:5]:
    print(f"{r['lambda1']:>8.2f} {r['threshold']:>8.2f} {r['tpr']:>8.4f} {r['fdr']:>8.4f} {r['f1']:>8.4f} {r['shd']:>8}")

print("\n" + "="*80)
print("TOP 5 DyCoLiDE by F1")
print("="*80)
print(f"{'lambda1':>8} {'thresh':>8} {'TPR':>8} {'FDR':>8} {'F1':>8} {'SHD':>8}")
print("-"*55)
for r in sorted(dycolide_results, key=lambda x: -x['f1'])[:5]:
    print(f"{r['lambda1']:>8.2f} {r['threshold']:>8.2f} {r['tpr']:>8.4f} {r['fdr']:>8.4f} {r['f1']:>8.4f} {r['shd']:>8}")
