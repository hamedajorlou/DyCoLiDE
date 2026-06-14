"""
Simple hyperparameter tuning for DyCoLiDE-EV.
30 nodes, 60 edges, 1000 timesteps
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
from SVAR.dycolide import DyCoLiDE_EV
from Utils import count_accuracy_svar
from Utils import generate_svar_data

def compute_f1(tpr, fdr):
    precision = 1 - fdr
    recall = tpr
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

# Generate data once
print("Generating data: 30 nodes, 60 edges, 1000 timesteps...")
X, B_true, A_list_true, _ = generate_svar_data(
    n_nodes=30, n_timesteps=1000, lag_order=1,
    instantaneous_edges=60, temporal_sparsity=0.1,
    temporal_strength=0.2, noise_scale=1, noise_type='ev', seed=42
)
A_true = np.vstack([A.T for A in A_list_true])
B_true = B_true.T
print(f"True W edges: {np.sum(np.abs(B_true) > 0)}, True A edges: {np.sum(np.abs(A_true) > 0)}")

# Test configurations (just 9 combinations)
configs = [
    {'lambda_W': 0.01, 'lambda_A': 0.02, 'lr': 0.0003, 'threshold': 0.06},


]

print(f"\nTesting {len(configs)} configurations...\n")
print("-"*90)

results = []
best_f1 = -1
best_config = None

for i, cfg in enumerate(configs):
    threshold = cfg['threshold']
    params = {
        'p': 1, 'lambda_W': cfg['lambda_W'], 'lambda_A': cfg['lambda_A'],
        'T': 4, 'mu_init': 2.0, 'mu_factor': 0.1, 's': [1.0, 0.9, 0.8, 0.7],
        'warm_iter': 15000, 'max_iter': 30000, 'lr': cfg['lr'], 'checkpoint': 5000
    }

    print(f"[{i+1}/{len(configs)}] lambda_W={cfg['lambda_W']}, lambda_A={cfg['lambda_A']}, threshold={threshold}")

    model = DyCoLiDE_EV(seed=42)
    start = time.time()
    W_est, A_est, _ = model.fit(X, **params)
    elapsed = time.time() - start

    metrics = count_accuracy_svar(B_true, W_est, A_true, A_est, threshold=threshold)
    w_f1 = compute_f1(metrics['W_tpr'], metrics['W_fdr'])
    a_f1 = compute_f1(metrics['A_tpr'], metrics['A_fdr'])
    total_f1 = (w_f1 + a_f1) / 2

    print(f"   W: TPR={metrics['W_tpr']:.3f}, FDR={metrics['W_fdr']:.3f}, F1={w_f1:.3f}")
    print(f"   A: TPR={metrics['A_tpr']:.3f}, FDR={metrics['A_fdr']:.3f}, F1={a_f1:.3f}")
    print(f"   Total F1={total_f1:.3f}, Time={elapsed:.1f}s")

    if total_f1 > best_f1:
        best_f1 = total_f1
        best_config = {**cfg, 'w_f1': w_f1, 'a_f1': a_f1, 'total_f1': total_f1,
                       'W_tpr': metrics['W_tpr'], 'W_fdr': metrics['W_fdr'],
                       'A_tpr': metrics['A_tpr'], 'A_fdr': metrics['A_fdr']}
        print("   *** NEW BEST ***")
    print()

print("="*90)
print("BEST CONFIGURATION:")
print(f"  lambda_W: {best_config['lambda_W']}")
print(f"  lambda_A: {best_config['lambda_A']}")
print(f"  threshold: {best_config['threshold']}")
print(f"\nResults:")
print(f"  W: TPR={best_config['W_tpr']:.3f}, FDR={best_config['W_fdr']:.3f}, F1={best_config['w_f1']:.3f}")
print(f"  A: TPR={best_config['A_tpr']:.3f}, FDR={best_config['A_fdr']:.3f}, F1={best_config['a_f1']:.3f}")
print(f"  Total F1: {best_config['total_f1']:.3f}")
print("="*90)
