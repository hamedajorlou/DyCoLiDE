"""
Test script for colide_nv_online with tunable batch_size.

Compares:
- Batch mode (batch_size=100): Mini-batch SGD
- Online mode (batch_size=1): True online learning
"""

from Colide_batch_nv_MOHEM import colide_nv_online
from Utils import simulate_sem, count_accuracy
import numpy as np
import time

# Experiment parameters
n_nodes = 100
n_samples = 1000
edges = 400
seed = 42

print("="*80)
print(f"Test: CoLiDE-NV Online vs Batch ({n_nodes} nodes, {edges} edges)")
print("="*80)

# Generate data with non-equal variance
X, W_true, var_nv = simulate_sem(
    n_nodes=n_nodes,
    n_samples=n_samples,
    edges=edges,
    graph_type='er',
    edge_type='weighted',
    var_type='nv',
    noise='normal',
    var=5.0,
    w_range=((-1.0, -0.25), (0.25, 1.0)),
    seed=seed
)

print(f"\nData: {n_nodes} nodes, {n_samples} samples, {edges} edges")
print(f"True edges: {np.sum(W_true != 0)}")
W_true_binary = (W_true != 0).astype(float)

# Target: TPR > 0.7, FDR < 0.2
# Best configuration found
lambda1 = 0.025
threshold = 0.25
T = 4
mu_init = 1.0
mu_factor = 0.1
s = [1.0, 0.9, 0.8, 0.7]
lr = 0.0002

n_batches_warm = 15000
n_batches_final = 45000

print(f"\nParameters: lambda1={lambda1}, threshold={threshold}")
print(f"Iterations: warm={n_batches_warm}, final={n_batches_final}")

# Test 1: Batch mode (batch_size=100)
print("\n" + "="*80)
print("1. Batch Mode (batch_size=400)")
print("="*80)

model_batch = colide_nv_online(seed=seed)
start = time.time()
W_batch, sigma_batch = model_batch.fit(
    X=X,
    lambda1=lambda1,
    T=T,
    mu_init=mu_init,
    mu_factor=mu_factor,
    s=s,
    batch_size=400,  # Balanced batch size
    n_batches_warm=n_batches_warm,
    n_batches_final=n_batches_final,
    lr=lr,
    checkpoint=500,
    sigma_0=0.1
)
time_batch = time.time() - start

W_batch_binary = (np.abs(W_batch) > threshold).astype(float)
shd_batch, tpr_batch, fdr_batch = count_accuracy(W_true_binary, W_batch_binary)
corr_batch = np.corrcoef(np.sqrt(var_nv), sigma_batch)[0, 1]

print(f"Time: {time_batch:.2f}s | TPR: {tpr_batch:.4f} | FDR: {fdr_batch:.4f} | SHD: {shd_batch}")
print(f"Variance correlation: {corr_batch:.4f}")

# Test 2: Online mode (batch_size=1)
print("\n" + "="*80)
print("2. Online Mode (batch_size=50)")
print("="*80)

model_online = colide_nv_online(seed=seed)
start = time.time()
W_online, sigma_online = model_online.fit(
    X=X,
    lambda1=lambda1,
    T=T,
    mu_init=mu_init,
    mu_factor=mu_factor,
    s=s,
    batch_size=50,  # True online mode
    n_batches_warm=n_batches_warm,
    n_batches_final=n_batches_final,
    lr=lr,
    checkpoint=500,
    sigma_0=0.1
)
time_online = time.time() - start

W_online_binary = (np.abs(W_online) > threshold).astype(float)
shd_online, tpr_online, fdr_online = count_accuracy(W_true_binary, W_online_binary)
corr_online = np.corrcoef(np.sqrt(var_nv), sigma_online)[0, 1]

print(f"Time: {time_online:.2f}s | TPR: {tpr_online:.4f} | FDR: {fdr_online:.4f} | SHD: {shd_online}")
print(f"Variance correlation: {corr_online:.4f}")

# Summary
print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"{'Method':<20} {'Time (s)':>10} {'TPR':>8} {'FDR':>8} {'SHD':>6} {'Var Corr':>10}")
print("-"*80)
print(f"{'Batch (bs=100)':<20} {time_batch:>10.2f} {tpr_batch:>8.4f} {fdr_batch:>8.4f} {shd_batch:>6} {corr_batch:>10.4f}")
print(f"{'Online (bs=1)':<20} {time_online:>10.2f} {tpr_online:>8.4f} {fdr_online:>8.4f} {shd_online:>6} {corr_online:>10.4f}")
print("="*80)

# Difference analysis
print("\nDifference (Online - Batch):")
print(f"  TPR diff: {tpr_online - tpr_batch:+.4f}")
print(f"  FDR diff: {fdr_online - fdr_batch:+.4f}")
print(f"  SHD diff: {shd_online - shd_batch:+d}")

# Final summary
print("\n" + "="*80)
print("Final Summary - Target: TPR > 0.7, FDR < 0.2")
print("="*80)
print(f"{'Method':<20} {'Time (s)':>10} {'TPR':>8} {'FDR':>8} {'SHD':>6} {'Var Corr':>10}")
print("-"*80)
print(f"{'Batch (bs=100)':<20} {time_batch:>10.2f} {tpr_batch:>8.4f} {fdr_batch:>8.4f} {shd_batch:>6} {corr_batch:>10.4f}")
print(f"{'Online (bs=1)':<20} {time_online:>10.2f} {tpr_online:>8.4f} {fdr_online:>8.4f} {shd_online:>6} {corr_online:>10.4f}")
print("="*80)

# Check targets
print("\nTarget Check:")
print(f"  Batch - TPR > 0.7: {'YES' if tpr_batch > 0.7 else 'NO'} ({tpr_batch:.4f})")
print(f"  Batch - FDR < 0.2: {'YES' if fdr_batch < 0.2 else 'NO'} ({fdr_batch:.4f})")
