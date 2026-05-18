"""
Debug script for Meta-D2AG to understand why it's performing poorly.
"""

import numpy as np
import torch
from meta_d2ag import MetaD2AG, compute_dag_constraint_logdet
from SVAR_data_generator import generate_svar_data
from DyCoLiDE import count_accuracy_svar

def compute_f1(tpr, fdr):
    precision = 1 - fdr
    recall = tpr
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

# Generate small test data
print("="*60)
print("DEBUG: Meta-D2AG")
print("="*60)

n_nodes = 10
n_edges = 20
n_timesteps = 500

print(f"\nGenerating data: {n_nodes} nodes, {n_edges} edges, {n_timesteps} timesteps")
X, B_true, A_list_true, _ = generate_svar_data(
    n_nodes=n_nodes,
    n_timesteps=n_timesteps,
    lag_order=1,
    instantaneous_edges=n_edges,
    temporal_sparsity=0.1,
    temporal_strength=0.2,
    noise_scale=1,
    noise_type='ev',
    seed=42
)

# Ground truth
B_true_t = B_true.T  # DyCoLiDE convention
A_true = np.vstack([A.T for A in A_list_true])

print(f"True W edges: {np.sum(np.abs(B_true_t) > 0)}")
print(f"True A edges: {np.sum(np.abs(A_true) > 0)}")
print(f"W range: [{B_true_t.min():.3f}, {B_true_t.max():.3f}]")
print(f"A range: [{A_true.min():.3f}, {A_true.max():.3f}]")

# Check what OLS would give us (baseline)
print("\n" + "-"*60)
print("Baseline: OLS solution (no DAG constraint)")
print("-"*60)

p = 1
X_current = X[p:]
Y_lagged = X[:-p]
n_eff = X_current.shape[0]

# OLS: minimize ||X - X@W - Y@A||^2
# Stack [X, Y] and solve for [W; A]
XY = np.hstack([X_current, Y_lagged])  # (n_eff, 2d)
# Normal equation: (XY'XY) @ [W;A] = XY' @ X
# But this doesn't account for X being on both sides...

# Instead, solve iteratively or use the closed form for SVAR
# X = X@W + Y@A + noise  =>  X@(I-W) = Y@A + noise
# This is tricky. Let's just check the data properties.

print(f"X_current shape: {X_current.shape}")
print(f"Y_lagged shape: {Y_lagged.shape}")
print(f"X variance: {np.var(X_current):.3f}")

# Test different hyperparameter configs
print("\n" + "="*60)
print("Testing different hyperparameter configurations")
print("="*60)

configs = [
    {'lambda_w': 0.01, 'lambda_a': 0.01, 'lr': 0.003, 'threshold': 0.1},
    {'lambda_w': 0.05, 'lambda_a': 0.05, 'lr': 0.003, 'threshold': 0.1},
    {'lambda_w': 0.1, 'lambda_a': 0.1, 'lr': 0.003, 'threshold': 0.1},
    {'lambda_w': 0.01, 'lambda_a': 0.01, 'lr': 0.001, 'threshold': 0.1},
    {'lambda_w': 0.01, 'lambda_a': 0.01, 'lr': 0.003, 'threshold': 0.2},
    {'lambda_w': 0.01, 'lambda_a': 0.01, 'lr': 0.003, 'threshold': 0.3},
]

for i, cfg in enumerate(configs):
    print(f"\n--- Config {i+1}: lambda_w={cfg['lambda_w']}, lambda_a={cfg['lambda_a']}, lr={cfg['lr']}, threshold={cfg['threshold']} ---")

    model = MetaD2AG(seed=42)
    W_est, A_est = model.fit(
        X, p=1,
        lambda_w=cfg['lambda_w'],
        lambda_a=cfg['lambda_a'],
        max_iter=2000,
        lr=cfg['lr'],
        w_threshold=cfg['threshold'],
        verbose=False
    )

    # Check estimated weights
    w_nnz = np.sum(np.abs(W_est) > 0)
    a_nnz = np.sum(np.abs(A_est) > 0)

    metrics = count_accuracy_svar(B_true_t, W_est, A_true, A_est, threshold=0.0)
    w_f1 = compute_f1(metrics['W_tpr'], metrics['W_fdr'])
    a_f1 = compute_f1(metrics['A_tpr'], metrics['A_fdr'])

    print(f"  W: nnz={w_nnz}, TPR={metrics['W_tpr']:.3f}, FDR={metrics['W_fdr']:.3f}, F1={w_f1:.3f}")
    print(f"  A: nnz={a_nnz}, TPR={metrics['A_tpr']:.3f}, FDR={metrics['A_fdr']:.3f}, F1={a_f1:.3f}")
    print(f"  W_est range: [{W_est.min():.3f}, {W_est.max():.3f}]")
    print(f"  A_est range: [{A_est.min():.3f}, {A_est.max():.3f}]")

# Detailed analysis of best config
print("\n" + "="*60)
print("Detailed analysis with verbose output")
print("="*60)

model = MetaD2AG(seed=42)
W_est, A_est = model.fit(
    X, p=1,
    lambda_w=0.01,
    lambda_a=0.01,
    max_iter=2000,
    lr=0.003,
    w_threshold=0.1,
    verbose=True
)

# Compare W_est vs B_true_t
print("\n--- W comparison (first 5x5) ---")
print("True W:")
print(B_true_t[:5, :5].round(3))
print("\nEstimated W:")
print(W_est[:5, :5].round(3))

# Check if the signs match
print("\n--- Sign agreement ---")
true_edges = np.abs(B_true_t) > 0
est_edges = np.abs(W_est) > 0
both_edges = true_edges & est_edges
if np.sum(both_edges) > 0:
    sign_agree = np.sum(np.sign(B_true_t[both_edges]) == np.sign(W_est[both_edges]))
    print(f"Sign agreement on true positive edges: {sign_agree}/{np.sum(both_edges)}")

# Check the residual
print("\n--- Residual analysis ---")
X_current = X[1:]
Y_lagged = X[:-1]
residual = X_current - X_current @ W_est - Y_lagged @ A_est
print(f"Residual norm: {np.linalg.norm(residual):.3f}")
print(f"Residual variance: {np.var(residual):.3f}")
print(f"Original X variance: {np.var(X_current):.3f}")
print(f"R^2: {1 - np.var(residual)/np.var(X_current):.3f}")
