"""
Test DyCoLiDE-EV on SVAR data with homoscedastic noise.
"""

import numpy as np
import time
from DyCoLiDE import DyCoLiDE_EV, count_accuracy_svar
from SVAR_data_generator import generate_svar_data

print("="*80)
print("DyCoLiDE-EV Test: Dynamic CoLiDE for SVAR Models")
print("="*80)

# ============================================
# Generate SVAR Data (Homoscedastic) - HARDER TEST
# ============================================
seed = 42
n_nodes = 30           
n_timesteps = 3000   
lag_order = 1        
instantaneous_edges = 60  
temporal_sparsity = 0.15  

print(f"\nGenerating SVAR data (HARDER TEST - homoscedastic noise)...")
print(f"  Nodes: {n_nodes}")
print(f"  Timesteps: {n_timesteps}")
print(f"  Lag order: {lag_order}")
print(f"  Instantaneous edges: {instantaneous_edges}")

X, B_true, A_list_true, params = generate_svar_data(
    n_nodes=n_nodes,
    n_timesteps=n_timesteps,
    lag_order=lag_order,
    instantaneous_edges=instantaneous_edges,
    temporal_sparsity=temporal_sparsity,
    temporal_strength=0.2,   # Much stronger temporal effects
    noise_scale=1,
    noise_type='ev',
    seed=seed
)

# Convert ground truth to DyCoLiDE convention (transpose due to row vs column vectors)
A_true = np.vstack([A.T for A in A_list_true])
B_true_dycolide = B_true.T

print(f"\nData generated:")
print(f"  X shape: {X.shape}")
print(f"  W (intra-slice) edges: {np.sum(np.abs(B_true_dycolide) > 0)}")
print(f"  A (inter-slice) edges: {np.sum(np.abs(A_true) > 0)}")

# ============================================
# Hyperparameters (tuned for better A estimation)
# Key insight: lambda_A should be much smaller than lambda_W
# because temporal effects are typically weaker
# ============================================
params_ev = {
    'p': lag_order,
    'lambda_W': 0.01,
    'lambda_A': 0.01,  
    'T': 6,
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    'warm_iter': 10000,
    'max_iter': 20000,
    'lr': 0.0001,
    'checkpoint': 1000
}

threshold = 0.08  # Slightly higher to reduce false positives

# ============================================
# Test DyCoLiDE-EV
# ============================================
print("\n" + "="*80)
print("DyCoLiDE-EV (Equal Variance / Homoscedastic)")
print("="*80)

model_ev = DyCoLiDE_EV(seed=seed)
start = time.time()
W_est, A_est, sigma_est = model_ev.fit(X, **params_ev)
elapsed = time.time() - start

metrics = count_accuracy_svar(B_true_dycolide, W_est, A_true, A_est, threshold=threshold)

print(f"\nResults (threshold={threshold}):")
print(f"  Time: {elapsed:.2f}s")
print(f"  Estimated sigma: {sigma_est:.4f}")
print(f"\n  Intra-slice (W - contemporaneous DAG):")
print(f"    TPR: {metrics['W_tpr']:.4f}")
print(f"    FDR: {metrics['W_fdr']:.4f}")
print(f"    SHD: {metrics['W_shd']}")
print(f"    Edges: {metrics['W_edges_est']} (true: {metrics['W_edges_true']})")
print(f"\n  Inter-slice (A - temporal effects):")
print(f"    TPR: {metrics['A_tpr']:.4f}")
print(f"    FDR: {metrics['A_fdr']:.4f}")
print(f"    SHD: {metrics['A_shd']}")
print(f"    Edges: {metrics['A_edges_est']} (true: {metrics['A_edges_true']})")


print("\n" + "="*80)
print("Test completed!")
print("="*80)
