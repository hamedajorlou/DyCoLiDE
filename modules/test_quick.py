"""Quick parameter sweep for batch mode only."""

from Colide_batch_nv_MOHEM import colide_nv_online
from Utils import simulate_sem, count_accuracy
import numpy as np

# Fixed experiment parameters
n_nodes = 100
n_samples = 1000
edges = 400
seed = 42

# Generate data once
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
W_true_binary = (W_true != 0).astype(float)
print(f"Data: {n_nodes} nodes, {n_samples} samples, {edges} target edges, {np.sum(W_true != 0)} actual edges")

# Test different parameter combinations
params = [
    (0.025, 0.25),  # Best from previous
    (0.025, 0.28),
    (0.02, 0.28),
    (0.02, 0.30),
    (0.015, 0.25),
    (0.015, 0.28),
]

print("\n" + "="*90)
print(f"{'lambda1':>10} {'threshold':>12} {'TPR':>10} {'FDR':>10} {'SHD':>8} {'Var Corr':>12} {'Target?':>10}")
print("="*90)

for lambda1, threshold in params:
    model = colide_nv_online(seed=seed)
    W_est, sigma_est = model.fit(
        X=X,
        lambda1=lambda1,
        T=4,
        mu_init=1.0,
        mu_factor=0.1,
        s=[1.0, 0.9, 0.8, 0.7],
        batch_size=200,
        n_batches_warm=8000,
        n_batches_final=25000,
        lr=0.0003,
        checkpoint=1000,
        sigma_0=0.1
    )

    W_binary = (np.abs(W_est) > threshold).astype(float)
    shd, tpr, fdr = count_accuracy(W_true_binary, W_binary)
    corr = np.corrcoef(np.sqrt(var_nv), sigma_est)[0, 1]

    target_met = "YES" if (tpr > 0.7 and fdr < 0.2) else "NO"
    print(f"{lambda1:>10.3f} {threshold:>12.2f} {tpr:>10.4f} {fdr:>10.4f} {shd:>8} {corr:>12.4f} {target_met:>10}")

print("="*90)
print("Target: TPR > 0.7, FDR < 0.2")
