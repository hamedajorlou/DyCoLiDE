"""
Comparison: DyCoLiDE-EV vs Dynotears
Uses the same SVAR data generation for fair comparison.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from DyCoLiDE import DyCoLiDE_EV, count_accuracy_svar
from SVAR_data_generator import generate_svar_data

def structure_to_matrices(structure_model, n_vars, lag_order):
    """Convert Dynotears StructureModel to W and A matrices.

    DyCoLiDE convention: W[i,j] means j->i (column causes row)
    Dynotears convention: edge (parent, child) with weight

    For fair comparison, we need to match conventions.
    """
    instantaneous = np.zeros((n_vars, n_vars))
    lagged = np.zeros((lag_order * n_vars, n_vars))

    for parent, child, data in structure_model.edges(data=True):
        weight = float(data.get("weight", 0.0))
        parent_var, parent_lag = parent.rsplit("_lag", 1)
        child_var, child_lag = child.rsplit("_lag", 1)

        child_lag = int(child_lag)
        parent_lag = int(parent_lag)
        if child_lag != 0:
            continue  # only look at edges ending in the current slice

        p_idx = int(parent_var)
        c_idx = int(child_var)

        if parent_lag == 0:
            # W[child, parent] = weight means parent -> child
            instantaneous[c_idx, p_idx] = weight
        elif 1 <= parent_lag <= lag_order:
            # A has shape (p*d, d) where rows are lagged vars, cols are current vars
            # A[lag_idx, child] means lagged parent -> current child
            lag_idx = (parent_lag - 1) * n_vars + p_idx
            lagged[lag_idx, c_idx] = weight

    return instantaneous, lagged


def run_comparison(n_nodes=20, n_timesteps=2000, lag_order=1,
                   instantaneous_edges=30, temporal_sparsity=0.2,
                   seed=42, threshold=0.08,
                   methods=None):
    """Run comparison between selected methods.

    Args:
        methods: List of methods to run. Options: ['dycolide', 'dynotears']
                 If None, runs all methods.
    """

    if methods is None:
        methods = ['dycolide', 'dynotears']

    # Normalize method names
    methods = [m.lower() for m in methods]

    print("="*80)
    print("COMPARISON: " + " vs ".join([m.upper() for m in methods]))
    print("="*80)

    # ============================================
    # Generate SVAR Data
    # ============================================
    print(f"\nGenerating SVAR data...")
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
        temporal_strength=0.2,
        noise_scale=1,
        noise_type='ev',
        seed=seed
    )

    # Ground truth for DyCoLiDE convention (transpose)
    A_true = np.vstack([A.T for A in A_list_true])
    B_true_dycolide = B_true.T

    print(f"\nData generated:")
    print(f"  X shape: {X.shape}")
    print(f"  W (intra-slice) edges: {np.sum(np.abs(B_true_dycolide) > 0)}")
    print(f"  A (inter-slice) edges: {np.sum(np.abs(A_true) > 0)}")

    results = {}

    # ============================================
    # Run DyCoLiDE-EV
    # ============================================
    if 'dycolide' in methods:
        print("\n" + "-"*80)
        print("Running DyCoLiDE-EV...")
        print("-"*80)

        params_ev = {
            'p': lag_order,
            'lambda_W': 0.01,
            'lambda_A': 0.0001,
            'T': 4,
            'mu_init': 1.0,
            'mu_factor': 0.1,
            's': [1.0, 0.9, 0.8, 0.7],
            'warm_iter': 10000,
            'max_iter': 20000,
            'lr': 0.0001,
            'checkpoint': 1000
        }

        model_ev = DyCoLiDE_EV(seed=seed)
        start = time.time()
        W_est_dycolide, A_est_dycolide, sigma_est = model_ev.fit(X, **params_ev)
        time_dycolide = time.time() - start

        metrics_dycolide = count_accuracy_svar(B_true_dycolide, W_est_dycolide,
                                                A_true, A_est_dycolide, threshold=threshold)
        results['dycolide'] = {
            'metrics': metrics_dycolide,
            'time': time_dycolide,
            'W_est': W_est_dycolide,
            'A_est': A_est_dycolide
        }

    # ============================================
    # Run Dynotears
    # ============================================
    if 'dynotears' in methods:
        print("\n" + "-"*80)
        print("Running Dynotears...")
        print("-"*80)

        from dynotears import from_pandas_dynamic

        # Convert to DataFrame for Dynotears
        df = pd.DataFrame(X, columns=[str(i) for i in range(n_nodes)])

        start = time.time()
        sm = from_pandas_dynamic(
            df,
            p=lag_order,
            lambda_w=0.01,
            lambda_a=0.01,
            max_iter=100,
        )
        time_dynotears = time.time() - start

        # Extract matrices from Dynotears result
        W_raw, A_raw = structure_to_matrices(sm, n_nodes, lag_order)

        # Dynotears convention: X(I-W) means W[j,i] is i->j
        # DyCoLiDE convention: X @ W means W[i,j] is j->i
        # So we need to transpose W to match conventions
        W_est_dynotears = W_raw.T
        A_est_dynotears = A_raw.T.reshape(n_nodes, lag_order * n_nodes).T

        # Compute metrics for Dynotears
        metrics_dynotears = count_accuracy_svar(B_true_dycolide, W_est_dynotears,
                                                 A_true, A_est_dynotears, threshold=threshold)
        results['dynotears'] = {
            'metrics': metrics_dynotears,
            'time': time_dynotears,
            'W_est': W_est_dynotears,
            'A_est': A_est_dynotears
        }

    # ============================================
    # Print Results
    # ============================================
    print("\n" + "="*80)
    print(f"RESULTS (threshold={threshold})")
    print("="*80)

    # Build header
    header = f"{'Metric':<25}"
    for method in methods:
        header += f" {method.upper():<20}"
    print(f"\n{header}")
    print("-"*(25 + 21*len(methods)))

    # Time
    time_row = f"{'Time (s)':<25}"
    for method in methods:
        time_row += f" {results[method]['time']:<20.2f}"
    print(time_row)
    print()

    # W metrics
    print(f"{'W (Intra-slice):':<25}")

    tpr_row = f"{'  TPR':<25}"
    for method in methods:
        tpr_row += f" {results[method]['metrics']['W_tpr']:<20.4f}"
    print(tpr_row)

    fdr_row = f"{'  FDR':<25}"
    for method in methods:
        fdr_row += f" {results[method]['metrics']['W_fdr']:<20.4f}"
    print(fdr_row)

    shd_row = f"{'  SHD':<25}"
    for method in methods:
        shd_row += f" {results[method]['metrics']['W_shd']:<20}"
    print(shd_row)

    # Get true edges from first method
    first_method = methods[0]
    true_w_edges = results[first_method]['metrics']['W_edges_true']
    edges_row = f"{'  Edges (true=' + str(true_w_edges) + ')':<25}"
    for method in methods:
        edges_row += f" {results[method]['metrics']['W_edges_est']:<20}"
    print(edges_row)

    print()

    # A metrics
    print(f"{'A (Inter-slice):':<25}")

    tpr_row = f"{'  TPR':<25}"
    for method in methods:
        tpr_row += f" {results[method]['metrics']['A_tpr']:<20.4f}"
    print(tpr_row)

    fdr_row = f"{'  FDR':<25}"
    for method in methods:
        fdr_row += f" {results[method]['metrics']['A_fdr']:<20.4f}"
    print(fdr_row)

    shd_row = f"{'  SHD':<25}"
    for method in methods:
        shd_row += f" {results[method]['metrics']['A_shd']:<20}"
    print(shd_row)

    true_a_edges = results[first_method]['metrics']['A_edges_true']
    edges_row = f"{'  Edges (true=' + str(true_a_edges) + ')':<25}"
    for method in methods:
        edges_row += f" {results[method]['metrics']['A_edges_est']:<20}"
    print(edges_row)

    print("\n" + "="*80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare causal discovery methods')
    parser.add_argument('--methods', nargs='+', default=['dycolide', 'dynotears'],
                        choices=['dycolide', 'dynotears'],
                        help='Methods to run (default: all)')
    parser.add_argument('--n_nodes', type=int, default=30, help='Number of nodes')
    parser.add_argument('--n_timesteps', type=int, default=3000, help='Number of timesteps')
    parser.add_argument('--lag_order', type=int, default=1, help='Lag order')
    parser.add_argument('--instantaneous_edges', type=int, default=60, help='Number of instantaneous edges')
    parser.add_argument('--temporal_sparsity', type=float, default=0.15, help='Temporal sparsity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.08, help='Edge threshold')

    args = parser.parse_args()

    results = run_comparison(
        n_nodes=args.n_nodes,
        n_timesteps=args.n_timesteps,
        lag_order=args.lag_order,
        instantaneous_edges=args.instantaneous_edges,
        temporal_sparsity=args.temporal_sparsity,
        seed=args.seed,
        threshold=args.threshold,
        methods=args.methods
    )
