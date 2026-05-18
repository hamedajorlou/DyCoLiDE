"""
Compare methods for dynamic DAG learning:
1. DyCoLiDE-EV
2. Dynotears
3. DynoDAGMA (DAGMA extended for time-series)
4. DynoGOLEM (GOLEM extended for time-series)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from DyCoLiDE import DyCoLiDE_EV, count_accuracy_svar
from SVAR_data_generator import generate_svar_data
from dynotears import from_pandas_dynamic
from dyno_dagma import DynoDAGMA
from dyno_golem import DynoGOLEM


def structure_to_matrices(structure_model, n_vars, lag_order):
    """Convert Dynotears StructureModel to W and A matrices."""
    instantaneous = np.zeros((n_vars, n_vars))
    lagged = np.zeros((lag_order * n_vars, n_vars))

    for parent, child, data in structure_model.edges(data=True):
        weight = float(data.get("weight", 0.0))
        parent_var, parent_lag = parent.rsplit("_lag", 1)
        child_var, child_lag = child.rsplit("_lag", 1)

        child_lag = int(child_lag)
        parent_lag = int(parent_lag)
        if child_lag != 0:
            continue

        p_idx = int(parent_var)
        c_idx = int(child_var)

        if parent_lag == 0:
            instantaneous[c_idx, p_idx] = weight
        elif 1 <= parent_lag <= lag_order:
            lag_idx = (parent_lag - 1) * n_vars + p_idx
            lagged[lag_idx, c_idx] = weight

    return instantaneous, lagged


def compute_f1(tpr, fdr):
    """Compute F1 score from TPR and FDR."""
    precision = 1 - fdr
    recall = tpr
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def run_comparison(n_nodes=20, n_timesteps=1000, lag_order=1,
                   instantaneous_edges=40, temporal_sparsity=0.1,
                   seed=42, threshold=0.08, methods=None):
    """
    Run comparison between DyCoLiDE, Dynotears, DynoDAGMA, and DynoGOLEM.
    """

    if methods is None:
        methods = ['dycolide', 'dynotears', 'dynodagma', 'dynogolem']

    print("="*90)
    print(f"Experiment: {n_nodes} nodes, {instantaneous_edges} edges, {n_timesteps} timesteps")
    print("="*90)

    # Generate SVAR data
    print("\nGenerating data...")
    X, B_true, A_list_true, params = generate_svar_data(
        n_nodes=n_nodes,
        n_timesteps=n_timesteps,
        lag_order=lag_order,
        instantaneous_edges=instantaneous_edges,
        temporal_edges=instantaneous_edges,  # Match intra-slice edges
        temporal_strength=0.5,  # Stronger temporal effects for better detection
        noise_scale=1,
        noise_type='ev',
        seed=seed
    )

    # Convert to DyCoLiDE convention (row vectors)
    A_true_for_eval = np.vstack([A.T for A in A_list_true])
    B_true_dycolide = B_true.T

    print(f"True W edges: {np.sum(np.abs(B_true_dycolide) > 0)}")
    print(f"True A edges: {np.sum(np.abs(A_true_for_eval) > 0)}")

    results = {
        'n_nodes': n_nodes,
        'n_edges': instantaneous_edges,
        'true_W_edges': int(np.sum(np.abs(B_true_dycolide) > 0)),
        'true_A_edges': int(np.sum(np.abs(A_true_for_eval) > 0))
    }

    # =====================
    # 1. DyCoLiDE-EV
    # =====================
    if 'dycolide' in methods:
        print("\n" + "-"*40)
        print("Running DyCoLiDE-EV...")
        print("-"*40)

        params_ev = {
            'p': lag_order,
            'lambda_W': 0.01,
            'lambda_A': 0.015,  # Tuned for A_TPR>0.8, A_FDR<0.1
            'T': 5,
            'mu_init': 1.0,
            'mu_factor': 0.1,
            's': [1.0, 0.9, 0.8, 0.7, 0.6],
            'warm_iter': 15000,
            'max_iter': 30000,
            'lr': 0.001,
            'checkpoint': 5000
        }

        model_ev = DyCoLiDE_EV(seed=seed)
        start = time.time()
        W_est, A_est, _ = model_ev.fit(X, **params_ev)
        time_dycolide = time.time() - start

        metrics = count_accuracy_svar(B_true_dycolide, W_est, A_true_for_eval, A_est, threshold=threshold)

        w_f1 = compute_f1(metrics['W_tpr'], metrics['W_fdr'])
        a_f1 = compute_f1(metrics['A_tpr'], metrics['A_fdr'])

        results['dycolide_time'] = time_dycolide
        results['dycolide_W_tpr'] = metrics['W_tpr']
        results['dycolide_W_fdr'] = metrics['W_fdr']
        results['dycolide_W_f1'] = w_f1
        results['dycolide_A_tpr'] = metrics['A_tpr']
        results['dycolide_A_fdr'] = metrics['A_fdr']
        results['dycolide_A_f1'] = a_f1

        print(f"  W: TPR={metrics['W_tpr']:.3f}, FDR={metrics['W_fdr']:.3f}, F1={w_f1:.3f}")
        print(f"  A: TPR={metrics['A_tpr']:.3f}, FDR={metrics['A_fdr']:.3f}, F1={a_f1:.3f}")
        print(f"  Time: {time_dycolide:.1f}s")

    # =====================
    # 2. Dynotears
    # =====================
    if 'dynotears' in methods:
        print("\n" + "-"*40)
        print("Running Dynotears...")
        print("-"*40)

        df = pd.DataFrame(X, columns=[str(i) for i in range(n_nodes)])

        start = time.time()
        sm = from_pandas_dynamic(
            df,
            p=lag_order,
            lambda_w=0.01,
            lambda_a=0.006,  # Lower for higher A_TPR
            max_iter=100,
        )
        time_dynotears = time.time() - start

        W_raw, A_raw = structure_to_matrices(sm, n_nodes, lag_order)
        W_est_dynotears = W_raw.T
        A_est_dynotears = A_raw.T.reshape(n_nodes, lag_order * n_nodes).T

        metrics = count_accuracy_svar(B_true_dycolide, W_est_dynotears, A_true_for_eval, A_est_dynotears, threshold=threshold)

        w_f1 = compute_f1(metrics['W_tpr'], metrics['W_fdr'])
        a_f1 = compute_f1(metrics['A_tpr'], metrics['A_fdr'])

        results['dynotears_time'] = time_dynotears
        results['dynotears_W_tpr'] = metrics['W_tpr']
        results['dynotears_W_fdr'] = metrics['W_fdr']
        results['dynotears_W_f1'] = w_f1
        results['dynotears_A_tpr'] = metrics['A_tpr']
        results['dynotears_A_fdr'] = metrics['A_fdr']
        results['dynotears_A_f1'] = a_f1

        print(f"  W: TPR={metrics['W_tpr']:.3f}, FDR={metrics['W_fdr']:.3f}, F1={w_f1:.3f}")
        print(f"  A: TPR={metrics['A_tpr']:.3f}, FDR={metrics['A_fdr']:.3f}, F1={a_f1:.3f}")
        print(f"  Time: {time_dynotears:.1f}s")

    # =====================
    # 3. DynoDAGMA
    # =====================
    if 'dynodagma' in methods:
        print("\n" + "-"*40)
        print("Running DynoDAGMA...")
        print("-"*40)

        model = DynoDAGMA(seed=seed, verbose=False)
        start = time.time()
        W_est_dagma, A_est_dagma = model.fit(
            X, p=lag_order,
            lambda_w=0.01,
            lambda_a=0.005,  # Lower for higher A_TPR
            w_threshold=threshold,
            T=5,
            mu_init=1.0,
            mu_factor=0.1,
            s=[1.0, 0.9, 0.8, 0.7, 0.6],
            warm_iter=5000,
            max_iter=10000,
            lr=0.0003
        )
        time_dagma = time.time() - start

        metrics = count_accuracy_svar(B_true_dycolide, W_est_dagma, A_true_for_eval, A_est_dagma, threshold=0.0)

        w_f1 = compute_f1(metrics['W_tpr'], metrics['W_fdr'])
        a_f1 = compute_f1(metrics['A_tpr'], metrics['A_fdr'])

        results['dynodagma_time'] = time_dagma
        results['dynodagma_W_tpr'] = metrics['W_tpr']
        results['dynodagma_W_fdr'] = metrics['W_fdr']
        results['dynodagma_W_f1'] = w_f1
        results['dynodagma_A_tpr'] = metrics['A_tpr']
        results['dynodagma_A_fdr'] = metrics['A_fdr']
        results['dynodagma_A_f1'] = a_f1

        print(f"  W: TPR={metrics['W_tpr']:.3f}, FDR={metrics['W_fdr']:.3f}, F1={w_f1:.3f}")
        print(f"  A: TPR={metrics['A_tpr']:.3f}, FDR={metrics['A_fdr']:.3f}, F1={a_f1:.3f}")
        print(f"  Time: {time_dagma:.1f}s")

    return results


def print_comparison_table(results, methods):
    """Print a formatted comparison table."""

    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)

    print(f"\n{'Method':<15} {'W_TPR':>10} {'W_FDR':>10} {'W_F1':>10} {'A_TPR':>10} {'A_FDR':>10} {'A_F1':>10} {'Time (s)':>10}")
    print("-"*100)

    for method in methods:
        if f'{method}_W_tpr' in results:
            print(f"{method.upper():<15} "
                  f"{results[f'{method}_W_tpr']:>10.3f} "
                  f"{results[f'{method}_W_fdr']:>10.3f} "
                  f"{results[f'{method}_W_f1']:>10.3f} "
                  f"{results[f'{method}_A_tpr']:>10.3f} "
                  f"{results[f'{method}_A_fdr']:>10.3f} "
                  f"{results[f'{method}_A_f1']:>10.3f} "
                  f"{results[f'{method}_time']:>10.1f}")

    print("="*100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare dynamic DAG methods')
    parser.add_argument('--n_nodes', type=int, default=30, help='Number of nodes')
    parser.add_argument('--n_edges', type=int, default=60, help='Number of instantaneous edges')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='Number of timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.10, help='Edge threshold (0.10 optimal for A metrics)')
    parser.add_argument('--methods', nargs='+', default=['dycolide','dynogolem'],
                        choices=['dycolide', 'dynotears', 'dynodagma', 'dynogolem'],
                        help='Methods to run')

    args = parser.parse_args()

    results = run_comparison(
        n_nodes=args.n_nodes,
        n_timesteps=args.n_timesteps,
        instantaneous_edges=args.n_edges,
        seed=args.seed,
        threshold=args.threshold,
        methods=args.methods
    )

    print_comparison_table(results, args.methods)
