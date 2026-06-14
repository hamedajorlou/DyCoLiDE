"""
Run experiments comparing DyCoLiDE-EV vs Dynotears across different graph sizes.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from SVAR.dycolide import DyCoLiDE_EV
from Utils import count_accuracy_svar
from Utils import generate_svar_data
from dynotears import from_pandas_dynamic


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


def run_single_experiment(n_nodes, n_edges, n_timesteps, lag_order, seed, threshold, methods):
    """Run experiment for a single configuration."""

    # Generate data
    X, B_true, A_list_true, params = generate_svar_data(
        n_nodes=n_nodes,
        n_timesteps=n_timesteps,
        lag_order=lag_order,
        instantaneous_edges=n_edges,
        temporal_sparsity=0.1,
        temporal_strength=0.2,
        noise_scale=1,
        noise_type='ev',
        seed=seed
    )

    A_true = np.vstack([A.T for A in A_list_true])
    B_true_dycolide = B_true.T

    results = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'true_W_edges': int(np.sum(np.abs(B_true_dycolide) > 0)),
        'true_A_edges': int(np.sum(np.abs(A_true) > 0))
    }

    # Run DyCoLiDE
    if 'dycolide' in methods:
        params_ev = {
            'p': lag_order,
            'lambda_W': 0.01,
            'lambda_A': 0.03,
            'T': 4,
            'mu_init': 1.0,
            'mu_factor': 0.1,
            's': [1.0, 0.9, 0.8, 0.7],
            'warm_iter': 15000,
            'max_iter': 30000,
            'lr': 0.0001,
            'checkpoint': 1000
        }

        model_ev = DyCoLiDE_EV(seed=seed)
        start = time.time()
        W_est, A_est, _ = model_ev.fit(X, **params_ev)
        time_dycolide = time.time() - start

        metrics = count_accuracy_svar(B_true_dycolide, W_est, A_true, A_est, threshold=threshold)
        results['dycolide_time'] = time_dycolide
        results['dycolide_W_tpr'] = metrics['W_tpr']
        results['dycolide_W_fdr'] = metrics['W_fdr']
        # Compute F1 score: F1 = 2 * precision * recall / (precision + recall)
        w_precision = 1 - metrics['W_fdr']
        w_recall = metrics['W_tpr']
        results['dycolide_W_f1'] = 2 * w_precision * w_recall / (w_precision + w_recall) if (w_precision + w_recall) > 0 else 0
        results['dycolide_A_tpr'] = metrics['A_tpr']
        results['dycolide_A_fdr'] = metrics['A_fdr']
        a_precision = 1 - metrics['A_fdr']
        a_recall = metrics['A_tpr']
        results['dycolide_A_f1'] = 2 * a_precision * a_recall / (a_precision + a_recall) if (a_precision + a_recall) > 0 else 0

    # Run Dynotears
    if 'dynotears' in methods:
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

        W_raw, A_raw = structure_to_matrices(sm, n_nodes, lag_order)
        W_est_dynotears = W_raw.T
        A_est_dynotears = A_raw.T.reshape(n_nodes, lag_order * n_nodes).T

        metrics = count_accuracy_svar(B_true_dycolide, W_est_dynotears, A_true, A_est_dynotears, threshold=threshold)
        results['dynotears_time'] = time_dynotears
        results['dynotears_W_tpr'] = metrics['W_tpr']
        results['dynotears_W_fdr'] = metrics['W_fdr']
        # Compute F1 score: F1 = 2 * precision * recall / (precision + recall)
        w_precision = 1 - metrics['W_fdr']
        w_recall = metrics['W_tpr']
        results['dynotears_W_f1'] = 2 * w_precision * w_recall / (w_precision + w_recall) if (w_precision + w_recall) > 0 else 0
        results['dynotears_A_tpr'] = metrics['A_tpr']
        results['dynotears_A_fdr'] = metrics['A_fdr']
        a_precision = 1 - metrics['A_fdr']
        a_recall = metrics['A_tpr']
        results['dynotears_A_f1'] = 2 * a_precision * a_recall / (a_precision + a_recall) if (a_precision + a_recall) > 0 else 0

    return results


def run_experiments(node_sizes=[10, 20, 30], edge_multiplier=2, n_timesteps=3000,
                    lag_order=1, seed=42, threshold=0.08, methods=['dycolide', 'dynotears']):
    """Run experiments across multiple node sizes."""

    all_results = []

    print("="*100)
    print("EXPERIMENT: DyCoLiDE-EV vs Dynotears Comparison")
    print("="*100)
    print(f"\nSettings:")
    print(f"  Node sizes: {node_sizes}")
    print(f"  Edge multiplier: {edge_multiplier}x nodes")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Lag order: {lag_order}")
    print(f"  Threshold: {threshold}")
    print(f"  Methods: {methods}")
    print()

    for n_nodes in node_sizes:
        n_edges = edge_multiplier * n_nodes

        print("-"*100)
        print(f"Running experiment: {n_nodes} nodes, {n_edges} edges")
        print("-"*100)

        results = run_single_experiment(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_timesteps=n_timesteps,
            lag_order=lag_order,
            seed=seed,
            threshold=threshold,
            methods=methods
        )

        all_results.append(results)

        # Print intermediate results
        print(f"\n  True W edges: {results['true_W_edges']}, True A edges: {results['true_A_edges']}")
        if 'dycolide' in methods:
            print(f"  DyCoLiDE:  W_TPR={results['dycolide_W_tpr']:.3f}, W_FDR={results['dycolide_W_fdr']:.3f}, W_F1={results['dycolide_W_f1']:.3f}, "
                  f"A_TPR={results['dycolide_A_tpr']:.3f}, A_FDR={results['dycolide_A_fdr']:.3f}, A_F1={results['dycolide_A_f1']:.3f}, "
                  f"Time={results['dycolide_time']:.1f}s")
        if 'dynotears' in methods:
            print(f"  Dynotears: W_TPR={results['dynotears_W_tpr']:.3f}, W_FDR={results['dynotears_W_fdr']:.3f}, W_F1={results['dynotears_W_f1']:.3f}, "
                  f"A_TPR={results['dynotears_A_tpr']:.3f}, A_FDR={results['dynotears_A_fdr']:.3f}, A_F1={results['dynotears_A_f1']:.3f}, "
                  f"Time={results['dynotears_time']:.1f}s")
        print()

    return all_results


def print_results_table(results, methods=['dycolide', 'dynotears']):
    """Print a formatted comparison table."""

    print("\n" + "="*120)
    print("RESULTS TABLE")
    print("="*120)

    # Header
    print(f"\n{'Nodes':<8} {'Edges':<8}", end="")
    for method in methods:
        print(f" | {method.upper():<50}", end="")
    print()

    print(f"{'':8} {'':8}", end="")
    for method in methods:
        print(f" | {'W_TPR':>8} {'W_FDR':>8} {'W_F1':>8} {'A_TPR':>8} {'A_FDR':>8} {'A_F1':>8} {'Time':>8}", end="")
    print()
    print("-"*140)

    # Data rows
    for r in results:
        print(f"{r['n_nodes']:<8} {r['true_W_edges']:<8}", end="")
        for method in methods:
            prefix = method
            print(f" | {r[f'{prefix}_W_tpr']:>8.3f} {r[f'{prefix}_W_fdr']:>8.3f} {r[f'{prefix}_W_f1']:>8.3f} "
                  f"{r[f'{prefix}_A_tpr']:>8.3f} {r[f'{prefix}_A_fdr']:>8.3f} {r[f'{prefix}_A_f1']:>8.3f} {r[f'{prefix}_time']:>7.1f}s", end="")
        print()

    print("="*140)

    # Summary statistics
    print("\n" + "="*90)
    print("SUMMARY (Average across all node sizes)")
    print("="*90)

    print(f"\n{'Method':<15} {'W_TPR':>10} {'W_FDR':>10} {'W_F1':>10} {'A_TPR':>10} {'A_FDR':>10} {'A_F1':>10} {'Time (s)':>10}")
    print("-"*90)

    for method in methods:
        avg_w_tpr = np.mean([r[f'{method}_W_tpr'] for r in results])
        avg_w_fdr = np.mean([r[f'{method}_W_fdr'] for r in results])
        avg_w_f1 = np.mean([r[f'{method}_W_f1'] for r in results])
        avg_a_tpr = np.mean([r[f'{method}_A_tpr'] for r in results])
        avg_a_fdr = np.mean([r[f'{method}_A_fdr'] for r in results])
        avg_a_f1 = np.mean([r[f'{method}_A_f1'] for r in results])
        avg_time = np.mean([r[f'{method}_time'] for r in results])

        print(f"{method.upper():<15} {avg_w_tpr:>10.3f} {avg_w_fdr:>10.3f} {avg_w_f1:>10.3f} "
              f"{avg_a_tpr:>10.3f} {avg_a_fdr:>10.3f} {avg_a_f1:>10.3f} {avg_time:>10.1f}")

    print("="*90)


def save_results_csv(results, filename='experiment_results.csv'):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('--node_sizes', nargs='+', type=int, default=[30],
                        help='Node sizes to test')
    parser.add_argument('--edge_multiplier', type=int, default=2,
                        help='Edge count = multiplier * nodes')
    parser.add_argument('--n_timesteps', type=int, default=1000,
                        help='Number of timesteps')
    parser.add_argument('--methods', nargs='+', default=['dycolide'],
                        choices=['dycolide', 'dynotears'],
                        help='Methods to compare')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.05, help='Edge threshold')
    parser.add_argument('--save_csv', action='store_true', help='Save results to CSV')

    args = parser.parse_args()

    results = run_experiments(
        node_sizes=args.node_sizes,
        edge_multiplier=args.edge_multiplier,
        n_timesteps=args.n_timesteps,
        methods=args.methods,
        seed=args.seed,
        threshold=args.threshold
    )

    print_results_table(results, methods=args.methods)

    if args.save_csv:
        save_results_csv(results)
