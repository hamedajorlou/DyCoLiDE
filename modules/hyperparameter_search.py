"""
Hyperparameter optimization for DyCoLiDE-EV.
Target: 30 nodes, 60 edges, 1000 timesteps
Goal: Maximize F1 score for both W and A
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
from itertools import product
from SVAR.dycolide import DyCoLiDE_EV
from Utils import count_accuracy_svar
from Utils import generate_svar_data

def compute_f1(tpr, fdr):
    """Compute F1 score from TPR and FDR."""
    precision = 1 - fdr
    recall = tpr
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def run_single_config(X, B_true, A_true, params_ev, threshold, seed):
    """Run DyCoLiDE with specific hyperparameters."""
    model = DyCoLiDE_EV(seed=seed)
    start = time.time()
    W_est, A_est, _ = model.fit(X, **params_ev)
    elapsed = time.time() - start

    metrics = count_accuracy_svar(B_true, W_est, A_true, A_est, threshold=threshold)

    w_f1 = compute_f1(metrics['W_tpr'], metrics['W_fdr'])
    a_f1 = compute_f1(metrics['A_tpr'], metrics['A_fdr'])

    return {
        'W_tpr': metrics['W_tpr'],
        'W_fdr': metrics['W_fdr'],
        'W_f1': w_f1,
        'A_tpr': metrics['A_tpr'],
        'A_fdr': metrics['A_fdr'],
        'A_f1': a_f1,
        'total_f1': (w_f1 + a_f1) / 2,
        'time': elapsed
    }

def main():
    # Fixed settings
    n_nodes = 30
    n_edges = 60
    n_timesteps = 1000
    lag_order = 1
    seed = 42

    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION FOR DyCoLiDE-EV")
    print("="*80)
    print(f"\nFixed settings:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {n_edges}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Lag order: {lag_order}")

    # Generate data once
    print("\nGenerating data...")
    X, B_true, A_list_true, _ = generate_svar_data(
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

    print(f"  True W edges: {np.sum(np.abs(B_true_dycolide) > 0)}")
    print(f"  True A edges: {np.sum(np.abs(A_true) > 0)}")

    # Hyperparameter grid
    param_grid = {
        'lambda_W': [0.005, 0.01, 0.02],
        'lambda_A': [0.001, 0.005, 0.01, 0.02],
        'threshold': [0.03, 0.05, 0.08, 0.1],
        'lr': [0.0001, 0.0003, 0.001],
    }

    # Fixed params that we won't grid search
    fixed_params = {
        'p': lag_order,
        'T': 4,
        'mu_init': 1.0,
        'mu_factor': 0.1,
        's': [1.0, 0.9, 0.8, 0.7],
        'warm_iter': 10000,
        'max_iter': 20000,
        'checkpoint': 5000
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    print(f"\nTotal configurations to test: {len(combinations)}")
    print("\n" + "-"*80)

    results = []
    best_result = None
    best_total_f1 = -1

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        threshold = config.pop('threshold')

        params_ev = {**fixed_params, **config}

        print(f"\n[{i+1}/{len(combinations)}] Testing: lambda_W={config['lambda_W']}, lambda_A={config['lambda_A']}, lr={config['lr']}, threshold={threshold}")

        try:
            result = run_single_config(X, B_true_dycolide, A_true, params_ev, threshold, seed)
            result['config'] = {**config, 'threshold': threshold}
            results.append(result)

            print(f"  W: TPR={result['W_tpr']:.3f}, FDR={result['W_fdr']:.3f}, F1={result['W_f1']:.3f}")
            print(f"  A: TPR={result['A_tpr']:.3f}, FDR={result['A_fdr']:.3f}, F1={result['A_f1']:.3f}")
            print(f"  Total F1={result['total_f1']:.3f}, Time={result['time']:.1f}s")

            if result['total_f1'] > best_total_f1:
                best_total_f1 = result['total_f1']
                best_result = result
                print(f"  *** NEW BEST ***")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Print best results
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"\nHyperparameters:")
    for k, v in best_result['config'].items():
        print(f"  {k}: {v}")
    print(f"\nResults:")
    print(f"  W: TPR={best_result['W_tpr']:.3f}, FDR={best_result['W_fdr']:.3f}, F1={best_result['W_f1']:.3f}")
    print(f"  A: TPR={best_result['A_tpr']:.3f}, FDR={best_result['A_fdr']:.3f}, F1={best_result['A_f1']:.3f}")
    print(f"  Total F1={best_result['total_f1']:.3f}")
    print(f"  Time={best_result['time']:.1f}s")

    # Print top 5 configurations
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS")
    print("="*80)
    sorted_results = sorted(results, key=lambda x: x['total_f1'], reverse=True)[:5]

    for i, r in enumerate(sorted_results):
        print(f"\n{i+1}. Total F1={r['total_f1']:.3f} (W_F1={r['W_f1']:.3f}, A_F1={r['A_f1']:.3f})")
        print(f"   Config: lambda_W={r['config']['lambda_W']}, lambda_A={r['config']['lambda_A']}, "
              f"lr={r['config']['lr']}, threshold={r['config']['threshold']}")

if __name__ == "__main__":
    main()
