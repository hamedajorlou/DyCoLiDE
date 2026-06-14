"""
Baseline Comparison Script for Static DAG Learning

Compares:
- Baselines: NOTEARS, GOLEM, DAGMA, PC, DirectLiNGAM
- Static CoLiDE (full batch)
- DyCoLiDE (mini-batch SGD) - can handle streaming/batch settings

This script demonstrates that DyCoLiDE achieves competitive performance
while supporting flexible batch sizes for online/streaming scenarios.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
from Utils import simulate_sem, count_accuracy

# Try importing baseline methods
AVAILABLE_METHODS = {}

# NOTEARS
try:
    from notears import notears_linear
    AVAILABLE_METHODS['NOTEARS'] = True
except ImportError:
    try:
        # Alternative: use local implementation or gcastle
        from castle.algorithms import Notears
        AVAILABLE_METHODS['NOTEARS'] = 'castle'
    except ImportError:
        AVAILABLE_METHODS['NOTEARS'] = False

# GOLEM
try:
    from golem import golem
    AVAILABLE_METHODS['GOLEM'] = True
except ImportError:
    AVAILABLE_METHODS['GOLEM'] = False

# DAGMA
try:
    from dagma.linear import DagmaLinear
    AVAILABLE_METHODS['DAGMA'] = True
except ImportError:
    AVAILABLE_METHODS['DAGMA'] = False

# PC Algorithm
try:
    from causallearn.search.ConstraintBased.PC import pc
    AVAILABLE_METHODS['PC'] = True
except ImportError:
    AVAILABLE_METHODS['PC'] = False

# DirectLiNGAM
try:
    import lingam
    AVAILABLE_METHODS['DirectLiNGAM'] = True
except ImportError:
    AVAILABLE_METHODS['DirectLiNGAM'] = False

# CoLiDE (always available - local)
from SEM.dycolide import colide_ev as StaticCoLiDE
AVAILABLE_METHODS['CoLiDE'] = True

# DyCoLiDE (our batch/online implementation)
from Colide_batch_MOHEM import colide_ev_batch as DyCoLiDE
AVAILABLE_METHODS['DyCoLiDE'] = True


def run_notears(X, lambda1=0.1):
    """Run NOTEARS algorithm."""
    if AVAILABLE_METHODS['NOTEARS'] == 'castle':
        from castle.algorithms import Notears
        model = Notears()
        model.learn(X)
        return model.causal_matrix
    else:
        from notears import notears_linear
        W_est = notears_linear.notears_linear(X, lambda1=lambda1, loss_type='l2')
        return W_est


def run_golem(X, lambda1=0.02, lambda2=5.0, equal_variances=False):
    """Run GOLEM algorithm."""
    from golem import golem
    W_est = golem(X, lambda_1=lambda1, lambda_2=lambda2,
                  equal_variances=equal_variances)
    return W_est


def run_dagma(X, lambda1=0.02):
    """Run DAGMA algorithm."""
    from dagma.linear import DagmaLinear
    model = DagmaLinear(loss_type='l2')
    W_est = model.fit(X, lambda1=lambda1)
    return W_est


def run_pc(X, alpha=0.05):
    """Run PC algorithm."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    cg = pc(X, alpha=alpha, indep_test=fisherz)
    # Convert to adjacency matrix
    W_est = cg.G.graph
    # PC returns CPDAG, convert to DAG-like adjacency
    # -1 means undirected, 1 means directed edge
    W_binary = np.zeros_like(W_est, dtype=float)
    n = W_est.shape[0]
    for i in range(n):
        for j in range(n):
            if W_est[i, j] == -1 and W_est[j, i] == 1:
                W_binary[j, i] = 1.0  # i -> j
            elif W_est[i, j] == -1 and W_est[j, i] == -1:
                # Undirected edge - use arbitrary direction
                if i < j:
                    W_binary[i, j] = 1.0
    return W_binary


def run_directlingam(X):
    """Run DirectLiNGAM algorithm."""
    import lingam
    model = lingam.DirectLiNGAM()
    model.fit(X)
    return model.adjacency_matrix_


def run_colide(X, lambda1=0.02, seed=42):
    """Run static CoLiDE (full batch)."""
    model = StaticCoLiDE(seed=seed)
    W_est, sigma_est = model.fit(
        X=X,
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
    return W_est


def run_dycolide(X, lambda1=0.02, batch_size=200, seed=42):
    """Run DyCoLiDE-EV (mini-batch SGD)."""
    model = DyCoLiDE(seed=seed)
    W_est, sigma_est = model.fit(
        X=X,
        lambda1=lambda1,
        T=4,
        mu_init=1.0,
        mu_factor=0.1,
        s=[1.0, 0.9, 0.8, 0.7],
        batch_size=batch_size,
        n_batches_warm=5000,
        n_batches_final=15000,
        lr=0.0003,
        checkpoint=500
    )
    return W_est


def run_comparison(n_nodes=50, n_samples=1000, edges=100, seed=42, threshold=0.3):
    """Run comparison across all available methods."""

    print("="*100)
    print(f"BASELINE COMPARISON: Static DAG Learning ({n_nodes} nodes, {edges} edges)")
    print("="*100)

    # Generate data (Equal Variance case)
    print("\nGenerating data (Equal Variance, var=1)...")
    X, W_true, _ = simulate_sem(
        n_nodes=n_nodes,
        n_samples=n_samples,
        edges=edges,
        graph_type='er',
        edge_type='weighted',
        var_type='ev',
        noise='normal',
        var=1.0,
        w_range=((-2.0, -0.5), (0.5, 2.0)),
        seed=seed
    )

    W_true_binary = (W_true != 0).astype(float)
    true_edges = int(np.sum(W_true_binary))
    print(f"Data: {n_nodes} nodes, {n_samples} samples, {true_edges} true edges")

    # Print available methods
    print("\nMethod availability:")
    for method, available in AVAILABLE_METHODS.items():
        status = "Available" if available else "Not installed"
        print(f"  {method}: {status}")

    results = {}

    # Run each available method
    methods_to_run = [
        ('NOTEARS', lambda: run_notears(X, lambda1=0.1), 0.3),
        ('GOLEM', lambda: run_golem(X, lambda1=0.02), 0.3),
        ('DAGMA', lambda: run_dagma(X, lambda1=0.02), 0.3),
        ('PC', lambda: run_pc(X, alpha=0.01), 0.5),
        ('DirectLiNGAM', lambda: run_directlingam(X), 0.3),
        ('CoLiDE', lambda: run_colide(X, lambda1=0.02, seed=seed), threshold),
        ('DyCoLiDE-B200', lambda: run_dycolide(X, lambda1=0.02, batch_size=200, seed=seed), threshold),
        ('DyCoLiDE-B50', lambda: run_dycolide(X, lambda1=0.02, batch_size=50, seed=seed), threshold),
    ]

    for method_name, run_fn, thresh in methods_to_run:
        base_method = method_name.split('-')[0]
        if base_method not in AVAILABLE_METHODS or not AVAILABLE_METHODS[base_method]:
            print(f"\n[SKIP] {method_name}: Not available")
            continue

        print(f"\n[RUN] {method_name}...")
        try:
            start = time.time()
            W_est = run_fn()
            elapsed = time.time() - start

            # Threshold and compute metrics
            W_est_binary = (np.abs(W_est) > thresh).astype(float)
            shd, tpr, fdr = count_accuracy(W_true_binary, W_est_binary)
            est_edges = int(np.sum(W_est_binary))

            results[method_name] = {
                'time': elapsed,
                'tpr': tpr,
                'fdr': fdr,
                'shd': shd,
                'edges': est_edges,
                'threshold': thresh
            }
            print(f"  Completed in {elapsed:.2f}s | TPR: {tpr:.4f} | FDR: {fdr:.4f} | SHD: {shd}")

        except Exception as e:
            print(f"  [ERROR] {method_name}: {str(e)[:100]}")
            results[method_name] = {'error': str(e)}

    # Print summary table
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"\nTrue edges: {true_edges}")
    print(f"\n{'Method':<20} {'Time (s)':>10} {'TPR':>10} {'FDR':>10} {'SHD':>8} {'Edges':>8} {'Threshold':>10}")
    print("-"*100)

    for method_name in ['NOTEARS', 'GOLEM', 'DAGMA', 'PC', 'DirectLiNGAM', 'CoLiDE', 'DyCoLiDE-B200', 'DyCoLiDE-B50']:
        if method_name not in results:
            continue
        r = results[method_name]
        if 'error' in r:
            print(f"{method_name:<20} {'ERROR':>10}")
        else:
            print(f"{method_name:<20} {r['time']:>10.2f} {r['tpr']:>10.4f} {r['fdr']:>10.4f} {r['shd']:>8} {r['edges']:>8} {r['threshold']:>10.2f}")

    print("="*100)

    # Highlight DyCoLiDE advantages
    print("\nKEY FINDINGS:")
    print("-"*60)
    if 'DyCoLiDE-B200' in results and 'error' not in results['DyCoLiDE-B200']:
        print("* DyCoLiDE supports flexible batch sizes (1 to n)")
        print("* Can process data in streaming/online fashion")
        if 'CoLiDE' in results and 'error' not in results['CoLiDE']:
            tpr_diff = results['DyCoLiDE-B200']['tpr'] - results['CoLiDE']['tpr']
            fdr_diff = results['DyCoLiDE-B200']['fdr'] - results['CoLiDE']['fdr']
            print(f"* vs Static CoLiDE: TPR diff={tpr_diff:+.4f}, FDR diff={fdr_diff:+.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare DAG learning methods')
    parser.add_argument('--n_nodes', type=int, default=50, help='Number of nodes')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--edges', type=int, default=100, help='Number of edges (ER graph)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold', type=float, default=0.3, help='Edge threshold for CoLiDE methods')

    args = parser.parse_args()

    results = run_comparison(
        n_nodes=args.n_nodes,
        n_samples=args.n_samples,
        edges=args.edges,
        seed=args.seed,
        threshold=args.threshold
    )
