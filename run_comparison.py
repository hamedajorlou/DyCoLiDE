"""
Comprehensive Comparison: GOLEM vs DAGMA vs CoLiDE vs DyCoLiDE

Configuration:
- 200 nodes, ER4 graph (800 edges), 1000 samples
- 10 runs for mean ± std
- EV case: two sigma values (σ=1, σ=5)
- NV case: single table
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import time

sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears/dagma/src/dagma')

from Utils import simulate_sem, to_bin, count_accuracy
from Colide import colide_ev, colide_ev_batch, colide_nv, colide_nv_batch_cov
from golem_sem import GOLEM_SEM
from linear import DagmaLinear

# =============================================================================
# CONFIGURATION
# =============================================================================

N_RUNS = 1
N_NODES = 200
N_SAMPLES = 1000
EDGES = 800  # ER4
GRAPH_TYPE = 'er'
THRESHOLD = 0.3

# Sigma values for EV case
SIGMA_VALUES = [1.0, 5.0]

# Model configs
GOLEM_CONFIG = {
    'lambda1': 0.02,
    'lambda2': 5.0,
    'num_iter': 50000,
    'lr': 0.001,
}

DAGMA_CONFIG = {
    'lambda1': 0.02,
    'T': 5,
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7, 0.6],
    'warm_iter': 20000,
    'max_iter': 70000,
    'lr': 0.0003,
}

COLIDE_CONFIG = {
    'lambda1': 0.05,
    'T': 4,
    'mu_init': 1.0,
    'mu_factor': 0.1,
    's': [1.0, 0.9, 0.8, 0.7],
    'warm_iter': 20000,
    'max_iter': 70000,
    'lr': 0.0003,
}

DYCOLIDE_CONFIG = {
    'lambda1': 0.05,
    'T': 4,
    'batch_size': 200,
    'n_batches_warm': 20000,
    'n_batches_final': 70000,
    'lr': 0.0003,
}

# =============================================================================
# MODEL RUNNERS
# =============================================================================

def run_golem_ev(X, seed):
    model = GOLEM_SEM(seed=seed, verbose=False)
    W_est = model.fit(
        X, lambda1=GOLEM_CONFIG['lambda1'], lambda2=GOLEM_CONFIG['lambda2'],
        equal_variances=True, num_iter=GOLEM_CONFIG['num_iter'],
        lr=GOLEM_CONFIG['lr'], w_threshold=0.0
    )
    return W_est

def run_golem_nv(X, seed):
    model = GOLEM_SEM(seed=seed, verbose=False)
    W_est = model.fit(
        X, lambda1=0.002, lambda2=GOLEM_CONFIG['lambda2'],
        equal_variances=False, num_iter=GOLEM_CONFIG['num_iter'],
        lr=GOLEM_CONFIG['lr'], w_threshold=0.0
    )
    return W_est

def run_dagma(X):
    model = DagmaLinear(loss_type='l2', verbose=False)
    W_est = model.fit(
        X, lambda1=DAGMA_CONFIG['lambda1'], T=DAGMA_CONFIG['T'],
        mu_init=DAGMA_CONFIG['mu_init'], mu_factor=DAGMA_CONFIG['mu_factor'],
        s=DAGMA_CONFIG['s'].copy(), warm_iter=DAGMA_CONFIG['warm_iter'],
        max_iter=DAGMA_CONFIG['max_iter'], lr=DAGMA_CONFIG['lr']
    )
    return W_est

def run_colide_ev(X, seed):
    model = colide_ev(seed=seed)
    W_est, _ = model.fit(
        X, lambda1=COLIDE_CONFIG['lambda1'], T=COLIDE_CONFIG['T'],
        mu_init=COLIDE_CONFIG['mu_init'], mu_factor=COLIDE_CONFIG['mu_factor'],
        s=COLIDE_CONFIG['s'].copy(), warm_iter=COLIDE_CONFIG['warm_iter'],
        max_iter=COLIDE_CONFIG['max_iter'], lr=COLIDE_CONFIG['lr']
    )
    return W_est

def run_colide_nv(X, seed):
    model = colide_nv(seed=seed)
    W_est, _ = model.fit(
        X, lambda1=COLIDE_CONFIG['lambda1'], T=COLIDE_CONFIG['T'],
        mu_init=COLIDE_CONFIG['mu_init'], mu_factor=COLIDE_CONFIG['mu_factor'],
        s=COLIDE_CONFIG['s'].copy(), warm_iter=COLIDE_CONFIG['warm_iter'],
        max_iter=COLIDE_CONFIG['max_iter'], lr=COLIDE_CONFIG['lr']
    )
    return W_est

def run_dycolide_ev(X, seed):
    model = colide_ev_batch(seed=seed)
    W_est, _ = model.fit(
        X, lambda1=DYCOLIDE_CONFIG['lambda1'], T=DYCOLIDE_CONFIG['T'],
        batch_size=DYCOLIDE_CONFIG['batch_size'],
        n_batches_warm=DYCOLIDE_CONFIG['n_batches_warm'],
        n_batches_final=DYCOLIDE_CONFIG['n_batches_final'],
        lr=DYCOLIDE_CONFIG['lr']
    )
    return W_est

def run_dycolide_nv(X, seed):
    model = colide_nv_batch_cov(seed=seed)
    W_est, _ = model.fit(
        X, lambda1=DYCOLIDE_CONFIG['lambda1'], T=DYCOLIDE_CONFIG['T'],
        batch_size=DYCOLIDE_CONFIG['batch_size'],
        n_batches_warm=DYCOLIDE_CONFIG['n_batches_warm'],
        n_batches_final=DYCOLIDE_CONFIG['n_batches_final'],
        lr=DYCOLIDE_CONFIG['lr']
    )
    return W_est

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(runner_fn, X, W_true_bin, seed):
    """Run a single experiment and return metrics."""
    try:
        start = time.time()
        W_est = runner_fn(X.copy(), seed) if 'seed' in runner_fn.__code__.co_varnames else runner_fn(X.copy())
        elapsed = time.time() - start

        W_est_bin = to_bin(W_est, thr=THRESHOLD)
        shd, tpr, fdr = count_accuracy(W_true_bin, W_est_bin)

        return {'tpr': tpr, 'fdr': fdr, 'shd': shd, 'time': elapsed, 'status': 'OK'}
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

def run_multiple_experiments(runner_fn, var_type, sigma, n_runs, base_seed=42):
    """Run multiple experiments and collect statistics."""
    results = {'tpr': [], 'fdr': [], 'shd': [], 'time': []}

    for run in range(n_runs):
        seed = base_seed + run

        # Generate data
        X, W_true, _ = simulate_sem(
            n_nodes=N_NODES, n_samples=N_SAMPLES, edges=EDGES,
            graph_type=GRAPH_TYPE, edge_type='weighted',
            var_type=var_type, noise='normal', var=sigma, seed=seed
        )
        W_true_bin = to_bin(W_true, thr=0.0)

        # Run experiment
        res = run_single_experiment(runner_fn, X, W_true_bin, seed)

        if res['status'] == 'OK':
            results['tpr'].append(res['tpr'])
            results['fdr'].append(res['fdr'])
            results['shd'].append(res['shd'])
            results['time'].append(res['time'])
        else:
            print(f"  Run {run+1} FAILED: {res.get('error', 'Unknown error')}")

    if len(results['tpr']) > 0:
        return {
            'tpr_mean': np.mean(results['tpr']), 'tpr_std': np.std(results['tpr']),
            'fdr_mean': np.mean(results['fdr']), 'fdr_std': np.std(results['fdr']),
            'shd_mean': np.mean(results['shd']), 'shd_std': np.std(results['shd']),
            'time_mean': np.mean(results['time']), 'time_std': np.std(results['time']),
            'n_success': len(results['tpr']),
            'status': 'OK'
        }
    else:
        return {'status': 'FAILED'}

# =============================================================================
# TABLE FORMATTING
# =============================================================================

def format_metric(mean, std):
    """Format metric as mean±std."""
    return f"{mean:.2f}±{std:.2f}"

def format_shd(mean, std):
    """Format SHD as integer mean±std."""
    return f"{int(round(mean))}±{int(round(std))}"

def format_time(mean):
    """Format time in seconds."""
    return f"{mean:.0f}s"

def print_ev_table(results):
    """Print EV results table with sigma as major columns."""
    print("\n" + "=" * 100)
    print(" " * 20 + "EV Case (200 nodes, ER4, 1000 samples, 10 runs)")
    print("=" * 100)

    # Header
    print(f"{'Method':<16} |{'σ = 1':^40}|{'σ = 5':^40}")
    print(f"{'':<16} |{'TPR':^9}|{'FDR':^9}|{'SHD':^9}|{'Time':^9}|{'TPR':^9}|{'FDR':^9}|{'SHD':^9}|{'Time':^9}")
    print("-" * 100)

    methods = ['GOLEM-EV', 'DAGMA', 'CoLiDE-EV', 'DyCoLiDE-EV']

    for method in methods:
        row = f"{method:<16} |"
        for sigma in SIGMA_VALUES:
            key = f"{method}_sigma{sigma}"
            if key in results and results[key]['status'] == 'OK':
                r = results[key]
                row += f"{format_metric(r['tpr_mean'], r['tpr_std']):^9}|"
                row += f"{format_metric(r['fdr_mean'], r['fdr_std']):^9}|"
                row += f"{format_shd(r['shd_mean'], r['shd_std']):^9}|"
                row += f"{format_time(r['time_mean']):^9}|"
            else:
                row += f"{'FAILED':^9}|" * 4
        print(row)

    print("=" * 100)

def print_nv_table(results):
    """Print NV results table."""
    print("\n" + "=" * 70)
    print(" " * 10 + "NV Case (200 nodes, ER4, 1000 samples, 10 runs)")
    print("=" * 70)

    print(f"{'Method':<18} |{'TPR':^10}|{'FDR':^10}|{'SHD':^10}|{'Time':^10}")
    print("-" * 70)

    methods = ['GOLEM-NV', 'DAGMA', 'CoLiDE-NV', 'DyCoLiDE-NV-Cov']

    for method in methods:
        if method in results and results[method]['status'] == 'OK':
            r = results[method]
            print(f"{method:<18} |{format_metric(r['tpr_mean'], r['tpr_std']):^10}|"
                  f"{format_metric(r['fdr_mean'], r['fdr_std']):^10}|"
                  f"{format_shd(r['shd_mean'], r['shd_std']):^10}|"
                  f"{format_time(r['time_mean']):^10}")
        else:
            print(f"{method:<18} |{'FAILED':^10}|{'FAILED':^10}|{'FAILED':^10}|{'FAILED':^10}")

    print("=" * 70)

# =============================================================================
# MAIN
# =============================================================================

def run_ev_experiments(n_runs=N_RUNS):
    """Run all EV experiments."""
    print("\n" + "#" * 60)
    print("# Running EV Case Experiments")
    print("#" * 60)

    results = {}

    ev_models = {
        'CoLiDE-EV': run_colide_ev,
        'DyCoLiDE-EV': run_dycolide_ev,
    }

    for sigma in SIGMA_VALUES:
        print(f"\n--- σ = {sigma} ---")
        for model_name, runner_fn in ev_models.items():
            print(f"\nRunning {model_name} (σ={sigma})...")
            key = f"{model_name}_sigma{sigma}"
            results[key] = run_multiple_experiments(runner_fn, 'ev', sigma, n_runs)

            if results[key]['status'] == 'OK':
                r = results[key]
                print(f"  TPR={r['tpr_mean']:.3f}±{r['tpr_std']:.3f}, "
                      f"FDR={r['fdr_mean']:.3f}±{r['fdr_std']:.3f}, "
                      f"SHD={r['shd_mean']:.1f}±{r['shd_std']:.1f}")

    return results

def run_nv_experiments(n_runs=N_RUNS):
    """Run all NV experiments."""
    print("\n" + "#" * 60)
    print("# Running NV Case Experiments")
    print("#" * 60)

    results = {}

    nv_models = {
        # 'GOLEM-NV': run_golem_nv,
        # 'DAGMA': run_dagma,
        'CoLiDE-NV': run_colide_nv,
        'DyCoLiDE-NV-Cov': run_dycolide_nv,
    }

    sigma = 1.0  # Default sigma for NV

    for model_name, runner_fn in nv_models.items():
        print(f"\nRunning {model_name}...")
        results[model_name] = run_multiple_experiments(runner_fn, 'nv', sigma, n_runs)

        if results[model_name]['status'] == 'OK':
            r = results[model_name]
            print(f"  TPR={r['tpr_mean']:.3f}±{r['tpr_std']:.3f}, "
                  f"FDR={r['fdr_mean']:.3f}±{r['fdr_std']:.3f}, "
                  f"SHD={r['shd_mean']:.1f}±{r['shd_std']:.1f}")

    return results

def main():
    """Run full comparison."""
    print("=" * 60)
    print("COMPREHENSIVE COMPARISON")
    print(f"Config: {N_NODES} nodes, ER4 ({EDGES} edges), {N_SAMPLES} samples")
    print(f"Runs: {N_RUNS}, Threshold: {THRESHOLD}")
    print("=" * 60)

    # Run EV experiments
    ev_results = run_ev_experiments()
    print_ev_table(ev_results)

    # Run NV experiments
    nv_results = run_nv_experiments()
    print_nv_table(nv_results)

    print("\nDone!")

# def sanity_check():
#     """Quick sanity check with 1 run and smaller config."""
#     global N_RUNS, N_NODES, N_SAMPLES, EDGES
#     global GOLEM_CONFIG, DAGMA_CONFIG, COLIDE_CONFIG, DYCOLIDE_CONFIG

#     # Override for quick test
#     N_RUNS = 1
#     N_NODES = 20
#     N_SAMPLES = 500
#     EDGES = 40

#     # Reduce iterations
#     GOLEM_CONFIG['num_iter'] = 5000
#     DAGMA_CONFIG['warm_iter'] = 2000
#     DAGMA_CONFIG['max_iter'] = 5000
#     COLIDE_CONFIG['warm_iter'] = 2000
#     COLIDE_CONFIG['max_iter'] = 5000
#     DYCOLIDE_CONFIG['n_batches_warm'] = 2000
#     DYCOLIDE_CONFIG['n_batches_final'] = 5000

#     print("=" * 60)
#     print("SANITY CHECK (small scale)")
#     print(f"Config: {N_NODES} nodes, {EDGES} edges, {N_SAMPLES} samples")
#     print("=" * 60)

#     # Quick EV test
#     ev_results = run_ev_experiments(n_runs=1)
#     print_ev_table(ev_results)

#     # Quick NV test
#     nv_results = run_nv_experiments(n_runs=1)
#     print_nv_table(nv_results)

#     print("\nSanity check complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanity', action='store_true', help='Run quick sanity check')
    args = parser.parse_args()

    # if args.sanity:
    #     sanity_check()
    # else:
    main()
