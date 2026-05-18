"""
EV Model Comparison: GOLEM vs DAGMA vs CoLiDE vs CoLiDE-Batch

Compares Equal Variance (EV) versions of:
- GOLEM (PyTorch implementation)
- DAGMA (linear, l2 loss)
- CoLiDE-EV (static full-batch)
- CoLiDE-EV-Batch (mini-batch SGD)
"""

import os
# Disable torch dynamo to avoid transformers version conflicts
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import warnings
warnings.filterwarnings("ignore")

# Workaround for PyTorch/transformers version conflict
# PyTorch's _dynamo tries to patch transformers.configuration_utils but it may not exist
try:
    import transformers
    if not hasattr(transformers, 'configuration_utils'):
        # Create a dummy module to prevent the error
        import types
        transformers.configuration_utils = types.ModuleType('configuration_utils')
        class DummyConfig:
            pass
        transformers.configuration_utils.PretrainedConfig = DummyConfig
except ImportError:
    pass

import sys
import numpy as np
import time

# Add DAGMA to path
sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears/dagma/src/dagma')

from Utils import simulate_sem, to_bin, count_accuracy

# =============================================================================
# MODEL IMPORTS
# =============================================================================

from Colide import colide_ev, colide_ev_batch
from linear import DagmaLinear

# GOLEM-SEM (clean implementation without torch._dynamo issues)
from golem_sem import GOLEM_SEM
GOLEM_AVAILABLE = True

# =============================================================================
# CONFIGURATION - EDIT THESE BEFORE RUNNING
# =============================================================================

DATA_CONFIG = {
    'n_nodes': 50,
    'n_samples': 1000,
    'edges': 200,           # For ER4: edges = 4 * n_nodes
    'graph_type': 'er',
    'edge_type': 'weighted',
    'noise': 'normal',
    'var': 1.0,
    'w_range': ((-2.0, -0.5), (0.5, 2.0)),
    'seed': 42,
}

# GOLEM-EV parameters
GOLEM_CONFIG = {
    'lambda1': 0.02,
    'lambda2': 5.0,
    'num_iter': 50000,
    'lr': 1e-3,
}

# DAGMA parameters
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

# CoLiDE-EV (static) parameters
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

# CoLiDE-EV-Batch parameters
COLIDE_BATCH_CONFIG = {
    'lambda1': 0.05,
    'T': 4,
    'batch_size': 200,       # Set to 1 for online mode
    'n_batches_warm': 20000,
    'n_batches_final': 70000,
    'lr': 0.0003,
}

# Post-processing
THRESHOLD = 0.3

# Which models to run
RUN_GOLEM = False
RUN_DAGMA = False
RUN_COLIDE = False
RUN_COLIDE_BATCH = True

# =============================================================================
# MODEL RUNNERS
# =============================================================================

def run_golem(X, config, seed):
    """Run GOLEM-EV (PyTorch implementation)."""
    if not GOLEM_AVAILABLE:
        raise ImportError("GOLEM not available due to import error")

    model = GOLEM_SEM(seed=seed, verbose=False)
    W_est = model.fit(
        X,
        lambda1=config['lambda1'],
        lambda2=config['lambda2'],
        equal_variances=True,
        num_iter=config['num_iter'],
        lr=config['lr'],
        w_threshold=0.0,  # Don't threshold here, we threshold later
    )
    return W_est


def run_dagma(X, config):
    """Run DAGMA (l2 loss for continuous data)."""
    model = DagmaLinear(loss_type='l2', verbose=False)
    W_est = model.fit(
        X,
        lambda1=config['lambda1'],
        T=config['T'],
        mu_init=config['mu_init'],
        mu_factor=config['mu_factor'],
        s=config['s'].copy(),
        warm_iter=config['warm_iter'],
        max_iter=config['max_iter'],
        lr=config['lr'],
    )
    return W_est


def run_colide(X, config, seed):
    """Run CoLiDE-EV (static full-batch)."""
    model = colide_ev(seed=seed)
    W_est, sigma_est = model.fit(
        X=X,
        lambda1=config['lambda1'],
        T=config['T'],
        mu_init=config['mu_init'],
        mu_factor=config['mu_factor'],
        s=config['s'].copy(),
        warm_iter=config['warm_iter'],
        max_iter=config['max_iter'],
        lr=config['lr'],
    )
    return W_est, sigma_est


def run_colide_batch(X, config, seed):
    """Run CoLiDE-EV-Batch (mini-batch SGD)."""
    model = colide_ev_batch(seed=seed)
    W_est, sigma_est = model.fit(
        X=X,
        lambda1=config['lambda1'],
        T=config['T'],
        batch_size=config['batch_size'],
        n_batches_warm=config['n_batches_warm'],
        n_batches_final=config['n_batches_final'],
        lr=config['lr'],
    )
    return W_est, sigma_est


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison():
    """Run full comparison of EV models."""

    print("=" * 80)
    print("EV MODEL COMPARISON: GOLEM vs DAGMA vs CoLiDE vs CoLiDE-Batch")
    print("=" * 80)

    # Print configuration
    print("\n[DATA CONFIGURATION]")
    print(f"  Nodes: {DATA_CONFIG['n_nodes']}")
    print(f"  Samples: {DATA_CONFIG['n_samples']}")
    print(f"  Edges: {DATA_CONFIG['edges']} (density: {DATA_CONFIG['edges'] / DATA_CONFIG['n_nodes']:.1f}x)")
    print(f"  Graph type: {DATA_CONFIG['graph_type'].upper()}")
    print(f"  Noise: {DATA_CONFIG['noise']}, var={DATA_CONFIG['var']}")
    print(f"  Seed: {DATA_CONFIG['seed']}")

    print(f"\n[THRESHOLD]: {THRESHOLD}")

    # Generate data
    print("\n" + "-" * 80)
    print("Generating data (Equal Variance)...")
    X, W_true, _ = simulate_sem(
        n_nodes=DATA_CONFIG['n_nodes'],
        n_samples=DATA_CONFIG['n_samples'],
        edges=DATA_CONFIG['edges'],
        graph_type=DATA_CONFIG['graph_type'],
        edge_type=DATA_CONFIG['edge_type'],
        var_type='ev',
        noise=DATA_CONFIG['noise'],
        var=DATA_CONFIG['var'],
        w_range=DATA_CONFIG['w_range'],
        seed=DATA_CONFIG['seed'],
    )

    W_true_bin = to_bin(W_true, thr=0.0)
    n_true_edges = int(np.sum(W_true_bin))
    print(f"Data generated: {X.shape[0]} samples, {X.shape[1]} nodes, {n_true_edges} true edges")

    results = {}

    # -------------------------------------------------------------------------
    # GOLEM
    # -------------------------------------------------------------------------
    if RUN_GOLEM:
        print("\n" + "-" * 80)
        print("[1] GOLEM-EV (PyTorch)")
        print(f"    lambda1={GOLEM_CONFIG['lambda1']}, lambda2={GOLEM_CONFIG['lambda2']}, "
              f"num_iter={GOLEM_CONFIG['num_iter']}, lr={GOLEM_CONFIG['lr']}")
        print("-" * 80)

        try:
            start = time.time()
            W_golem = run_golem(X.copy(), GOLEM_CONFIG, DATA_CONFIG['seed'])
            elapsed = time.time() - start

            W_golem_bin = to_bin(W_golem, thr=THRESHOLD)
            shd, tpr, fdr = count_accuracy(W_true_bin, W_golem_bin)
            n_est = int(np.sum(W_golem_bin))

            results['GOLEM'] = {
                'status': 'OK', 'time': elapsed,
                'tpr': tpr, 'fdr': fdr, 'shd': shd, 'edges': n_est
            }
            print(f"    TPR={tpr:.4f}, FDR={fdr:.4f}, SHD={shd}, edges={n_est}/{n_true_edges}, time={elapsed:.2f}s")

        except Exception as e:
            results['GOLEM'] = {'status': 'FAILED', 'error': str(e)}
            print(f"    FAILED: {e}")

    # -------------------------------------------------------------------------
    # DAGMA
    # -------------------------------------------------------------------------
    if RUN_DAGMA:
        print("\n" + "-" * 80)
        print("[2] DAGMA (l2 loss)")
        print(f"    lambda1={DAGMA_CONFIG['lambda1']}, T={DAGMA_CONFIG['T']}, lr={DAGMA_CONFIG['lr']}")
        print("-" * 80)

        try:
            start = time.time()
            W_dagma = run_dagma(X.copy(), DAGMA_CONFIG)
            elapsed = time.time() - start

            W_dagma_bin = to_bin(W_dagma, thr=THRESHOLD)
            shd, tpr, fdr = count_accuracy(W_true_bin, W_dagma_bin)
            n_est = int(np.sum(W_dagma_bin))

            results['DAGMA'] = {
                'status': 'OK', 'time': elapsed,
                'tpr': tpr, 'fdr': fdr, 'shd': shd, 'edges': n_est
            }
            print(f"    TPR={tpr:.4f}, FDR={fdr:.4f}, SHD={shd}, edges={n_est}/{n_true_edges}, time={elapsed:.2f}s")

        except Exception as e:
            results['DAGMA'] = {'status': 'FAILED', 'error': str(e)}
            print(f"    FAILED: {e}")

    # -------------------------------------------------------------------------
    # CoLiDE-EV (static)
    # -------------------------------------------------------------------------
    if RUN_COLIDE:
        print("\n" + "-" * 80)
        print("[3] CoLiDE-EV (static full-batch)")
        print(f"    lambda1={COLIDE_CONFIG['lambda1']}, T={COLIDE_CONFIG['T']}, lr={COLIDE_CONFIG['lr']}")
        print("-" * 80)

        try:
            start = time.time()
            W_colide, sigma_colide = run_colide(X.copy(), COLIDE_CONFIG, DATA_CONFIG['seed'])
            elapsed = time.time() - start

            W_colide_bin = to_bin(W_colide, thr=THRESHOLD)
            shd, tpr, fdr = count_accuracy(W_true_bin, W_colide_bin)
            n_est = int(np.sum(W_colide_bin))

            results['CoLiDE'] = {
                'status': 'OK', 'time': elapsed,
                'tpr': tpr, 'fdr': fdr, 'shd': shd, 'edges': n_est,
                'sigma': sigma_colide
            }
            print(f"    TPR={tpr:.4f}, FDR={fdr:.4f}, SHD={shd}, edges={n_est}/{n_true_edges}, "
                  f"sigma={sigma_colide:.4f}, time={elapsed:.2f}s")

        except Exception as e:
            results['CoLiDE'] = {'status': 'FAILED', 'error': str(e)}
            print(f"    FAILED: {e}")

    # -------------------------------------------------------------------------
    # CoLiDE-EV-Batch
    # -------------------------------------------------------------------------
    if RUN_COLIDE_BATCH:
        bs = COLIDE_BATCH_CONFIG['batch_size']
        mode_str = "online" if bs == 1 else f"batch_size={bs}"

        print("\n" + "-" * 80)
        print(f"[4] CoLiDE-EV-Batch ({mode_str})")
        print(f"    lambda1={COLIDE_BATCH_CONFIG['lambda1']}, T={COLIDE_BATCH_CONFIG['T']}, lr={COLIDE_BATCH_CONFIG['lr']}")
        print("-" * 80)

        try:
            start = time.time()
            W_batch, sigma_batch = run_colide_batch(X.copy(), COLIDE_BATCH_CONFIG, DATA_CONFIG['seed'])
            elapsed = time.time() - start

            W_batch_bin = to_bin(W_batch, thr=THRESHOLD)
            shd, tpr, fdr = count_accuracy(W_true_bin, W_batch_bin)
            n_est = int(np.sum(W_batch_bin))

            results['CoLiDE-Batch'] = {
                'status': 'OK', 'time': elapsed,
                'tpr': tpr, 'fdr': fdr, 'shd': shd, 'edges': n_est,
                'sigma': sigma_batch
            }
            print(f"    TPR={tpr:.4f}, FDR={fdr:.4f}, SHD={shd}, edges={n_est}/{n_true_edges}, "
                  f"sigma={sigma_batch:.4f}, time={elapsed:.2f}s")

        except Exception as e:
            results['CoLiDE-Batch'] = {'status': 'FAILED', 'error': str(e)}
            print(f"    FAILED: {e}")

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTrue edges: {n_true_edges} | Threshold: {THRESHOLD}")
    print(f"\n{'Method':<18} {'Time (s)':>10} {'TPR':>8} {'FDR':>8} {'SHD':>8} {'Edges':>10}")
    print("-" * 70)

    for method in ['CoLiDE', 'CoLiDE-Batch']:
        if method in results:
            r = results[method]
            if r['status'] == 'OK':
                print(f"{method:<18} {r['time']:>10.2f} {r['tpr']:>8.4f} {r['fdr']:>8.4f} "
                      f"{r['shd']:>8} {r['edges']:>10}")
            elif r['status'] == 'SKIPPED':
                print(f"{method:<18} {'SKIPPED':>10}")
            else:
                print(f"{method:<18} {'FAILED':>10}")

    print("=" * 70)

    # Check all passed (OK or SKIPPED counts as passed)
    all_passed = all(
        r['status'] in ('OK', 'SKIPPED')
        for r in results.values()
    )
    n_skipped = sum(1 for r in results.values() if r['status'] == 'SKIPPED')
    if all_passed and n_skipped == 0:
        print("\nAll models completed successfully!")
    elif all_passed:
        print(f"\nAll models completed ({n_skipped} skipped due to environment issues).")
    else:
        print("\nSome models FAILED!")

    return results


if __name__ == "__main__":
    run_comparison()
