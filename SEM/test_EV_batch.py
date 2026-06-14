"""
EV Model Comparison: GOLEM vs DAGMA vs CoLiDE vs CoLiDE-Batch

Compares Equal Variance (EV) versions of:
- GOLEM (PyTorch implementation)
- DAGMA (linear, l2 loss)
- CoLiDE-EV (static full-batch)
- CoLiDE-EV-Batch (mini-batch SGD)
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import warnings
warnings.filterwarnings("ignore")

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
import json
import numpy as np
import time
from datetime import datetime

sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears/dagma/src/dagma')

# Make repo root importable when running this file from the SEM/ subfolder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils import simulate_sem, to_bin, count_accuracy

# =============================================================================
# MODEL IMPORTS
# =============================================================================

from SEM.dycolide import colide_ev, colide_ev_batch

try:
    from linear import DagmaLinear
    DAGMA_AVAILABLE = True
except ImportError:
    DAGMA_AVAILABLE = False

try:
    from golem_sem import GOLEM_SEM
    GOLEM_AVAILABLE = True
except ImportError:
    GOLEM_AVAILABLE = False


DATA_CONFIG = {
    'n_nodes': 50,
    'n_samples': 1000,
    'edges': 200,           
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
# SCALING SWEEP CONFIG  (used by run_scaling())
# =============================================================================

NODES_LIST  = [20, 40, 60, 80, 100]
BATCH_SIZES = [100, 50, 1]                       # bs=1 → true online
ER_LEVELS   = [2, 3, 4]                          # edges per node → ER2, ER3, ER4
SEEDS       = [42, 43, 44, 45, 46]               # multi-seed averaging

# Same iteration budget across all methods (matches the original
# COLIDE_CONFIG / COLIDE_BATCH_CONFIG above).
SWEEP_STATIC_KW = dict(T=4, mu_init=1.0, mu_factor=0.1,
                       s=[1.0, 0.9, 0.8, 0.7],
                       warm_iter=20000, max_iter=70000, lr=0.0003)

SWEEP_BATCH_KW  = dict(T=4, mu_init=1.0, mu_factor=0.1,
                       s=[1.0, 0.9, 0.8, 0.7],
                       n_batches_warm=20000, n_batches_final=70000, lr=0.0003)

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
        w_threshold=0.0, 
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


# =============================================================================
# SCALING SWEEP  —  CoLiDE static vs mini-batch (bs=100, 50, 1) as d grows
# =============================================================================

def nmse(W_est, W_true):
    """Normalized mean squared error of estimated weighted DAG."""
    denom = np.sum(W_true ** 2)
    if denom <= 1e-12:
        return float(np.sum(W_est ** 2))
    return float(np.sum((W_est - W_true) ** 2) / denom)


def _empty_cells():
    """Bookkeeping container: (er, d, method) -> list of per-seed dicts."""
    cells = {}
    for er in ER_LEVELS:
        for d in NODES_LIST:
            for m in ['Static'] + [f'bs={bs}' for bs in BATCH_SIZES]:
                cells[(er, d, m)] = []
    return cells


def _agg(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'values': [float(x) for x in a]}


def _save_partial(cells, path):
    """Dump current state to JSON. Called after every fit."""
    methods = ['Static'] + [f'bs={bs}' for bs in BATCH_SIZES]
    results = {f'ER{er}': {m: {} for m in methods} for er in ER_LEVELS}
    for (er, d, m), seed_rows in cells.items():
        if not seed_rows:
            continue
        key = f'd={d}'
        results[f'ER{er}'][m][key] = {
            'tpr':    _agg([r['tpr'] for r in seed_rows]),
            'fdr':    _agg([r['fdr'] for r in seed_rows]),
            'shd':    _agg([r['shd'] for r in seed_rows]),
            'nmse':   _agg([r['nmse'] for r in seed_rows]),
            'time_s': _agg([r['time_s'] for r in seed_rows]),
            'seeds_done': [r['seed'] for r in seed_rows],
        }

    payload = {
        'experiment': 'colide_ev_scaling_sweep_multiseed',
        'description': (
            'CoLiDE-EV static vs mini-batch SGD (batch_size in {100, 50, 1}) '
            'across n_nodes in {20,40,60,80,100} and ER densities {2,3,4}, '
            'averaged over 5 seeds. Each metric stores mean, std, and the '
            'per-seed values list. JSON is rewritten after every fit so the '
            'file always reflects work completed so far.'
        ),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': {
            'nodes_list':  NODES_LIST,
            'batch_sizes': BATCH_SIZES,
            'er_levels':   ER_LEVELS,
            'seeds':       SEEDS,
            'graph_type':  DATA_CONFIG['graph_type'],
            'edge_type':   DATA_CONFIG['edge_type'],
            'n_samples':   DATA_CONFIG['n_samples'],
            'noise':       DATA_CONFIG['noise'],
            'var':         DATA_CONFIG['var'],
            'w_range':     [list(r) for r in DATA_CONFIG['w_range']],
            'threshold':   THRESHOLD,
            'lambda1':     COLIDE_CONFIG['lambda1'],
            'static_kw':   SWEEP_STATIC_KW,
            'batch_kw':    SWEEP_BATCH_KW,
        },
        'results': results,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def run_scaling():
    """Multi-seed sweep over d × ER_level × method × seed.

    Loop order: ER → d → method → seed. After each individual fit completes,
    the partial JSON is written so progress is never lost if the run is
    interrupted.
    """
    json_path = os.path.join(os.path.dirname(__file__),
                             'colide_ev_scaling_5seeds.json')
    cells = _empty_cells()

    for er in ER_LEVELS:
        print(f"\n{'#' * 80}\n#  ER{er} (edges = {er} * d)\n{'#' * 80}", flush=True)

        for d in NODES_LIST:
            edges = er * d

            for seed in SEEDS:
                X, W_true, _ = simulate_sem(
                    n_nodes=d, n_samples=DATA_CONFIG['n_samples'], edges=edges,
                    graph_type=DATA_CONFIG['graph_type'],
                    edge_type=DATA_CONFIG['edge_type'],
                    var_type='ev', noise=DATA_CONFIG['noise'],
                    var=DATA_CONFIG['var'], w_range=DATA_CONFIG['w_range'],
                    seed=seed,
                )
                W_true_bin = to_bin(W_true, thr=0.0)
                n_true = int(W_true_bin.sum())
                print(f"\n--- ER{er}, d={d}, seed={seed}, true edges={n_true} ---",
                      flush=True)

                # Static CoLiDE
                t0 = time.time()
                W_est, _ = colide_ev(seed=seed).fit(
                    X=X.copy(), lambda1=COLIDE_CONFIG['lambda1'], **SWEEP_STATIC_KW)
                dt = time.time() - t0
                shd, tpr, fdr = count_accuracy(W_true_bin, to_bin(W_est, thr=THRESHOLD))
                err = nmse(W_est, W_true)
                cells[(er, d, 'Static')].append({
                    'seed': seed, 'tpr': float(tpr), 'fdr': float(fdr),
                    'shd': int(shd), 'nmse': err, 'time_s': dt})
                print(f"  Static     SHD={shd:>3}  NMSE={err:.4f}  "
                      f"TPR={tpr:.3f}  FDR={fdr:.3f}   t={dt:6.1f}s", flush=True)
                _save_partial(cells, json_path)

                # Mini-batch CoLiDE at each batch_size
                for bs in BATCH_SIZES:
                    t0 = time.time()
                    W_est, _ = colide_ev_batch(seed=seed).fit(
                        X=X.copy(), lambda1=COLIDE_BATCH_CONFIG['lambda1'],
                        batch_size=bs, **SWEEP_BATCH_KW)
                    dt = time.time() - t0
                    shd, tpr, fdr = count_accuracy(W_true_bin, to_bin(W_est, thr=THRESHOLD))
                    err = nmse(W_est, W_true)
                    cells[(er, d, f'bs={bs}')].append({
                        'seed': seed, 'tpr': float(tpr), 'fdr': float(fdr),
                        'shd': int(shd), 'nmse': err, 'time_s': dt})
                    print(f"  bs={bs:<5}  SHD={shd:>3}  NMSE={err:.4f}  "
                          f"TPR={tpr:.3f}  FDR={fdr:.3f}   t={dt:6.1f}s", flush=True)
                    _save_partial(cells, json_path)

    return cells, json_path


def print_summary(cells):
    methods = ['Static'] + [f'bs={bs}' for bs in BATCH_SIZES]

    def stat(er, m, d, field):
        seeds = cells.get((er, d, m), [])
        if not seeds:
            return None
        vals = np.asarray([r[field] for r in seeds], dtype=float)
        return vals.mean(), vals.std()

    for er in ER_LEVELS:
        print("\n" + "=" * 110)
        print(f"ER{er}  —  SHD (mean ± std over {len(SEEDS)} seeds)")
        print("=" * 110)
        print(f"{'Method':<10}" + "".join(f"{'d='+str(d):>20}" for d in NODES_LIST))
        print("-" * 110)
        for m in methods:
            line = f"{m:<10}"
            for d in NODES_LIST:
                s = stat(er, m, d, 'shd')
                line += f"   SHD={s[0]:5.1f}±{s[1]:4.1f}" if s else f"{'-':>20}"
            print(line)

        print(f"\nER{er}  —  NMSE (mean ± std)")
        print("-" * 110)
        print(f"{'Method':<10}" + "".join(f"{'d='+str(d):>20}" for d in NODES_LIST))
        for m in methods:
            line = f"{m:<10}"
            for d in NODES_LIST:
                s = stat(er, m, d, 'nmse')
                line += f" NMSE={s[0]:.4f}±{s[1]:.4f}" if s else f"{'-':>20}"
            print(line)


def save_results_json(rows, path=None):
    """Write the sweep results to a JSON file in the SEM/ folder.

    Structure:
        {
          "experiment": ...,
          "timestamp": ISO-8601,
          "config": {...everything needed to reproduce...},
          "results": {
              "ER2": { "<method>": { "d=<N>": {tpr, fdr, shd, nmse, time_s} } },
              "ER3": {...},
              "ER4": {...}
          }
        }
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'colide_ev_scaling_1seed.json')

    methods = ['Static'] + [f'bs={bs}' for bs in BATCH_SIZES]
    results = {f'ER{er}': {m: {} for m in methods} for er in ER_LEVELS}
    for r in rows:
        results[f"ER{r['er']}"][r['method']][f"d={r['d']}"] = {
            'tpr':    r['tpr'],
            'fdr':    r['fdr'],
            'shd':    r['shd'],
            'nmse':   r['nmse'],
            'time_s': r['time_s'],
        }

    payload = {
        'experiment': 'colide_ev_scaling_sweep',
        'description': (
            'CoLiDE-EV static vs mini-batch SGD (batch_size in {100, 50, 1}) '
            'across n_nodes in {20,40,60,80,100} and ER densities {2,3,4}. '
            'Same iteration budget across all methods. Metrics: SHD on '
            'thresholded W_est; NMSE on raw weighted W_est.'
        ),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': {
            'nodes_list':  NODES_LIST,
            'batch_sizes': BATCH_SIZES,
            'er_levels':   ER_LEVELS,
            'graph_type':  DATA_CONFIG['graph_type'],
            'edge_type':   DATA_CONFIG['edge_type'],
            'n_samples':   DATA_CONFIG['n_samples'],
            'noise':       DATA_CONFIG['noise'],
            'var':         DATA_CONFIG['var'],
            'w_range':     [list(r) for r in DATA_CONFIG['w_range']],
            'seed':        DATA_CONFIG['seed'],
            'threshold':   THRESHOLD,
            'lambda1':     COLIDE_CONFIG['lambda1'],
            'static_kw':   SWEEP_STATIC_KW,
            'batch_kw':    SWEEP_BATCH_KW,
        },
        'results': results,
    }

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to {path}")
    return path


if __name__ == "__main__":
    cells, json_path = run_scaling()
    print_summary(cells)
    print(f"\nFinal results saved to {json_path}")
