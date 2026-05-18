"""
Test script for colide_ev_batch, colide_nv_batch, and colide_nv_batch_cov.
"""

import numpy as np
from Colide import colide_ev_batch, colide_nv_batch, colide_nv_batch_cov
from Utils import simulate_sem, to_bin, count_accuracy

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    'n_nodes': 200,
    'n_samples': 1000,
    'edges': 800,
    'graph_type': 'er',
    'edge_type': 'weighted',
    'noise': 'normal',
    'var': 1.0,
    'seed': 42,
}

MODEL_CONFIG = {
    'lambda1': 0.05,
    'T': 4,
    'lr': 0.0003,
    'n_batches_warm': 20000,
    'n_batches_final': 70000,
    'threshold': 0.3,
}

BATCH_SIZES = [50, 200]  # Skip online (bs=1) for faster testing

TEST_EV = False
TEST_NV = False
TEST_NV_COV = True  # New: test colide_nv_batch_cov

# =============================================================================

def run_test(model_class, X, W_true_bin, batch_sizes, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    n_true = int(np.sum(W_true_bin))
    results = {}

    for bs in batch_sizes:
        mode = "(online)" if bs == 1 else f"(bs={bs})"
        print(f"\nBatch size = {bs} {mode}")

        model = model_class(seed=DATA_CONFIG['seed'])
        try:
            W_est, sigma_est = model.fit(
                X.copy(),
                lambda1=MODEL_CONFIG['lambda1'],
                T=MODEL_CONFIG['T'],
                batch_size=bs,
                n_batches_warm=MODEL_CONFIG['n_batches_warm'],
                n_batches_final=MODEL_CONFIG['n_batches_final'],
                lr=MODEL_CONFIG['lr']
            )

            W_est_bin = to_bin(W_est, thr=MODEL_CONFIG['threshold'])
            shd, tpr, fdr = count_accuracy(W_true_bin, W_est_bin)
            n_est = int(np.sum(W_est_bin))

            results[bs] = {'status': 'OK', 'tpr': tpr, 'fdr': fdr, 'shd': shd}

            sigma_str = f"{sigma_est:.4f}" if np.isscalar(sigma_est) else f"mean={np.mean(sigma_est):.4f}"
            print(f"  TPR={tpr:.3f}, FDR={fdr:.3f}, SHD={shd}, edges={n_est}/{n_true}, sigma={sigma_str}")

        except Exception as e:
            results[bs] = {'status': 'FAILED', 'error': str(e)}
            print(f"  FAILED: {e}")

    return results


def main():
    print(f"\n{'#'*60}\n# CoLiDE Batch Tests\n{'#'*60}")
    print(f"Data: d={DATA_CONFIG['n_nodes']}, n={DATA_CONFIG['n_samples']}, edges={DATA_CONFIG['edges']}")
    print(f"Model: lambda1={MODEL_CONFIG['lambda1']}, T={MODEL_CONFIG['T']}, lr={MODEL_CONFIG['lr']}")

    all_results = {}

    if TEST_EV:
        X, W_true, _ = simulate_sem(**DATA_CONFIG, var_type='ev')
        W_true_bin = to_bin(W_true, thr=0.0)
        all_results['EV'] = run_test(colide_ev_batch, X, W_true_bin, BATCH_SIZES, "colide_ev_batch")

    if TEST_NV:
        X, W_true, _ = simulate_sem(**DATA_CONFIG, var_type='nv')
        W_true_bin = to_bin(W_true, thr=0.0)
        all_results['NV'] = run_test(colide_nv_batch, X, W_true_bin, BATCH_SIZES, "colide_nv_batch")

    if TEST_NV_COV:
        X, W_true, _ = simulate_sem(**DATA_CONFIG, var_type='nv')
        W_true_bin = to_bin(W_true, thr=0.0)
        all_results['NV_COV'] = run_test(colide_nv_batch_cov, X, W_true_bin, BATCH_SIZES, "colide_nv_batch_cov")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for bs, res in results.items():
            status = res['status']
            if status == 'OK':
                print(f"  bs={bs:3d}: TPR={res['tpr']:.3f}, FDR={res['fdr']:.3f}, SHD={res['shd']}")
            else:
                print(f"  bs={bs:3d}: {status}")

    all_passed = all(
        res['status'] == 'OK'
        for results in all_results.values()
        for res in results.values()
    )
    print(f"\n{'='*60}")
    print("All tests PASSED!" if all_passed else "Some tests FAILED!")

    return all_passed


if __name__ == "__main__":
    main()
