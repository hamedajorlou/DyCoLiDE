"""
Large-scale comparison: Testing CoLiDE variants on very large graphs
Tests scalability: 100, 200, and 300 node graphs
"""
from Utils import simulate_sem
import numpy as np
import time
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx


# def simulate_sem_simple(n_nodes, n_samples, edges, seed=123):
#     """Simplified SEM simulation"""
#     rng = np.random.default_rng(seed=seed)

#     # Generate ER graph
#     prob = float(edges*2)/float(n_nodes**2 - n_nodes)
#     G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
#     adj = nx.to_numpy_array(G)
#     U_mask = np.triu(adj, k=1)

#     # Random permutation
#     P = np.eye(n_nodes)
#     P = P[:, rng.permutation(n_nodes)]
#     W = P @ U_mask @ P.T

#     # Add weights
#     W_weighted = np.zeros(W.shape)
#     S = rng.integers(2, size=W.shape)
#     for i, (low, high) in enumerate([(-2.0, -0.5), (0.5, 2.0)]):
#         weights = rng.uniform(low=low, high=high, size=W.shape)
#         W_weighted += W * (S == i) * weights

#     # Generate data
#     G_sem = nx.DiGraph(W_weighted)
#     X = np.zeros((n_samples, n_nodes))
#     ordered_vertices = list(nx.topological_sort(G_sem))

#     for j in ordered_vertices:
#         parents = list(G_sem.predecessors(j))
#         eta = X[:, parents].dot(W_weighted[parents, j])
#         X[:, j] = eta + rng.normal(scale=1.0, size=(n_samples))

#     return X, W_weighted


def to_bin(A, thr):
    """Convert to binary adjacency matrix"""
    B = A.copy().astype(float)
    B[np.abs(B) < thr] = 0.0
    B[B != 0.0] = 1.0
    np.fill_diagonal(B, 0.0)
    return B


def count_accuracy(W_bin_true, W_bin_est):
    """Compute SHD, TPR, FDR"""
    pred = np.flatnonzero(W_bin_est == 1)
    cond = np.flatnonzero(W_bin_true)
    cond_reversed = np.flatnonzero(W_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # SHD
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    pred_lower = np.flatnonzero(np.tril(W_bin_est + W_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(W_bin_true + W_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # TPR
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    tpr = float(len(true_pos)) / max(len(cond), 1)

    # FDR
    pred_size = len(pred)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)

    return shd, tpr, fdr


# Import CoLiDE methods
from Colide_batch_MOHEM import colide_ev_batch
from Colide_online_MOHEM import colide_ev_online


def test_scalability(graph_sizes=[(100, 400), (200, 1600), (300, 3600)], seed=123):
    """Test batch and online methods on increasingly large graphs"""
    print("="*90)
    print(" Large-Scale Scalability Test: Batch vs Online CoLiDE-EV")
    print("="*90)
    print("\n Testing Batch and Online methods (Standard is too slow for large graphs)")
    print("-"*90)

    all_results = []

    for n_nodes, edges in graph_sizes:
        # Scale samples with graph size
        n_samples = 1000

        print(f"\n{'='*90}")
        print(f" GRAPH SIZE: {n_nodes} nodes, {edges} edges, {n_samples} samples")
        print(f"{'='*90}")

        # Generate data
        print(f"\n1. Generating data...")
        X, W_true, var_ev = simulate_sem(
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
        print(f"   Data shape: {X.shape}, True edges: {np.sum(W_true != 0)}")

        results = {'n_nodes': n_nodes, 'edges': edges, 'n_samples': n_samples}

        # Batch CoLiDE-EV
        print(f"\n2. Running Batch CoLiDE-EV (Mini-batch SGD)")
        print("-"*90)

        model_batch = colide_ev_batch(seed=seed)

        # Adaptive scaling for larger graphs
        n_batches_warm = int(500 * (n_nodes / 20))    # Warm-up iterations (was 3000)
        n_batches_final = int(1500 * (n_nodes / 20))  # Final iterations (was 8000)
        batch_size = min(100, max(50, n_samples // 20))

        print(f"   Batches: warm={n_batches_warm}, final={n_batches_final}, size={batch_size}")

        t_start = time.time()
        try:
            W_batch, sigma_batch = model_batch.fit(
                X=X, lambda1=0.05, T=5, mu_init=1.0, mu_factor=0.1,
                s=[1.0, 0.9, 0.8, 0.7, 0.6], batch_size=batch_size,
                n_batches_warm=n_batches_warm, n_batches_final=n_batches_final,
                lr=0.0002, checkpoint=500,
            )
            t_batch = time.time() - t_start

            W_true_bin = to_bin(W_true, thr=0.0)
            W_batch_bin = to_bin(W_batch, thr=0.3)
            shd_batch, tpr_batch, fdr_batch = count_accuracy(W_true_bin, W_batch_bin)

            results['batch'] = {
                'time': t_batch,
                'sigma': sigma_batch,
                'shd': shd_batch,
                'tpr': tpr_batch,
                'fdr': fdr_batch,
                'edges_found': np.sum(np.abs(W_batch) > 0.3)
            }

            print(f"   Time: {t_batch:.2f}s")
            print(f"   SHD: {shd_batch}, TPR: {tpr_batch:.4f}, FDR: {fdr_batch:.4f}")
            print(f"   Sigma: {sigma_batch:.4f}, Edges found: {results['batch']['edges_found']}")

        except Exception as e:
            print(f"   FAILED: {e}")
            results['batch'] = None

        # Online CoLiDE-EV
        print(f"\n3. Running Online CoLiDE-EV (Streaming, one-at-a-time)")
        print("-"*90)

        model_online = colide_ev_online(seed=seed)

        # For online: more samples needed for convergence
        n_samples_warm = 5000
        n_samples_final = 10000
        update_freq = 1  # True online

        print(f"   Samples: warm={n_samples_warm}, final={n_samples_final}")
        print(f"   Update frequency: every {update_freq} sample")

        t_start = time.time()
        try:
            W_online, sigma_online = model_online.fit(
                X=X, lambda1=0.05, T=5, mu_init=1.0, mu_factor=0.1,
                s=[1.0, 0.9, 0.8, 0.7, 0.6],
                n_samples_warm=n_samples_warm, n_samples_final=n_samples_final,
                lr=0.0002, checkpoint=500, update_freq=update_freq,
            )
            t_online = time.time() - t_start

            W_online_bin = to_bin(W_online, thr=0.3)
            shd_online, tpr_online, fdr_online = count_accuracy(W_true_bin, W_online_bin)

            results['online'] = {
                'time': t_online,
                'sigma': sigma_online,
                'shd': shd_online,
                'tpr': tpr_online,
                'fdr': fdr_online,
                'edges_found': np.sum(np.abs(W_online) > 0.3)
            }

            print(f"   Time: {t_online:.2f}s")
            print(f"   SHD: {shd_online}, TPR: {tpr_online:.4f}, FDR: {fdr_online:.4f}")
            print(f"   Sigma: {sigma_online:.4f}, Edges found: {results['online']['edges_found']}")

        except Exception as e:
            print(f"   FAILED: {e}")
            results['online'] = None

        all_results.append(results)

    # Final summary
    print(f"\n{'='*90}")
    print(f" SCALABILITY SUMMARY")
    print(f"{'='*90}")

    print(f"\n{'Nodes':<8} {'Method':<12} {'Time(s)':<12} {'SHD':<10} {'TPR':<12} {'FDR':<12}")
    print("-"*90)

    for res in all_results:
        n = res['n_nodes']
        for method in ['batch', 'online']:
            if res[method] is not None:
                m = res[method]
                sigma_val = m['sigma'] if isinstance(m['sigma'], float) else m['sigma'].mean()
                print(f"{n:<8} {method.capitalize():<12} {m['time']:<12.1f} "
                      f"{m['shd']:<10d} {m['tpr']:<12.4f} {m['fdr']:<12.4f}")

    print("="*90)

    # Analyze scaling
    print(f"\n SCALING ANALYSIS:")
    print("-"*90)

    for method in ['batch', 'online']:
        times = [r[method]['time'] for r in all_results if r[method] is not None]
        sizes = [r['n_nodes'] for r in all_results if r[method] is not None]

        if len(times) >= 2:
            # Estimate scaling factor
            if len(times) >= 2:
                ratio = times[-1] / times[0]
                size_ratio = sizes[-1] / sizes[0]
                print(f"   {method.capitalize()}: {sizes[0]}→{sizes[-1]} nodes took {times[0]:.1f}→{times[-1]:.1f}s")
                print(f"     Time increased {ratio:.2f}x for {size_ratio:.1f}x graph size")

    print("="*90)

    return all_results


if __name__ == "__main__":
    # Test on progressively larger graphs
    # Start with , 200 nodes (300 might be very slow)
    results = test_scalability(
        graph_sizes=[
            # (100, 400),      # 100 nodes, 400 edges
            (200, 800),     # 200 nodes, 1600 edges
        ],
        seed=123
    )
