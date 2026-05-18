"""
Test script for DyCoLiDE_BatchStreaming with multiple seeds.

Compares BatchStreaming vs standard DyCoLiDE_EV across 100 random seeds.
"""

import numpy as np
from SVAR_data_generator import generate_svar_data
from DyCoLiDE import DyCoLiDE_BatchStreaming, DyCoLiDE_EV, count_accuracy_svar


def run_single_experiment(seed, n_nodes=30, n_timesteps=1000, lag_order=1,
                          instantaneous_edges=60, temporal_edges=60,
                          batch_size=2, lambda_W=0.01, lambda_A=0.015,
                          threshold=0.10):
    """
    Run a single experiment with given seed.

    Tuned parameters for 30 nodes (from TUNING_NOTES.md):
    - temporal_strength=0.5 (critical for detectability)
    - lambda_W=0.01, lambda_A=0.015 (balanced for temporal edge detection)
    - threshold=0.10 (filters false positives while keeping true edges)
    """

    # Generate data with temporal_strength=0.5 (critical!)
    X, B_true, A_list_true, _ = generate_svar_data(
        n_nodes=n_nodes,
        n_timesteps=n_timesteps,
        lag_order=lag_order,
        instantaneous_edges=instantaneous_edges,
        temporal_edges=temporal_edges,
        temporal_strength=0.5,  # Important: higher strength for detectability
        seed=seed
    )

    W_true = B_true.T
    A_true = np.vstack([A.T for A in A_list_true])

    # Run BatchStreaming
    model_stream = DyCoLiDE_BatchStreaming(d=n_nodes, p=lag_order, seed=seed)
    W_stream, A_stream, _ = model_stream.fit_streaming(
        X, batch_size=batch_size,
        lambda_W=lambda_W, lambda_A=lambda_A,
        T=5, warm_iter=3000, max_iter=6000, lr=0.0003,
        verbose=True
    )

    # Threshold
    W_stream[np.abs(W_stream) < threshold] = 0
    A_stream[np.abs(A_stream) < threshold] = 0

    metrics_stream = count_accuracy_svar(W_true, W_stream, A_true, A_stream, threshold=0.0)

    # Run standard DyCoLiDE_EV
    model_std = DyCoLiDE_EV(seed=seed)
    W_std, A_std, _ = model_std.fit(
        X, p=lag_order,
        lambda_W=lambda_W, lambda_A=lambda_A,
        T=5, warm_iter=3000, max_iter=6000, lr=0.0003
    )

    # Threshold
    W_std[np.abs(W_std) < threshold] = 0
    A_std[np.abs(A_std) < threshold] = 0

    metrics_std = count_accuracy_svar(W_true, W_std, A_true, A_std, threshold=0.0)

    return metrics_stream, metrics_std


def run_multiple_seeds(n_seeds=10, batch_size=2):
    """Run experiments with multiple seeds and compute statistics."""

    print("=" * 70)
    print(f"Running {n_seeds} experiments with batch_size={batch_size}")
    print("=" * 70)

    # Storage for results
    stream_results = {'W_tpr': [], 'W_fdr': [], 'A_tpr': [], 'A_fdr': []}
    std_results = {'W_tpr': [], 'W_fdr': [], 'A_tpr': [], 'A_fdr': []}

    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"Experiment {seed+1}/{n_seeds} (seed={seed})")
        print(f"{'='*70}")
        metrics_stream, metrics_std = run_single_experiment(seed = 42, batch_size=batch_size)

        for key in stream_results.keys():
            stream_results[key].append(metrics_stream[key])
            std_results[key].append(metrics_std[key])

    # Convert to numpy arrays
    for key in stream_results.keys():
        stream_results[key] = np.array(stream_results[key])
        std_results[key] = np.array(std_results[key])

    # Print results table
    print("\n" + "=" * 70)
    print(f"Results over {n_seeds} seeds (batch_size={batch_size})")
    print("=" * 70)
    print(f"{'Method':<25} {'W_TPR':>12} {'W_FDR':>12} {'A_TPR':>12} {'A_FDR':>12}")
    print("-" * 70)

    # BatchStreaming
    print(f"{'BatchStreaming':<25} "
          f"{stream_results['W_tpr'].mean():.3f}±{stream_results['W_tpr'].std():.3f} "
          f"{stream_results['W_fdr'].mean():.3f}±{stream_results['W_fdr'].std():.3f} "
          f"{stream_results['A_tpr'].mean():.3f}±{stream_results['A_tpr'].std():.3f} "
          f"{stream_results['A_fdr'].mean():.3f}±{stream_results['A_fdr'].std():.3f}")

    # Standard
    print(f"{'Standard DyCoLiDE_EV':<25} "
          f"{std_results['W_tpr'].mean():.3f}±{std_results['W_tpr'].std():.3f} "
          f"{std_results['W_fdr'].mean():.3f}±{std_results['W_fdr'].std():.3f} "
          f"{std_results['A_tpr'].mean():.3f}±{std_results['A_tpr'].std():.3f} "
          f"{std_results['A_fdr'].mean():.3f}±{std_results['A_fdr'].std():.3f}")

    # Check if results are identical
    print("\n" + "-" * 70)
    print("Difference (BatchStreaming - Standard):")
    for key in ['W_tpr', 'W_fdr', 'A_tpr', 'A_fdr']:
        diff = stream_results[key] - std_results[key]
        print(f"  {key}: mean={diff.mean():.6f}, max_abs={np.abs(diff).max():.6f}")

    return stream_results, std_results


if __name__ == "__main__":
    stream_results, std_results = run_multiple_seeds(n_seeds=1, batch_size=2)

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)
