"""
Evolving DAG Experiment: Testing DyCoLiDE's ability to track changing networks.

This experiment demonstrates that online/batch methods can adapt to structural changes
in the underlying causal graph, while static methods cannot.

Scenario: The true DAG changes abruptly at time T/2
- Phase 1 (t = 0 to T/2): Data generated from DAG W₁
- Phase 2 (t = T/2 to T): Data generated from DAG W₂

We track how DyCoLiDE's estimates evolve over time and measure:
- How quickly it adapts to the new structure
- Performance metrics (TPR, FDR, SHD) over time
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time
import networkx as nx

# Import DyCoLiDE
from SEM.dycolide import colide_ev_batch

# Import existing utilities
from Utils import create_dag


# =============================================================================
# NON-STATIONARY DATA GENERATOR
# =============================================================================

def generate_dag(d: int, expected_edges: int, seed: int = None) -> np.ndarray:
    """
    Generate a random DAG using the existing create_dag utility.

    Parameters
    ----------
    d : int
        Number of nodes
    expected_edges : int
        Expected number of edges
    seed : int
        Random seed

    Returns
    -------
    W : np.ndarray
        (d, d) weighted adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Use create_dag from Utils.py
    W, _ = create_dag(
        n_nodes=d,
        graph_type='er',
        edges=expected_edges,
        permute=False,
        edge_type='weighted',
        w_range=((-2.0, -0.5), (0.5, 2.0))
    )

    return W


def generate_evolved_dag(W1: np.ndarray, change_fraction: float = 0.5,
                         seed: int = None) -> np.ndarray:
    """
    Generate W2 by modifying W1:
    - Keep some edges (persistence)
    - Remove some edges (disappearing)
    - Add some new edges (emerging)

    Parameters
    ----------
    W1 : np.ndarray
        Original DAG
    change_fraction : float
        Fraction of edges to change (0.5 = 50% of edges change)
    seed : int
        Random seed

    Returns
    -------
    W2 : np.ndarray
        Modified DAG
    """
    rng = np.random.default_rng(seed=seed)

    d = W1.shape[0]
    W2 = W1.copy()

    # Get existing edges
    existing_edges = np.argwhere(np.abs(W1) > 0)
    n_existing = len(existing_edges)

    # Number of edges to change
    n_remove = int(n_existing * change_fraction / 2)
    n_add = n_remove

    # Remove some edges
    if n_remove > 0 and n_existing > 0:
        remove_idx = rng.choice(n_existing, min(n_remove, n_existing), replace=False)
        for idx in remove_idx:
            i, j = existing_edges[idx]
            W2[i, j] = 0

    # Add some new edges - sample weights from same range as W1
    # Match the weight distribution: w_range=((-2.0, -0.5), (0.5, 2.0))
    w_ranges = [(-2.0, -0.5), (0.5, 2.0)]
    added = 0
    attempts = 0
    max_edges = d * (d - 1) // 2

    # Get potential positions (lower triangular, excluding diagonal)
    potential_positions = []
    for i in range(1, d):
        for j in range(i):
            if W2[i, j] == 0:
                potential_positions.append((i, j))

    # Shuffle and try to add edges
    rng.shuffle(potential_positions)

    for i, j in potential_positions:
        if added >= n_add:
            break
        # Sample from one of the weight ranges randomly
        w_range = w_ranges[rng.integers(0, 2)]
        weight = rng.uniform(w_range[0], w_range[1])
        W2[i, j] = weight
        added += 1

    return W2


def generate_nonstationary_data(d: int, T: int, W1: np.ndarray, W2: np.ndarray,
                                 changepoint: int, sigma: float = 1.0,
                                 seed: int = None) -> Tuple[np.ndarray, int]:
    """
    Generate data from a non-stationary SEM where the DAG changes at changepoint.

    Model: X = X @ W + noise (where W changes at changepoint)
    Uses topological ordering to respect causal structure.

    Parameters
    ----------
    d : int
        Number of nodes
    T : int
        Total number of samples
    W1 : np.ndarray
        DAG for phase 1 (t < changepoint)
    W2 : np.ndarray
        DAG for phase 2 (t >= changepoint)
    changepoint : int
        Time index where structure changes
    sigma : float
        Noise standard deviation
    seed : int
        Random seed

    Returns
    -------
    X : np.ndarray
        (T, d) data matrix
    changepoint : int
        The changepoint index
    """
    rng = np.random.default_rng(seed=seed)

    X = np.zeros((T, d))

    # Phase 1: Generate from W1 using topological ordering
    G1 = nx.DiGraph(W1)
    ordered_vertices_1 = list(nx.topological_sort(G1))

    for t in range(changepoint):
        for j in ordered_vertices_1:
            parents = list(G1.predecessors(j))
            if len(parents) > 0:
                eta = X[t, parents].dot(W1[parents, j])
            else:
                eta = 0
            X[t, j] = eta + rng.normal(scale=sigma)

    # Phase 2: Generate from W2 using topological ordering
    G2 = nx.DiGraph(W2)
    ordered_vertices_2 = list(nx.topological_sort(G2))

    for t in range(changepoint, T):
        for j in ordered_vertices_2:
            parents = list(G2.predecessors(j))
            if len(parents) > 0:
                eta = X[t, parents].dot(W2[parents, j])
            else:
                eta = 0
            X[t, j] = eta + rng.normal(scale=sigma)

    return X, changepoint


# =============================================================================
# ONLINE TRACKING
# =============================================================================

def to_bin(W: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Convert weighted adjacency to binary."""
    return (np.abs(W) > threshold).astype(int)


def compute_metrics(W_true: np.ndarray, W_est: np.ndarray,
                    threshold: float = 0.3, use_topk: bool = False) -> Dict[str, float]:
    """Compute TPR, FDR, SHD between true and estimated DAG.

    Parameters
    ----------
    W_true : np.ndarray
        True adjacency matrix
    W_est : np.ndarray
        Estimated adjacency matrix
    threshold : float
        Weight threshold for edge detection
    use_topk : bool
        If True, select top-k edges by magnitude (k = number of true edges)
        This can help reduce FDR by being more selective
    """
    W_true_bin = to_bin(W_true, 0.0)  # True is already sparse

    if use_topk:
        # Select top-k edges by magnitude where k = number of true edges
        k = int(W_true_bin.sum())
        W_flat = np.abs(W_est).flatten()
        if k > 0:
            threshold_topk = np.partition(W_flat, -k)[-k]
            W_est_bin = (np.abs(W_est) >= threshold_topk).astype(int)
        else:
            W_est_bin = np.zeros_like(W_est, dtype=int)
    else:
        W_est_bin = to_bin(W_est, threshold)

    # True positives, false positives, false negatives
    tp = np.sum((W_true_bin == 1) & (W_est_bin == 1))
    fp = np.sum((W_true_bin == 0) & (W_est_bin == 1))
    fn = np.sum((W_true_bin == 1) & (W_est_bin == 0))

    # Metrics
    tpr = tp / max(tp + fn, 1)
    fdr = fp / max(tp + fp, 1)
    shd = fp + fn  # Structural Hamming Distance

    return {'tpr': tpr, 'fdr': fdr, 'shd': shd, 'tp': tp, 'fp': fp, 'fn': fn}


class OnlineDyCoLiDE:
    """
    Online wrapper for DyCoLiDE that tracks evolving structure.

    Uses a sliding window of real data for adaptation. As new batches arrive,
    old samples are forgotten by sliding the window forward. This is more
    principled than generating synthetic data from covariance and enables
    faster adaptation to structural changes.
    """

    def __init__(self, d: int, forgetting_factor: float = 0.99,
                 window_size: int = 1000, seed: int = 42):
        """
        Parameters
        ----------
        d : int
            Number of nodes
        forgetting_factor : float
            EMA decay factor (0.99 = slow forgetting, 0.9 = fast forgetting)
        window_size : int
            Size of sliding window for real data (default: 1000)
        seed : int
            Random seed
        """
        self.d = d
        self.gamma = forgetting_factor
        self.window_size = window_size
        self.seed = seed

        # Initialize running statistics
        self.cov = np.eye(d)  # Running covariance estimate
        self.mean = np.zeros(d)
        self.n_samples = 0

        # Sliding window of real data
        self.data_window = None

        # Current estimate
        self.W_est = np.zeros((d, d))

    def update_statistics(self, X_batch: np.ndarray):
        """Update running mean and covariance with new batch using EMA."""
        batch_mean = X_batch.mean(axis=0)
        X_centered = X_batch - batch_mean
        batch_cov = X_centered.T @ X_centered / len(X_batch)

        if self.n_samples == 0:
            self.mean = batch_mean
            self.cov = batch_cov
        else:
            # Exponential moving average
            self.mean = self.gamma * self.mean + (1 - self.gamma) * batch_mean
            self.cov = self.gamma * self.cov + (1 - self.gamma) * batch_cov

        self.n_samples += len(X_batch)

    def fit_incremental(self, X_batch: np.ndarray,
                        lambda1: float = 0.05,
                        sgd_batch_size: int = 100,
                        n_iter_warm: int = 10000,
                        n_iter_final: int = 20000,
                        lr: float = 0.001) -> np.ndarray:
        """
        Fit DyCoLiDE on sliding window of real data.

        Uses a sliding window of recent samples to track evolving structure.
        This is more principled than generating synthetic data from covariance.

        Parameters
        ----------
        X_batch : np.ndarray
            New batch of streaming data
        lambda1 : float
            Sparsity penalty
        sgd_batch_size : int
            Batch size for SGD inside CoLiDE training
        n_iter_warm : int
            Warm iterations
        n_iter_final : int
            Final iterations
        lr : float
            Learning rate
        """
        # Update statistics with new batch (for monitoring/debugging)
        self.update_statistics(X_batch)

        # Maintain sliding window of REAL data
        if self.data_window is None:
            self.data_window = X_batch.copy()
        else:
            # Append new data
            self.data_window = np.vstack([self.data_window, X_batch])
            # Keep only the last window_size samples
            if len(self.data_window) > self.window_size:
                self.data_window = self.data_window[-self.window_size:]

        # Run DyCoLiDE batch on REAL data window
        # Adjust batch size if we don't have enough data yet
        effective_batch_size = min(sgd_batch_size, len(self.data_window))

        model = colide_ev_batch(seed=self.seed)
        W_est, sigma_est = model.fit(
            X=self.data_window,  # Use real data!
            lambda1=lambda1,
            T=4,
            batch_size=effective_batch_size,
            n_batches_warm=n_iter_warm,
            n_batches_final=n_iter_final,
            lr=lr,
        )

        self.W_est = W_est
        return W_est


def run_online_experiment(X: np.ndarray, W1: np.ndarray, W2: np.ndarray,
                          changepoint: int,
                          batch_size: int = 50,
                          sgd_batch_size: int = 100,
                          window_size: int = 1000,
                          forgetting_factor: float = 0.95,
                          lambda1: float = 0.05,
                          threshold: float = 0.3,
                          eval_every: int = 5,
                          n_iter_warm: int = 20000,
                          n_iter_final: int = 70000,
                          seed: int = 42) -> Dict:
    """
    Run online DyCoLiDE experiment tracking how estimates evolve.

    Parameters
    ----------
    X : np.ndarray
        (T, d) non-stationary data
    W1 : np.ndarray
        True DAG for phase 1
    W2 : np.ndarray
        True DAG for phase 2
    changepoint : int
        Index where structure changes
    batch_size : int
        Number of samples per streaming batch
    sgd_batch_size : int
        Batch size for SGD inside CoLiDE training
    window_size : int
        Size of sliding window for real data
    forgetting_factor : float
        EMA decay (lower = faster adaptation) - kept for statistics tracking
    lambda1 : float
        Sparsity penalty
    threshold : float
        Edge threshold
    eval_every : int
        Evaluate every N batches
    seed : int
        Random seed

    Returns
    -------
    results : Dict
        Contains time series of metrics and estimates
    """
    T, d = X.shape
    n_batches = T // batch_size

    print(f"Running online experiment: {T} samples, {n_batches} batches of size {batch_size}")
    print(f"Changepoint at sample {changepoint} (batch {changepoint // batch_size})")
    print(f"Sliding window size: {window_size}")

    # Initialize tracker
    tracker = OnlineDyCoLiDE(d, forgetting_factor=forgetting_factor,
                             window_size=window_size, seed=seed)

    # Storage for results
    times = []
    metrics_w1 = []  # Metrics vs W1
    metrics_w2 = []  # Metrics vs W2
    W_estimates = []

    start_time = time.time()

    for batch_idx in range(n_batches):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]

        # Determine which phase we're in
        current_time = end_idx
        in_phase2 = current_time > changepoint

        # Update and fit
        W_est = tracker.fit_incremental(X_batch, lambda1=lambda1,
                                        sgd_batch_size=sgd_batch_size,
                                        n_iter_warm=n_iter_warm,
                                        n_iter_final=n_iter_final)

        # Evaluate every N batches
        if batch_idx % eval_every == 0 or batch_idx == n_batches - 1:
            m1 = compute_metrics(W1, W_est, threshold)
            m2 = compute_metrics(W2, W_est, threshold)

            times.append(current_time)
            metrics_w1.append(m1)
            metrics_w2.append(m2)
            W_estimates.append(W_est.copy())

            # Progress
            phase = "Phase 2" if in_phase2 else "Phase 1"
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx+1}/{n_batches} ({phase}): "
                  f"TPR(W1)={m1['tpr']:.3f}, TPR(W2)={m2['tpr']:.3f}, "
                  f"time={elapsed:.1f}s")

    return {
        'times': np.array(times),
        'metrics_w1': metrics_w1,
        'metrics_w2': metrics_w2,
        'W_estimates': W_estimates,
        'changepoint': changepoint,
        'W1': W1,
        'W2': W2,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_evolution(results: Dict, save_path: str = None):
    """
    Plot how DyCoLiDE's performance evolves over time.

    Shows:
    - TPR vs active DAG (W1 before, W2 after changepoint)
    - FDR vs active DAG
    - SHD vs W2 (only after changepoint - shows adaptation)
    - Combined error metric
    """
    times = results['times']
    changepoint = results['changepoint']

    # Extract metrics
    tpr_w1 = np.array([m['tpr'] for m in results['metrics_w1']])
    tpr_w2 = np.array([m['tpr'] for m in results['metrics_w2']])
    fdr_w1 = np.array([m['fdr'] for m in results['metrics_w1']])
    fdr_w2 = np.array([m['fdr'] for m in results['metrics_w2']])
    shd_w2 = np.array([m['shd'] for m in results['metrics_w2']])  # Only need W2 for post-change adaptation

    # Create "active" metrics - track the currently relevant DAG
    active_tpr = np.where(times <= changepoint, tpr_w1, tpr_w2)
    active_fdr = np.where(times <= changepoint, fdr_w1, fdr_w2)

    # Find changepoint index
    cp_idx = np.searchsorted(times, changepoint)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # =========================================================================
    # Plot 1: TPR vs Active DAG (main performance metric)
    # =========================================================================
    ax = axes[0, 0]
    # Phase 1 (blue)
    ax.plot(times[:cp_idx+1], active_tpr[:cp_idx+1], 'b-', linewidth=2.5,
            label='Phase 1: TPR vs W₁', marker='o', markersize=4)
    # Phase 2 (red)
    ax.plot(times[cp_idx:], active_tpr[cp_idx:], 'r-', linewidth=2.5,
            label='Phase 2: TPR vs W₂', marker='s', markersize=4)
    ax.axvline(x=changepoint, color='green', linestyle='--', linewidth=2.5,
               label=f'DAG Change (t={changepoint})')
    ax.axhspan(0.8, 1.0, alpha=0.1, color='green', label='Good TPR (>0.8)')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('TPR vs Active DAG', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([times[0], times[-1]])
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: FDR vs Active DAG
    # =========================================================================
    ax = axes[0, 1]
    ax.plot(times[:cp_idx+1], active_fdr[:cp_idx+1], 'b-', linewidth=2.5,
            label='Phase 1: FDR vs W₁', marker='o', markersize=4)
    ax.plot(times[cp_idx:], active_fdr[cp_idx:], 'r-', linewidth=2.5,
            label='Phase 2: FDR vs W₂', marker='s', markersize=4)
    ax.axvline(x=changepoint, color='green', linestyle='--', linewidth=2.5)
    ax.axhspan(0, 0.2, alpha=0.1, color='green', label='Good FDR (<0.2)')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('False Discovery Rate', fontsize=12)
    ax.set_title('FDR vs Active DAG', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([times[0], times[-1]])
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: SHD vs W2 (only after changepoint - shows adaptation curve)
    # =========================================================================
    ax = axes[1, 0]
    # Only plot after changepoint
    times_after = times[cp_idx:]
    shd_after = shd_w2[cp_idx:]

    ax.plot(times_after, shd_after, 'r-', linewidth=3,
            label='SHD vs W₂ (New DAG)', marker='s', markersize=5)
    ax.axvline(x=changepoint, color='green', linestyle='--', linewidth=2.5,
               label=f'DAG Change (t={changepoint})')

    # Highlight the adaptation region
    ax.fill_between(times_after, 0, shd_after, alpha=0.2, color='red')

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Structural Hamming Distance', fontsize=12)
    ax.set_title('SHD vs W₂ After Changepoint', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([changepoint - (times[-1] - changepoint) * 0.1, times[-1]])
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 4: Combined Error Over Time (1 - TPR + FDR)
    # =========================================================================
    ax = axes[1, 1]
    error = (1 - active_tpr) + active_fdr  # Combined error metric

    ax.fill_between(times[:cp_idx+1], 0, error[:cp_idx+1], alpha=0.4, color='blue',
                    label='Phase 1 Error')
    ax.fill_between(times[cp_idx:], 0, error[cp_idx:], alpha=0.4, color='red',
                    label='Phase 2 Error')
    ax.plot(times, error, 'k-', linewidth=2, label='Combined Error = (1-TPR) + FDR')
    ax.axvline(x=changepoint, color='green', linestyle='--', linewidth=2.5)

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Error = (1-TPR) + FDR', fontsize=12)
    ax.set_title('Combined Error Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, max(error) * 1.3])
    ax.set_xlim([times[0], times[-1]])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    return fig


def plot_dag_comparison(W1: np.ndarray, W2: np.ndarray, W_est_before: np.ndarray,
                        W_est_after: np.ndarray, threshold: float = 0.3,
                        save_path: str = None):
    """
    Plot DAG comparison: W1, W2, estimates before and after changepoint.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    matrices = [
        (W1, 'True W₁ (Phase 1)'),
        (W2, 'True W₂ (Phase 2)'),
        (to_bin(W_est_before, threshold), 'Estimated (Before Change)'),
        (to_bin(W_est_after, threshold), 'Estimated (After Adaptation)')
    ]

    for ax, (W, title) in zip(axes.flat, matrices):
        im = ax.imshow(np.abs(W) > 0, cmap='Blues', aspect='equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Node')
        ax.set_ylabel('Node')

        # Add edge count
        n_edges = np.sum(np.abs(W) > 0)
        ax.text(0.02, 0.98, f'Edges: {n_edges}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    return fig


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_full_experiment(
    d: int = 20,
    T: int = 10000,
    expected_edges: int = 40,
    change_fraction: float = 0.5,
    batch_size: int = 50,
    sgd_batch_size: int = 100,
    window_size: int = 1000,
    forgetting_factor: float = 0.95,
    lambda1: float = 0.05,
    threshold: float = 0.3,
    n_iter_warm: int = 20000,
    n_iter_final: int = 70000,
    seed: int = 42,
    save_plots: bool = True
):
    """
    Run the full evolving DAG experiment.

    Parameters
    ----------
    d : int
        Number of nodes
    T : int
        Total samples
    expected_edges : int
        Expected number of edges in DAG
    change_fraction : float
        Fraction of edges that change at changepoint
    batch_size : int
        Samples per batch for streaming data processing
    sgd_batch_size : int
        Batch size for SGD inside CoLiDE training
    window_size : int
        Size of sliding window for real data (smaller = faster adaptation)
    forgetting_factor : float
        EMA decay (kept for statistics tracking, not used for adaptation)
    lambda1 : float
        Sparsity penalty
    threshold : float
        Edge threshold
    n_iter_warm : int
        Warm-up iterations per batch (for convergence)
    n_iter_final : int
        Final iterations per batch (for convergence)
    seed : int
        Random seed
    save_plots : bool
        Whether to save plots to files
    """
    print("=" * 80)
    print("EVOLVING DAG EXPERIMENT")
    print("=" * 80)

    # Generate DAGs
    print("\n[1] Generating DAGs...")
    W1 = generate_dag(d, expected_edges, seed=seed)
    W2 = generate_evolved_dag(W1, change_fraction=change_fraction, seed=seed+1)

    n_edges_w1 = np.sum(np.abs(W1) > 0)
    n_edges_w2 = np.sum(np.abs(W2) > 0)

    # Compute overlap
    w1_bin = to_bin(W1, 0)
    w2_bin = to_bin(W2, 0)
    overlap = np.sum((w1_bin == 1) & (w2_bin == 1))
    only_w1 = np.sum((w1_bin == 1) & (w2_bin == 0))
    only_w2 = np.sum((w1_bin == 0) & (w2_bin == 1))

    print(f"  W₁: {n_edges_w1} edges")
    print(f"  W₂: {n_edges_w2} edges")
    print(f"  Overlap: {overlap} edges persist")
    print(f"  Removed: {only_w1} edges (in W₁ only)")
    print(f"  Added: {only_w2} edges (in W₂ only)")

    # Generate data
    print("\n[2] Generating non-stationary data...")
    changepoint = T // 2
    X, _ = generate_nonstationary_data(d, T, W1, W2, changepoint, sigma=1.0, seed=seed)
    print(f"  Total samples: {T}")
    print(f"  Changepoint: {changepoint}")
    print(f"  Phase 1: samples 0-{changepoint-1}")
    print(f"  Phase 2: samples {changepoint}-{T-1}")

    # Run online experiment
    print("\n[3] Running online DyCoLiDE...")
    results = run_online_experiment(
        X, W1, W2, changepoint,
        batch_size=batch_size,
        sgd_batch_size=sgd_batch_size,
        window_size=window_size,
        forgetting_factor=forgetting_factor,
        lambda1=lambda1,
        threshold=threshold,
        eval_every=1,
        n_iter_warm=n_iter_warm,
        n_iter_final=n_iter_final,
        seed=seed
    )

    # Find estimates before and after adaptation
    times = results['times']
    before_idx = np.argmax(times >= changepoint) - 1
    after_idx = len(times) - 1

    W_est_before = results['W_estimates'][max(0, before_idx)]
    W_est_after = results['W_estimates'][after_idx]

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    m1_before = results['metrics_w1'][max(0, before_idx)]
    m2_before = results['metrics_w2'][max(0, before_idx)]
    m1_after = results['metrics_w1'][after_idx]
    m2_after = results['metrics_w2'][after_idx]

    print(f"\nBefore changepoint (tracking W₁):")
    print(f"  TPR vs W₁: {m1_before['tpr']:.3f}")
    print(f"  FDR vs W₁: {m1_before['fdr']:.3f}")
    print(f"  SHD vs W₁: {m1_before['shd']}")

    print(f"\nAfter adaptation (tracking W₂):")
    print(f"  TPR vs W₂: {m2_after['tpr']:.3f}")
    print(f"  FDR vs W₂: {m2_after['fdr']:.3f}")
    print(f"  SHD vs W₂: {m2_after['shd']}")

    # Plot results
    print("\n[4] Generating plots...")

    plot_path = 'evolving_dag_metrics.png' if save_plots else None
    plot_evolution(results, save_path=plot_path)

    dag_path = 'evolving_dag_comparison.png' if save_plots else None
    plot_dag_comparison(W1, W2, W_est_before, W_est_after, threshold, save_path=dag_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evolving DAG Experiment')
    parser.add_argument('--d', type=int, default=50, help='Number of nodes')
    parser.add_argument('--T', type=int, default=2000, help='Total samples')
    parser.add_argument('--edges', type=int, default=200, help='Expected edges')
    parser.add_argument('--change', type=float, default=0.5, help='Fraction of edges that change')
    parser.add_argument('--batch', type=int, default=25, help='Streaming data batch size')
    parser.add_argument('--sgd-batch', type=int, default=100, help='SGD batch size for CoLiDE training')
    parser.add_argument('--window-size', type=int, default=100, help='Sliding window size for real data')
    parser.add_argument('--gamma', type=float, default=0.9, help='Forgetting factor (for statistics tracking)')
    parser.add_argument('--lambda1', type=float, default=0.05, help='Sparsity penalty')
    parser.add_argument('--threshold', type=float, default=0.3, help='Edge threshold')
    parser.add_argument('--warm-iter', type=int, default=20000, help='Warm-up iterations per batch')
    parser.add_argument('--final-iter', type=int, default=70000, help='Final iterations per batch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots')

    args = parser.parse_args()
    
    results = run_full_experiment(
        d=args.d,
        T=args.T,
        expected_edges=args.edges,
        change_fraction=args.change,
        batch_size=args.batch,
        sgd_batch_size=args.sgd_batch,
        window_size=args.window_size,
        forgetting_factor=args.gamma,
        lambda1=args.lambda1,
        threshold=args.threshold,
        n_iter_warm=args.warm_iter,
        n_iter_final=args.final_iter,
        seed=args.seed,
        save_plots=not args.no_save
    )
