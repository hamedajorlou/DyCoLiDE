"""
SVAR Data Generator

Generates time series data from Structural Vector Autoregression (SVAR) models:
    X_t = (I - B)^{-1} * (sum_{k=1}^p A_k X_{t-k} + epsilon_t)

Where:
- B: Instantaneous causal structure (DAG) - what we want to learn
- A_k: Temporal lag matrices (lag k)
- p: Lag order
- epsilon_t: Independent noise (typically Gaussian)

Usage:
    from SVAR_data_generator import generate_svar_data

    X, B_true, A_true, params = generate_svar_data(
        n_nodes=20, n_timesteps=1000, lag_order=2,
        instantaneous_edges=30, temporal_sparsity=0.3
    )
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional


def generate_dag_structure(n_nodes: int, n_edges: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random DAG structure using Erdos-Renyi graph.

    Parameters:
    -----------
    n_nodes : int
        Number of nodes
    n_edges : int
        Number of edges in the DAG
    seed : int, optional
        Random seed

    Returns:
    --------
    W : np.ndarray, shape (n_nodes, n_nodes)
        Weighted adjacency matrix (with random weights)
    """
    rng = np.random.default_rng(seed=seed)

    # Generate ER graph
    prob = float(n_edges * 2) / float(n_nodes**2 - n_nodes)
    G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
    adj = nx.to_numpy_array(G)
    U_mask = np.triu(adj, k=1)

    # Random permutation to create DAG
    P = np.eye(n_nodes)
    P = P[:, rng.permutation(n_nodes)]
    W = P @ U_mask @ P.T

    # Add random weights (mixed signs)
    W_weighted = np.zeros(W.shape)
    S = rng.integers(2, size=W.shape)
    for i, (low, high) in enumerate([(-2.0, -0.5), (0.5, 2.0)]):
        weights = rng.uniform(low=low, high=high, size=W.shape)
        W_weighted += W * (S == i) * weights

    return W_weighted


def generate_temporal_structure(n_nodes: int, lag_order: int, sparsity: float = 0.3,
                                strength: float = 0.3, n_edges: Optional[int] = None,
                                seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Generate temporal lag matrices A_1, ..., A_p.

    Parameters:
    -----------
    n_nodes : int
        Number of nodes
    lag_order : int
        Number of lags (p)
    sparsity : float
        Proportion of non-zero entries in each A_k (0 to 1). Ignored if n_edges is set.
    strength : float
        Maximum absolute value of temporal coefficients
    n_edges : int, optional
        Exact number of edges per lag matrix. If set, overrides sparsity.
    seed : int, optional
        Random seed

    Returns:
    --------
    A_list : list of np.ndarray
        List of lag matrices [A_1, A_2, ..., A_p]
    """
    rng = np.random.default_rng(seed=seed)

    A_list = []

    for k in range(lag_order):
        # Create sparse random matrix
        A_k = np.zeros((n_nodes, n_nodes))

        if n_edges is not None:
            # Select exactly n_edges random positions
            all_positions = [(i, j) for i in range(n_nodes) for j in range(n_nodes)]
            selected = rng.choice(len(all_positions), size=min(n_edges, len(all_positions)), replace=False)
            mask = np.zeros((n_nodes, n_nodes), dtype=bool)
            for idx in selected:
                i, j = all_positions[idx]
                mask[i, j] = True
        else:
            # Randomly select non-zero entries based on sparsity
            mask = rng.random((n_nodes, n_nodes)) < sparsity

        # Add random weights with mixed signs
        S = rng.integers(2, size=(n_nodes, n_nodes))
        for i, (low, high) in enumerate([(-strength, -strength/2), (strength/2, strength)]):
            weights = rng.uniform(low=low, high=high, size=(n_nodes, n_nodes))
            A_k += mask * (S == i) * weights

        # Decay strength with lag order (earlier lags have stronger effects)
        decay_factor = 0.7 ** k
        A_k = A_k * decay_factor

        A_list.append(A_k)

    return A_list


def check_stability(A_list: List[np.ndarray], n_nodes: int, B: Optional[np.ndarray] = None) -> bool:
    """
    Check if SVAR process is stable.

    For SVAR: X_t = (I-B)^{-1} * (sum A_k X_{t-k} + eps_t)
    Define C_k = (I-B)^{-1} @ A_k, then stability requires all eigenvalues
    of the companion matrix built from C_k to have magnitude < 1.

    Parameters:
    -----------
    A_list : list of np.ndarray
        Lag matrices [A_1, ..., A_p]
    n_nodes : int
        Number of nodes
    B : np.ndarray, optional
        Instantaneous structure. If provided, check SVAR stability.
        If None, check VAR stability.

    Returns:
    --------
    is_stable : bool
        True if process is stable
    """
    p = len(A_list)

    # If B is provided, transform A_k to C_k = (I-B)^{-1} @ A_k
    if B is not None:
        I = np.eye(n_nodes)
        try:
            B_inv = np.linalg.inv(I - B)
            # Check if B_inv has reasonable magnitude
            if np.max(np.abs(B_inv)) > 100:
                return False
            C_list = [B_inv @ A_k for A_k in A_list]
        except np.linalg.LinAlgError:
            return False
    else:
        C_list = A_list

    # Construct companion matrix
    # [C_1  C_2  ...  C_{p-1}  C_p]
    # [I    0    ...    0        0 ]
    # [0    I    ...    0        0 ]
    # ...
    # [0    0    ...    I        0 ]

    companion = np.zeros((n_nodes * p, n_nodes * p))

    # First row: [C_1, C_2, ..., C_p]
    for i, C_k in enumerate(C_list):
        companion[:n_nodes, i*n_nodes:(i+1)*n_nodes] = C_k

    # Identity blocks below
    if p > 1:
        companion[n_nodes:, :n_nodes*(p-1)] = np.eye(n_nodes * (p-1))

    # Check eigenvalues
    eigenvalues = np.linalg.eigvals(companion)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    return max_eigenvalue < 0.95  # Use 0.95 for safety margin


def simulate_svar(B: np.ndarray, A_list: List[np.ndarray], n_timesteps: int,
                  noise_scale: float = 1.0, noise_type: str = 'ev',
                  noise_scales: Optional[np.ndarray] = None, burnin: int = 500,
                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate SVAR time series.

    Model: X_t = (I - B)^{-1} * (sum_{k=1}^p A_k X_{t-k} + epsilon_t)

    Parameters:
    -----------
    B : np.ndarray, shape (n_nodes, n_nodes)
        Instantaneous causal structure (DAG)
    A_list : list of np.ndarray
        Temporal lag matrices [A_1, ..., A_p]
    n_timesteps : int
        Number of time steps to generate
    noise_scale : float
        Standard deviation of noise (used for EV, or as base for NV)
    noise_type : str
        'ev' for equal variance (homoscedastic), 'nv' for non-equal variance (heteroscedastic)
    noise_scales : np.ndarray, optional
        Node-specific noise scales for NV case. If None, will be generated randomly.
    burnin : int
        Number of initial timesteps to discard
    seed : int, optional
        Random seed

    Returns:
    --------
    X : np.ndarray, shape (n_timesteps, n_nodes)
        Generated time series
    sigma_true : np.ndarray, shape (n_nodes,)
        True noise standard deviations per node
    """
    rng = np.random.default_rng(seed=seed)

    n_nodes = B.shape[0]
    p = len(A_list)

    # Compute (I - B)^{-1}
    I = np.eye(n_nodes)
    try:
        B_inv = np.linalg.inv(I - B)
    except np.linalg.LinAlgError:
        raise ValueError("(I - B) is singular! B must be a valid DAG structure.")

    # Set up noise scales based on noise_type
    if noise_type == 'ev':
        # Equal variance: all nodes have the same noise scale
        sigma_true = np.ones(n_nodes) * noise_scale
    elif noise_type == 'nv':
        # Non-equal variance: each node has different noise scale
        if noise_scales is not None:
            sigma_true = noise_scales
        else:
            # Generate random noise scales in range [0.5*noise_scale, 2.0*noise_scale]
            sigma_true = rng.uniform(0.5 * noise_scale, 2.0 * noise_scale, size=n_nodes)
    else:
        raise ValueError(f"noise_type must be 'ev' or 'nv', got '{noise_type}'")

    # Total timesteps including burnin
    total_timesteps = n_timesteps + burnin

    # Initialize with zeros
    X = np.zeros((total_timesteps, n_nodes))

    # Generate time series
    for t in range(p, total_timesteps):
        # Temporal component: sum_{k=1}^p A_k X_{t-k}
        temporal_effect = np.zeros(n_nodes)
        for k, A_k in enumerate(A_list):
            temporal_effect += A_k @ X[t - k - 1]

        # Innovation with node-specific variance
        epsilon_t = rng.normal(scale=sigma_true, size=n_nodes)

        # SVAR equation: X_t = (I - B)^{-1} * (temporal + noise)
        X[t] = B_inv @ (temporal_effect + epsilon_t)

    # Remove burnin period
    X = X[burnin:]

    return X, sigma_true


def generate_svar_data(n_nodes: int = 20, n_timesteps: int = 1000,
                       lag_order: int = 2, instantaneous_edges: int = 30,
                       temporal_sparsity: float = 0.3, temporal_edges: Optional[int] = None,
                       temporal_strength: float = 0.3,
                       noise_scale: float = 1.0, noise_type: str = 'ev',
                       seed: Optional[int] = None,
                       max_stability_attempts: int = 10) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict]:
    """
    Generate complete SVAR dataset with known ground truth.

    Parameters:
    -----------
    n_nodes : int
        Number of nodes/variables
    n_timesteps : int
        Number of time steps
    lag_order : int
        VAR lag order (p)
    instantaneous_edges : int
        Number of edges in instantaneous DAG (B)
    temporal_sparsity : float
        Sparsity of temporal matrices (proportion of non-zero entries). Ignored if temporal_edges is set.
    temporal_edges : int, optional
        Exact number of edges per lag matrix. If set, overrides temporal_sparsity.
        Use this to match instantaneous_edges for balanced comparison.
    temporal_strength : float
        Maximum strength of temporal effects
    noise_scale : float
        Standard deviation of innovations (base scale for NV)
    noise_type : str
        'ev' for equal variance (homoscedastic), 'nv' for non-equal variance (heteroscedastic)
    seed : int, optional
        Random seed for reproducibility
    max_stability_attempts : int
        Maximum attempts to generate stable VAR

    Returns:
    --------
    X : np.ndarray, shape (n_timesteps, n_nodes)
        Generated time series
    B : np.ndarray, shape (n_nodes, n_nodes)
        True instantaneous causal structure (DAG)
    A_list : list of np.ndarray
        True temporal lag matrices [A_1, ..., A_p]
    params : dict
        Generation parameters and statistics (includes 'sigma_true' for NV case)
    """
    rng = np.random.default_rng(seed=seed)

    # Generate structures with stability check
    stable = False
    attempts = 0

    while not stable and attempts < max_stability_attempts:
        # Reduce temporal strength if previous attempts failed
        current_strength = temporal_strength * (0.8 ** attempts)

        # Generate instantaneous DAG structure (B)
        B = generate_dag_structure(n_nodes, instantaneous_edges,
                                  seed=None if seed is None else seed + attempts)

        # Generate temporal structures
        A_list = generate_temporal_structure(
            n_nodes, lag_order, sparsity=temporal_sparsity,
            strength=current_strength, n_edges=temporal_edges,
            seed=None if seed is None else seed + attempts + 1000
        )

        # Check SVAR stability (includes both B and A_k)
        stable = check_stability(A_list, n_nodes, B=B)
        attempts += 1

    if not stable:
        raise ValueError(f"Could not generate stable SVAR after {max_stability_attempts} attempts. "
                        f"Try reducing temporal_strength or temporal_sparsity.")

    # Generate time series
    X, sigma_true = simulate_svar(B, A_list, n_timesteps, noise_scale=noise_scale,
                                  noise_type=noise_type,
                                  seed=None if seed is None else seed + 1000)

    # Compute statistics
    A_edges_list = [int(np.sum(np.abs(A) > 0)) for A in A_list]
    params = {
        'n_nodes': n_nodes,
        'n_timesteps': n_timesteps,
        'lag_order': lag_order,
        'instantaneous_edges': int(np.sum(np.abs(B) > 0)),
        'temporal_sparsity': temporal_sparsity,
        'temporal_edges': temporal_edges,
        'temporal_strength': temporal_strength,
        'noise_scale': noise_scale,
        'actual_temporal_strength': current_strength,
        'stability_attempts': attempts,
        'B_edges': int(np.sum(np.abs(B) > 0)),
        'A_edges_per_lag': A_edges_list,
        'A_total_edges': sum(A_edges_list),
        'A_avg_edges': np.mean(A_edges_list),
    }

    return X, B, A_list, params


def print_svar_summary(X: np.ndarray, B: np.ndarray, A_list: List[np.ndarray], params: Dict):
    """Print summary of generated SVAR data."""
    print("=" * 80)
    print(" SVAR DATA GENERATION SUMMARY")
    print("=" * 80)

    print(f"\nData Dimensions:")
    print(f"  Time steps: {X.shape[0]}")
    print(f"  Variables:  {X.shape[1]}")
    print(f"  Lag order:  {len(A_list)}")

    print(f"\nInstantaneous Structure (B):")
    print(f"  Edges: {params['B_edges']}")
    print(f"  Density: {params['B_edges'] / (X.shape[1]**2 - X.shape[1]):.3f}")
    print(f"  Weight range: [{np.min(B[B != 0]):.3f}, {np.max(B[B != 0]):.3f}]")

    print(f"\nTemporal Structures (A_k):")
    for k, A_k in enumerate(A_list):
        n_edges = np.sum(np.abs(A_k) > 0)
        if n_edges > 0:
            weight_range = f"[{np.min(A_k[A_k != 0]):.3f}, {np.max(A_k[A_k != 0]):.3f}]"
        else:
            weight_range = "[N/A]"
        print(f"  A_{k+1}: {n_edges} edges, weights {weight_range}")

    print(f"\nData Statistics:")
    print(f"  Mean: {np.mean(X):.4f}")
    print(f"  Std:  {np.std(X):.4f}")
    print(f"  Min:  {np.min(X):.4f}")
    print(f"  Max:  {np.max(X):.4f}")

    print(f"\nStability:")
    print(f"  Attempts to generate stable VAR: {params['stability_attempts']}")
    print(f"  Final temporal strength: {params['actual_temporal_strength']:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    # Example 1: Small SVAR for testing
    print("\nExample 1: Small SVAR (20 nodes, lag 2)")
    print("-" * 80)

    X, B, A_list, params = generate_svar_data(
        n_nodes=20,
        n_timesteps=1000,
        lag_order=2,
        instantaneous_edges=30,
        temporal_sparsity=0.3,
        temporal_strength=0.3,
        seed=123
    )

    print_svar_summary(X, B, A_list, params)

    # # Example 2: Medium SVAR
    # print("\n\nExample 2: Medium SVAR (50 nodes, lag 3)")
    # print("-" * 80)

    # X, B, A_list, params = generate_svar_data(
    #     n_nodes=50,
    #     n_timesteps=2000,
    #     lag_order=3,
    #     instantaneous_edges=150,
    #     temporal_sparsity=0.2,
    #     temporal_strength=0.25,
    #     seed=456
    # )

    # print_svar_summary(X, B, A_list, params)

    # # Example 3: High-order lags
    # print("\n\nExample 3: High-order SVAR (20 nodes, lag 5)")
    # print("-" * 80)

    # X, B, A_list, params = generate_svar_data(
    #     n_nodes=20,
    #     n_timesteps=1500,
    #     lag_order=5,
    #     instantaneous_edges=25,
    #     temporal_sparsity=0.25,
    #     temporal_strength=0.2,
    #     seed=789
    # )

    # print_svar_summary(X, B, A_list, params)
