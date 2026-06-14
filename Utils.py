from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import networkx as nx
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import time

def is_dag(W):
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def create_dag(n_nodes, graph_type, edges, permute=False, edge_type='positive', w_range=(0, 0.5),
               rew_prob=.1):
    """
    edge_type can be binary, positive, or weighted
    """
    if 'er' in graph_type:
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob)
        W = np.tril(nx.to_numpy_array(G), k=-1)

    elif graph_type == 'sf' or graph_type == 'sf_t':
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m)
        adj = nx.to_numpy_array(G)
        W = np.tril(adj, k=-1)

    elif graph_type == 'sw' or graph_type == 'sw_t':
        G = nx.watts_strogatz_graph(n_nodes, int(2*round(edges/n_nodes)), rew_prob)
        adj = nx.to_numpy_array(G)
        W = np.tril(adj, k=-1)

    else:
        raise ValueError('Unknown graph type')

    assert nx.is_weighted(G) == False
    assert nx.is_empty(G) == False

    if permute:
        P = np.eye(n_nodes)
        P = P[:, np.random.permutation(n_nodes)]
        W = P @ W @ P.T

    if edge_type == 'binary':
        W_weighted = W.copy()
    elif edge_type == 'positive':
        weights = np.random.uniform(w_range[0], w_range[1], size=W.shape)
        W_weighted = weights * W
    elif edge_type == 'weighted':
        # Default range: w_range=((-2.0, -0.5), (0.5, 2.0))
        W_weighted = np.zeros(W.shape)
        S = np.random.randint(len(w_range), size=W.shape)
        for i, (low, high) in enumerate(w_range):
            weights = np.random.uniform(low=low, high=high, size=W.shape)
            W_weighted += W * (S == i) * weights
    else:
        raise ValueError('Unknown edge type')

    dag = nx.from_numpy_array(W_weighted, create_using=nx.DiGraph)
    # assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
    return W_weighted, dag
    # return W_weighted, dag



try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional networkx dependency
    nx = None  # type: ignore


def _structure_to_adjacency(
    structure: object, node_order: Optional[Sequence] = None
) -> Tuple[np.ndarray, Tuple]:
    """
    Convert a StructureModel / DiGraph / adjacency matrix into a numpy array.
    Returns the adjacency matrix (float) and the node ordering used.
    """
    if isinstance(structure, np.ndarray):
        adj = np.array(structure, dtype=float, copy=True)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        return adj, tuple(range(adj.shape[0]))

    if nx is None or not isinstance(structure, nx.DiGraph):
        raise TypeError(
            "structure must be either a square numpy array or a networkx.DiGraph."
        )

    nodes = tuple(node_order) if node_order is not None else tuple(structure.nodes())
    if len(nodes) == 0:
        raise ValueError("Provided structure contains no nodes.")

    adj = nx.to_numpy_array(structure, nodelist=nodes, weight="weight", dtype=float)
    return adj, nodes


def plot_svar_structure(dag, A_lags, node_order=None, seed=0):
    if plt is None:
        print("matplotlib.pyplot is not installed. Skipping plot.")
        return
    nodes = list(node_order) if node_order is not None else list(dag.nodes())
    matrices = [("B0", nx.to_numpy_array(dag, nodelist=nodes), dag)]
    matrices += [(f"A{idx+1}", mat, None) for idx, mat in enumerate(A_lags)]

    fig, axes = plt.subplots(2, len(matrices), figsize=(5 * len(matrices), 8), constrained_layout=True)
    if len(matrices) == 1:
        axes = axes.reshape(2, 1)

    for col, (label, mat, graph) in enumerate(matrices):
        vmax = np.abs(mat).max() or 1.0
        im = axes[0, col].imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("parent")
        axes[0, col].set_ylabel("child")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        axes[1, col].axis("off")
        G = graph or nx.DiGraph()
        if graph is None:
            G.add_nodes_from(nodes)
            rows, cols_idx = np.where(np.abs(mat) > 1e-8)
            G.add_edges_from((nodes[j], nodes[i]) for i, j in zip(rows, cols_idx))
        if G.number_of_edges():
            pos = nx.spring_layout(G, seed=seed + col)
            nx.draw_networkx(G, pos, ax=axes[1, col], node_color="#87CEEB", edge_color="#444", width=2.5, font_size=9)

    plt.suptitle("Contemporaneous and lagged SVAR structure")
    plt.show()




def to_bin(A, thr):
    B = A.copy().astype(float)
    B[np.abs(B) < thr] = 0.0
    B[B != 0.0] = 1.0
    np.fill_diagonal(B, 0.0)
    return B


def count_accuracy(W_bin_true, W_bin_est):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1},
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    pred_und = np.flatnonzero(W_bin_est == -1)
    pred = np.flatnonzero(W_bin_est == 1)
    cond = np.flatnonzero(W_bin_true)
    cond_reversed = np.flatnonzero(W_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # Compute SHD
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    pred_lower = np.flatnonzero(np.tril(W_bin_est + W_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(W_bin_true + W_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # Compute TPR
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    tpr = float(len(true_pos)) / max(len(cond), 1)

    # Compute FDR
    pred_size = len(pred) + len(pred_und)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)

    return shd, tpr, fdr


def simulate_sem(n_nodes, n_samples, edges, graph_type='er', edge_type='weighted', var_type='ev', noise='normal', var=1.0, w_range=((-2.0, -0.5), (0.5, 2.0)), seed=123):

    rng = np.random.default_rng(seed=seed)
    if graph_type == 'er':
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
        adj = nx.to_numpy_array(G)
        U_mask = np.triu(adj, k=1)
        P = np.eye(n_nodes)
        P = P[:, rng.permutation(n_nodes)]
        W = P @ U_mask @ P.T
    elif graph_type == 'sf':
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m, seed=seed)
        adj = nx.to_numpy_array(G)
        W = np.tril(adj, k=-1)
    else:
        raise ValueError('Unknown graph type')
        
    assert nx.is_weighted(G)==False
    assert nx.is_empty(G)==False

    if edge_type == 'binary':
        W_weighted = W.copy()
    elif edge_type == 'weighted':
        W_weighted = np.zeros(W.shape)
        S = np.random.randint(len(w_range), size=W.shape)
        for i, (low, high) in enumerate(w_range):
            weights = np.random.uniform(low=low, high=high, size=W.shape)
            W_weighted += W * (S == i) * weights
    else:
        raise ValueError('Unknown edge type')
    G_sem = nx.DiGraph(W_weighted)

    X = np.zeros((n_samples, n_nodes))
    ordered_vertices = list(nx.topological_sort(G_sem))
    assert len(ordered_vertices) == n_nodes
    var_nv = rng.uniform(0.5,10.0,n_nodes)
    
    t_start = time.time()
    for j in ordered_vertices:
        parents = list(G_sem.predecessors(j))
        eta = X[:, parents].dot(W_weighted[parents, j])
        if var_type =='ev':
            if noise == 'normal':
                scale = np.sqrt(var)
                X[:, j] = eta + rng.normal(scale=scale, size=(n_samples))
            elif noise == 'exp':
                scale = np.sqrt(var)
                X[:, j] = eta + rng.exponential(scale=scale, size=(n_samples))
            elif noise == 'laplace':
                scale = np.sqrt(var / 2.0)
                X[:, j] = eta + rng.laplace(loc=0.0, scale=scale, size=(n_samples))
            elif noise == 'gumbel':
                scale = np.sqrt(6.0 * var) / np.pi
                X[:, j] = eta + rng.gumbel(loc=0.0, scale=scale, size=(n_samples))
            else:
                raise ValueError('Noise type error!')
        elif var_type =='nv':
            if noise == 'normal':
                scale = np.sqrt(var_nv[j])
                X[:, j] = eta + rng.normal(scale=scale, size=(n_samples))
            elif noise == 'exp':
                scale = np.sqrt(var_nv[j])
                X[:, j] = eta + rng.exponential(scale=scale, size=(n_samples))
            elif noise == 'laplace':
                scale = np.sqrt(var_nv[j] / 2.0)
                X[:, j] = eta + rng.laplace(loc=0.0, scale=scale, size=(n_samples))
            elif noise == 'gumbel':
                scale = np.sqrt(6.0 * var_nv[j]) / np.pi
                X[:, j] = eta + rng.gumbel(loc=0.0, scale=scale, size=(n_samples))
            else:
                raise ValueError('Noise type error!')
        else:
            raise ValueError('Variance type error!')

    t_end = time.time()
    assert is_dag(W_weighted)==True
    print('The data generation is finished! It took', t_end-t_start, 'seconds.')
    
    return X, W_weighted, var_nv

def is_dag(W):
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def to_dag(W, thr=0.3):
    A = np.copy(W)
    A[np.abs(A) <= thr] = 0

    if is_dag(A):
        return A

    nonzero_indices = np.where(A != 0)
    weight_indices_ls = list(zip(A[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))
    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(A):
            break
        A[j, i] = 0

    return A


# =============================================================================
# SVAR data generation.
#
# Model: X_t = (I - B)^{-1} (sum_{k=1}^p A_k X_{t-k} + epsilon_t)
# =============================================================================
def generate_dag_structure(n_nodes: int, n_edges: int, seed: Optional[int] = None) -> np.ndarray:
    """Random ER-style DAG with mixed-sign uniform weights on
    [-2, -0.5] ∪ [0.5, 2.0]."""
    rng = np.random.default_rng(seed=seed)
    prob = float(n_edges * 2) / float(n_nodes ** 2 - n_nodes)
    G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
    adj = nx.to_numpy_array(G)
    U_mask = np.triu(adj, k=1)
    P = np.eye(n_nodes)
    P = P[:, rng.permutation(n_nodes)]
    W = P @ U_mask @ P.T
    W_weighted = np.zeros(W.shape)
    S = rng.integers(2, size=W.shape)
    for i, (low, high) in enumerate([(-2.0, -0.5), (0.5, 2.0)]):
        weights = rng.uniform(low=low, high=high, size=W.shape)
        W_weighted += W * (S == i) * weights
    return W_weighted


def generate_temporal_structure(n_nodes: int, lag_order: int, sparsity: float = 0.3,
                                strength: float = 0.3, n_edges: Optional[int] = None,
                                seed: Optional[int] = None) -> List[np.ndarray]:
    """Build lag matrices A_1, ..., A_p. If `n_edges` is given, each
    matrix has exactly that many non-zeros; otherwise the density is
    set by `sparsity`. Earlier lags get stronger coefficients via a
    0.7^k decay."""
    rng = np.random.default_rng(seed=seed)
    A_list = []
    for k in range(lag_order):
        A_k = np.zeros((n_nodes, n_nodes))
        if n_edges is not None:
            all_positions = [(i, j) for i in range(n_nodes) for j in range(n_nodes)]
            selected = rng.choice(len(all_positions),
                                  size=min(n_edges, len(all_positions)),
                                  replace=False)
            mask = np.zeros((n_nodes, n_nodes), dtype=bool)
            for idx in selected:
                i, j = all_positions[idx]
                mask[i, j] = True
        else:
            mask = rng.random((n_nodes, n_nodes)) < sparsity
        S = rng.integers(2, size=(n_nodes, n_nodes))
        for i, (low, high) in enumerate([(-strength, -strength / 2),
                                         (strength / 2, strength)]):
            weights = rng.uniform(low=low, high=high, size=(n_nodes, n_nodes))
            A_k += mask * (S == i) * weights
        A_k = A_k * (0.7 ** k)
        A_list.append(A_k)
    return A_list


def check_stability(A_list: List[np.ndarray], n_nodes: int,
                    B: Optional[np.ndarray] = None) -> bool:
    """Companion-matrix stability test for the SVAR

        X_t = (I-B)^{-1} (sum_k A_k X_{t-k} + eps_t).

    Evolution operator is C_k = (I-B)^{-1} A_k (no transpose — A acts
    directly on x_{t-1}); the companion matrix built from {C_k} must
    have spectral radius below the safety margin 0.95.
    """
    p = len(A_list)
    if B is not None:
        I = np.eye(n_nodes)
        try:
            B_inv = np.linalg.inv(I - B)
            if np.max(np.abs(B_inv)) > 100:
                return False
            C_list = [B_inv @ A_k for A_k in A_list]
        except np.linalg.LinAlgError:
            return False
    else:
        C_list = A_list
    companion = np.zeros((n_nodes * p, n_nodes * p))
    for i, C_k in enumerate(C_list):
        companion[:n_nodes, i * n_nodes:(i + 1) * n_nodes] = C_k
    if p > 1:
        companion[n_nodes:, :n_nodes * (p - 1)] = np.eye(n_nodes * (p - 1))
    max_eigenvalue = np.max(np.abs(np.linalg.eigvals(companion)))
    return max_eigenvalue < 0.95


def simulate_svar(B: np.ndarray, A_list: List[np.ndarray], n_timesteps: int,
                  noise_scale: float = 1.0, noise_type: str = 'ev',
                  noise_scales: Optional[np.ndarray] = None, burnin: int = 500,
                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one SVAR phase

        X_t = (I-B)^{-1} (sum_k A_k X_{t-k} + epsilon_t).

    Returns (X, sigma_true). For `noise_type='nv'`, `noise_scales` can
    fix the per-node sigmas; if None they are drawn from
    [0.5*noise_scale, 2.0*noise_scale].
    """
    rng = np.random.default_rng(seed=seed)
    n_nodes = B.shape[0]
    p = len(A_list)
    I = np.eye(n_nodes)
    try:
        B_inv = np.linalg.inv(I - B)
    except np.linalg.LinAlgError:
        raise ValueError("(I - B) is singular! B must be a valid DAG structure.")
    if noise_type == 'ev':
        sigma_true = np.ones(n_nodes) * noise_scale
    elif noise_type == 'nv':
        if noise_scales is not None:
            sigma_true = noise_scales
        else:
            sigma_true = rng.uniform(0.5 * noise_scale, 2.0 * noise_scale,
                                     size=n_nodes)
    else:
        raise ValueError(f"noise_type must be 'ev' or 'nv', got '{noise_type}'")
    total_timesteps = n_timesteps + burnin
    X = np.zeros((total_timesteps, n_nodes))
    for t in range(p, total_timesteps):
        temporal_effect = np.zeros(n_nodes)
        for k, A_k in enumerate(A_list):
            temporal_effect += A_k @ X[t - k - 1]
        epsilon_t = rng.normal(scale=sigma_true, size=n_nodes)
        X[t] = B_inv @ (temporal_effect + epsilon_t)
    return X[burnin:], sigma_true


def generate_svar_data(n_nodes: int = 20, n_timesteps: int = 1000,
                       lag_order: int = 2, instantaneous_edges: int = 30,
                       temporal_sparsity: float = 0.3,
                       temporal_edges: Optional[int] = None,
                       temporal_strength: float = 0.3,
                       noise_scale: float = 1.0, noise_type: str = 'ev',
                       seed: Optional[int] = None,
                       max_stability_attempts: int = 10
                       ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict]:
    """Sample a stable SVAR and simulate it.

    Model:  X_t = (I - B)^{-1} ( sum_k A_k X_{t-k} + epsilon_t ).

    Tries up to ``max_stability_attempts`` random (B, A_1..A_p) draws,
    decaying ``temporal_strength`` by 0.8 each retry, until
    :func:`check_stability` passes (companion spectral radius < 0.95).
    Then simulates ``n_timesteps`` post-burn-in samples.

    Returns
    -------
    X : (n_timesteps, n_nodes) array.
    B : (n_nodes, n_nodes) contemporaneous DAG, mixed-sign weights in
        [-2, -0.5] u [0.5, 2].
    A_list : list of p lag matrices (n_nodes, n_nodes).
    params : dict with realized edge counts, ``stability_attempts``,
        ``actual_temporal_strength``, and ``sigma_true``.

    Raises ``ValueError`` if no stable draw is found in the retry budget
    — reduce ``temporal_strength`` or the edge density.

    Convention: a row of ``X`` is one sample. The corresponding
    DyCoLiDE / CoLiDE estimands are the transposes (``W = B^T``,
    ``A_k_DyCoLiDE = A_k^T``).
    """
    stable = False
    attempts = 0
    while not stable and attempts < max_stability_attempts:
        current_strength = temporal_strength * (0.8 ** attempts)
        if seed is not None:
            import random as _random
            _random.seed(seed + attempts)
            np.random.seed(seed + attempts)
        B, _ = create_dag(n_nodes, 'er', instantaneous_edges, permute=True,
                          edge_type='weighted',
                          w_range=((-2.0, -0.5), (0.5, 2.0)))
        A_list = generate_temporal_structure(
            n_nodes, lag_order, sparsity=temporal_sparsity,
            strength=current_strength, n_edges=temporal_edges,
            seed=None if seed is None else seed + attempts + 1000)
        stable = check_stability(A_list, n_nodes, B=B)
        attempts += 1
    if not stable:
        raise ValueError(
            f"Could not generate stable SVAR after {max_stability_attempts} "
            f"attempts. Try reducing temporal_strength or temporal_sparsity.")
    X, sigma_true = simulate_svar(B, A_list, n_timesteps,
                                  noise_scale=noise_scale,
                                  noise_type=noise_type,
                                  seed=None if seed is None else seed + 1000)
    A_edges_list = [int(np.sum(np.abs(A) > 0)) for A in A_list]
    params = {
        'n_nodes': n_nodes, 'n_timesteps': n_timesteps,
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
        'A_avg_edges': float(np.mean(A_edges_list)),
        'sigma_true': sigma_true,
    }
    return X, B, A_list, params


def print_svar_summary(X: np.ndarray, B: np.ndarray,
                       A_list: List[np.ndarray], params: Dict):
    """Pretty-print structure/density/stability stats for a generated SVAR."""
    print("=" * 80)
    print(" SVAR DATA GENERATION SUMMARY")
    print("=" * 80)
    print(f"\nData Dimensions:")
    print(f"  Time steps: {X.shape[0]}")
    print(f"  Variables:  {X.shape[1]}")
    print(f"  Lag order:  {len(A_list)}")
    print(f"\nInstantaneous Structure (B):")
    print(f"  Edges: {params['B_edges']}")
    print(f"  Density: {params['B_edges'] / (X.shape[1] ** 2 - X.shape[1]):.3f}")
    if params['B_edges']:
        print(f"  Weight range: [{np.min(B[B != 0]):.3f}, "
              f"{np.max(B[B != 0]):.3f}]")
    print(f"\nTemporal Structures (A_k):")
    for k, A_k in enumerate(A_list):
        n_edges = int(np.sum(np.abs(A_k) > 0))
        if n_edges > 0:
            wr = f"[{np.min(A_k[A_k != 0]):.3f}, {np.max(A_k[A_k != 0]):.3f}]"
        else:
            wr = "[N/A]"
        print(f"  A_{k + 1}: {n_edges} edges, weights {wr}")
    print(f"\nData Statistics:")
    print(f"  Mean: {np.mean(X):.4f}")
    print(f"  Std:  {np.std(X):.4f}")
    print(f"  Min:  {np.min(X):.4f}")
    print(f"  Max:  {np.max(X):.4f}")
    print(f"\nStability:")
    print(f"  Attempts to generate stable VAR: {params['stability_attempts']}")
    print(f"  Final temporal strength: "
          f"{params['actual_temporal_strength']:.4f}")
    print("=" * 80)

# =============================================================================
# SVAR accuracy metric (W- and A-side TPR / FDR / SHD).
# =============================================================================
def count_accuracy_svar(W_true: np.ndarray, W_est: np.ndarray,
                        A_true: np.ndarray = None, A_est: np.ndarray = None,
                        threshold: float = 0.3) -> dict:
    """
    Compute accuracy metrics for SVAR estimation.

    Parameters
    ----------
    W_true, W_est : np.ndarray
        True and estimated intra-slice (contemporaneous) DAG
    A_true, A_est : np.ndarray, optional
        True and estimated inter-slice (temporal) weights
    threshold : float
        Threshold for binarizing estimated weights

    Returns
    -------
    metrics : dict
        Dictionary with TPR, FDR, SHD for both W and A
    """
    # Binarize
    W_true_bin = (np.abs(W_true) > 0).astype(int)
    W_est_bin = (np.abs(W_est) > threshold).astype(int)

    # Metrics for W (intra-slice)
    TP_W = np.sum(W_true_bin * W_est_bin)
    FP_W = np.sum(W_est_bin * (1 - W_true_bin))
    FN_W = np.sum((1 - W_est_bin) * W_true_bin)

    tpr_W = TP_W / max(TP_W + FN_W, 1)
    fdr_W = FP_W / max(TP_W + FP_W, 1)
    shd_W = FP_W + FN_W

    metrics = {
        'W_tpr': tpr_W,
        'W_fdr': fdr_W,
        'W_shd': shd_W,
        'W_edges_true': int(np.sum(W_true_bin)),
        'W_edges_est': int(np.sum(W_est_bin))
    }

    # Metrics for A (inter-slice) if provided
    if A_true is not None and A_est is not None:
        A_true_bin = (np.abs(A_true) > 0).astype(int)
        A_est_bin = (np.abs(A_est) > threshold).astype(int)

        TP_A = np.sum(A_true_bin * A_est_bin)
        FP_A = np.sum(A_est_bin * (1 - A_true_bin))
        FN_A = np.sum((1 - A_est_bin) * A_true_bin)

        tpr_A = TP_A / max(TP_A + FN_A, 1)
        fdr_A = FP_A / max(TP_A + FP_A, 1)
        shd_A = FP_A + FN_A

        metrics.update({
            'A_tpr': tpr_A,
            'A_fdr': fdr_A,
            'A_shd': shd_A,
            'A_edges_true': int(np.sum(A_true_bin)),
            'A_edges_est': int(np.sum(A_est_bin))
        })

    return metrics
