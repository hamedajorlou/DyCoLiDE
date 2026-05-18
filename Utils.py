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


def simulate_svar(
    n_obs: int,
    n_vars: int,
    lag_order: int,
    burn_in: int = 250,
    stability_radius: float = 0.95,
    shock_scale: float = 1.0,
    seed: Optional[int] = None,
    structure: Optional[object] = None,
    node_order: Optional[Sequence] = None,
) -> Dict[str, object]:
    """
    Simulate a Structural VAR(p) process and return the simulated objects.

    The structural model is
        B0 * y_t = sum_{i=1}^p Bi * y_{t-i} + ε_t,     ε_t ~ N(0, I),
    where B0 is full rank. The routine draws random coefficient matrices,
    rescales them to satisfy the requested stability radius, and simulates
    (`burn_in` + `n_obs`) observations. The burn-in samples are discarded.

    Args:
        n_obs: Number of observations to keep after burn-in.
        n_vars: Number of variables (dimension of y_t). Must match the size of
            `structure` when one is supplied.
        lag_order: VAR order p. Must be >= 1.
        burn_in: Extra samples discarded to wash out initial conditions.
        stability_radius: Maximum allowed spectral radius of the companion matrix.
        shock_scale: Standard deviation of structural shocks.
        seed: Optional random seed for reproducibility.
        structure: Optional DAG describing contemporaneous edges. Accepts either
            a networkx `DiGraph`/`StructureModel` or a square numpy array whose
            (i, j) entry encodes the edge weight i -> j. When supplied, B0 is set
            to `I - adjacency.T`, so parents affect their children
            contemporaneously, and lag matrices are masked to respect those edges
            (self-lags remain allowed).
        node_order: Optional node ordering to use when `structure` is a graph.

    Returns:
        Dictionary with the simulated series and parameters:
            data: np.ndarray of shape (n_obs, n_vars), the simulated y_t.
            structural_shocks: np.ndarray of shape (n_obs, n_vars), ε_t.
            reduced_form_shocks: np.ndarray of shape (n_obs, n_vars), u_t.
            B0: np.ndarray (n_vars, n_vars), contemporaneous matrix.
            B_lags: tuple of np.ndarray, structural lag matrices.
            A_lags: tuple of np.ndarray, reduced-form lag matrices.
            nodes: tuple giving the node ordering matching the arrays.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if n_vars <= 0:
        raise ValueError("n_vars must be positive.")
    if lag_order <= 0:
        raise ValueError("lag_order must be at least 1.")
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative.")
    if stability_radius <= 0:
        raise ValueError("stability_radius must be positive.")
    if shock_scale <= 0:
        raise ValueError("shock_scale must be positive.")

    rng = np.random.default_rng(seed)

    adjacency: Optional[np.ndarray] = None
    if structure is not None:
        adjacency, nodes = _structure_to_adjacency(structure, node_order)
        if adjacency.shape[0] != n_vars:
            raise ValueError(
                f"n_vars={n_vars} does not match structure size {adjacency.shape[0]}"
            )
        B0 = np.eye(n_vars) - adjacency.T
    else:
        nodes = tuple(range(n_vars))
        tril_mask = np.tril(np.ones((n_vars, n_vars), dtype=bool))
        B0 = rng.normal(loc=0.0, scale=0.3, size=(n_vars, n_vars))
        B0[~tril_mask] = 0.0
        np.fill_diagonal(B0, 1.0)

    lag_mask = np.ones((n_vars, n_vars), dtype=bool)
    if adjacency is not None:
        lag_mask = adjacency.T != 0.0
        np.fill_diagonal(lag_mask, True)

    B_lags_list: List[np.ndarray] = []
    for _ in range(lag_order):
        lag = rng.normal(loc=0.0, scale=0.25, size=(n_vars, n_vars))
        lag *= lag_mask
        B_lags_list.append(lag)

    A_lags_list: List[np.ndarray] = [
        np.linalg.solve(B0, lag) for lag in B_lags_list
    ]

    companion_dim = n_vars * lag_order
    companion = np.zeros((companion_dim, companion_dim))
    for k, coeff in enumerate(A_lags_list):
        start = k * n_vars
        companion[:n_vars, start : start + n_vars] = coeff
    if lag_order > 1:
        companion[n_vars:, :-n_vars] = np.eye(n_vars * (lag_order - 1))

    radius = max(np.abs(np.linalg.eigvals(companion)))
    if radius >= stability_radius:
        scale = stability_radius / (radius + 1e-12)
        A_lags_list = [coeff * scale for coeff in A_lags_list]

    A_lags = tuple(A_lags_list)
    B_lags = tuple(B0 @ coeff for coeff in A_lags)

    total_samples = n_obs + burn_in
    eps = rng.normal(
        loc=0.0, scale=shock_scale, size=(total_samples, n_vars)
    )
    u = np.linalg.solve(B0, eps.T).T

    y = np.zeros((total_samples, n_vars))
    for t in range(lag_order, total_samples):
        state = np.zeros(n_vars)
        for k, coeff in enumerate(A_lags, start=1):
            state += coeff @ y[t - k]
        y[t] = state + u[t]

    keep_slice = slice(burn_in, None)
    return {
        "data": y[keep_slice],
        "structural_shocks": eps[keep_slice],
        "reduced_form_shocks": u[keep_slice],
        "B0": B0,
        "B_lags": B_lags,
        "A_lags": A_lags,
        "nodes": nodes,
    }


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