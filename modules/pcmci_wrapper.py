"""
PCMCI+ wrapper for comparison with DyCoLiDE and Dynotears.

PCMCI+ is a constraint-based causal discovery method for time series.
For single-domain data without context variables, we use standard PCMCI+.
"""

import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


class PCMCIWrapper:
    """
    Wrapper for PCMCI+ to match the interface of DyCoLiDE/Dynotears.

    PCMCI+ discovers the time series causal graph using conditional
    independence tests.
    """

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def fit(self, X, p=1, pc_alpha=0.05, tau_max=None, threshold=0.0):
        """
        Fit PCMCI+ to discover causal graph.

        Args:
            X: Data matrix (n_samples, d)
            p: Lag order (used as tau_max if tau_max not specified)
            pc_alpha: Significance level for independence tests
            tau_max: Maximum time lag to consider
            threshold: Not used (PCMCI+ uses significance testing)

        Returns:
            W_est: Estimated intra-slice adjacency (d, d) - contemporaneous
            A_est: Estimated inter-slice adjacency (p*d, d) - lagged
        """
        n, d = X.shape

        if tau_max is None:
            tau_max = p

        # Create tigramite dataframe
        dataframe = pp.DataFrame(X, var_names=[str(i) for i in range(d)])

        # Use ParCorr for linear relationships
        parcorr = ParCorr(significance='analytic')

        # Initialize PCMCI
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=0
        )

        # Run PCMCI+
        results = pcmci.run_pcmciplus(
            tau_min=0,
            tau_max=tau_max,
            pc_alpha=pc_alpha
        )

        # Extract graph
        # graph shape: (d, d, tau_max+1)
        # graph[i,j,tau] indicates link from i at t-tau to j at t
        # Values: '-->' means i->j, '<--' means j->i, 'o-o' undirected, '' no link
        graph = results['graph']
        val_matrix = results['val_matrix']

        # Convert to W (contemporaneous) and A (lagged) matrices
        # W[i,j] = weight from j to i at lag 0 (in DyCoLiDE convention: row = effect)
        # A[lag*d + j, i] = weight from j at t-lag to i at t

        W_est = np.zeros((d, d))
        A_est = np.zeros((p * d, d))

        # Contemporaneous links (tau=0)
        for i in range(d):
            for j in range(d):
                if i != j:
                    link = graph[j, i, 0]  # j -> i at lag 0
                    if link in ['-->', 'o-o', 'x-x']:
                        # Use the test statistic as weight proxy
                        W_est[i, j] = val_matrix[j, i, 0]

        # Lagged links (tau > 0)
        for lag in range(1, p + 1):
            if lag <= tau_max:
                for i in range(d):
                    for j in range(d):
                        link = graph[j, i, lag]  # j at t-lag -> i at t
                        if link in ['-->', 'o-o', 'x-x']:
                            lag_idx = (lag - 1) * d + j
                            A_est[lag_idx, i] = val_matrix[j, i, lag]

        return W_est, A_est


def run_pcmci_on_svar(X, p=1, pc_alpha=0.05, seed=42):
    """
    Convenience function to run PCMCI+ on SVAR data.

    Args:
        X: Data matrix (n_samples, d)
        p: Lag order
        pc_alpha: Significance level
        seed: Random seed

    Returns:
        W_est, A_est: Estimated adjacency matrices
    """
    model = PCMCIWrapper(seed=seed)
    return model.fit(X, p=p, pc_alpha=pc_alpha)


if __name__ == "__main__":
    # Quick test
    from SVAR_data_generator import generate_svar_data
    from DyCoLiDE import count_accuracy_svar

    print("Testing PCMCI+ wrapper...")

    # Generate data
    X, B_true, A_list_true, _ = generate_svar_data(
        n_nodes=10,
        n_timesteps=500,
        lag_order=1,
        instantaneous_edges=20,
        temporal_sparsity=0.1,
        temporal_strength=0.2,
        noise_scale=1,
        noise_type='ev',
        seed=42
    )

    # Ground truth
    B_true_t = B_true.T
    A_true = np.vstack([A.T for A in A_list_true])

    print(f"Data shape: {X.shape}")
    print(f"True W edges: {np.sum(np.abs(B_true_t) > 0)}")
    print(f"True A edges: {np.sum(np.abs(A_true) > 0)}")

    # Run PCMCI+
    model = PCMCIWrapper(seed=42)
    W_est, A_est = model.fit(X, p=1, pc_alpha=0.05)

    print(f"\nEstimated W edges: {np.sum(np.abs(W_est) > 0)}")
    print(f"Estimated A edges: {np.sum(np.abs(A_est) > 0)}")

    # Evaluate
    # For PCMCI+, we threshold based on whether edges exist (non-zero)
    metrics = count_accuracy_svar(B_true_t, W_est, A_true, A_est, threshold=0.0)

    def compute_f1(tpr, fdr):
        precision = 1 - fdr
        recall = tpr
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    w_f1 = compute_f1(metrics['W_tpr'], metrics['W_fdr'])
    a_f1 = compute_f1(metrics['A_tpr'], metrics['A_fdr'])

    print(f"\nW: TPR={metrics['W_tpr']:.3f}, FDR={metrics['W_fdr']:.3f}, F1={w_f1:.3f}")
    print(f"A: TPR={metrics['A_tpr']:.3f}, FDR={metrics['A_fdr']:.3f}, F1={a_f1:.3f}")
