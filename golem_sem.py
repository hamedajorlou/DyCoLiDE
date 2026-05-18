"""
GOLEM-SEM: GOLEM for static structural equation models.

A clean PyTorch implementation of GOLEM (Ng et al., NeurIPS 2020) for
learning linear DAGs from observational data.

Model: X = X @ W + noise  (or equivalently, x = W^T @ x + e)

Uses likelihood-based objective with trace-exponential DAG constraint.

Reference: "On the Role of Sparsity and DAG Constraints for Learning Linear DAGs"
"""

import os
# Prevent multiprocessing issues on macOS
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch
from tqdm import tqdm


class GOLEM_SEM:
    """
    GOLEM for static structural equation models.

    Uses likelihood-based objective with trace-exponential DAG constraint.

    Two variants:
    - GOLEM-EV (equal_variances=True): Assumes equal noise variances across variables
    - GOLEM-NV (equal_variances=False): Allows different noise variances per variable

    Hyperparameters from paper:
        GOLEM-EV: lambda1=0.02, lambda2=5.0
        GOLEM-NV: lambda1=0.002, lambda2=5.0
    """

    def __init__(self, seed=None, verbose=False, dtype=torch.float64):
        """
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            If True, print optimization progress.
        dtype : torch.dtype
            Float precision (float32 or float64).
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.verbose = verbose
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

    def _h_trace_exp(self, W):
        """
        Trace-exponential DAG constraint (NOTEARS formulation):
        h(W) = tr(exp(W ◦ W)) - d

        Returns 0 if and only if W represents a DAG.

        Args:
            W: (d, d) weighted adjacency matrix

        Returns:
            Scalar DAG constraint value
        """
        d = W.shape[0]
        W_sq = W * W  # Element-wise square
        exp_W_sq = torch.matrix_exp(W_sq)
        h = torch.trace(exp_W_sq) - d
        return h

    def _likelihood_ev(self, X, W):
        """
        Negative log-likelihood assuming equal noise variances (GOLEM-EV).

        Model: X = X @ W + noise, with noise ~ N(0, sigma^2 * I)

        Likelihood: 0.5 * d * log(||X - X @ W||^2_F) - log|det(I - W)|

        Args:
            X: (n, d) centered data matrix
            W: (d, d) weighted adjacency matrix

        Returns:
            Scalar negative log-likelihood
        """
        n, d = X.shape
        I = torch.eye(d, dtype=self.dtype)

        # Residual: X - X @ W
        R = X - X @ W

        # Squared Frobenius norm
        R_sq_norm = torch.sum(R ** 2)

        # Negative log-likelihood
        nll = 0.5 * d * torch.log(R_sq_norm) - torch.linalg.slogdet(I - W)[1]

        return nll

    def _likelihood_nv(self, X, W):
        """
        Negative log-likelihood assuming non-equal noise variances (GOLEM-NV).

        Model: X = X @ W + noise, with noise_j ~ N(0, sigma_j^2)

        Likelihood: 0.5 * sum_j log(||R_j||^2) - log|det(I - W)|
        where R_j is the j-th column of the residual matrix.

        Args:
            X: (n, d) centered data matrix
            W: (d, d) weighted adjacency matrix

        Returns:
            Scalar negative log-likelihood
        """
        n, d = X.shape
        I = torch.eye(d, dtype=self.dtype)

        # Residual
        R = X - X @ W

        # Sum of log of squared column norms (one variance per variable)
        col_sq_norms = torch.sum(R ** 2, dim=0)  # (d,)
        log_sum = 0.5 * torch.sum(torch.log(col_sq_norms + 1e-10))

        # Negative log-likelihood
        nll = log_sum - torch.linalg.slogdet(I - W)[1]

        return nll

    def fit(self, X, lambda1=0.02, lambda2=5.0, equal_variances=True,
            w_threshold=0.3, num_iter=50000, lr=0.001, checkpoint=1000,
            tol=1e-6, beta1=0.9, beta2=0.999):
        """
        Fit GOLEM to observational data.

        Minimizes: likelihood + lambda1 * ||W||_1 + lambda2 * h(W)

        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, d_variables)
        lambda1 : float
            L1 sparsity penalty. Default: 0.02 (EV) or 0.002 (NV)
        lambda2 : float
            DAG penalty coefficient. Default: 5.0
        equal_variances : bool
            If True, use GOLEM-EV. If False, use GOLEM-NV. Default: True
        w_threshold : float
            Threshold for removing weak edges. Default: 0.3
        num_iter : int
            Maximum optimization iterations. Default: 50000
        lr : float
            Learning rate for Adam. Default: 0.001
        checkpoint : int
            Iterations between convergence checks. Default: 1000
        tol : float
            Relative tolerance for convergence. Default: 1e-6
        beta1 : float
            Adam beta1 parameter. Default: 0.9
        beta2 : float
            Adam beta2 parameter. Default: 0.999

        Returns
        -------
        W_est : np.ndarray
            Estimated weighted adjacency matrix (d, d)
        """
        # Convert to tensor and center
        X_np = X - X.mean(axis=0, keepdims=True)
        X_t = torch.tensor(X_np, dtype=self.dtype)
        n, d = X_t.shape

        # Initialize W at zeros
        W = torch.zeros((d, d), dtype=self.dtype, requires_grad=True)

        # Select likelihood function
        likelihood_fn = self._likelihood_ev if equal_variances else self._likelihood_nv
        variant = "GOLEM-EV" if equal_variances else "GOLEM-NV"

        # Manual Adam optimizer state (avoids torch._dynamo issues)
        m = torch.zeros_like(W)  # First moment
        v = torch.zeros_like(W)  # Second moment
        eps = 1e-8

        self.vprint(f"\n{variant}: d={d}, n={n}")
        self.vprint(f"lambda1={lambda1}, lambda2={lambda2}, lr={lr}")

        prev_score = float('inf')

        # Training loop
        pbar = tqdm(range(1, num_iter + 1), desc=variant, disable=not self.verbose)

        for iteration in pbar:
            # Zero out diagonal of W (no self-loops)
            W_masked = W - torch.diag(torch.diag(W))

            # Compute objective components
            likelihood = likelihood_fn(X_t, W_masked)
            l1_penalty = lambda1 * torch.sum(torch.abs(W_masked))
            h = self._h_trace_exp(W_masked)

            # Total score (objective to minimize)
            score = likelihood + l1_penalty + lambda2 * h

            # Backward pass
            if W.grad is not None:
                W.grad.zero_()
            score.backward()

            # Manual Adam update
            with torch.no_grad():
                g = W.grad
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                m_hat = m / (1 - beta1 ** iteration)
                v_hat = v / (1 - beta2 ** iteration)
                W -= lr * m_hat / (torch.sqrt(v_hat) + eps)

            # Convergence check at checkpoints
            if iteration % checkpoint == 0:
                current_score = score.item()
                pbar.set_postfix({'score': f'{current_score:.4f}', 'h': f'{h.item():.4f}'})

                if abs(prev_score - current_score) / (abs(prev_score) + 1e-8) < tol:
                    self.vprint(f"\nConverged at iteration {iteration}")
                    break
                prev_score = current_score

        # Extract result
        with torch.no_grad():
            W_est = W.numpy().copy()
            np.fill_diagonal(W_est, 0)

        # Store final values
        self.h_final = h.item()
        self.score_final = score.item()

        # Threshold weak edges
        W_est[np.abs(W_est) < w_threshold] = 0

        return W_est


def run_golem_sem(X, lambda1=0.02, lambda2=5.0, equal_variances=True,
                  num_iter=50000, lr=0.001, w_threshold=0.3, seed=None):
    """
    Convenience function to run GOLEM-SEM.

    Args:
        X: (n, d) data matrix
        lambda1: L1 sparsity penalty
        lambda2: DAG penalty coefficient
        equal_variances: Use GOLEM-EV (True) or GOLEM-NV (False)
        num_iter: Number of iterations
        lr: Learning rate
        w_threshold: Edge threshold
        seed: Random seed

    Returns:
        W_est: (d, d) estimated adjacency matrix
    """
    model = GOLEM_SEM(seed=seed, verbose=False)
    W_est = model.fit(
        X,
        lambda1=lambda1,
        lambda2=lambda2,
        equal_variances=equal_variances,
        w_threshold=w_threshold,
        num_iter=num_iter,
        lr=lr
    )
    return W_est


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears')
    from Utils import simulate_sem, to_bin, count_accuracy

    print("Testing GOLEM-SEM...")

    # Generate data
    X, W_true, _ = simulate_sem(
        n_nodes=20,
        n_samples=1000,
        edges=40,
        graph_type='er',
        edge_type='weighted',
        var_type='ev',
        noise='normal',
        var=1.0,
        seed=42
    )

    W_true_bin = to_bin(W_true, thr=0.0)
    print(f"True edges: {int(np.sum(W_true_bin))}")

    # Test GOLEM-EV
    print("\n--- GOLEM-EV ---")
    model = GOLEM_SEM(seed=42, verbose=True)
    W_est = model.fit(
        X,
        lambda1=0.02,
        lambda2=5.0,
        equal_variances=True,
        w_threshold=0.3,
        num_iter=30000,
        lr=0.001
    )

    W_est_bin = to_bin(W_est, thr=0.0)
    shd, tpr, fdr = count_accuracy(W_true_bin, W_est_bin)
    print(f"\nResults: TPR={tpr:.4f}, FDR={fdr:.4f}, SHD={shd}")
    print(f"Estimated edges: {int(np.sum(W_est_bin))}")

    # Test GOLEM-NV
    print("\n--- GOLEM-NV ---")
    X_nv, W_true_nv, _ = simulate_sem(
        n_nodes=20,
        n_samples=1000,
        edges=40,
        graph_type='er',
        edge_type='weighted',
        var_type='nv',
        noise='normal',
        var=1.0,
        seed=42
    )
    
    W_true_nv_bin = to_bin(W_true_nv, thr=0.0)

    model_nv = GOLEM_SEM(seed=42, verbose=True)
    W_est_nv = model_nv.fit(
        X_nv,
        lambda1=0.002,  # Lower lambda1 for NV
        lambda2=5.0,
        equal_variances=False,
        w_threshold=0.3,
        num_iter=30000,
        lr=0.001
    )

    W_est_nv_bin = to_bin(W_est_nv, thr=0.0)
    shd_nv, tpr_nv, fdr_nv = count_accuracy(W_true_nv_bin, W_est_nv_bin)
    print(f"\nResults: TPR={tpr_nv:.4f}, FDR={fdr_nv:.4f}, SHD={shd_nv}")
    print(f"Estimated edges: {int(np.sum(W_est_nv_bin))}")
