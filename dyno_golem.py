"""
DynoGOLEM: GOLEM extended for dynamic/time-series DAG learning.

Extends the GOLEM algorithm (Ng et al., NeurIPS 2020) to handle SVAR models
with both instantaneous (W) and lagged (A) effects.

Model: X_t = X_t @ W + X_{t-1:t-p} @ A + noise

Uses GOLEM's likelihood-based objective with trace-exponential DAG constraint.

Original GOLEM paper: "On the Role of Sparsity and DAG Constraints for Learning Linear DAGs"
"""

import os
# Prevent multiprocessing issues on macOS
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm


class DynoGOLEM:
    """
    GOLEM extended for dynamic structural equation models (SVAR).

    Uses likelihood-based objective with trace-exponential DAG constraint.

    Two variants:
    - GOLEM-EV (equal_variances=True): Assumes equal noise variances
    - GOLEM-NV (equal_variances=False): Allows different noise variances per variable
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
            Float precision.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.verbose = verbose
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

    def _h_trace_exp(self, W):
        """
        Trace-exponential DAG constraint (same as NOTEARS):
        h(W) = tr(exp(W◦W)) - d

        Returns 0 iff W is a DAG.
        """
        d = W.shape[0]
        W_sq = W * W  # Element-wise square
        exp_W_sq = torch.matrix_exp(W_sq)
        h = torch.trace(exp_W_sq) - d
        return h

    def _likelihood_ev(self, X_t, Y_t, W, A):
        """
        Compute negative log-likelihood assuming equal noise variances.

        For SVAR: X_t = X_t @ W + Y_t @ A + noise
        Residual: R = X_t - X_t @ W - Y_t @ A

        Likelihood (EV): 0.5 * d * log(||R||²) - log det(I - W)
        """
        n_eff, d = X_t.shape
        I = torch.eye(d, dtype=self.dtype)

        # Compute residual
        R = X_t - X_t @ W - Y_t @ A

        # Squared Frobenius norm of residual
        R_sq_norm = torch.sum(R ** 2)

        # Log-likelihood term (negative)
        nll = 0.5 * d * torch.log(R_sq_norm) - torch.linalg.slogdet(I - W)[1]

        return nll

    def _likelihood_nv(self, X_t, Y_t, W, A):
        """
        Compute negative log-likelihood assuming non-equal noise variances.

        Likelihood (NV): 0.5 * Σ_j log(||R_j||²) - log det(I - W)
        where R_j is the j-th column of the residual matrix.
        """
        n_eff, d = X_t.shape
        I = torch.eye(d, dtype=self.dtype)

        # Compute residual
        R = X_t - X_t @ W - Y_t @ A

        # Sum of log of squared column norms
        col_sq_norms = torch.sum(R ** 2, dim=0)  # (d,)
        log_sum = 0.5 * torch.sum(torch.log(col_sq_norms + 1e-10))

        # Log-likelihood term (negative)
        nll = log_sum - torch.linalg.slogdet(I - W)[1]

        return nll

    def fit(self, X, p=1, lambda_1_w=0.02, lambda_1_a=0.02, lambda_2=5.0,
            equal_variances=True, w_threshold=0.3,
            num_iter=50000, lr=0.001, checkpoint=5000):
        """
        Fit DynoGOLEM to time-series data.

        Model: X_t = X_t @ W + X_{t-1:t-p} @ A + noise

        The objective is:
            score = likelihood + λ₁_w * ||W||₁ + λ₁_a * ||A||₁ + λ₂ * h(W)

        Parameters
        ----------
        X : np.ndarray
            Time-series data (n_timesteps, d)
        p : int
            Lag order
        lambda_1_w : float
            L1 penalty coefficient for W. Default: 0.02
        lambda_1_a : float
            L1 penalty coefficient for A. Default: 0.02
        lambda_2 : float
            DAG penalty coefficient. Default: 5.0
        equal_variances : bool
            If True, use GOLEM-EV (equal noise variances).
            If False, use GOLEM-NV (non-equal variances). Default: True
        w_threshold : float
            Threshold for removing weak edges. Default: 0.3
        num_iter : int
            Number of optimization iterations. Default: 50000
        lr : float
            Learning rate. Default: 0.001
        checkpoint : int
            Print frequency (if verbose). Default: 5000

        Returns
        -------
        W_est : np.ndarray
            Estimated instantaneous adjacency (d, d)
        A_est : np.ndarray
            Estimated lagged adjacency (p*d, d)
        """
        # Prepare data
        n_timesteps, d = X.shape
        n_eff = n_timesteps - p

        # Build lagged matrix Y_t = [X_{t-1}, X_{t-2}, ..., X_{t-p}]
        X_t = torch.tensor(X[p:], dtype=self.dtype)  # (n_eff, d)
        Y_list = [X[p-k-1:n_timesteps-k-1] for k in range(p)]
        Y_t = torch.tensor(np.hstack(Y_list), dtype=self.dtype)  # (n_eff, p*d)

        # Center data
        X_t = X_t - X_t.mean(dim=0, keepdim=True)
        Y_t = Y_t - Y_t.mean(dim=0, keepdim=True)

        # Initialize W and A at zeros
        W = torch.zeros((d, d), dtype=self.dtype, requires_grad=True)
        A = torch.zeros((p * d, d), dtype=self.dtype, requires_grad=True)

        # Select likelihood function
        likelihood_fn = self._likelihood_ev if equal_variances else self._likelihood_nv

        # Optimizer
        optimizer = Adam([W, A], lr=lr)

        self.vprint(f"\nDynoGOLEM: d={d}, p={p}, equal_variances={equal_variances}")
        self.vprint(f"lambda_1_w={lambda_1_w}, lambda_1_a={lambda_1_a}, lambda_2={lambda_2}")

        # Training loop
        for iteration in tqdm(range(num_iter + 1), disable=not self.verbose):
            optimizer.zero_grad()

            # Zero out diagonal of W
            W_masked = W - torch.diag(torch.diag(W))

            # Compute objective components
            likelihood = likelihood_fn(X_t, Y_t, W_masked, A)
            l1_penalty_w = lambda_1_w * torch.sum(torch.abs(W_masked))
            l1_penalty_a = lambda_1_a * torch.sum(torch.abs(A))
            h = self._h_trace_exp(W_masked)

            # Total score (objective to minimize)
            score = likelihood + l1_penalty_w + l1_penalty_a + lambda_2 * h

            if iteration > 0:
                score.backward()
                optimizer.step()

            # Checkpoint
            if iteration % checkpoint == 0:
                self.vprint(f"  iter {iteration}: score={score.item():.3e}, "
                           f"likelihood={likelihood.item():.3e}, h={h.item():.3e}")

        # Extract results
        with torch.no_grad():
            W_est = W.numpy()
            A_est = A.numpy()

            # Zero diagonal
            np.fill_diagonal(W_est, 0)

        # Store final h value
        self.h_final = h.item()
        self.vprint(f"\nFinal h(W) = {self.h_final:.4e}")

        # Threshold small edges
        W_est[np.abs(W_est) < w_threshold] = 0
        A_est[np.abs(A_est) < w_threshold] = 0

        return W_est, A_est


def test():
    """Test DynoGOLEM on synthetic SVAR data."""
    import sys
    sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears')
    from SVAR_data_generator import generate_svar_data
    from DyCoLiDE import count_accuracy_svar

    print("Testing DynoGOLEM...")

    # Generate data
    X, B_true, A_list_true, params = generate_svar_data(
        n_nodes=20,
        n_timesteps=1000,
        lag_order=1,
        instantaneous_edges=40,
        temporal_sparsity=0.1,
        seed=42
    )

    # Convert to DyCoLiDE convention
    A_true = np.vstack([A.T for A in A_list_true])
    B_true_dycolide = B_true.T

    print(f"True W edges: {np.sum(np.abs(B_true_dycolide) > 0)}")
    print(f"True A edges: {np.sum(np.abs(A_true) > 0)}")

    # Fit DynoGOLEM-EV
    print("\n--- DynoGOLEM-EV ---")
    model = DynoGOLEM(seed=42, verbose=True)
    W_est, A_est = model.fit(
        X, p=1,
        lambda_1_w=0.02,
        lambda_1_a=0.02,
        lambda_2=5.0,
        equal_variances=True,
        w_threshold=0.1,
        num_iter=30000,
        lr=0.001,
        checkpoint=5000
    )

    # Evaluate
    metrics = count_accuracy_svar(B_true_dycolide, W_est, A_true, A_est, threshold=0.0)

    print(f"\nResults:")
    print(f"  W: TPR={metrics['W_tpr']:.3f}, FDR={metrics['W_fdr']:.3f}")
    print(f"  A: TPR={metrics['A_tpr']:.3f}, FDR={metrics['A_fdr']:.3f}")


if __name__ == '__main__':
    test()
