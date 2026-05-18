"""
GOLEM PyTorch Implementation

A PyTorch re-implementation of GOLEM (Gradient-based Optimization of
DAG LEarning with M-matrices) for systems without TensorFlow.

Reference: Ng et al. "On the Role of Sparsity and DAG Constraints for
Learning Linear DAGs" (NeurIPS 2020)
"""

import sys
import numpy as np
import torch
from tqdm import tqdm


class GolemTorch:
    """PyTorch implementation of GOLEM for DAG learning.

    Hyperparameters (from paper):
        GOLEM-EV: equal_variances=True, lambda1=2e-2, lambda2=5.0
        GOLEM-NV: equal_variances=False, lambda1=2e-3, lambda2=5.0
    """

    def __init__(self, equal_variances=True, seed=1):
        """Initialize GOLEM model.

        Args:
            equal_variances: If True, use GOLEM-EV (equal variance).
                           If False, use GOLEM-NV (non-equal variance).
            seed: Random seed.
        """
        self.equal_variances = equal_variances
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Use CPU (MPS doesn't support all linalg operations yet)
        self.device = torch.device("cpu")

    def _compute_likelihood(self, X, B):
        """Compute negative log-likelihood.

        Args:
            X: [n, d] data matrix
            B: [d, d] weighted adjacency matrix

        Returns:
            Negative log-likelihood (scalar)
        """
        n, d = X.shape

        # Residuals: X - X @ B
        residuals = X - X @ B

        if self.equal_variances:
            # GOLEM-EV: assuming equal noise variances
            # 0.5 * d * log(||X - XB||^2) - log|det(I-B)|
            likelihood = 0.5 * d * torch.log(torch.sum(residuals ** 2))
        else:
            # GOLEM-NV: assuming non-equal noise variances
            # 0.5 * sum_j log(||X_j - XB_j||^2) - log|det(I-B)|
            likelihood = 0.5 * torch.sum(torch.log(torch.sum(residuals ** 2, dim=0)))

        # Subtract log determinant of (I - B)
        I = torch.eye(d, device=self.device)
        sign, logabsdet = torch.linalg.slogdet(I - B)
        likelihood = likelihood - logabsdet

        return likelihood

    def _compute_h(self, B):
        """Compute DAG constraint h(B) = tr(exp(B*B)) - d.

        When h(B) = 0, B represents a DAG.

        Args:
            B: [d, d] weighted adjacency matrix

        Returns:
            DAG penalty (scalar)
        """
        d = B.shape[0]
        # h(B) = tr(exp(B ⊙ B)) - d
        M = B * B  # element-wise square
        h = torch.trace(torch.matrix_exp(M)) - d
        return h

    def _compute_score(self, X, B, lambda1, lambda2):
        """Compute total score (objective function).

        score = likelihood + lambda1 * ||B||_1 + lambda2 * h(B)

        Args:
            X: [n, d] data matrix
            B: [d, d] weighted adjacency matrix
            lambda1: L1 penalty coefficient
            lambda2: DAG penalty coefficient

        Returns:
            Total score (scalar)
        """
        likelihood = self._compute_likelihood(X, B)
        l1_penalty = torch.sum(torch.abs(B))
        h = self._compute_h(B)

        score = likelihood + lambda1 * l1_penalty + lambda2 * h
        return score, likelihood, h

    def fit(self, X, lambda1=0.02, lambda2=5.0, num_iter=50000,
            lr=1e-3, checkpoint=1000, tol=1e-6):
        """Fit the GOLEM model to data.

        Args:
            X: [n, d] numpy array of data
            lambda1: L1 sparsity penalty (default: 0.02 for EV, 0.002 for NV)
            lambda2: DAG penalty coefficient (default: 5.0)
            num_iter: Maximum number of iterations
            lr: Learning rate for Adam optimizer
            checkpoint: Iterations between convergence checks
            tol: Convergence tolerance

        Returns:
            B_est: [d, d] estimated weighted adjacency matrix
        """
        # Center data
        X = X - X.mean(axis=0, keepdims=True)
        n, d = X.shape

        # Convert to torch tensor
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Initialize B as zeros (with gradient tracking)
        B = torch.zeros((d, d), dtype=torch.float32, device=self.device, requires_grad=True)

        # Adam optimizer
        optimizer = torch.optim.Adam([B], lr=lr)

        # For convergence checking
        prev_score = float('inf')

        # Progress bar update frequency (every 100 iterations for efficiency)
        update_freq = 100

        pbar = tqdm(total=num_iter, desc="GOLEM", file=sys.stderr,
                    dynamic_ncols=True, mininterval=0.5)

        try:
            for i in range(1, num_iter + 1):
                optimizer.zero_grad()

                # Zero out diagonal (no self-loops)
                B_masked = B - torch.diag(torch.diag(B))

                # Compute score
                score, likelihood, h = self._compute_score(X_torch, B_masked, lambda1, lambda2)

                # Backward pass
                score.backward()

                # Update
                optimizer.step()

                # Update progress bar periodically
                if i % update_freq == 0:
                    pbar.update(update_freq)

                # Check convergence at checkpoints
                if i % checkpoint == 0:
                    current_score = score.item()
                    pbar.set_postfix({'score': f'{current_score:.4f}', 'h': f'{h.item():.4f}'})

                    if abs(prev_score - current_score) / (abs(prev_score) + 1e-8) < tol:
                        pbar.update(num_iter - i)
                        break
                    prev_score = current_score
        finally:
            pbar.close()

        # Get final B with zeroed diagonal
        with torch.no_grad():
            B_est = B - torch.diag(torch.diag(B))
            B_est = B_est.cpu().numpy()

        return B_est


def run_golem(X, lambda1=0.02, lambda2=5.0, equal_variances=True,
              num_iter=50000, lr=1e-3, seed=1):
    """Convenience function to run GOLEM.

    Args:
        X: [n, d] data matrix
        lambda1: L1 penalty (default: 0.02 for EV)
        lambda2: DAG penalty (default: 5.0)
        equal_variances: Whether to use equal variance model
        num_iter: Number of iterations
        lr: Learning rate
        seed: Random seed

    Returns:
        B_est: [d, d] estimated adjacency matrix
    """
    model = GolemTorch(equal_variances=equal_variances, seed=seed)
    B_est = model.fit(X, lambda1=lambda1, lambda2=lambda2,
                      num_iter=num_iter, lr=lr)
    return B_est


if __name__ == "__main__":
    # Quick test
    import numpy as np

    print("Testing GOLEM PyTorch implementation...")

    # Generate simple test data
    np.random.seed(42)
    d = 10
    n = 500

    # True DAG: simple chain 0 -> 1 -> 2 -> ... -> d-1
    B_true = np.zeros((d, d))
    for i in range(d - 1):
        B_true[i, i + 1] = np.random.uniform(0.5, 1.5)

    # Generate data: X = X @ B + noise => X = noise @ (I - B)^{-1}
    I = np.eye(d)
    noise = np.random.randn(n, d)
    X = noise @ np.linalg.inv(I - B_true)

    print(f"Data shape: {X.shape}")
    print(f"True edges: {np.sum(B_true != 0)}")

    # Run GOLEM
    B_est = run_golem(X, lambda1=0.02, lambda2=5.0, num_iter=10000, lr=1e-3)

    # Threshold
    threshold = 0.3
    B_est_binary = (np.abs(B_est) > threshold).astype(float)
    B_true_binary = (B_true != 0).astype(float)

    # Metrics
    tp = np.sum((B_est_binary == 1) & (B_true_binary == 1))
    fp = np.sum((B_est_binary == 1) & (B_true_binary == 0))
    fn = np.sum((B_est_binary == 0) & (B_true_binary == 1))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\nResults (threshold={threshold}):")
    print(f"  TPR: {tpr:.4f}")
    print(f"  FDR: {fdr:.4f}")
    print(f"  Estimated edges: {int(np.sum(B_est_binary))}")
