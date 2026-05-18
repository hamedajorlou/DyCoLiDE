"""
Meta-D2AG: Meta-Learning for Dynamic DAG Learning
Based on: "Meta-D2AG: Causal Graph Learning with Interventional Dynamic Data" (NeurIPS 2025)

This is a simplified implementation for comparison with DYNOTEARS and DyCoLiDE.
For single-domain data, the formulation reduces to standard continuous DAG learning.
"""

import numpy as np
import torch
from torch.optim import Adam


def compute_dag_constraint_logdet(W, s=1.0):
    """
    DAG constraint using log-determinant:
    h(W) = -log det(sI - W◦W) + d*log(s)
    """
    d = W.shape[0]
    W_squared = W * W
    M = s * torch.eye(d, device=W.device) - W_squared
    sign, logdet = torch.linalg.slogdet(M)
    h = -logdet + d * np.log(s)
    return h


class MetaD2AG:
    """
    Meta-D2AG for Dynamic DAG Learning.

    For single domain (stationary) data, this reduces to standard continuous
    DAG optimization with the bilevel structure simplified.

    Model: X_t = X_t @ W + X_{t-1} @ A + Z
    where W is intra-slice (contemporaneous) and A is inter-slice (temporal)
    """

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def fit(self, X, p=1, lambda_w=0.01, lambda_a=0.01,
            max_iter=20000, lr=0.003,
            rho_init=1.0, rho_max=1e16, alpha=2.0,
            h_tol=1e-8, w_threshold=0.3,
            use_logdet=True, s_init=1.0,
            checkpoint=1000, verbose=True):
        """
        Fit Meta-D2AG model using augmented Lagrangian optimization.

        Args:
            X: Data matrix (n_samples, d)
            p: Lag order
            lambda_w: L1 regularization for W (intra-slice)
            lambda_a: L1 regularization for A (inter-slice)
            max_iter: Maximum iterations per subproblem
            lr: Learning rate
            rho_init: Initial augmented Lagrangian penalty
            rho_max: Maximum penalty
            alpha: Penalty increase factor
            h_tol: DAG constraint tolerance
            w_threshold: Threshold for edge detection
            use_logdet: Use log-det constraint
            s_init: Initial s for log-det constraint
            checkpoint: Print frequency
            verbose: Print progress

        Returns:
            W_est: Estimated intra-slice adjacency (d, d)
            A_est: Estimated inter-slice adjacency (p*d, d)
        """

        n, d = X.shape

        # Construct lagged data matrices
        X_current = X[p:]
        Y_list = [X[p-i:-i] if i < p else X[:-(p)] for i in range(1, p+1)]
        Y_lagged = np.hstack(Y_list)  # (n-p, p*d)

        n_eff = X_current.shape[0]

        # Force CPU to avoid GPU issues
        device = torch.device('cpu')
        X_t = torch.tensor(X_current, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y_lagged, dtype=torch.float32, device=device)

        # Initialize parameters with small random values (not zeros)
        W = torch.randn(d, d, device=device) * 0.01
        W = W.requires_grad_(True)
        A = torch.randn(p * d, d, device=device) * 0.01
        A = A.requires_grad_(True)

        # Augmented Lagrangian parameters
        rho = rho_init
        mu = 0.0  # Lagrange multiplier (scalar)
        s = s_init

        # Two-phase optimization similar to NOTEARS/DYNOTEARS
        for phase in range(100):  # Outer loop for augmented Lagrangian updates
            optimizer = Adam([W, A], lr=lr)

            for iteration in range(max_iter // 10):  # Inner optimization
                optimizer.zero_grad()

                # Forward pass: X_t ≈ X_t @ W + Y_t @ A
                # The model is X_t = X_t @ W + Y_t @ A + noise
                # So loss = || X_t - X_t @ W - Y_t @ A ||^2
                residual = X_t - X_t @ W - Y_t @ A
                loss_mse = 0.5 / n_eff * torch.sum(residual ** 2)

                # L1 regularization (excluding diagonal of W)
                W_off_diag = W - torch.diag(torch.diag(W))
                loss_l1 = lambda_w * torch.sum(torch.abs(W_off_diag)) + lambda_a * torch.sum(torch.abs(A))

                # DAG constraint using log-det
                h = compute_dag_constraint_logdet(W, s=s)

                # Augmented Lagrangian: L = loss + mu*h + (rho/2)*h^2
                loss_aug = mu * h + 0.5 * rho * h * h

                # Total loss
                loss = loss_mse + loss_l1 + loss_aug

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([W, A], max_norm=10.0)

                optimizer.step()

                # Zero out diagonal of W (no self-loops)
                with torch.no_grad():
                    W.fill_diagonal_(0)

            # Get current h value
            with torch.no_grad():
                h_val = compute_dag_constraint_logdet(W, s=s).item()

            if verbose and phase % 10 == 0:
                w_nnz = torch.sum(torch.abs(W) > w_threshold).item()
                a_nnz = torch.sum(torch.abs(A) > w_threshold).item()
                print(f"Phase {phase}: loss={loss.item():.4f}, h={h_val:.2e}, rho={rho:.1e}, W_nnz={w_nnz}, A_nnz={a_nnz}")

            # Check convergence
            if h_val < h_tol:
                if verbose:
                    print(f"Converged at phase {phase}, h={h_val:.2e}")
                break

            # Update Lagrangian parameters
            mu = mu + rho * h_val
            rho = min(rho * alpha, rho_max)

            # Anneal s for better convergence
            if phase > 0 and phase % 20 == 0:
                s = max(s * 0.9, 0.5)

        # Extract results
        W_est = W.detach().cpu().numpy()
        A_est = A.detach().cpu().numpy()

        # Apply threshold
        W_est[np.abs(W_est) < w_threshold] = 0
        A_est[np.abs(A_est) < w_threshold] = 0

        return W_est, A_est


def count_accuracy(W_true, W_est, threshold=0.0):
    """Compute accuracy metrics for graph recovery."""
    # Binarize
    W_true_bin = (np.abs(W_true) > threshold).astype(int)
    W_est_bin = (np.abs(W_est) > threshold).astype(int)

    # True positives, false positives, false negatives
    TP = np.sum((W_true_bin == 1) & (W_est_bin == 1))
    FP = np.sum((W_true_bin == 0) & (W_est_bin == 1))
    FN = np.sum((W_true_bin == 1) & (W_est_bin == 0))

    # Metrics
    TPR = TP / max(TP + FN, 1)  # Recall/Sensitivity
    FDR = FP / max(TP + FP, 1)  # False Discovery Rate
    SHD = FP + FN  # Structural Hamming Distance

    return {'tpr': TPR, 'fdr': FDR, 'shd': SHD}


if __name__ == "__main__":
    # Quick test
    from SVAR_data_generator import generate_svar_data

    print("Testing Meta-D2AG...")

    # Generate data
    X, B_true, A_list_true, params = generate_svar_data(
        n_nodes=10,
        n_timesteps=1000,
        lag_order=1,
        instantaneous_edges=20,
        temporal_sparsity=0.1,
        temporal_strength=0.2,
        noise_scale=1,
        noise_type='ev',
        seed=42
    )

    # Ground truth in DyCoLiDE convention
    B_true_t = B_true.T
    A_true = np.vstack([A.T for A in A_list_true])

    print(f"Data shape: {X.shape}")
    print(f"True W edges: {np.sum(np.abs(B_true_t) > 0)}")
    print(f"True A edges: {np.sum(np.abs(A_true) > 0)}")

    # Fit Meta-D2AG
    model = MetaD2AG(seed=42)
    W_est, A_est = model.fit(
        X, p=1,
        lambda_w=0.01, lambda_a=0.01,
        max_iter=2000, lr=0.003,
        w_threshold=0.1,
        use_logdet=True,
        verbose=True
    )

    # Evaluate
    w_metrics = count_accuracy(B_true_t, W_est)
    a_metrics = count_accuracy(A_true, A_est)

    print(f"\nW metrics: TPR={w_metrics['tpr']:.3f}, FDR={w_metrics['fdr']:.3f}, SHD={w_metrics['shd']}")
    print(f"A metrics: TPR={a_metrics['tpr']:.3f}, FDR={a_metrics['fdr']:.3f}, SHD={a_metrics['shd']}")
