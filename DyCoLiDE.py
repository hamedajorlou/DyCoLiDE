"""
DyCoLiDE: Dynamic Concomitant Linear DAG Estimation

Extension of CoLiDE for time-series data with SVAR (Structural Vector Autoregressive) models.

SVAR Model:
    X = XW + YA + Z

where:
- X: (n, d) current observations
- Y: (n, pd) lagged observations [Y₁ | Y₂ | ... | Yₚ]
- W: (d, d) intra-slice (contemporaneous) weights - DAG
- A: (pd, d) inter-slice (temporal) weights
- Z: (n, d) noise
- p: autoregressive order

Two variants:
- DyCoLiDE-EV: Equal variance (homoscedastic)
- DyCoLiDE-NV: Non-equal variance (heteroscedastic)
"""

import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from tqdm import tqdm
from typing import Tuple, List, Optional


def create_lagged_data(X: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged observation matrix Y from time series X.

    Parameters
    ----------
    X : np.ndarray, shape (T, d)
        Full time series data
    p : int
        Number of lags (autoregressive order)

    Returns
    -------
    X_current : np.ndarray, shape (T-p, d)
        Current observations (excluding first p timesteps)
    Y_lagged : np.ndarray, shape (T-p, p*d)
        Lagged observations [Y₁ | Y₂ | ... | Yₚ] where Yₖ = X_{t-k}
    """
    T, d = X.shape
    n = T - p  # Number of usable samples

    X_current = X[p:]  # Current observations: X_p, X_{p+1}, ..., X_{T-1}

    # Build lagged matrix: each row t contains [X_{t-1}, X_{t-2}, ..., X_{t-p}]
    Y_lagged = np.zeros((n, p * d))
    for k in range(p):
        # Lag k+1: X_{t-(k+1)} for t = p, p+1, ..., T-1
        Y_lagged[:, k*d:(k+1)*d] = X[p-k-1:T-k-1]

    return X_current, Y_lagged


class DyCoLiDE_EV:
    """
    DyCoLiDE with Equal Variance (homoscedastic noise).

    Score function:
        S(W, A, σ) = (1/(2nσ)) * ||X - XW - YA||²_F + (d*σ)/2 + λ_W*||W||₁ + λ_A*||A||₁
    """

    def __init__(self, dtype=np.float64, seed=0):
        np.random.seed(seed)
        self.dtype = dtype

    def _score(self, W: np.ndarray, A: np.ndarray, sigma: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute score and gradients.

        Returns: (loss, grad_W, grad_A)
        """
        # Residual: R = X - XW - YA
        R = self.X - self.X @ W - self.Y @ A

        # Score loss
        loss = (1.0 / (2 * self.n * sigma)) * np.sum(R ** 2) + (self.d * sigma) / 2.0

        # Gradients
        grad_W = -(1.0 / (self.n * sigma)) * (self.X.T @ R)
        grad_A = -(1.0 / (self.n * sigma)) * (self.Y.T @ R)

        return loss, grad_W, grad_A

    def _h(self, W: np.ndarray, s: float = 1.0) -> Tuple[float, np.ndarray]:
        """DAG constraint using log-det formulation."""
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W: np.ndarray, A: np.ndarray, sigma: float, mu: float, s: float = 1.0) -> float:
        """Compute augmented Lagrangian objective."""
        score, _, _ = self._score(W, A, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda_W * np.abs(W).sum() + self.lambda_A * np.abs(A).sum()) + h
        return obj

    def _adam_update(self, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
                     iter: int, beta_1: float, beta_2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adam optimizer update."""
        m = m * beta_1 + (1 - beta_1) * grad
        v = v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = m / (1 - beta_1 ** iter)
        v_hat = v / (1 - beta_2 ** iter)
        update = m_hat / (np.sqrt(v_hat) + 1e-8)
        return update, m, v

    def _update_sigma(self, W: np.ndarray, A: np.ndarray) -> float:
        """Update noise variance estimate (closed-form solution)."""
        R = self.X - self.X @ W - self.Y @ A
        sigma = np.sqrt(np.sum(R ** 2) / (self.n * self.d))
        return max(sigma, 1e-6)  # Prevent zero variance

    def minimize(self, W: np.ndarray, A: np.ndarray, sigma: float, mu: float,
                 max_iter: int, s: float, lr: float, tol: float = 1e-6,
                 beta_1: float = 0.99, beta_2: float = 0.999, pbar=None) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Minimize objective using Adam optimizer."""
        obj_prev = 1e16

        # Adam states for W and A
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_A, v_A = np.zeros_like(A), np.zeros_like(A)

        for iter in range(1, max_iter + 1):
            # Check DAG constraint
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, A, sigma, False
                else:
                    W += lr * grad_W_update
                    A += lr * grad_A_update
                    lr *= 0.5
                    if lr <= 1e-16:
                        return W, A, sigma, True
                    W -= lr * grad_W_update
                    A -= lr * grad_A_update
                    sigma = self._update_sigma(W, A)
                    M = sla.inv(s * self.Id - W * W) + 1e-16

            # Compute gradients
            _, grad_W_score, grad_A_score = self._score(W, A, sigma)
            _, G_h = self._h(W, s)

            # Full gradient for W (includes DAG constraint)
            grad_W = mu * (grad_W_score + self.lambda_W * np.sign(W)) + G_h
            # Full gradient for A (no DAG constraint)
            grad_A = mu * (grad_A_score + self.lambda_A * np.sign(A))

            # Adam updates
            grad_W_update, m_W, v_W = self._adam_update(grad_W, m_W, v_W, iter, beta_1, beta_2)
            grad_A_update, m_A, v_A = self._adam_update(grad_A, m_A, v_A, iter, beta_1, beta_2)

            W -= lr * grad_W_update
            A -= lr * grad_A_update

            # Update sigma (closed-form)
            sigma = self._update_sigma(W, A)

            # Check convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new = self._func(W, A, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, A, sigma, True

    def fit(self, X_full: np.ndarray, p: int, lambda_W: float = 0.05, lambda_A: float = 0.05,
            T: int = 5, mu_init: float = 1.0, mu_factor: float = 0.1,
            s: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6],
            warm_iter: int = 3000, max_iter: int = 6000, lr: float = 0.0003,
            checkpoint: int = 1000, beta_1: float = 0.99, beta_2: float = 0.999,
            W_init: Optional[np.ndarray] = None, A_init: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit DyCoLiDE-EV model.

        Parameters
        ----------
        X_full : np.ndarray, shape (T_total, d)
            Full time series data
        p : int
            Lag order (number of lags)
        lambda_W : float
            L1 regularization for intra-slice weights
        lambda_A : float
            L1 regularization for inter-slice weights

        Returns
        -------
        W_est : np.ndarray, shape (d, d)
            Estimated intra-slice (contemporaneous) DAG
        A_est : np.ndarray, shape (p*d, d)
            Estimated inter-slice (temporal) weights
        sigma_est : float
            Estimated noise standard deviation
        """
        # Create lagged data
        self.X, self.Y = create_lagged_data(X_full, p)
        self.n, self.d = self.X.shape
        self.p = p
        self.lambda_W = lambda_W
        self.lambda_A = lambda_A
        self.checkpoint = checkpoint

        # Center data
        self.X = self.X - self.X.mean(axis=0, keepdims=True)
        self.Y = self.Y - self.Y.mean(axis=0, keepdims=True)

        self.Id = np.eye(self.d).astype(self.dtype)

        # Initialize parameters
        if W_init is None:
            self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        else:
            self.W_est = W_init.astype(self.dtype)

        if A_init is None:
            self.A_est = np.zeros((p * self.d, self.d)).astype(self.dtype)
        else:
            self.A_est = A_init.astype(self.dtype)

        # Initialize sigma
        self.sig_est = self._update_sigma(self.W_est, self.A_est)

        # Setup schedule
        mu = mu_init
        if isinstance(s, list):
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        else:
            s = T * [s]

        # Optimization loop
        with tqdm(total=int((T-1)*warm_iter + max_iter), desc="DyCoLiDE-EV") as pbar:
            for i in range(T):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)

                while not success:
                    W_temp, A_temp, sig_temp, success = self.minimize(
                        self.W_est.copy(), self.A_est.copy(), self.sig_est,
                        mu, inner_iters, s[i], lr=lr_adam,
                        beta_1=beta_1, beta_2=beta_2, pbar=pbar
                    )
                    if not success:
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.A_est = A_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.A_est, self.sig_est


class DyCoLiDE_NV:
    """
    DyCoLiDE with Non-equal Variance (heteroscedastic noise).

    Score function:
        S(W, A, Σ) = (1/(2n)) * Tr[(X - XW - YA)ᵀ Σ⁻¹ (X - XW - YA)]
                   + (1/2)*Tr(Σ) + λ_W*||W||₁ + λ_A*||A||₁

    where Σ = diag(σ₁², σ₂², ..., σ_d²)
    """

    def __init__(self, dtype=np.float64, seed=0):
        np.random.seed(seed)
        self.dtype = dtype

    def _score(self, W: np.ndarray, A: np.ndarray, sigma: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute score and gradients for heteroscedastic case.

        sigma : array of shape (d,) - node-specific standard deviations
        """
        # Residual: R = X - XW - YA, shape (n, d)
        R = self.X - self.X @ W - self.Y @ A

        # Inverse variance: 1/σ² for each node, shape (d,)
        inv_var = 1.0 / (sigma ** 2 + 1e-8)

        # Score loss: (1/2n) * sum_j (1/σ_j²) * sum_i R_{ij}² + (1/2) * sum(σ²)
        # Efficient computation: sum over all elements of (R² * inv_var)
        loss = (1.0 / (2 * self.n)) * np.sum(R ** 2 * inv_var) + 0.5 * np.sum(sigma ** 2)

        # Gradients: scale each column of R by its inverse variance
        R_scaled = R * inv_var  # Broadcasting: (n, d) * (d,) -> (n, d)
        grad_W = -(1.0 / self.n) * (self.X.T @ R_scaled)
        grad_A = -(1.0 / self.n) * (self.Y.T @ R_scaled)

        return loss, grad_W, grad_A

    def _h(self, W: np.ndarray, s: float = 1.0) -> Tuple[float, np.ndarray]:
        """DAG constraint using log-det formulation."""
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W: np.ndarray, A: np.ndarray, sigma: np.ndarray, mu: float, s: float = 1.0) -> float:
        """Compute augmented Lagrangian objective."""
        score, _, _ = self._score(W, A, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda_W * np.abs(W).sum() + self.lambda_A * np.abs(A).sum()) + h
        return obj

    def _adam_update(self, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
                     iter: int, beta_1: float, beta_2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adam optimizer update."""
        m = m * beta_1 + (1 - beta_1) * grad
        v = v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = m / (1 - beta_1 ** iter)
        v_hat = v / (1 - beta_2 ** iter)
        update = m_hat / (np.sqrt(v_hat) + 1e-8)
        return update, m, v

    def _update_sigma(self, W: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Update node-specific variance estimates (closed-form solution)."""
        R = self.X - self.X @ W - self.Y @ A
        # σ_j = sqrt((1/n) * sum_i R_{ij}²)
        sigma = np.sqrt(np.mean(R ** 2, axis=0))
        return np.maximum(sigma, 1e-6)  # Prevent zero variance

    def minimize(self, W: np.ndarray, A: np.ndarray, sigma: np.ndarray, mu: float,
                 max_iter: int, s: float, lr: float, tol: float = 1e-6,
                 beta_1: float = 0.99, beta_2: float = 0.999, pbar=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Minimize objective using Adam optimizer."""
        obj_prev = 1e16

        # Adam states
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_A, v_A = np.zeros_like(A), np.zeros_like(A)

        for iter in range(1, max_iter + 1):
            # Check DAG constraint
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, A, sigma, False
                else:
                    W += lr * grad_W_update
                    A += lr * grad_A_update
                    lr *= 0.5
                    if lr <= 1e-16:
                        return W, A, sigma, True
                    W -= lr * grad_W_update
                    A -= lr * grad_A_update
                    sigma = self._update_sigma(W, A)
                    M = sla.inv(s * self.Id - W * W) + 1e-16

            # Compute gradients
            _, grad_W_score, grad_A_score = self._score(W, A, sigma)
            _, G_h = self._h(W, s)

            # Full gradients
            grad_W = mu * (grad_W_score + self.lambda_W * np.sign(W)) + G_h
            grad_A = mu * (grad_A_score + self.lambda_A * np.sign(A))

            # Adam updates
            grad_W_update, m_W, v_W = self._adam_update(grad_W, m_W, v_W, iter, beta_1, beta_2)
            grad_A_update, m_A, v_A = self._adam_update(grad_A, m_A, v_A, iter, beta_1, beta_2)

            W -= lr * grad_W_update
            A -= lr * grad_A_update

            # Update sigma (closed-form)
            sigma = self._update_sigma(W, A)

            # Check convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new = self._func(W, A, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, A, sigma, True

    def fit(self, X_full: np.ndarray, p: int, lambda_W: float = 0.05, lambda_A: float = 0.05,
            T: int = 5, mu_init: float = 1.0, mu_factor: float = 0.1,
            s: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6],
            warm_iter: int = 3000, max_iter: int = 6000, lr: float = 0.0003,
            checkpoint: int = 1000, beta_1: float = 0.99, beta_2: float = 0.999,
            W_init: Optional[np.ndarray] = None, A_init: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit DyCoLiDE-NV model.

        Returns
        -------
        W_est : np.ndarray, shape (d, d)
            Estimated intra-slice (contemporaneous) DAG
        A_est : np.ndarray, shape (p*d, d)
            Estimated inter-slice (temporal) weights
        sigma_est : np.ndarray, shape (d,)
            Estimated node-specific noise standard deviations
        """
        # Create lagged data
        self.X, self.Y = create_lagged_data(X_full, p)
        self.n, self.d = self.X.shape
        self.p = p
        self.lambda_W = lambda_W
        self.lambda_A = lambda_A
        self.checkpoint = checkpoint

        # Center data
        self.X = self.X - self.X.mean(axis=0, keepdims=True)
        self.Y = self.Y - self.Y.mean(axis=0, keepdims=True)

        self.Id = np.eye(self.d).astype(self.dtype)

        # Initialize parameters
        if W_init is None:
            self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        else:
            self.W_est = W_init.astype(self.dtype)

        if A_init is None:
            self.A_est = np.zeros((p * self.d, self.d)).astype(self.dtype)
        else:
            self.A_est = A_init.astype(self.dtype)

        # Initialize sigma (node-specific)
        self.sig_est = self._update_sigma(self.W_est, self.A_est)

        # Setup schedule
        mu = mu_init
        if isinstance(s, list):
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        else:
            s = T * [s]

        # Optimization loop
        with tqdm(total=int((T-1)*warm_iter + max_iter), desc="DyCoLiDE-NV") as pbar:
            for i in range(T):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)

                while not success:
                    W_temp, A_temp, sig_temp, success = self.minimize(
                        self.W_est.copy(), self.A_est.copy(), self.sig_est.copy(),
                        mu, inner_iters, s[i], lr=lr_adam,
                        beta_1=beta_1, beta_2=beta_2, pbar=pbar
                    )
                    if not success:
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.A_est = A_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.A_est, self.sig_est


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


class DyCoLiDE_BatchStreaming:
    """
    DyCoLiDE with Batch Streaming for SVAR data.

    Processes time-series data in sequential batches of consecutive time points.
    Accumulates data across batches and runs full optimization after all data arrives.

    Model: X_t = X_t @ W + Y_t @ A + noise

    Key features:
    - Maintains buffer of last p observations for cross-batch lagged data
    - Accumulates all (X, Y) pairs across batches
    - After all batches arrive, runs full optimization identical to DyCoLiDE_EV
    """

    def __init__(self, d: int, p: int = 1, dtype=np.float64, seed: int = 0):
        """
        Initialize the streaming model.

        Parameters
        ----------
        d : int
            Number of variables/nodes
        p : int
            Lag order (autoregressive order)
        dtype : type
            Data type for arrays
        seed : int
            Random seed
        """
        np.random.seed(seed)
        self.dtype = dtype
        self.d = d
        self.p = p
        self.Id = np.eye(d).astype(dtype)

        # Initialize parameters
        self.W_est = np.zeros((d, d)).astype(dtype)
        self.A_est = np.zeros((p * d, d)).astype(dtype)
        self.sig_est = 1.0

        # Buffer for cross-batch lagged data (stores last p observations)
        self.buffer = None  # Shape: (p, d)

        # Accumulated data across batches
        self.X_accumulated = []
        self.Y_accumulated = []
        self.n_samples_seen = 0

    def _create_batch_lagged_data(self, X_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lagged data for current batch, using buffer from previous batch.
        """
        batch_size = X_batch.shape[0]

        if self.buffer is None:
            # First batch: can only use samples after first p observations
            if batch_size <= self.p:
                return None, None

            X_current = X_batch[self.p:]
            Y_lagged = np.zeros((batch_size - self.p, self.p * self.d))
            for k in range(self.p):
                Y_lagged[:, k*self.d:(k+1)*self.d] = X_batch[self.p-k-1:batch_size-k-1]
        else:
            # Subsequent batches: prepend buffer
            full_data = np.vstack([self.buffer, X_batch])
            n_full = full_data.shape[0]

            X_current = full_data[self.p:]
            Y_lagged = np.zeros((n_full - self.p, self.p * self.d))
            for k in range(self.p):
                Y_lagged[:, k*self.d:(k+1)*self.d] = full_data[self.p-k-1:n_full-k-1]

        return X_current, Y_lagged

    def _update_buffer(self, X_batch: np.ndarray):
        """Store last p observations for next batch."""
        self.buffer = X_batch[-self.p:].copy()

    def process_batch(self, X_batch: np.ndarray):
        """
        Process a single batch: accumulate data for later optimization.

        Parameters
        ----------
        X_batch : np.ndarray, shape (batch_size, d)
            Batch of consecutive observations
        """
        # Create lagged data for this batch
        X_current, Y_lagged = self._create_batch_lagged_data(X_batch)

        if X_current is not None:
            self.X_accumulated.append(X_current)
            self.Y_accumulated.append(Y_lagged)
            self.n_samples_seen += X_current.shape[0]

        # Update buffer for next batch
        self._update_buffer(X_batch)

    def _score(self, W: np.ndarray, A: np.ndarray, sigma: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute score and gradients."""
        R = self.X - self.X @ W - self.Y @ A
        loss = (1.0 / (2 * self.n * sigma)) * np.sum(R ** 2) + (self.d * sigma) / 2.0
        grad_W = -(1.0 / (self.n * sigma)) * (self.X.T @ R)
        grad_A = -(1.0 / (self.n * sigma)) * (self.Y.T @ R)
        return loss, grad_W, grad_A

    def _h(self, W: np.ndarray, s: float = 1.0) -> Tuple[float, np.ndarray]:
        """DAG constraint using log-det formulation."""
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W: np.ndarray, A: np.ndarray, sigma: float, mu: float, s: float = 1.0) -> float:
        """Compute augmented Lagrangian objective."""
        score, _, _ = self._score(W, A, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda_W * np.abs(W).sum() + self.lambda_A * np.abs(A).sum()) + h
        return obj

    def _adam_update(self, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
                     iter: int, beta_1: float, beta_2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adam optimizer update."""
        m = m * beta_1 + (1 - beta_1) * grad
        v = v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = m / (1 - beta_1 ** iter)
        v_hat = v / (1 - beta_2 ** iter)
        update = m_hat / (np.sqrt(v_hat) + 1e-8)
        return update, m, v

    def _update_sigma(self, W: np.ndarray, A: np.ndarray) -> float:
        """Update noise variance estimate."""
        R = self.X - self.X @ W - self.Y @ A
        sigma = np.sqrt(np.sum(R ** 2) / (self.n * self.d))
        return max(sigma, 1e-6)

    def _minimize(self, W: np.ndarray, A: np.ndarray, sigma: float, mu: float,
                  max_iter: int, s: float, lr: float, tol: float = 1e-6,
                  beta_1: float = 0.99, beta_2: float = 0.999, pbar=None) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Minimize objective using Adam optimizer (identical to DyCoLiDE_EV)."""
        obj_prev = 1e16

        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_A, v_A = np.zeros_like(A), np.zeros_like(A)

        for iter in range(1, max_iter + 1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, A, sigma, False
                else:
                    W += lr * grad_W_update
                    A += lr * grad_A_update
                    lr *= 0.5
                    if lr <= 1e-16:
                        return W, A, sigma, True
                    W -= lr * grad_W_update
                    A -= lr * grad_A_update
                    sigma = self._update_sigma(W, A)
                    M = sla.inv(s * self.Id - W * W) + 1e-16

            _, grad_W_score, grad_A_score = self._score(W, A, sigma)
            _, G_h = self._h(W, s)

            grad_W = mu * (grad_W_score + self.lambda_W * np.sign(W)) + G_h
            grad_A = mu * (grad_A_score + self.lambda_A * np.sign(A))

            grad_W_update, m_W, v_W = self._adam_update(grad_W, m_W, v_W, iter, beta_1, beta_2)
            grad_A_update, m_A, v_A = self._adam_update(grad_A, m_A, v_A, iter, beta_1, beta_2)

            W -= lr * grad_W_update
            A -= lr * grad_A_update

            sigma = self._update_sigma(W, A)

            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new = self._func(W, A, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, A, sigma, True

    def finalize(self, lambda_W: float = 0.05, lambda_A: float = 0.05,
                 T: int = 5, mu_init: float = 1.0, mu_factor: float = 0.1,
                 s: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6],
                 warm_iter: int = 3000, max_iter: int = 6000, lr: float = 0.0003,
                 checkpoint: int = 1000, beta_1: float = 0.99, beta_2: float = 0.999,
                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run full optimization on accumulated data (call after all batches processed).

        Uses identical optimization to DyCoLiDE_EV.
        """
        if len(self.X_accumulated) == 0:
            raise ValueError("No data accumulated. Call process_batch first.")

        # Combine accumulated data
        self.X = np.vstack(self.X_accumulated)
        self.Y = np.vstack(self.Y_accumulated)
        self.n = self.X.shape[0]

        # Center data (global centering)
        self.X = self.X - self.X.mean(axis=0, keepdims=True)
        self.Y = self.Y - self.Y.mean(axis=0, keepdims=True)

        self.lambda_W = lambda_W
        self.lambda_A = lambda_A
        self.checkpoint = checkpoint

        # Initialize parameters
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        self.A_est = np.zeros((self.p * self.d, self.d)).astype(self.dtype)
        self.sig_est = self._update_sigma(self.W_est, self.A_est)

        # Setup schedule
        mu = mu_init
        if isinstance(s, list):
            s = s.copy()
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        else:
            s = T * [s]

        # Optimization loop (identical to DyCoLiDE_EV)
        with tqdm(total=int((T-1)*warm_iter + max_iter), desc="DyCoLiDE-BatchStreaming", disable=not verbose) as pbar:
            for i in range(T):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)

                while not success:
                    W_temp, A_temp, sig_temp, success = self._minimize(
                        self.W_est.copy(), self.A_est.copy(), self.sig_est,
                        mu, inner_iters, s[i], lr=lr_adam,
                        beta_1=beta_1, beta_2=beta_2, pbar=pbar
                    )
                    if not success:
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.A_est = A_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.A_est, self.sig_est

    def fit_streaming(self, X_full: np.ndarray, batch_size: int = 100,
                      lambda_W: float = 0.05, lambda_A: float = 0.05,
                      T: int = 5, mu_init: float = 1.0, mu_factor: float = 0.1,
                      s: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6],
                      warm_iter: int = 3000, max_iter: int = 6000, lr: float = 0.0003,
                      checkpoint: int = 1000, beta_1: float = 0.99, beta_2: float = 0.999,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit model by simulating streaming batches from full data.

        Parameters
        ----------
        X_full : np.ndarray, shape (T, d)
            Full time series data
        batch_size : int
            Number of time points per batch
        lambda_W, lambda_A : float
            L1 regularization parameters
        T : int
            Number of outer iterations
        mu_init : float
            Initial penalty weight
        mu_factor : float
            Decay factor for mu
        s : list
            Schedule for DAG constraint parameter
        warm_iter : int
            Iterations for warmup phases
        max_iter : int
            Iterations for final phase
        lr : float
            Learning rate
        checkpoint : int
            Checkpoint frequency
        verbose : bool
            Whether to show progress

        Returns
        -------
        W_est, A_est, sigma_est
        """
        T_total = X_full.shape[0]
        n_batches = (T_total + batch_size - 1) // batch_size

        # Phase 1: Accumulate data from batches
        if verbose:
            print(f"Processing {n_batches} batches...")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, T_total)
            X_batch = X_full[start_idx:end_idx]
            self.process_batch(X_batch)

        if verbose:
            print(f"Accumulated {self.n_samples_seen} samples. Running optimization...")

        # Phase 2: Run full optimization
        return self.finalize(
            lambda_W=lambda_W, lambda_A=lambda_A,
            T=T, mu_init=mu_init, mu_factor=mu_factor,
            s=s, warm_iter=warm_iter, max_iter=max_iter, lr=lr,
            checkpoint=checkpoint, beta_1=beta_1, beta_2=beta_2,
            verbose=verbose
        )

    def get_estimates(self):
        """Return current parameter estimates."""
        return self.W_est.copy(), self.A_est.copy(), self.sig_est
