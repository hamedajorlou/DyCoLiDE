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


class DyCoLiDE_EV_batch:
    """
    DyCoLiDE-EV with mini-batch stochastic gradient descent.

    Direct SVAR analog of colide_ev_batch (SEM): subsamples (X, Y) row-pairs
    each SGD step, maintains running covariances Σ_XX, Σ_XY, Σ_YY, and does
    Adam updates on W and A jointly.

    Score function (full-data form, matched by the running-covariance form
    that's actually used inside the SGD loop):
        S(W, A, σ) = (1/(2σ)) · ‖X(I-W) - YA‖²_F / n  +  (dσ)/2
                   + λ_W·‖W‖_1 + λ_A·‖A‖_1

    Running covariances:
        Σ_XX_t  ← (t-1)/t · Σ_XX_{t-1} + 1/t · X_batch.T @ X_batch / B
        Σ_XY_t  ← (t-1)/t · Σ_XY_{t-1} + 1/t · X_batch.T @ Y_batch / B
        Σ_YY_t  ← (t-1)/t · Σ_YY_{t-1} + 1/t · Y_batch.T @ Y_batch / B

    Gradients from the running covariances:
        grad_W = (Σ_XY · A  -  Σ_XX · (I-W)) / σ
        grad_A = (Σ_YY · A  -  Σ_XY^T · (I-W)) / σ

    Sigma is updated closed-form from the running covariances (same scheme
    as colide_ev_batch / DyCoLiDE_EV — never EMA-on-residuals).

    Supports:
        - batch_size == 1 → true online (sequential walk with Welford updates)
        - batch_size > 1  → mini-batch SGD (random row-pair subsets)
    """

    def __init__(self, dtype=np.float64, seed=0):
        np.random.seed(seed)
        self.dtype = dtype
        self.online_mode = False

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _score_from_cov(self, W, A, sigma):
        """Score value using the running covariances (no fresh batch)."""
        dif = self.Id - W
        # tr((I-W)^T Σ_XX (I-W)) - 2 tr((I-W)^T Σ_XY A) + tr(A^T Σ_YY A)
        term1 = np.trace(dif.T @ self.cov_XX @ dif)
        term2 = np.trace(dif.T @ self.cov_XY @ A)
        term3 = np.trace(A.T @ self.cov_YY @ A)
        rss_over_n = term1 - 2 * term2 + term3
        loss = (1.0 / (2 * sigma)) * rss_over_n + (self.d * sigma) / 2.0
        return loss

    def _func(self, W, A, sigma, mu, s=1.0):
        score = self._score_from_cov(W, A, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda_W * np.abs(W).sum()
                          + self.lambda_A * np.abs(A).sum()) + h
        return obj

    def _adam_update(self, grad, m, v, iter, beta_1, beta_2):
        m = m * beta_1 + (1 - beta_1) * grad
        v = v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = m / (1 - beta_1 ** iter)
        v_hat = v / (1 - beta_2 ** iter)
        update = m_hat / (np.sqrt(v_hat) + 1e-8)
        return update, m, v

    def _update_covariance(self, X_batch, Y_batch, t):
        """Update running Σ_XX, Σ_XY, Σ_YY from a (centered) mini-batch."""
        n_batch = X_batch.shape[0]
        batch_XX = X_batch.T @ X_batch / n_batch
        batch_XY = X_batch.T @ Y_batch / n_batch
        batch_YY = Y_batch.T @ Y_batch / n_batch
        if t == 1:
            self.cov_XX = batch_XX
            self.cov_XY = batch_XY
            self.cov_YY = batch_YY
        else:
            w_old = (t - 1) / t
            w_new = 1.0 / t
            self.cov_XX = w_old * self.cov_XX + w_new * batch_XX
            self.cov_XY = w_old * self.cov_XY + w_new * batch_XY
            self.cov_YY = w_old * self.cov_YY + w_new * batch_YY

    def _update_covariance_online(self, x, y, t):
        """Welford-style update for batch_size==1: tracks joint mean/M2 of [x|y]."""
        delta_x = x - self.online_mean_x
        delta_y = y - self.online_mean_y
        self.online_mean_x = self.online_mean_x + delta_x / t
        self.online_mean_y = self.online_mean_y + delta_y / t
        delta_x2 = x - self.online_mean_x
        delta_y2 = y - self.online_mean_y
        self.online_M2_XX = self.online_M2_XX + np.outer(delta_x, delta_x2)
        self.online_M2_XY = self.online_M2_XY + np.outer(delta_x, delta_y2)
        self.online_M2_YY = self.online_M2_YY + np.outer(delta_y, delta_y2)
        if t > 1:
            self.cov_XX = self.online_M2_XX / t
            self.cov_XY = self.online_M2_XY / t
            self.cov_YY = self.online_M2_YY / t
        else:
            self.cov_XX = self.online_M2_XX.copy()
            self.cov_XY = self.online_M2_XY.copy()
            self.cov_YY = self.online_M2_YY.copy()

    def _update_sigma(self, W, A):
        """Closed-form sigma from running covariances."""
        dif = self.Id - W
        term1 = np.trace(dif.T @ self.cov_XX @ dif)
        term2 = np.trace(dif.T @ self.cov_XY @ A)
        term3 = np.trace(A.T @ self.cov_YY @ A)
        sigma_sq = (term1 - 2 * term2 + term3) / self.d
        return max(np.sqrt(max(sigma_sq, 0.0)), 1e-8)

    def minimize_batch(self, W, A, sigma, mu, n_batches, batch_size, s, lr,
                       tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_A, v_A = np.zeros_like(A), np.zeros_like(A)

        n_total = self.X.shape[0]
        indices = np.arange(n_total)

        for batch_idx in range(1, n_batches + 1):
            if self.online_mode:
                x = self.X[self.online_sample_idx]
                y = self.Y[self.online_sample_idx]
                self.online_sample_idx = (self.online_sample_idx + 1) % n_total
                self.online_count += 1
                self._update_covariance_online(x, y, self.online_count)
            else:
                batch_indices = np.random.choice(indices, size=batch_size, replace=False)
                X_batch = self.X[batch_indices]
                Y_batch = self.Y[batch_indices]
                X_batch = X_batch - X_batch.mean(axis=0, keepdims=True)
                Y_batch = Y_batch - Y_batch.mean(axis=0, keepdims=True)
                self._update_covariance(X_batch, Y_batch, batch_idx)

            # Feasibility check for log-det (sI - W⊙W must be PD)
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if batch_idx == 1 or s <= 0.9:
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

            # Gradients from running covariances
            dif = self.Id - W
            G_score_W = (self.cov_XY @ A - self.cov_XX @ dif) / sigma
            G_score_A = (self.cov_YY @ A - self.cov_XY.T @ dif) / sigma
            _, G_h = self._h(W, s)

            grad_W = mu * (G_score_W + self.lambda_W * np.sign(W)) + G_h
            grad_A = mu * (G_score_A + self.lambda_A * np.sign(A))

            grad_W_update, m_W, v_W = self._adam_update(grad_W, m_W, v_W,
                                                       batch_idx, beta_1, beta_2)
            grad_A_update, m_A, v_A = self._adam_update(grad_A, m_A, v_A,
                                                       batch_idx, beta_1, beta_2)

            W -= lr * grad_W_update
            A -= lr * grad_A_update

            sigma = self._update_sigma(W, A)

            # Convergence check
            if batch_idx % self.checkpoint == 0 or batch_idx == n_batches:
                obj_new = self._func(W, A, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(n_batches - batch_idx + 1)
                    break
                obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, A, sigma, True

    def fit(self, X_full: np.ndarray, p: int,
            lambda_W: float = 0.05, lambda_A: float = 0.05,
            T: int = 4, mu_init: float = 1.0, mu_factor: float = 0.1,
            s=(1.0, 0.9, 0.8, 0.7),
            batch_size: int = 100,
            n_batches_warm: int = 300, n_batches_final: int = 600,
            lr: float = 0.0003, checkpoint: int = 100,
            beta_1: float = 0.99, beta_2: float = 0.999,
            W_init: Optional[np.ndarray] = None,
            A_init: Optional[np.ndarray] = None):
        """
        Fit DyCoLiDE-EV with mini-batch SGD on SVAR data.

        Args mirror DyCoLiDE_EV.fit; differences:
          - batch_size (1 → online, >1 → mini-batch SGD)
          - n_batches_warm / n_batches_final replace warm_iter / max_iter
            (one SGD step per "batch")
          - W_init / A_init: optional warm-start adjacency / lag matrices
            for sliding-window streaming.
        """
        # Build lagged data (DyCoLiDE_EV convention)
        self.X, self.Y = create_lagged_data(X_full, p)
        self.n, self.d = self.X.shape
        self.p = p
        self.lambda_W = lambda_W
        self.lambda_A = lambda_A
        self.checkpoint = checkpoint
        self.Id = np.eye(self.d).astype(self.dtype)

        # Center globally (mini-batches will also recenter, see minimize_batch)
        self.X = self.X - self.X.mean(axis=0, keepdims=True)
        self.Y = self.Y - self.Y.mean(axis=0, keepdims=True)

        pd_ = p * self.d
        self.cov_XX = np.zeros((self.d, self.d), dtype=self.dtype)
        self.cov_XY = np.zeros((self.d, pd_), dtype=self.dtype)
        self.cov_YY = np.zeros((pd_, pd_), dtype=self.dtype)
        if W_init is None:
            self.W_est = np.zeros((self.d, self.d), dtype=self.dtype)
        else:
            self.W_est = np.asarray(W_init, dtype=self.dtype).copy()
        if A_init is None:
            self.A_est = np.zeros((pd_, self.d), dtype=self.dtype)
        else:
            self.A_est = np.asarray(A_init, dtype=self.dtype).copy()
        self.sig_est = 1.0

        self.online_mode = (batch_size == 1)
        if self.online_mode:
            self.online_mean_x = np.zeros(self.d, dtype=self.dtype)
            self.online_mean_y = np.zeros(pd_, dtype=self.dtype)
            self.online_M2_XX = np.zeros((self.d, self.d), dtype=self.dtype)
            self.online_M2_XY = np.zeros((self.d, pd_), dtype=self.dtype)
            self.online_M2_YY = np.zeros((pd_, pd_), dtype=self.dtype)
            self.online_sample_idx = 0
            self.online_count = 0

        # s schedule
        s_list = list(s) if isinstance(s, (list, tuple)) else [float(s)] * T
        if len(s_list) < T:
            s_list = s_list + (T - len(s_list)) * [s_list[-1]]

        mu = mu_init
        total = (T - 1) * n_batches_warm + n_batches_final
        desc = ("DyCoLiDE-EV-batch (online)" if batch_size == 1
                else f"DyCoLiDE-EV-batch (bs={batch_size})")

        with tqdm(total=int(total), desc=desc) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                n_inner = n_batches_final if i == T - 1 else n_batches_warm

                while not success:
                    W_temp, A_temp, sig_temp, success = self.minimize_batch(
                        self.W_est.copy(), self.A_est.copy(), self.sig_est,
                        mu, n_inner, batch_size, s_list[i], lr=lr_adam,
                        beta_1=beta_1, beta_2=beta_2, pbar=pbar
                    )
                    if not success:
                        lr_adam *= 0.5
                        s_list[i] += 0.1

                self.W_est = W_temp
                self.A_est = A_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.A_est, self.sig_est


class DyCoLiDE_NV_batch:
    """
    DyCoLiDE-NV with mini-batch SGD on SVAR data.

    Same machinery as DyCoLiDE_EV_batch (running covariances Σ_XX, Σ_XY, Σ_YY,
    mini-batch SGD with Adam, log-det acyclicity on W only) but with a vector
    of per-node noise scales σ ∈ R^d updated by the covariance-based closed
    form (never EMA-on-residuals).

    Gradients:
        grad_W = (Σ_XY A − Σ_XX (I−W)) · diag(1/σ)
        grad_A = (Σ_YY A − Σ_XY^T (I−W)) · diag(1/σ)

    σ update:
        σ_j² = [(I−W)^T Σ_XX (I−W) − 2 (I−W)^T Σ_XY A + A^T Σ_YY A]_jj
    """

    def __init__(self, dtype=np.float64, seed=0):
        np.random.seed(seed)
        self.dtype = dtype
        self.online_mode = False

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _score_from_cov(self, W, A, sigma):
        dif = self.Id - W
        # M = (I-W)^T Σ_XX (I-W) - 2 (I-W)^T Σ_XY A + A^T Σ_YY A
        M_jj = (np.diag(dif.T @ self.cov_XX @ dif)
                - 2.0 * np.diag(dif.T @ self.cov_XY @ A)
                + np.diag(A.T @ self.cov_YY @ A))
        # NV loss: 0.5 * sum_j (M_jj / σ_j) + 0.5 * sum σ
        loss = 0.5 * np.sum(M_jj / np.maximum(sigma, 1e-8)) + 0.5 * np.sum(sigma)
        return loss

    def _func(self, W, A, sigma, mu, s=1.0):
        score = self._score_from_cov(W, A, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda_W * np.abs(W).sum()
                          + self.lambda_A * np.abs(A).sum()) + h
        return obj

    def _adam_update(self, grad, m, v, iter, beta_1, beta_2):
        m = m * beta_1 + (1 - beta_1) * grad
        v = v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = m / (1 - beta_1 ** iter)
        v_hat = v / (1 - beta_2 ** iter)
        update = m_hat / (np.sqrt(v_hat) + 1e-8)
        return update, m, v

    def _update_covariance(self, X_batch, Y_batch, t):
        n_batch = X_batch.shape[0]
        batch_XX = X_batch.T @ X_batch / n_batch
        batch_XY = X_batch.T @ Y_batch / n_batch
        batch_YY = Y_batch.T @ Y_batch / n_batch
        if t == 1:
            self.cov_XX = batch_XX
            self.cov_XY = batch_XY
            self.cov_YY = batch_YY
        else:
            w_old = (t - 1) / t
            w_new = 1.0 / t
            self.cov_XX = w_old * self.cov_XX + w_new * batch_XX
            self.cov_XY = w_old * self.cov_XY + w_new * batch_XY
            self.cov_YY = w_old * self.cov_YY + w_new * batch_YY

    def _update_covariance_online(self, x, y, t):
        delta_x = x - self.online_mean_x
        delta_y = y - self.online_mean_y
        self.online_mean_x = self.online_mean_x + delta_x / t
        self.online_mean_y = self.online_mean_y + delta_y / t
        delta_x2 = x - self.online_mean_x
        delta_y2 = y - self.online_mean_y
        self.online_M2_XX = self.online_M2_XX + np.outer(delta_x, delta_x2)
        self.online_M2_XY = self.online_M2_XY + np.outer(delta_x, delta_y2)
        self.online_M2_YY = self.online_M2_YY + np.outer(delta_y, delta_y2)
        if t > 1:
            self.cov_XX = self.online_M2_XX / t
            self.cov_XY = self.online_M2_XY / t
            self.cov_YY = self.online_M2_YY / t
        else:
            self.cov_XX = self.online_M2_XX.copy()
            self.cov_XY = self.online_M2_XY.copy()
            self.cov_YY = self.online_M2_YY.copy()

    def _update_sigma(self, W, A):
        dif = self.Id - W
        M_jj = (np.diag(dif.T @ self.cov_XX @ dif)
                - 2.0 * np.diag(dif.T @ self.cov_XY @ A)
                + np.diag(A.T @ self.cov_YY @ A))
        return np.maximum(np.sqrt(np.maximum(M_jj, 0.0)), 1e-8)

    def minimize_batch(self, W, A, sigma, mu, n_batches, batch_size, s, lr,
                       tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        m_A, v_A = np.zeros_like(A), np.zeros_like(A)

        n_total = self.X.shape[0]
        indices = np.arange(n_total)

        for batch_idx in range(1, n_batches + 1):
            if self.online_mode:
                x = self.X[self.online_sample_idx]
                y = self.Y[self.online_sample_idx]
                self.online_sample_idx = (self.online_sample_idx + 1) % n_total
                self.online_count += 1
                self._update_covariance_online(x, y, self.online_count)
            else:
                batch_indices = np.random.choice(indices, size=batch_size, replace=False)
                X_batch = self.X[batch_indices]
                Y_batch = self.Y[batch_indices]
                X_batch = X_batch - X_batch.mean(axis=0, keepdims=True)
                Y_batch = Y_batch - Y_batch.mean(axis=0, keepdims=True)
                self._update_covariance(X_batch, Y_batch, batch_idx)

            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if batch_idx == 1 or s <= 0.9:
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

            # NV gradients: right-multiply by diag(1/σ)
            inv_sigma = 1.0 / np.maximum(sigma, 1e-8)        # (d,)
            dif = self.Id - W
            G_score_W = (self.cov_XY @ A - self.cov_XX @ dif) * inv_sigma[np.newaxis, :]
            G_score_A = (self.cov_YY @ A - self.cov_XY.T @ dif) * inv_sigma[np.newaxis, :]
            _, G_h = self._h(W, s)

            grad_W = mu * (G_score_W + self.lambda_W * np.sign(W)) + G_h
            grad_A = mu * (G_score_A + self.lambda_A * np.sign(A))

            grad_W_update, m_W, v_W = self._adam_update(grad_W, m_W, v_W,
                                                       batch_idx, beta_1, beta_2)
            grad_A_update, m_A, v_A = self._adam_update(grad_A, m_A, v_A,
                                                       batch_idx, beta_1, beta_2)
            W -= lr * grad_W_update
            A -= lr * grad_A_update

            sigma = self._update_sigma(W, A)

            if batch_idx % self.checkpoint == 0 or batch_idx == n_batches:
                obj_new = self._func(W, A, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(n_batches - batch_idx + 1)
                    break
                obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, A, sigma, True

    def fit(self, X_full, p, lambda_W=0.05, lambda_A=0.05, T=4,
            mu_init=1.0, mu_factor=0.1, s=(1.0, 0.9, 0.8, 0.7),
            batch_size=100, n_batches_warm=300, n_batches_final=600,
            lr=0.0003, checkpoint=100, beta_1=0.99, beta_2=0.999):
        self.X, self.Y = create_lagged_data(X_full, p)
        self.n, self.d = self.X.shape
        self.p = p
        self.lambda_W = lambda_W
        self.lambda_A = lambda_A
        self.checkpoint = checkpoint
        self.Id = np.eye(self.d).astype(self.dtype)

        self.X = self.X - self.X.mean(axis=0, keepdims=True)
        self.Y = self.Y - self.Y.mean(axis=0, keepdims=True)

        pd_ = p * self.d
        self.cov_XX = np.zeros((self.d, self.d), dtype=self.dtype)
        self.cov_XY = np.zeros((self.d, pd_), dtype=self.dtype)
        self.cov_YY = np.zeros((pd_, pd_), dtype=self.dtype)
        self.W_est = np.zeros((self.d, self.d), dtype=self.dtype)
        self.A_est = np.zeros((pd_, self.d), dtype=self.dtype)
        self.sig_est = np.ones(self.d, dtype=self.dtype)

        self.online_mode = (batch_size == 1)
        if self.online_mode:
            self.online_mean_x = np.zeros(self.d, dtype=self.dtype)
            self.online_mean_y = np.zeros(pd_, dtype=self.dtype)
            self.online_M2_XX = np.zeros((self.d, self.d), dtype=self.dtype)
            self.online_M2_XY = np.zeros((self.d, pd_), dtype=self.dtype)
            self.online_M2_YY = np.zeros((pd_, pd_), dtype=self.dtype)
            self.online_sample_idx = 0
            self.online_count = 0

        s_list = list(s) if isinstance(s, (list, tuple)) else [float(s)] * T
        if len(s_list) < T:
            s_list = s_list + (T - len(s_list)) * [s_list[-1]]

        mu = mu_init
        total = (T - 1) * n_batches_warm + n_batches_final
        desc = ("DyCoLiDE-NV-batch (online)" if batch_size == 1
                else f"DyCoLiDE-NV-batch (bs={batch_size})")

        with tqdm(total=int(total), desc=desc) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                n_inner = n_batches_final if i == T - 1 else n_batches_warm

                while not success:
                    W_temp, A_temp, sig_temp, success = self.minimize_batch(
                        self.W_est.copy(), self.A_est.copy(), self.sig_est.copy(),
                        mu, n_inner, batch_size, s_list[i], lr=lr_adam,
                        beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if not success:
                        lr_adam *= 0.5
                        s_list[i] += 0.1

                self.W_est = W_temp
                self.A_est = A_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.A_est, self.sig_est


# =============================================================================
# Static (full-batch) DyCoLiDE-NV — heteroscedastic reference baseline.
# =============================================================================
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


