"""
DAGMA-SVAR: DAGMA for Structural Vector Autoregression models.

Extends DAGMA to handle time series data with both instantaneous and lagged effects.

Model: X_t = X_t @ W + X_{t-1} @ A + noise

Where:
- W: (d x d) instantaneous effects matrix (must be a DAG - acyclicity constraint applied)
- A: (d x d) lagged effects matrix (no acyclicity constraint - temporal ordering is causal)

When A = 0, this reduces to the standard SEM model: X = X @ W + noise

Reference: Based on DAGMA (Bello et al., 2022) extended for SVAR models.
"""

import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from tqdm.auto import tqdm
import typing


__all__ = ["DagmaSVAR"]


class DagmaSVAR:
    """
    DAGMA for Structural Vector Autoregression (SVAR) models.

    Learns both instantaneous effects (W) and lagged effects (A) from time series data.
    The acyclicity constraint is only applied to W, not A.

    When no lagged data is provided (or A is constrained to zero), this reduces
    to the standard DAGMA for SEM.
    """

    def __init__(self, loss_type: str = 'l2', verbose: bool = False,
                 dtype: type = np.float64) -> None:
        """
        Parameters
        ----------
        loss_type : str
            One of ["l2", "logistic"]. For continuous data use "l2".
        verbose : bool
            If True, print optimization progress.
        dtype : type
            Float precision. Default: np.float64
        """
        super().__init__()
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

    def _score(self, W: np.ndarray, A: np.ndarray) -> typing.Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate value and gradient of the score function for SVAR model.

        Model: X_t = X_t @ W + X_{t-1} @ A + noise
        Residual: R = X_t - X_t @ W - X_{t-1} @ A = X_t @ (I - W) - X_{t-1} @ A

        For l2 loss: 0.5 * ||R||_F^2 / n

        Parameters
        ----------
        W : np.ndarray
            (d, d) instantaneous effects matrix
        A : np.ndarray
            (d, d) lagged effects matrix (None for SEM mode)

        Returns
        -------
        loss : float
            Score function value
        G_W : np.ndarray
            Gradient w.r.t. W
        G_A : np.ndarray
            Gradient w.r.t. A (zeros if A is None or SEM mode)
        """
        if self.loss_type == 'l2':
            if self.svar_mode:
                # SVAR mode: X_t = X_t @ W + X_{t-1} @ A + noise
                # Residual: R = X_t - X_t @ W - X_{t-1} @ A
                R = self.X_t - self.X_t @ W - self.X_lag @ A
                loss = 0.5 * np.sum(R ** 2) / self.n

                # Gradients
                G_W = -self.X_t.T @ R / self.n
                G_A = -self.X_lag.T @ R / self.n
            else:
                # SEM mode: X = X @ W + noise (same as original DAGMA)
                dif = self.Id - W
                rhs = self.cov @ dif
                loss = 0.5 * np.trace(dif.T @ rhs)
                G_W = -rhs
                G_A = np.zeros_like(W)

        elif self.loss_type == 'logistic':
            if self.svar_mode:
                # Logistic loss for SVAR
                linear = self.X_t @ W + self.X_lag @ A
                from scipy.special import expit as sigmoid
                loss = 1.0 / self.n * (np.logaddexp(0, linear) - self.X_t * linear).sum()
                sig_linear = sigmoid(linear)
                G_W = (1.0 / self.n * self.X_t.T) @ sig_linear - self.cov_t
                G_A = (1.0 / self.n * self.X_lag.T) @ sig_linear - self.cov_lag
            else:
                # SEM logistic (original DAGMA)
                from scipy.special import expit as sigmoid
                R = self.X @ W
                loss = 1.0 / self.n * (np.logaddexp(0, R) - self.X * R).sum()
                G_W = (1.0 / self.n * self.X.T) @ sigmoid(R) - self.cov
                G_A = np.zeros_like(W)

        return loss, G_W, G_A

    def _h(self, W: np.ndarray, s: float = 1.0) -> typing.Tuple[float, np.ndarray]:
        """
        Evaluate value and gradient of the logdet acyclicity constraint.

        h(W) = -log|sI - W◦W| + d*log(s)

        Only applied to W (instantaneous effects), not to A (lagged effects).

        Parameters
        ----------
        W : np.ndarray
            (d, d) adjacency matrix
        s : float
            Controls the domain of M-matrices. Default: 1.0

        Returns
        -------
        h : float
            Acyclicity constraint value
        G_h : np.ndarray
            Gradient of h w.r.t. W
        """
        M = s * self.Id - W * W
        h = -la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W: np.ndarray, A: np.ndarray, mu: float,
              s: float = 1.0) -> typing.Tuple[float, float, float]:
        """
        Evaluate value of the penalized objective function.

        obj = mu * (score + lambda1 * ||W||_1 + lambda2 * ||A||_1) + h(W)

        Note: Acyclicity constraint h(W) only applies to W, not A.

        Parameters
        ----------
        W : np.ndarray
            Instantaneous effects matrix
        A : np.ndarray
            Lagged effects matrix
        mu : float
            Weight of the score function
        s : float
            Controls the domain of M-matrices

        Returns
        -------
        obj : float
            Objective value
        score : float
            Score function value
        h : float
            Acyclicity constraint value
        """
        score, _, _ = self._score(W, A)
        h, _ = self._h(W, s)

        # L1 penalty on both W and A
        l1_penalty = self.lambda1 * np.abs(W).sum()
        if self.svar_mode:
            l1_penalty += self.lambda2 * np.abs(A).sum()

        obj = mu * (score + l1_penalty) + h
        return obj, score, h

    def _adam_update(self, grad: np.ndarray, iter: int,
                     m: np.ndarray, v: np.ndarray,
                     beta_1: float, beta_2: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs one update of Adam optimizer.

        Parameters
        ----------
        grad : np.ndarray
            Current gradient
        iter : int
            Current iteration number
        m : np.ndarray
            First moment estimate
        v : np.ndarray
            Second moment estimate
        beta_1 : float
            Adam hyperparameter
        beta_2 : float
            Adam hyperparameter

        Returns
        -------
        update : np.ndarray
            The Adam update direction
        m : np.ndarray
            Updated first moment
        v : np.ndarray
            Updated second moment
        """
        m = m * beta_1 + (1 - beta_1) * grad
        v = v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = m / (1 - beta_1 ** iter)
        v_hat = v / (1 - beta_2 ** iter)
        update = m_hat / (np.sqrt(v_hat) + 1e-8)
        return update, m, v

    def minimize(self, W: np.ndarray, A: np.ndarray, mu: float,
                 max_iter: int, s: float, lr: float,
                 tol: float = 1e-6, beta_1: float = 0.99,
                 beta_2: float = 0.999,
                 pbar: typing.Optional[tqdm] = None) -> typing.Tuple[np.ndarray, np.ndarray, bool]:
        """
        Solves the optimization problem using (sub)gradient descent.

        Parameters
        ----------
        W : np.ndarray
            Initial instantaneous effects matrix
        A : np.ndarray
            Initial lagged effects matrix
        mu : float
            Weights the score function
        max_iter : int
            Maximum number of iterations
        s : float
            Controls the domain of M-matrices
        lr : float
            Learning rate
        tol : float
            Tolerance for convergence
        beta_1 : float
            Adam hyperparameter
        beta_2 : float
            Adam hyperparameter
        pbar : tqdm
            Progress bar

        Returns
        -------
        W : np.ndarray
            Optimized instantaneous effects
        A : np.ndarray
            Optimized lagged effects
        success : bool
            Whether optimization succeeded
        """
        obj_prev = 1e16

        # Adam states for W
        m_W, v_W = np.zeros_like(W), np.zeros_like(W)
        # Adam states for A
        m_A, v_A = np.zeros_like(A), np.zeros_like(A)

        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr:{lr} -- s:{s} -- '
                    f'lambda1:{self.lambda1} -- lambda2:{self.lambda2} for {max_iter} iterations')

        for iter in range(1, max_iter + 1):
            # Check M-matrix condition for acyclicity
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0):
                if iter == 1 or s <= 0.9:
                    self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                    return W, A, False
                else:
                    W += lr * grad_W
                    lr *= 0.5
                    if lr <= 1e-16:
                        return W, A, True
                    W -= lr * grad_W
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f'Learning rate decreased to lr: {lr}')

            # Compute gradients
            _, G_score_W, G_score_A = self._score(W, A)
            _, G_h = self._h(W, s)

            # Gradient for W: score + L1 + acyclicity
            G_W = mu * G_score_W + mu * self.lambda1 * np.sign(W) + G_h

            # Gradient for A: score + L1 (no acyclicity constraint)
            if self.svar_mode:
                G_A = mu * G_score_A + mu * self.lambda2 * np.sign(A)
            else:
                G_A = np.zeros_like(A)

            # Adam updates
            grad_W, m_W, v_W = self._adam_update(G_W, iter, m_W, v_W, beta_1, beta_2)
            W -= lr * grad_W

            if self.svar_mode:
                grad_A, m_A, v_A = self._adam_update(G_A, iter, m_A, v_A, beta_1, beta_2)
                A -= lr * grad_A

            # Check convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, A, mu, s)
                self.vprint(f'\nInner iteration {iter}')
                self.vprint(f'\th(W): {h:.4e}')
                self.vprint(f'\tscore: {score:.4e}')
                self.vprint(f'\tobj: {obj_new:.4e}')

                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, A, True

    def fit(self, X: np.ndarray,
            X_lag: typing.Optional[np.ndarray] = None,
            lambda1: float = 0.03,
            lambda2: float = 0.03,
            w_threshold: float = 0.3,
            T: int = 5,
            mu_init: float = 1.0,
            mu_factor: float = 0.1,
            s: typing.Union[typing.List[float], float] = [1.0, 0.9, 0.8, 0.7, 0.6],
            warm_iter: int = 3e4,
            max_iter: int = 6e4,
            lr: float = 0.0003,
            checkpoint: int = 1000,
            beta_1: float = 0.99,
            beta_2: float = 0.999,
            ) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
        """
        Runs the DAGMA-SVAR algorithm.

        For SEM (static) data: Pass only X, leave X_lag as None.
        For SVAR (time series) data: Pass X as X_t (current) and X_lag as X_{t-1} (lagged).

        Parameters
        ----------
        X : np.ndarray
            For SEM: (n, d) data matrix
            For SVAR: (n-1, d) current time observations X_t
        X_lag : np.ndarray, optional
            For SVAR: (n-1, d) lagged observations X_{t-1}
            If None, runs in SEM mode (standard DAGMA)
        lambda1 : float
            L1 penalty coefficient for W. Default: 0.03
        lambda2 : float
            L1 penalty coefficient for A. Default: 0.03 (only used in SVAR mode)
        w_threshold : float
            Threshold for removing weak edges. Default: 0.3
        T : int
            Number of DAGMA iterations. Default: 5
        mu_init : float
            Initial value of mu. Default: 1.0
        mu_factor : float
            Decay factor for mu. Default: 0.1
        s : list or float
            Controls the domain of M-matrices. Default: [1.0, 0.9, 0.8, 0.7, 0.6]
        warm_iter : int
            Number of iterations for t < T. Default: 3e4
        max_iter : int
            Number of iterations for t = T. Default: 6e4
        lr : float
            Learning rate. Default: 0.0003
        checkpoint : int
            Print frequency if verbose. Default: 1000
        beta_1 : float
            Adam hyperparameter. Default: 0.99
        beta_2 : float
            Adam hyperparameter. Default: 0.999

        Returns
        -------
        W_est : np.ndarray
            Estimated instantaneous effects (d, d)
        A_est : np.ndarray or None
            Estimated lagged effects (d, d) if SVAR mode, else None
        """
        # Determine mode: SVAR or SEM
        self.svar_mode = X_lag is not None

        if self.svar_mode:
            # SVAR mode
            self.X_t = X.astype(self.dtype)
            self.X_lag = X_lag.astype(self.dtype)
            self.n, self.d = self.X_t.shape

            assert self.X_t.shape == self.X_lag.shape, \
                "X and X_lag must have the same shape"

            # Center data
            if self.loss_type == 'l2':
                self.X_t = self.X_t - self.X_t.mean(axis=0, keepdims=True)
                self.X_lag = self.X_lag - self.X_lag.mean(axis=0, keepdims=True)

            # Covariances for logistic loss
            self.cov_t = self.X_t.T @ self.X_t / float(self.n)
            self.cov_lag = self.X_lag.T @ self.X_lag / float(self.n)

            self.vprint(f"SVAR mode: n={self.n}, d={self.d}")
        else:
            # SEM mode (standard DAGMA)
            self.X = X.astype(self.dtype)
            self.n, self.d = self.X.shape

            if self.loss_type == 'l2':
                self.X = self.X - self.X.mean(axis=0, keepdims=True)

            self.cov = self.X.T @ self.X / float(self.n)

            self.vprint(f"SEM mode: n={self.n}, d={self.d}")

        # Store parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.checkpoint = checkpoint
        self.Id = np.eye(self.d).astype(self.dtype)

        # Initialize W and A at zeros
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        self.A_est = np.zeros((self.d, self.d)).astype(self.dtype)

        # Setup s schedule
        mu = mu_init
        if isinstance(s, list):
            if len(s) < T:
                self.vprint(f"Length of s is {len(s)}, using last value for t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif isinstance(s, (int, float)):
            s = T * [s]
        else:
            raise ValueError("s should be a list, int, or float")

        # Run DAGMA
        desc = "DAGMA-SVAR" if self.svar_mode else "DAGMA"
        with tqdm(total=(T - 1) * int(warm_iter) + int(max_iter), desc=desc) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nIteration -- {i + 1}:')
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)

                while not success:
                    W_temp, A_temp, success = self.minimize(
                        self.W_est.copy(), self.A_est.copy(),
                        mu, inner_iters, s[i], lr=lr_adam,
                        beta_1=beta_1, beta_2=beta_2, pbar=pbar
                    )
                    if not success:
                        self.vprint('Retrying with larger s')
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.A_est = A_temp
                mu *= mu_factor

        # Store final values
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _, _ = self._score(self.W_est, self.A_est)

        # Threshold weak edges
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        if self.svar_mode:
            self.A_est[np.abs(self.A_est) < w_threshold] = 0
            return self.W_est, self.A_est
        else:
            return self.W_est, None


def prepare_svar_data(X_timeseries: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Prepare time series data for SVAR model.

    Parameters
    ----------
    X_timeseries : np.ndarray
        (T, d) time series data with T time points and d variables

    Returns
    -------
    X_t : np.ndarray
        (T-1, d) current time observations
    X_lag : np.ndarray
        (T-1, d) lagged observations
    """
    X_t = X_timeseries[1:, :]      # X_t for t = 1, ..., T-1
    X_lag = X_timeseries[:-1, :]   # X_{t-1} for t = 1, ..., T-1
    return X_t, X_lag


def test_sem():
    """Test DAGMA-SVAR in SEM mode (should match original DAGMA)."""
    from Utils import simulate_sem, to_bin, count_accuracy

    print("=" * 60)
    print("Testing DAGMA-SVAR in SEM mode")
    print("=" * 60)

    # Generate SEM data
    X, W_true, _ = simulate_sem(
        n_nodes=20, n_samples=1000, edges=40,
        graph_type='er', edge_type='weighted',
        var_type='ev', noise='normal', var=1.0, seed=42
    )

    W_true_bin = to_bin(W_true, thr=0.0)
    print(f"True edges: {int(np.sum(W_true_bin))}")

    # Run DAGMA-SVAR in SEM mode
    model = DagmaSVAR(loss_type='l2', verbose=False)
    W_est, A_est = model.fit(
        X,
        X_lag=None,  # SEM mode
        lambda1=0.02,
        w_threshold=0.3,
        T=5,
        warm_iter=5000,
        max_iter=8000,
    )

    assert A_est is None, "A should be None in SEM mode"

    W_est_bin = to_bin(W_est, thr=0.0)
    shd, tpr, fdr = count_accuracy(W_true_bin, W_est_bin)

    print(f"\nSEM mode results:")
    print(f"  TPR: {tpr:.4f}")
    print(f"  FDR: {fdr:.4f}")
    print(f"  SHD: {shd}")
    print(f"  Estimated edges: {int(np.sum(W_est_bin))}")

    return W_est


def test_svar():
    """Test DAGMA-SVAR in SVAR mode with synthetic time series data."""
    from Utils import to_bin

    print("\n" + "=" * 60)
    print("Testing DAGMA-SVAR in SVAR mode")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic SVAR data
    d = 10  # number of variables
    T = 1000  # number of time points (more data helps)

    # True W (instantaneous, must be DAG - lower triangular)
    # Use smaller coefficients for stability
    W_true = np.zeros((d, d))
    for i in range(1, d):
        for j in range(i):
            if np.random.rand() < 0.3:
                W_true[i, j] = np.random.uniform(0.3, 0.6) * np.random.choice([-1, 1])

    # True A (lagged, no DAG constraint)
    # Keep spectral radius < 1 for stability
    A_true = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if np.random.rand() < 0.2:
                A_true[i, j] = np.random.uniform(0.2, 0.4) * np.random.choice([-1, 1])

    # Scale A to ensure stability (spectral radius < 1)
    spec_radius = np.max(np.abs(np.linalg.eigvals(A_true)))
    if spec_radius >= 0.95:
        A_true = A_true * 0.9 / spec_radius

    print(f"True W edges: {np.sum(np.abs(W_true) > 0)}")
    print(f"True A edges: {np.sum(np.abs(A_true) > 0)}")
    print(f"A spectral radius: {np.max(np.abs(np.linalg.eigvals(A_true))):.3f}")

    # Generate time series: X_t = X_t @ W + X_{t-1} @ A + noise
    # Rearranged: X_t @ (I - W) = X_{t-1} @ A + noise
    # X_t = (X_{t-1} @ A + noise) @ inv(I - W)
    I_minus_W_inv = np.linalg.inv(np.eye(d) - W_true)

    X = np.zeros((T, d))
    X[0] = np.random.randn(d)  # Initial state

    for t in range(1, T):
        noise = np.random.randn(d) * 1.0
        X[t] = (X[t-1] @ A_true + noise) @ I_minus_W_inv

    # Prepare data
    X_t, X_lag = prepare_svar_data(X)
    print(f"Data shape: X_t={X_t.shape}, X_lag={X_lag.shape}")
    print(f"Data scale: X_t std={np.std(X_t):.3f}")

    # Run DAGMA-SVAR
    model = DagmaSVAR(loss_type='l2', verbose=False)
    W_est, A_est = model.fit(
        X_t, X_lag,
        lambda1=0.02,  # smaller regularization
        lambda2=0.02,
        w_threshold=0.15,  # smaller threshold
        T=5,
        warm_iter=5000,
        max_iter=10000,
        lr=0.001,  # slightly higher learning rate
    )

    # Debug: print scale of estimates before thresholding
    print(f"\nW_est max abs value: {np.max(np.abs(model.W_est)):.4f}")
    print(f"A_est max abs value: {np.max(np.abs(model.A_est)):.4f}")

    # Evaluate W
    W_true_bin = to_bin(W_true, thr=0.0)
    W_est_bin = to_bin(W_est, thr=0.0)

    # Simple accuracy metrics
    tp_W = np.sum((W_true_bin == 1) & (W_est_bin == 1))
    fp_W = np.sum((W_true_bin == 0) & (W_est_bin == 1))
    fn_W = np.sum((W_true_bin == 1) & (W_est_bin == 0))

    tpr_W = tp_W / max(tp_W + fn_W, 1)
    fdr_W = fp_W / max(tp_W + fp_W, 1)

    print(f"\nW (instantaneous) results:")
    print(f"  TPR: {tpr_W:.4f}")
    print(f"  FDR: {fdr_W:.4f}")
    print(f"  True edges: {int(np.sum(W_true_bin))}")
    print(f"  Estimated edges: {int(np.sum(W_est_bin))}")

    # Evaluate A
    A_true_bin = to_bin(A_true, thr=0.0)
    A_est_bin = to_bin(A_est, thr=0.0)

    tp_A = np.sum((A_true_bin == 1) & (A_est_bin == 1))
    fp_A = np.sum((A_true_bin == 0) & (A_est_bin == 1))
    fn_A = np.sum((A_true_bin == 1) & (A_est_bin == 0))

    tpr_A = tp_A / max(tp_A + fn_A, 1)
    fdr_A = fp_A / max(tp_A + fp_A, 1)

    print(f"\nA (lagged) results:")
    print(f"  TPR: {tpr_A:.4f}")
    print(f"  FDR: {fdr_A:.4f}")
    print(f"  True edges: {int(np.sum(A_true_bin))}")
    print(f"  Estimated edges: {int(np.sum(A_est_bin))}")

    return W_est, A_est


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/hamedajorlou/Documents/Dynotears')

    # Test SEM mode
    test_sem()

    # Test SVAR mode
    test_svar()
