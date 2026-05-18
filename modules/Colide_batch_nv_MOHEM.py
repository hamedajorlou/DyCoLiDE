import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from tqdm.auto import tqdm


class colide_nv_batch:
    """
    Batch processing version of CoLiDE-NV (heteroscedastic case).

    Processes data in mini-batches to compute covariance estimates,
    while maintaining the same optimization approach as the standard colide_nv.
    """

    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype

    def _score(self, W, sigma):
        """Compute score loss using current covariance estimate."""
        dif = self.Id - W
        rhs = self.cov @ dif
        inv_SigMa = np.diag(1.0/(sigma))
        loss = (np.trace(inv_SigMa @ (dif.T @ rhs)) + np.sum(sigma)) / (2.0)
        G_loss = (-rhs @ inv_SigMa)
        return loss, G_loss

    def _h(self, W, s=1.0):
        """DAG constraint and its gradient."""
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        """Objective function."""
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h
        return obj, score, h

    def _adam_update(self, grad, iter, beta_1, beta_2):
        """Adam optimizer update."""
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad

    def _compute_batch_covariance(self, X_batch):
        """Compute covariance for a batch."""
        n_batch = X_batch.shape[0]
        return X_batch.T @ X_batch / float(n_batch)

    def _update_covariance_estimate(self, X, batch_size, update_freq):
        """
        Update covariance estimate using batch processing.

        Parameters
        ----------
        X : array, shape (n, d)
            Data matrix
        batch_size : int
            Size of mini-batches
        update_freq : int
            How often to sample new batches (in iterations)
        """
        n = X.shape[0]
        n_batches = n // batch_size

        if n_batches == 0:
            # If batch_size > n, use full data
            self.cov = X.T @ X / float(n)
        else:
            # Sample random batches and average their covariances
            cov_sum = np.zeros((self.d, self.d))

            for _ in range(min(n_batches, 10)):  # Use up to 10 batches for estimate
                indices = np.random.choice(n, batch_size, replace=False)
                X_batch = X[indices]
                cov_sum += self._compute_batch_covariance(X_batch)

            self.cov = cov_sum / min(n_batches, 10)

    def minimize(self, W, sigma, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999,
                 pbar=None, batch_size=None, X=None, update_freq=100):
        """
        Minimize objective using Adam optimizer with batch covariance updates.

        Parameters
        ----------
        W : array
            Current weight matrix
        sigma : array
            Current variance estimates
        mu : float
            Penalty parameter
        max_iter : int
            Maximum iterations
        s : float
            DAG constraint parameter
        lr : float
            Learning rate
        tol : float
            Convergence tolerance
        beta_1, beta_2 : float
            Adam parameters
        pbar : tqdm progress bar
        batch_size : int or None
            Batch size for covariance updates (None = full data)
        X : array or None
            Data matrix (required if batch_size is not None)
        update_freq : int
            Frequency of batch covariance updates
        """
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0

        for iter in range(1, max_iter+1):
            # Update covariance estimate periodically using batches
            if batch_size is not None and X is not None and iter % update_freq == 0:
                self._update_covariance_estimate(X, batch_size, update_freq)

            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, sigma, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, sigma, True
                    W -= lr * grad
                    dif = self.Id - W
                    rhs = self.cov @ dif
                    sigma = np.sqrt(np.diag(dif.T @ rhs))
                    M = sla.inv(s * self.Id - W * W) + 1e-16

            inv_SigMa = np.diag(1.0/(sigma))
            G_score = -mu * (self.cov @ (self.Id - W) @ inv_SigMa)
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad

            dif = self.Id - W
            rhs = self.cov @ dif
            sigma = np.sqrt(np.diag(dif.T @ rhs))

            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, _, _ = self._func(W, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, sigma, True

    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6],
            warm_iter=3e4, max_iter=6e4, lr=0.0003,
            checkpoint=1000, beta_1=0.99, beta_2=0.999, w_init=None,
            batch_size=None, update_freq=100,
        ):
        """
        Fit CoLiDE-NV with batch processing.

        Parameters
        ----------
        X : array, shape (n, d)
            Data matrix
        lambda1 : float
            L1 regularization parameter
        T : int
            Number of stages
        mu_init : float
            Initial penalty parameter
        mu_factor : float
            Penalty reduction factor
        s : list or float
            DAG constraint parameter schedule
        warm_iter : int
            Warm-up iterations per stage
        max_iter : int
            Maximum iterations in final stage
        lr : float
            Learning rate
        checkpoint : int
            Checkpoint frequency
        beta_1, beta_2 : float
            Adam parameters
        w_init : array or None
            Initial weight matrix
        batch_size : int or None
            Batch size for covariance estimation
            If None, uses full data (standard CoLiDE-NV)
            If int, uses batch processing
        update_freq : int
            How often to update batch covariance (in iterations)

        Returns
        -------
        W_est : array
            Estimated weight matrix
        sig_est : array
            Estimated node-specific variances
        """
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.X -= X.mean(axis=0, keepdims=True)

        # Initial covariance estimate
        if batch_size is None:
            # Full data
            self.cov = X.T @ X / float(self.n)
        else:
            # Batch estimate
            self._update_covariance_estimate(X, batch_size, update_freq)

        # Initialize W and sigma
        if w_init is None:
            self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
            self.sig_est = (np.linalg.norm(self.X, axis=0) / np.sqrt(self.n)).astype(self.dtype)
        else:
            self.W_est = np.copy(w_init).astype(self.dtype)
            self.sig_est = (np.linalg.norm(self.X @ (self.Id - w_init), axis=0) / np.sqrt(self.n)).astype(self.dtype)

        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, sig_temp, success = self.minimize(
                        self.W_est.copy(), self.sig_est.copy(), mu, inner_iters, s[i],
                        lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar,
                        batch_size=batch_size, X=self.X, update_freq=update_freq
                    )
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est


class colide_nv_online:
    """
    CoLiDE-NV with mini-batch stochastic gradient descent.
    Based on Appendix B of the CoLiDE paper.

    Supports:
    - batch_size=1: True online learning (sequential processing with Welford's algorithm)
    - batch_size>1: Mini-batch SGD (random sampling)

    For online mode (batch_size=1), implements equations (36-38):
    - ϵ_{j,t} = (x_t)_j - (Ŵ_{t-1}^T x_t)_j)^2
    - e_{j,t} = e_{j,t-1} + ϵ_{j,t}
    - σ̂_{j,t} = max(√(1/t e_{j,t}), σ_0)
    """

    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
        self.online_mode = False  # Set True when batch_size=1

    def _score(self, W, sigma):
        """Compute score loss using current covariance estimate."""
        dif = self.Id - W
        rhs = self.cov @ dif
        inv_SigMa = np.diag(1.0/(sigma + 1e-8))
        loss = (np.trace(inv_SigMa @ (dif.T @ rhs)) + np.sum(sigma)) / (2.0)
        G_loss = (-rhs @ inv_SigMa)
        return loss, G_loss

    def _h(self, W, s=1.0):
        """DAG constraint and its gradient."""
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        """Objective function."""
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h
        return obj, score, h

    def _adam_update(self, grad, iter, beta_1, beta_2):
        """Adam optimizer update."""
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad

    def _update_covariance_online(self, x, t):
        """
        Update mean and covariance for a single sample using Welford's algorithm.
        More numerically stable for online (batch_size=1) mode.
        """
        delta = x - self.online_mean
        self.online_mean = self.online_mean + delta / t
        delta2 = x - self.online_mean
        # Update sum of squared deviations
        self.online_M2 = self.online_M2 + np.outer(delta, delta2)
        # Covariance = M2 / t
        if t > 1:
            self.cov = self.online_M2 / t
        else:
            self.cov = self.online_M2

    def _update_covariance(self, X_batch, t):
        """
        Update sample covariance using online algorithm.
        cov(X_t) = (t-1)/t * cov(X_{t-1}) + 1/t * batch_cov
        """
        n_batch = X_batch.shape[0]
        batch_cov = X_batch.T @ X_batch / n_batch

        if t == 1:
            self.cov = batch_cov
        else:
            self.cov = ((t - 1) / t) * self.cov + (1 / t) * batch_cov

    def _update_sufficient_statistics(self, x_t, t):
        """
        Update sufficient statistics online based on equations (36-38).

        Parameters
        ----------
        x_t : array, shape (d,)
            Current sample
        t : int
            Current time step (1-indexed)
        """
        # Equation (36): ϵ_{j,t} = ((x_t)_j - (Ŵ_{t-1}^T x_t)_j)^2
        prediction = self.W_est.T @ x_t  # Shape (d,)
        epsilon_t = (x_t - prediction) ** 2  # Squared residuals

        # Equation (37): e_{j,t} = e_{j,t-1} + ϵ_{j,t}
        self.e_sum += epsilon_t

        # Equation (38): σ̂_{j,t} = max(√(1/t e_{j,t}), σ_0)
        sigma_new = np.sqrt(self.e_sum / float(t))
        self.sig_est = np.maximum(sigma_new, self.sigma_0)

        # Update covariance estimate online: C_t = (t-1)/t * C_{t-1} + 1/t * x_t x_t^T
        # self.cov = ((t - 1.0) / t) * self.cov + (1.0 / t) * np.outer(x_t, x_t)

    def minimize_batch(self, W, sigma, mu, n_batches, batch_size, s, lr,
                       tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        """
        Minimize using mini-batch stochastic gradient descent.

        Args:
            W: Initial adjacency matrix
            sigma: Initial noise variances per node
            mu: Penalty parameter
            n_batches: Number of batches/samples to process
            batch_size: Size of each mini-batch (1 for online mode)
            s: Acyclicity parameter
            lr: Learning rate
        """
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0

        n_total = self.X.shape[0]
        indices = np.arange(n_total)

        for batch_idx in range(1, n_batches + 1):
            if self.online_mode:
                # Online mode: process samples sequentially one by one
                x = self.X[self.online_sample_idx]
                self.online_sample_idx = (self.online_sample_idx + 1) % n_total
                self.online_count += 1

                # Update covariance using Welford's algorithm (persistent across stages)
                self._update_covariance_online(x, self.online_count)
                self.cov_batch = self.cov

                # Update sufficient statistics for sigma
                self._update_sufficient_statistics(x, self.online_count)
            else:
                # Mini-batch mode: sample random batch
                batch_indices = np.random.choice(indices, size=batch_size, replace=False)
                X_batch = self.X[batch_indices]

                # Center the batch
                X_batch = X_batch - X_batch.mean(axis=0, keepdims=True)

                # Update covariance matrix estimate
                self._update_covariance(X_batch, batch_idx)
                self.cov_batch = self.cov

            # Check feasibility
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if batch_idx == 1 or s <= 0.9:
                    return W, sigma, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, sigma, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16

            # Compute gradient
            inv_SigMa = np.diag(1.0 / (sigma + 1e-8))
            G_score = -mu * (self.cov_batch @ (self.Id - W) @ inv_SigMa)
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

            # ADAM step
            grad = self._adam_update(Gobj, batch_idx, beta_1, beta_2)
            W -= lr * grad

            # Update sigma
            if self.online_mode:
                # Use sigma from sufficient statistics
                sigma = self.sig_est.copy()
            else:
                # Update sigma using the cumulative covariance estimate
                # For each node: sigma_j^2 = [(I-W)^T cov(X) (I-W)]_jj
                dif = self.Id - W
                rhs = self.cov @ dif
                sigma = np.sqrt(np.maximum(np.diag(dif.T @ rhs), 1e-8))

            # Check convergence periodically
            if batch_idx % self.checkpoint == 0 or batch_idx == n_batches:
                obj_new, _, _ = self._func(W, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(n_batches - batch_idx + 1)
                    break
                obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, sigma, True

    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6],
            batch_size=100, n_batches_warm=300, n_batches_final=600,
            lr=0.0003, checkpoint=100, beta_1=0.99, beta_2=0.999,
            sigma_0=0.1,
        ):
        """
        Fit CoLiDE-NV using mini-batch SGD.

        Parameters
        ----------
        X : array, shape (n, d)
            Data matrix
        lambda1 : float
            L1 regularization parameter
        T : int
            Number of outer iterations (stages)
        mu_init : float
            Initial penalty parameter
        mu_factor : float
            Factor to decrease mu each stage
        s : list or float
            Sequence of acyclicity parameters
        batch_size : int
            Size of mini-batches (1 for online mode)
        n_batches_warm : int
            Number of batches for warm-up iterations
        n_batches_final : int
            Number of batches for final iteration
        lr : float
            Learning rate
        checkpoint : int
            Check convergence every this many batches
        sigma_0 : float
            Minimum variance threshold (from equation 38)
        """
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.sigma_0 = sigma_0

        # Initialize
        self.cov = np.zeros((self.d, self.d)).astype(self.dtype)
        self.cov_batch = np.zeros((self.d, self.d)).astype(self.dtype)
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        self.sig_est = np.ones(self.d).astype(self.dtype) * sigma_0

        # Initialize sufficient statistics for sigma (for online mode)
        self.e_sum = np.zeros(self.d).astype(self.dtype)  # e_{j,0} = 0 for all j

        # Initialize online statistics once (persists across stages)
        self.online_mode = (batch_size == 1)
        if self.online_mode:
            self.online_mean = np.zeros(self.d)
            self.online_M2 = np.zeros((self.d, self.d))
            self.online_sample_idx = 0  # Track position across stages
            self.online_count = 0       # Total samples processed

        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        total_batches = (T - 1) * n_batches_warm + n_batches_final
        mode_desc = "Online CoLiDE-NV" if batch_size == 1 else "Batch CoLiDE-NV"
        with tqdm(total=total_batches, desc=mode_desc) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                n_batches = n_batches_final if i == T - 1 else n_batches_warm

                while success is False:
                    W_temp, sig_temp, success = self.minimize_batch(
                        self.W_est.copy(), self.sig_est.copy(), mu, n_batches,
                        batch_size, s[i], lr=lr_adam, beta_1=beta_1,
                        beta_2=beta_2, pbar=pbar
                    )
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est
