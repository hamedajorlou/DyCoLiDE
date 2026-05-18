import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from tqdm.auto import tqdm


#===================================#
#   Online CoLiDE-EV                #
#===================================#

class colide_ev_online:
    """
    CoLiDE-EV with true online learning (one sample at a time).
    Implements online stochastic gradient descent where samples arrive sequentially.
    """

    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype

    def _score(self, W, sigma):
        """Compute score function and gradient"""
        dif = self.Id - W
        rhs = self.cov @ dif
        loss = ((0.5 * np.trace(dif.T @ rhs)) / sigma) + (0.5 * sigma * self.d)
        G_loss = -rhs / sigma
        return loss, G_loss

    def _h(self, W, s=1.0):
        """Acyclicity function (log-determinant)"""
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        """Objective function"""
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h
        return obj, score, h

    def _adam_update(self, grad, iter, beta_1, beta_2):
        """ADAM optimizer update"""
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad

    def _update_covariance_online(self, x_sample, t):
        """
        Online covariance update (Welford's algorithm for covariance).

        For a single sample x_t:
        mean_t = mean_{t-1} + (x_t - mean_{t-1}) / t
        cov_t = (t-1)/t * cov_{t-1} + (1/t) * (x_t - mean_{t-1}) * (x_t - mean_t)^T

        For centered data (mean=0):
        cov_t = (t-1)/t * cov_{t-1} + (1/t) * x_t * x_t^T
        """
        # Assuming data is already centered (subtract running mean if needed)
        outer_product = np.outer(x_sample, x_sample)

        if t == 1:
            self.cov = outer_product
        else:
            self.cov = ((t - 1) / t) * self.cov + (1 / t) * outer_product

    def minimize_online(self, W, sigma, mu, n_samples, s, lr,
                        tol=1e-6, beta_1=0.99, beta_2=0.999,
                        update_freq=1, pbar=None):
        """
        Minimize using online SGD (one sample at a time).

        Args:
            W: Initial adjacency matrix
            sigma: Initial noise scale
            mu: Penalty parameter
            n_samples: Number of samples to process
            s: Acyclicity parameter
            lr: Learning rate
            update_freq: Update W every this many samples (1 = true online)
        """
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0

        # Indices for sampling
        n_total = self.X.shape[0]
        indices = np.arange(n_total)

        # Sample counter
        sample_count = 0

        for sample_idx in range(1, n_samples + 1):
            # Get one sample at a time
            idx = np.random.choice(indices)
            x_sample = self.X[idx]

            # Center the sample (subtract global mean)
            x_sample = x_sample - self.global_mean

            # Update covariance matrix estimate
            self._update_covariance_online(x_sample, sample_idx)

            # Only update W every update_freq samples
            if sample_idx % update_freq == 0:
                sample_count += 1

                # Check feasibility
                M = sla.inv(s * self.Id - W * W) + 1e-16
                while np.any(M < -1e-6):
                    if sample_idx == 1 or s <= 0.9:
                        return W, sigma, False
                    else:
                        W += lr * grad
                        lr *= .5
                        if lr <= 1e-16:
                            return W, sigma, True
                        W -= lr * grad
                        M = sla.inv(s * self.Id - W * W) + 1e-16

                # Compute gradient using accumulated covariance
                G_score = -mu * self.cov @ (self.Id - W) / sigma
                Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

                # ADAM step
                grad = self._adam_update(Gobj, sample_count, beta_1, beta_2)
                W -= lr * grad

                # Update sigma using cumulative covariance
                dif = self.Id - W
                rhs = self.cov @ dif
                sigma = max(np.sqrt(np.trace(dif.T @ rhs) / self.d), 1e-8)

                # Check convergence periodically
                if sample_count % self.checkpoint == 0 or sample_idx == n_samples:
                    obj_new, _, _ = self._func(W, sigma, mu, s)
                    if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                        if pbar:
                            pbar.update(n_samples - sample_idx + 1)
                        break
                    obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, sigma, True

    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6],
            n_samples_warm=5000, n_samples_final=10000,
            lr=0.0003, checkpoint=500, beta_1=0.99, beta_2=0.999,
            update_freq=1,
        ):
        """
        Fit CoLiDE-EV using online SGD (one sample at a time).

        Args:
            X: Data matrix (n_samples, n_features)
            lambda1: L1 regularization parameter
            T: Number of outer iterations
            mu_init: Initial penalty parameter
            mu_factor: Factor to decrease mu
            s: Sequence of acyclicity parameters
            n_samples_warm: Number of samples for warm-up iterations
            n_samples_final: Number of samples for final iteration
            lr: Learning rate
            checkpoint: Check convergence every this many W updates
            update_freq: Update W every this many samples (1=true online, >1=mini-batch)
        """
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        # Compute and store global mean for centering
        self.global_mean = X.mean(axis=0)

        # Initialize
        self.cov = np.zeros((self.d, self.d)).astype(self.dtype)
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        self.sig_est = 1.0

        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        total_samples = (T - 1) * n_samples_warm + n_samples_final
        with tqdm(total=total_samples, desc="Online CoLiDE-EV") as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                n_samples = n_samples_final if i == T - 1 else n_samples_warm

                while success is False:
                    W_temp, sig_temp, success = self.minimize_online(
                        self.W_est.copy(), self.sig_est, mu, n_samples,
                        s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2,
                        update_freq=update_freq, pbar=pbar
                    )
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est


#===================================#
#   Online CoLiDE-NV                #
#===================================#

class colide_nv_online:
    """
    CoLiDE-NV with true online learning (one sample at a time).
    Implements online stochastic gradient descent for non-equal variance case.
    """

    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype

    def _score(self, W, sigma):
        """Compute score function and gradient"""
        dif = self.Id - W
        rhs = self.cov @ dif
        inv_SigMa = np.diag(1.0 / sigma)
        loss = (np.trace(inv_SigMa @ (dif.T @ rhs)) + np.sum(sigma)) / 2.0
        G_loss = (-rhs @ inv_SigMa)
        return loss, G_loss

    def _h(self, W, s=1.0):
        """Acyclicity function (log-determinant)"""
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        """Objective function"""
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h
        return obj, score, h

    def _adam_update(self, grad, iter, beta_1, beta_2):
        """ADAM optimizer update"""
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad

    def _update_covariance_online(self, x_sample, t):
        """Online covariance update"""
        outer_product = np.outer(x_sample, x_sample)

        if t == 1:
            self.cov = outer_product
        else:
            self.cov = ((t - 1) / t) * self.cov + (1 / t) * outer_product

    def minimize_online(self, W, sigma, mu, n_samples, s, lr,
                        tol=1e-6, beta_1=0.99, beta_2=0.999,
                        update_freq=1, pbar=None):
        """Minimize using online SGD"""
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0

        n_total = self.X.shape[0]
        indices = np.arange(n_total)
        sample_count = 0

        for sample_idx in range(1, n_samples + 1):
            idx = np.random.choice(indices)
            x_sample = self.X[idx]
            x_sample = x_sample - self.global_mean

            self._update_covariance_online(x_sample, sample_idx)

            if sample_idx % update_freq == 0:
                sample_count += 1

                M = sla.inv(s * self.Id - W * W) + 1e-16
                while np.any(M < -1e-6):
                    if sample_idx == 1 or s <= 0.9:
                        return W, sigma, False
                    else:
                        W += lr * grad
                        lr *= .5
                        if lr <= 1e-16:
                            return W, sigma, True
                        W -= lr * grad
                        M = sla.inv(s * self.Id - W * W) + 1e-16

                inv_SigMa = np.diag(1.0 / sigma)
                G_score = -mu * (self.cov @ (self.Id - W) @ inv_SigMa)
                Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

                grad = self._adam_update(Gobj, sample_count, beta_1, beta_2)
                W -= lr * grad

                dif = self.Id - W
                rhs = self.cov @ dif
                sigma = np.sqrt(np.diag(dif.T @ rhs))
                sigma = np.maximum(sigma, 1e-8)

                if sample_count % self.checkpoint == 0 or sample_idx == n_samples:
                    obj_new, _, _ = self._func(W, sigma, mu, s)
                    if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                        if pbar:
                            pbar.update(n_samples - sample_idx + 1)
                        break
                    obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, sigma, True

    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6],
            n_samples_warm=5000, n_samples_final=10000,
            lr=0.0003, checkpoint=500, beta_1=0.99, beta_2=0.999,
            update_freq=1,
        ):
        """Fit CoLiDE-NV using online SGD"""
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        self.global_mean = X.mean(axis=0)
        self.cov = np.zeros((self.d, self.d)).astype(self.dtype)
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        self.sig_est = np.ones(self.d).astype(self.dtype)

        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        total_samples = (T - 1) * n_samples_warm + n_samples_final
        with tqdm(total=total_samples, desc="Online CoLiDE-NV") as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                n_samples = n_samples_final if i == T - 1 else n_samples_warm

                while success is False:
                    W_temp, sig_temp, success = self.minimize_online(
                        self.W_est.copy(), self.sig_est.copy(), mu, n_samples,
                        s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2,
                        update_freq=update_freq, pbar=pbar
                    )
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est




class colide_nv_online_correction:
    """
    CoLiDE-NV with true online learning (one sample at a time).
    Implements online stochastic gradient descent for non-equal variance case.
    """

    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype

    def _score(self, W, sigma):
        """Compute score function and gradient"""
        dif = self.Id - W
        rhs = self.cov @ dif
        inv_SigMa = np.diag(1.0 / sigma)
        loss = (np.trace(inv_SigMa @ (dif.T @ rhs)) + np.sum(sigma)) / 2.0
        G_loss = (-rhs @ inv_SigMa)
        return loss, G_loss

    def _h(self, W, s=1.0):
        """Acyclicity function (log-determinant)"""
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        """Objective function"""
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h
        return obj, score, h

    def _adam_update(self, grad, iter, beta_1, beta_2):
        """ADAM optimizer update"""
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad

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
        self.cov = ((t - 1.0) / t) * self.cov + (1.0 / t) * np.outer(x_t, x_t)

    def minimize_online(self, W, sigma, mu, n_samples, s, lr,
                        tol=1e-6, beta_1=0.99, beta_2=0.999,
                        update_freq=1, pbar=None):
        """Minimize using online SGD"""
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0

        n_total = self.X.shape[0]
        indices = np.arange(n_total)
        sample_count = 0

        for sample_idx in range(1, n_samples + 1):
            idx = np.random.choice(indices)
            x_sample = self.X[idx]
            x_sample = x_sample - self.global_mean

            self._update_sufficient_statistics(x_sample, sample_idx)

            if sample_idx % update_freq == 0:
                sample_count += 1

                M = sla.inv(s * self.Id - W * W) + 1e-16
                while np.any(M < -1e-6):
                    if sample_idx == 1 or s <= 0.9:
                        return W, sigma, False
                    else:
                        W += lr * grad
                        lr *= .5
                        if lr <= 1e-16:
                            return W, sigma, True
                        W -= lr * grad
                        M = sla.inv(s * self.Id - W * W) + 1e-16

                inv_SigMa = np.diag(1.0 / self.sig_est)
                G_score = -mu * (self.cov @ (self.Id - W) @ inv_SigMa)
                Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

                grad = self._adam_update(Gobj, sample_count, beta_1, beta_2)
                W -= lr * grad

                # sigma is now updated in _update_sufficient_statistics
                sigma = self.sig_est

                if sample_count % self.checkpoint == 0 or sample_idx == n_samples:
                    obj_new, _, _ = self._func(W, sigma, mu, s)
                    if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                        if pbar:
                            pbar.update(n_samples - sample_idx + 1)
                        break
                    obj_prev = obj_new

            if pbar:
                pbar.update(1)

        return W, sigma, True

    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6],
            n_samples_warm=None, n_samples_final=None,
            warm_iter=None, max_iter=None,  # Aliases for compatibility
            lr=0.0003, checkpoint=500, beta_1=0.99, beta_2=0.999,
            update_freq=1, sigma_0=1e-8,
        ):
        """Fit CoLiDE-NV using online SGD"""
        # Handle parameter aliases for compatibility
        if warm_iter is not None and n_samples_warm is None:
            n_samples_warm = warm_iter
        if max_iter is not None and n_samples_final is None:
            n_samples_final = max_iter
        if n_samples_warm is None:
            n_samples_warm = 5000
        if n_samples_final is None:
            n_samples_final = 10000

        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        self.global_mean = X.mean(axis=0)
        self.cov = np.zeros((self.d, self.d)).astype(self.dtype)
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        self.sig_est = np.ones(self.d).astype(self.dtype)

        # Initialize for online sufficient statistics update (equations 36-38)
        self.e_sum = np.zeros(self.d).astype(self.dtype)  # Accumulated squared residuals
        self.sigma_0 = sigma_0  # Minimum sigma threshold

        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        total_samples = (T - 1) * n_samples_warm + n_samples_final
        with tqdm(total=total_samples, desc="Online CoLiDE-NV") as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                n_samples = n_samples_final if i == T - 1 else n_samples_warm

                while success is False:
                    W_temp, sig_temp, success = self.minimize_online(
                        self.W_est.copy(), self.sig_est.copy(), mu, n_samples,
                        s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2,
                        update_freq=update_freq, pbar=pbar
                    )
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1

                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est


