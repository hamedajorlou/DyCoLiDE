import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from tqdm.auto import tqdm


#===================================#
#   Equal variance  CoLiDE-EV       #
#===================================#

class colide_ev:
    
    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
    
    def _score(self, W, sigma):
        dif = self.Id - W 
        rhs = self.cov @ dif
        loss = ((0.5 * np.trace(dif.T @ rhs)) / sigma) + (0.5 * sigma * self.d)
        G_loss = -rhs / sigma
        return loss, G_loss

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, W, sigma, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        
        for iter in range(1, max_iter+1):
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
                    sigma = np.sqrt(np.trace(dif.T @ rhs) / (self.d))
                    M = sla.inv(s * self.Id - W * W) + 1e-16
            
            G_score = -mu * self.cov @ (self.Id - W) / sigma
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad

            dif = self.Id - W 
            rhs = self.cov @ dif
            sigma = np.sqrt(np.trace(dif.T @ rhs) / (self.d))
            
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
            checkpoint=1000, beta_1=0.99, beta_2=0.999,
        ):
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
        self.sig_est = np.min(np.linalg.norm(self.X, axis=0) / np.sqrt(self.n)).astype(self.dtype)
        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        with tqdm(total=(T-1)*warm_iter+max_iter, desc="CoLiDE-EV") as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, sig_temp, success = self.minimize(self.W_est.copy(), self.sig_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est

#===================================#
#   Non-equal variance  CoLiDE-NV   #
#===================================#

class colide_nv:
    
    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
            
    def _score(self, W, sigma):
        dif = self.Id - W 
        rhs = self.cov @ dif
        inv_SigMa = np.diag(1.0/(sigma))
        loss = (np.trace(inv_SigMa @ (dif.T @ rhs)) + np.sum(sigma)) / (2.0)
        G_loss = (-rhs @ inv_SigMa)
        return loss, G_loss

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, W, sigma, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        
        for iter in range(1, max_iter+1):
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
        ):
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)
        if w_init is None:    
            self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
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

        with tqdm(total=(T-1)*warm_iter+max_iter, desc="CoLiDE-NV") as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, sig_temp, success = self.minimize(self.W_est.copy(), self.sig_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est
    



#===================================#
#   Equal variance  CoLiDE-EV       #
#===================================#


class colide_ev_batch:
    """
    CoLiDE-EV with mini-batch stochastic gradient descent.
    Based on Appendix B of the CoLiDE paper.

    Supports:
    - batch_size=1: True online learning (sequential processing)
    - batch_size>1: Mini-batch SGD (random sampling)
    """

    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
        self.online_mode = False  # Set True when batch_size=1

    def _score(self, W, sigma):
        """Compute score function and gradient for current batch"""
        dif = self.Id - W
        rhs = self.cov_batch @ dif
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
        # Covariance = M2 / (t - 1) for unbiased, or M2 / t for biased
        if t > 1:
            self.cov = self.online_M2 / t
        else:
            self.cov = self.online_M2

    def _update_covariance(self, X_batch, t):
        """
        Update sample covariance using online algorithm.
        cov(X_t) = (t-1)/t * cov(X_{t-1}) + 1/(t*n_batch) * X_batch^T X_batch
        """
        n_batch = X_batch.shape[0]
        batch_cov = X_batch.T @ X_batch / n_batch

        if t == 1:
            self.cov = batch_cov
        else:
            self.cov = ((t - 1) / t) * self.cov + (1 / t) * batch_cov

    def minimize_batch(self, W, sigma, mu, n_batches, batch_size, s, lr,
                       tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        """
        Minimize using mini-batch stochastic gradient descent.

        Args:
            W: Initial adjacency matrix
            sigma: Initial noise scale
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
            G_score = -mu * self.cov_batch @ (self.Id - W) / sigma
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

            # ADAM step
            grad = self._adam_update(Gobj, batch_idx, beta_1, beta_2)
            W -= lr * grad

            # Update sigma using the cumulative covariance estimate
            # sigma^2 = tr((I-W)^T cov(X) (I-W)) / d
            dif = self.Id - W
            rhs = self.cov @ dif
            sigma = max(np.sqrt(np.trace(dif.T @ rhs) / self.d), 1e-8)

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
            W_init=None,
        ):
        """
        Fit CoLiDE-EV using mini-batch SGD.

        Args:
            X: Data matrix (n_samples, n_features)
            lambda1: L1 regularization parameter
            T: Number of outer iterations
            mu_init: Initial penalty parameter
            mu_factor: Factor to decrease mu
            s: Sequence of acyclicity parameters
            batch_size: Size of mini-batches
            n_batches_warm: Number of batches for warm-up iterations
            n_batches_final: Number of batches for final iteration
            lr: Learning rate
            checkpoint: Check convergence every this many batches
            W_init: Optional warm-start adjacency matrix. If None, start
                from zero. Used for sliding-window streaming where the
                previous batch's estimate is a good init for the next.
        """
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        # Initialize
        self.cov = np.zeros((self.d, self.d)).astype(self.dtype)
        if W_init is None:
            self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        else:
            self.W_est = np.asarray(W_init, dtype=self.dtype).copy()
        self.sig_est = 1.0

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
        mode_desc = "DyCoLiDE-EV (online)" if batch_size == 1 else f"DyCoLiDE-EV (bs={batch_size})"
        with tqdm(total=total_batches, desc=mode_desc) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                n_batches = n_batches_final if i == T - 1 else n_batches_warm

                while success is False:
                    W_temp, sig_temp, success = self.minimize_batch(
                        self.W_est.copy(), self.sig_est, mu, n_batches,
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


#===================================#
#  Non-equal variance CoLiDE-NV     #
#  (covariance-based sigma update)  #
#===================================#

class colide_nv_batch_cov:
    """
    CoLiDE-NV with mini-batch stochastic gradient descent and
    covariance-based per-node sigma update (mirrors colide_ev_batch):

        sigma_j^2 = [(I-W)^T cov(X) (I-W)]_{jj}.

    Maintains a running sample covariance `cov(X)` via Welford / online
    batch updates, then plugs the closed form in at every step.

    Supports:
    - batch_size=1: True online learning (sequential processing)
    - batch_size>1: Mini-batch SGD (random sampling)
    """

    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
        self.online_mode = False

    def _score(self, W, sigma):
        """Compute score function and gradient for NV model"""
        dif = self.Id - W
        rhs = self.cov_batch @ dif
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

    def minimize_batch(self, W, sigma, mu, n_batches, batch_size, s, lr,
                       tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        """
        Minimize using mini-batch stochastic gradient descent.

        NV sigma is updated from cumulative covariance estimate:
        sigma_j^2 = [(I-W)^T cov(X) (I-W)]_jj
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

            # Compute gradient using NV score
            inv_SigMa = np.diag(1.0 / sigma)
            G_score = -mu * (self.cov_batch @ (self.Id - W) @ inv_SigMa)
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T

            # ADAM step
            grad = self._adam_update(Gobj, batch_idx, beta_1, beta_2)
            W -= lr * grad

            # Update sigma using the cumulative covariance estimate (NV version)
            # sigma_j^2 = [(I-W)^T cov(X) (I-W)]_jj (diagonal elements)
            dif = self.Id - W
            rhs = self.cov @ dif
            sigma_sq = np.diag(dif.T @ rhs)
            sigma = np.maximum(np.sqrt(np.maximum(sigma_sq, 0)), 1e-8)

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
        ):
        """
        Fit CoLiDE-NV using mini-batch SGD with covariance-based sigma.

        Args:
            X: Data matrix (n_samples, n_features)
            lambda1: L1 regularization parameter
            T: Number of outer iterations
            mu_init: Initial penalty parameter
            mu_factor: Factor to decrease mu
            s: Sequence of acyclicity parameters
            batch_size: Size of mini-batches (1 for online mode)
            n_batches_warm: Number of batches for warm-up iterations
            n_batches_final: Number of batches for final iteration
            lr: Learning rate
            checkpoint: Check convergence every this many batches
        """
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        # Initialize
        self.cov = np.zeros((self.d, self.d)).astype(self.dtype)
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        # Initialize sigma as vector (per-node variances)
        self.sig_est = np.ones(self.d).astype(self.dtype)

        # Initialize online statistics once (persists across stages)
        self.online_mode = (batch_size == 1)
        if self.online_mode:
            self.online_mean = np.zeros(self.d)
            self.online_M2 = np.zeros((self.d, self.d))
            self.online_sample_idx = 0
            self.online_count = 0

        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            raise ValueError("s should be a list, int, or float.")

        total_batches = (T - 1) * n_batches_warm + n_batches_final
        mode_desc = "CoLiDE-NV-Cov (online)" if batch_size == 1 else f"CoLiDE-NV-Cov (bs={batch_size})"
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
