"""
SVAR structure-learning methods for the DyCoLiDE comparison, gathered behind a
single import. All return (W, A) in the DyCoLiDE row-vector convention
(model: X = X W + Y A + Z, with Y = [X_{t-1}|...|X_{t-p}], W the d x d
contemporaneous DAG, A the p*d x d lagged matrix):

  * ``run_dagma_svar`` / ``run_dynotears`` -- the two continuous-optimization
    baselines defined below.
  * ``run_varlingam`` -- ICA/SCM external baseline (VARLiNGAM, lazy-imports
    ``lingam``).
  * ``DyCoLiDE_EV`` / ``DyCoLiDE_NV`` -- the proposed method, re-exported from
    ``dycolide`` so the whole comparison imports from one place.

Details of the two continuous-opt baselines:

  * DagmaSVAR  -- DAGMA extended to the SVAR setting. Same least-squares score
    and DAGMA log-det acyclicity h_s(W) = -logdet(sI - W∘W) + d log s as the
    static DAGMA [Bello et al.'22], plus a lagged term A (no acyclicity on A).
    Solved by Adam with the central-path schedule and the M-matrix domain
    safeguard. This is DyCoLiDE's objective WITHOUT the CoLiDE noise scale σ --
    equivalently, DYNOTEARS's objective with DAGMA's log-det acyclicity (and
    Adam) instead of NOTEARS' matrix-exponential (and L-BFGS).

  * DYNOTEARS  -- [Pamfil et al.'20] NOTEARS-style SVAR learner: matrix-exp
    acyclicity tr(e^{W∘W}) - d, augmented-Lagrangian dual ascent over L-BFGS-B.
    Core ported verbatim from causalnex (Apache-2.0).

Both share DyCoLiDE's lag layout via create_lagged_data, so A aligns with the
ground-truth A_true used by count_accuracy_svar.
"""

import os
import sys

import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import scipy.optimize as sopt
from tqdm.auto import tqdm

# create_lagged_data lives in dycolide.py (same SVAR/ package).
sys.path.insert(0, os.path.dirname(__file__))
from dycolide import create_lagged_data, DyCoLiDE_EV, DyCoLiDE_NV  # re-exported


# =============================================================================
# DAGMA-SVAR
# =============================================================================

class DagmaSVAR:
    """DAGMA for structural vector autoregressive models.

    Minimizes, over a central path of penalty subproblems t = 0..T-1,

        mu_t * [ Q(W, A) + lambda_w||W||_1 + lambda_a||A||_1 ] + h_s(W),

    with the least-squares score
        Q(W, A) = 1/2 [ tr((I-W)^T Sxx (I-W)) - 2 tr((I-W)^T Sxy A)
                        + tr(A^T Syy A) ]
                = 1/(2n) ||X(I-W) - Y A||_F^2,
    second-order statistics Sxx, Sxy, Syy, and the DAGMA log-det acyclicity
    h_s(W) = -logdet(sI - W∘W) + d log s on the contemporaneous W only.

    No noise scale: this is DyCoLiDE-EV with sigma fixed to 1.
    """

    def __init__(self, seed: int = 42, dtype: type = np.float64):
        self.seed = seed
        self.dtype = dtype
        np.random.seed(seed)

    # ---- objective pieces ----------------------------------------------------
    def _score(self, W, A):
        diff = self.Id - W
        return 0.5 * (np.trace(diff.T @ self.Sxx @ diff)
                      - 2.0 * np.trace(diff.T @ self.Sxy @ A)
                      + np.trace(A.T @ self.Syy @ A))

    def _h(self, W, s):
        M = s * self.Id - W * W
        return -la.slogdet(M)[1] + self.d * np.log(s)

    def _func(self, W, A, mu, s):
        return mu * (self._score(W, A)
                     + self.lambda_w * np.abs(W).sum()
                     + self.lambda_a * np.abs(A).sum()) + self._h(W, s)

    @staticmethod
    def _adam(grad, m, v, t, b1, b2):
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * (grad ** 2)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        return m_hat / (np.sqrt(v_hat) + 1e-8), m, v

    # ---- inner minimization (Adam over one penalty subproblem) ---------------
    def minimize(self, W, A, mu, max_iter, s, lr, tol, b1, b2, pbar):
        obj_prev = 1e16
        m_W = np.zeros_like(W); v_W = np.zeros_like(W)
        m_A = np.zeros_like(A); v_A = np.zeros_like(A)
        upd_W = np.zeros_like(W); upd_A = np.zeros_like(A)

        for it in range(1, max_iter + 1):
            # M-matrix feasibility safeguard (sI - W∘W must stay an M-matrix)
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if it == 1 or s <= 0.9:
                    return W, A, False
                W += lr * upd_W; A += lr * upd_A
                lr *= 0.5
                if lr <= 1e-16:
                    return W, A, True
                W -= lr * upd_W; A -= lr * upd_A
                M = sla.inv(s * self.Id - W * W) + 1e-16

            diff = self.Id - W
            g_W_score = mu * (self.Sxy @ A - self.Sxx @ diff)
            g_A_score = mu * (self.Syy @ A - self.Sxy.T @ diff)
            G_W = g_W_score + mu * self.lambda_w * np.sign(W) + 2 * W * M.T
            G_A = g_A_score + mu * self.lambda_a * np.sign(A)

            upd_W, m_W, v_W = self._adam(G_W, m_W, v_W, it, b1, b2)
            upd_A, m_A, v_A = self._adam(G_A, m_A, v_A, it, b1, b2)
            W -= lr * upd_W
            A -= lr * upd_A

            if it % self.checkpoint == 0 or it == max_iter:
                obj_new = self._func(W, A, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar:
                        pbar.update(max_iter - it + 1)
                    break
                obj_prev = obj_new
            if pbar:
                pbar.update(1)
        return W, A, True

    # ---- public fit ----------------------------------------------------------
    def fit(self, X_full, p=1, lambda_w=0.01, lambda_a=0.015, w_threshold=0.0,
            T=4, mu_init=1.0, mu_factor=0.1, s=(1.0, 0.9, 0.8, 0.7),
            warm_iter=20000, max_iter=70000, lr=1e-3, checkpoint=5000,
            beta_1=0.99, beta_2=0.999):
        """Fit DAGMA-SVAR; returns (W_est, A_est).

        X_full : (T_total, d) time series (one row per time point).
        Defaults mirror the DyCoLiDE EV schedule for a like-for-like comparison.
        """
        X, Y = create_lagged_data(X_full, p)
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        n, d = X.shape
        self.d = d
        self.lambda_w, self.lambda_a, self.checkpoint = lambda_w, lambda_a, checkpoint
        self.Id = np.eye(d).astype(self.dtype)
        self.Sxx = (X.T @ X / n).astype(self.dtype)
        self.Sxy = (X.T @ Y / n).astype(self.dtype)
        self.Syy = (Y.T @ Y / n).astype(self.dtype)

        W = np.zeros((d, d), dtype=self.dtype)
        A = np.zeros((p * d, d), dtype=self.dtype)
        s = list(s)
        if len(s) < T:
            s = s + (T - len(s)) * [s[-1]]

        mu = mu_init
        with tqdm(total=int((T - 1) * warm_iter + max_iter), desc="DAGMA-SVAR") as pbar:
            for i in range(T):
                inner = int(max_iter) if i == T - 1 else int(warm_iter)
                lr_i, success = lr, False
                while not success:
                    W_t, A_t, success = self.minimize(
                        W.copy(), A.copy(), mu, inner, s[i], lr_i,
                        tol=1e-6, b1=beta_1, b2=beta_2, pbar=pbar)
                    if not success:
                        lr_i *= 0.5
                        s[i] += 0.1
                W, A = W_t, A_t
                mu *= mu_factor

        if w_threshold > 0:
            W[np.abs(W) < w_threshold] = 0
            A[np.abs(A) < w_threshold] = 0
        self.W_est, self.A_est = W, A
        return W, A


def run_dagma_svar(X, p=1, lambda_w=0.01, lambda_a=0.015, seed=42, **kw):
    """Convenience: fit DAGMA-SVAR and return (W, A)."""
    return DagmaSVAR(seed=seed).fit(X, p=p, lambda_w=lambda_w,
                                    lambda_a=lambda_a, **kw)


# =============================================================================
# DYNOTEARS  (core ported from causalnex, Apache-2.0)
# =============================================================================

def _reshape_wa(wa_vec, d_vars, p_orders):
    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars: 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (w_tilde[2 * d_vars:]
              .reshape(2 * p_orders, d_vars ** 2)[::2]
              .reshape(d_vars * p_orders, d_vars))
    a_minus = (w_tilde[2 * d_vars:]
               .reshape(2 * p_orders, d_vars ** 2)[1::2]
               .reshape(d_vars * p_orders, d_vars))
    return w_mat, a_plus - a_minus


def _learn_dynamic_structure(X, Xlags, bnds, lambda_w=0.1, lambda_a=0.1,
                             max_iter=100, h_tol=1e-8):
    n, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    def _h(wa_vec):
        _w, _ = _reshape_wa(wa_vec, d_vars, p_orders)
        return np.trace(sla.expm(_w * _w)) - d_vars

    def _func(wa_vec):
        _w, _a = _reshape_wa(wa_vec, d_vars, p_orders)
        loss = 0.5 / n * np.square(np.linalg.norm(
            X.dot(np.eye(d_vars) - _w) - Xlags.dot(_a), "fro"))
        _hv = _h(wa_vec)
        l1 = lambda_w * wa_vec[: 2 * d_vars ** 2].sum() + \
            lambda_a * wa_vec[2 * d_vars ** 2:].sum()
        return loss + 0.5 * rho * _hv * _hv + alpha * _hv + l1

    def _grad(wa_vec):
        _w, _a = _reshape_wa(wa_vec, d_vars, p_orders)
        e_mat = sla.expm(_w * _w)
        loss_grad_w = -1.0 / n * (X.T.dot(
            X.dot(np.eye(d_vars) - _w) - Xlags.dot(_a)))
        obj_grad_w = loss_grad_w + \
            (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * _w * 2
        obj_grad_a = -1.0 / n * (Xlags.T.dot(
            X.dot(np.eye(d_vars) - _w) - Xlags.dot(_a)))
        grad_w = np.append(obj_grad_w, -obj_grad_w, axis=0).flatten() \
            + lambda_w * np.ones(2 * d_vars ** 2)
        grad_a = obj_grad_a.reshape(p_orders, d_vars ** 2)
        grad_a = np.hstack((grad_a, -grad_a)).flatten() \
            + lambda_a * np.ones(2 * p_orders * d_vars ** 2)
        return np.append(grad_w, grad_a, axis=0)

    wa_est = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    wa_new = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf
    for _ in range(max_iter):
        while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
            wa_new = sopt.minimize(_func, wa_est, method="L-BFGS-B",
                                   jac=_grad, bounds=bnds).x
            h_new = _h(wa_new)
            if h_new > 0.25 * h_value:
                rho *= 10
        wa_est, h_value = wa_new, h_new
        alpha += rho * h_value
        if h_value <= h_tol:
            break
    return _reshape_wa(wa_est, d_vars, p_orders)


def _dyno_bounds(d, p):
    bnds_w = 2 * [(0, 0) if i == j else (0, None)
                  for i in range(d) for j in range(d)]
    bnds_a = []
    for _ in range(1, p + 1):
        bnds_a.extend(2 * [(0, None) for _i in range(d) for _j in range(d)])
    return bnds_w + bnds_a


def run_dynotears(X, p=1, lambda_w=0.01, lambda_a=0.015, max_iter=100,
                  h_tol=1e-8, w_threshold=0.0, **_ignore):
    """Fit DYNOTEARS on an SVAR series; returns (W, A) in DyCoLiDE form."""
    X_cur, Xlags = create_lagged_data(X, p)
    d = X_cur.shape[1]
    W, A = _learn_dynamic_structure(X_cur, Xlags, _dyno_bounds(d, p),
                                    lambda_w, lambda_a, max_iter=max_iter,
                                    h_tol=h_tol)
    if w_threshold > 0:
        W = W.copy(); A = A.copy()
        W[np.abs(W) < w_threshold] = 0
        A[np.abs(A) < w_threshold] = 0
    return W, A


def run_varlingam(X, p=1, w_threshold=0.0, prune=True, **_ignore):
    """VARLiNGAM [Hyvarinen et al.'10]: VAR + LiNGAM (ICA exploiting NON-Gaussianity)
    on the residuals to orient the contemporaneous structure -- the standard
    SCM/ICA-based external SVAR baseline (different paradigm from the continuous-opt
    family). Needs non-Gaussian shocks for contemporaneous identifiability.

    Uses the official ``lingam`` package. adjacency_matrices_ is (p+1, d, d) =
    [B_0, B_1, ..., B_p]; B_0 contemporaneous, B_k lagged, B[i,j] = effect of x_j
    on x_i. Mapped to the DyCoLiDE row-vector convention: W = B_0^T, A = [B_1^T; ...].
    """
    import lingam   # lazy: only this baseline needs lingam
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    model = lingam.VARLiNGAM(lags=p, criterion=None, prune=prune, random_state=0)
    model.fit(X)
    M = np.asarray(model.adjacency_matrices_)        # (p+1, d, d)
    W = M[0].T.copy()                                 # B_0^T
    A = (np.vstack([M[k].T for k in range(1, p + 1)])
         if p >= 1 else np.zeros((0, d)))
    if w_threshold > 0:
        W[np.abs(W) < w_threshold] = 0
        A[np.abs(A) < w_threshold] = 0
    return W, A


# =============================================================================
# smoke test
# =============================================================================

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Utils import generate_svar_data, count_accuracy_svar

    X, B_true, A_list_true, _ = generate_svar_data(
        n_nodes=10, n_timesteps=500, lag_order=1, instantaneous_edges=20,
        temporal_edges=20, temporal_strength=0.5, noise_scale=1.0,
        noise_type='ev', seed=42)
    W_true = B_true.T
    A_true = np.vstack([A.T for A in A_list_true])
    print(f"True  W edges={int((np.abs(W_true) > 0).sum())}, "
          f"A edges={int((np.abs(A_true) > 0).sum())}")

    for name, fn in [("DAGMA-SVAR", run_dagma_svar), ("DYNOTEARS", run_dynotears)]:
        W_e, A_e = fn(X.copy(), p=1, lambda_w=0.01, lambda_a=0.015)
        mW = count_accuracy_svar(W_true, W_e, threshold=0.10)
        mA = count_accuracy_svar(A_true, A_e, threshold=0.08)
        print(f"{name:<11} W: SHD={mW['W_shd']:>3} TPR={mW['W_tpr']:.2f} "
              f"FDR={mW['W_fdr']:.2f}   A: SHD={mA['W_shd']:>3} "
              f"TPR={mA['W_tpr']:.2f} FDR={mA['W_fdr']:.2f}")
