"""
Meta-D2AG-Linear: online meta-learning for dynamic (drifting) SVAR DAG learning.

Faithful first-order implementation of Meta-D2AG [Gao, Lu et al., NeurIPS'25]
restricted to the LINEAR SEM (their Meta-D2AG-Linear variant), for comparison
against DyCoLiDE on multi-domain drifting data.

Defining mechanisms (from the paper):
  * Shared params W_s (cross-domain) + private params W_p,i (per domain),
    combined as a Hadamard product  G_i = W_s o W_p,i  (Remark 3.2), for both
    the contemporaneous Wa and the lagged Wb. The shared params are fit jointly
    across a sliding window of the last `w` domains -- this pooling of
    (w x few-shot) samples is the source of the meta advantage.
  * Online sliding window of the last `w` domains (eq. 9).
  * Warm-start of private params from the previous domain (Alg. 1, line 4).
  * Acyclicity via the NOTEARS matrix-exponential h(Wa)=tr(e^{Wa o Wa})-d on the
    contemporaneous graph, solved by the augmented-Lagrangian method (the
    convergent recipe used by NOTEARS/DYNOTEARS).

First-order: a single joint Adam optimization over (W_s, {W_p,i in window})
of the windowed penalized objective; no second-order hypergradients (the paper
also uses a first-order approach). Requires oracle domain boundaries -- the
method's design, and the limitation the paper itself states.

Output per domain in DyCoLiDE convention: W (d,d), A (p*d, d).
"""

import os
import sys
import numpy as np
import torch


def _matexp_h(Wa):
    d = Wa.shape[0]
    return torch.trace(torch.matrix_exp(Wa * Wa)) - d


class MetaD2AGLinear:
    def __init__(self, d, p=1, w=3, lambda_s=0.05, lambda_p=0.05, lr=1e-2,
                 inner_iters=800, max_phases=30, rho_init=1.0, rho_max=1e8,
                 alpha=2.0, h_tol=1e-6, w_threshold=0.25, seed=42,
                 device='cpu', verbose=False):
        torch.manual_seed(seed); np.random.seed(seed)
        self.d, self.p, self.w = d, p, w
        self.lambda_s, self.lambda_p = lambda_s, lambda_p
        self.lr, self.inner_iters, self.max_phases = lr, inner_iters, max_phases
        self.rho_init, self.rho_max, self.alpha = rho_init, rho_max, alpha
        self.h_tol, self.w_threshold = h_tol, w_threshold
        self.dev = torch.device(device); self.verbose = verbose

        g = torch.Generator(device=self.dev).manual_seed(seed)
        self.Wa_s = (0.1 * torch.randn(d, d, generator=g, device=self.dev)).requires_grad_(True)
        self.Wb_s = (0.1 * torch.randn(p * d, d, generator=g, device=self.dev)).requires_grad_(True)
        self.Wa_p, self.Wb_p = {}, {}     # per-domain private
        self.data = {}                    # domain -> (X, Y) centered tensors
        self._last = None

    def _lagged(self, X):
        p = self.p
        Xc = X[p:]
        Y = np.hstack([X[p - k - 1:len(X) - k - 1] for k in range(p)])
        Xc = Xc - Xc.mean(0, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True)
        return (torch.tensor(Xc, dtype=torch.float32, device=self.dev),
                torch.tensor(Y, dtype=torch.float32, device=self.dev))

    def _fit(self, X, Y, Ga, Gb):
        R = X - X @ Ga - Y @ Gb
        return 0.5 / X.shape[0] * torch.sum(R ** 2)

    def observe_domain(self, X_m, m):
        """Process domain m (oracle boundary); returns (W_est, A_est)."""
        self.data[m] = self._lagged(X_m)
        if self._last is not None and self._last in self.Wa_p:
            wa0 = self.Wa_p[self._last].detach().clone()
            wb0 = self.Wb_p[self._last].detach().clone()
        else:
            wa0 = torch.ones(self.d, self.d, device=self.dev)
            wb0 = torch.ones(self.p * self.d, self.d, device=self.dev)
        self.Wa_p[m] = wa0.requires_grad_(True)
        self.Wb_p[m] = wb0.requires_grad_(True)

        window = [i for i in range(max(0, m - self.w + 1), m + 1) if i in self.data]
        params = [self.Wa_s, self.Wb_s]
        for i in window:
            params += [self.Wa_p[i], self.Wb_p[i]]
        opt = torch.optim.Adam(params, lr=self.lr)

        mu, rho = 0.0, self.rho_init
        for phase in range(self.max_phases):
            for _ in range(self.inner_iters):
                opt.zero_grad()
                obj, hsum = 0.0, 0.0
                for i in window:
                    X_i, Y_i = self.data[i]
                    Ga = self.Wa_s * self.Wa_p[i]
                    Gb = self.Wb_s * self.Wb_p[i]
                    obj = obj + self._fit(X_i, Y_i, Ga, Gb) \
                        + self.lambda_p * (self.Wa_p[i].abs().sum() + self.Wb_p[i].abs().sum())
                    hsum = hsum + _matexp_h(Ga)
                nw = len(window)
                obj = obj / nw + self.lambda_s * (self.Wa_s.abs().sum() + self.Wb_s.abs().sum())
                h_mean = hsum / nw
                loss = obj + mu * h_mean + 0.5 * rho * h_mean ** 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                with torch.no_grad():
                    self.Wa_s.fill_diagonal_(0)
                    for i in window:
                        self.Wa_p[i].fill_diagonal_(0)
            with torch.no_grad():
                h_val = float(sum(_matexp_h(self.Wa_s * self.Wa_p[i]) for i in window) / nw)
            if self.verbose:
                print(f"  [meta m={m}] phase {phase}: h={h_val:.2e}, rho={rho:.1e}")
            if h_val < self.h_tol:
                break
            mu += rho * h_val
            rho = min(rho * self.alpha, self.rho_max)

        self._last = m
        for i in list(self.data):
            if i < m - self.w + 1:
                self.data.pop(i, None)

        W = (self.Wa_s * self.Wa_p[m]).detach().cpu().numpy()
        A = (self.Wb_s * self.Wb_p[m]).detach().cpu().numpy()
        if self.w_threshold > 0:
            W = W.copy(); A = A.copy()
            W[np.abs(W) < self.w_threshold] = 0
            A[np.abs(A) < self.w_threshold] = 0
        return W, A


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'EXP'))
    from Utils import count_accuracy_svar
    from multidomain import generate_multidomain_svar

    doms = generate_multidomain_svar(Q=5, d=10, n_per_domain=20, seed=1)
    model = MetaD2AGLinear(d=10, p=1, w=3)
    print("domain |  W SHD  W TPR  W FDR")
    for m, dom in enumerate(doms):
        W_e, A_e = model.observe_domain(dom['X'], m)
        mW = count_accuracy_svar(dom['W_true'], W_e, threshold=0.25)
        print(f"   {m}    |  {mW['W_shd']:>4}  {mW['W_tpr']:.2f}  {mW['W_fdr']:.2f}")
