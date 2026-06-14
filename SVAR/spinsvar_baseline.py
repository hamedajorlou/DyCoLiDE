"""
SpinSVAR baseline for the SVAR comparison.

SpinSVAR [Misiakos & Puschel, UAI'25] estimates a structural VAR under a SPARSE
/ LAPLACIAN exogenous-input assumption: it minimizes a least-absolute-error
(L1) reconstruction -- the Laplacian MLE -- plus the NOTEARS matrix-exp
acyclicity on the contemporaneous block, by Adam. This is fundamentally a
different noise model from DyCoLiDE (Gaussian + concomitant sigma), so it is
expected to shine on Laplacian/sparse-shock data and to be off-assumption on
Gaussian data -- the comparison should cover BOTH regimes.

Core (`SpinSVAR`, `spinsvar_solver`) is ported verbatim from
github.com/pmisiakos/SpinSVAR (sparserc/spinsvar.py), with two changes for
standalone use: the `X_past` helper is inlined, and the best-model checkpoint is
kept in memory instead of written to results/best_model.pl.

Output mapped to DyCoLiDE convention: W (d,d) = B_0, A (p*d, d) = [B_1; ...; B_p].
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float32


def _X_past(X, k, device="cpu"):
    """Lagged design: X_past[:, t, :] = [x_{t-k}, ..., x_{t-1}, x_t] (zero-padded
    at the start). X is (n, T, d); returns (n, T, (k+1)*d). Ported verbatim."""
    n, T, d = X.shape
    X = X.reshape(n, d * T)
    X_past = torch.zeros((n, T, (k + 1) * d), device=device)
    for t in range(T):
        if t < k:
            X_past[:, t, :] = torch.cat(
                [torch.zeros((n, (k - t) * d), device=device), X[:, :(t + 1) * d]], dim=1)
        else:
            X_past[:, t, :] = X[:, (t - k) * d: (t + 1) * d]
    return X_past


class SpinSVAR(nn.Module):
    """Sparse-root-causes model for time-series SVAR data (ported)."""

    def __init__(self, X, lambda1, lambda2, time_lag=1, constraint='notears',
                 omega=0.3, T=10):
        super().__init__()
        self.X = X.clone().detach()
        self.n, self.T, self.d = torch.tensor(X.shape).to(_device)
        self.eye = torch.eye(self.d).to(_device)
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.p = time_lag
        self.constraint = constraint
        self.omega = omega
        self.T = T
        # linear layer = window graph [B_p.T, ..., B_1.T, B_0.T] (reversed, transposed)
        self.fc = torch.nn.Linear(self.d * (self.p + 1), self.d, bias=False)
        self.X_past = _X_past(X, self.p, _device).to(_device).to(_dtype)

    def postprocess_A(self):
        """Threshold and return the window graph [B_0, B_1, ..., B_p]."""
        A = self.fc.weight
        A_est = torch.where(torch.abs(A) > self.omega, A, torch.zeros_like(A))
        res = torch.zeros(A_est.shape)
        for i in range(self.p + 1):
            res[:, i * self.d: (i + 1) * self.d] = \
                A_est[:, (self.p - i) * self.d: (self.p + 1 - i) * self.d].T
        return res.detach().cpu().numpy()

    def l1_reg(self):
        return torch.sum(torch.abs(self.fc.weight))

    def logdet(self):
        A = self.fc.weight[:self.d, self.d * self.p:]      # B_0
        return torch.abs(torch.linalg.det(self.eye - A))

    def acyclicity(self):
        A = self.fc.weight[:self.d, self.d * self.p:]      # B_0
        if self.constraint == 'notears':
            return torch.trace(torch.matrix_exp(A * A)) - self.d
        elif self.constraint == 'dag-gnn':
            M = torch.eye(self.d) + A * A / self.d
            return torch.trace(torch.linalg.matrix_power(M, self.d)) - self.d
        elif self.constraint == 'frobenius':
            return torch.sum((A * A.T) ** 2)

    def forward(self):
        return self.fc(self.X_past)


def spinsvar_solver(X, lambda1, lambda2, epochs=3000, time_lag=1,
                    constraint="notears", omega=0.3, T=10, verbose=False):
    """SpinSVAR solver. X: (n, T, d). Returns window graph [B_0,...,B_p]
    (np.ndarray, d x d(p+1)). Best-iterate kept in memory (no file I/O)."""
    X = torch.tensor(X, device=_device, dtype=_dtype)
    N = X.shape[0]
    model = SpinSVAR(X, lambda1, lambda2, time_lag, constraint, omega, T).to(_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    early_stop, best_loss, best_state = 40, float('inf'), None
    for i in range(epochs):
        def closure():
            nonlocal best_loss, early_stop, best_state
            optimizer.zero_grad()
            output = model()
            loss1 = torch.log(torch.norm((X - output), p=1)) \
                - (1 / model.d) * torch.log(model.logdet())
            loss = N * loss1 + lambda1 * model.l1_reg() + lambda2 * model.acyclicity()
            loss.backward()
            if loss.item() >= best_loss:
                early_stop -= 1
            else:
                early_stop = 40
                best_loss = loss.item()
                best_state = copy.deepcopy(model.state_dict())
            return loss
        optimizer.step(closure)
        if early_stop == 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.postprocess_A()


def run_spinsvar(X, p=1, lambda1=0.05, lambda2=0.5, omega=0.3, epochs=3000,
                 **_ignore):
    """Fit SpinSVAR on a single SVAR series X (T_total, d); return (W, A) in
    DyCoLiDE convention. W = B_0 (d,d); A = [B_1; ...; B_p] (p*d, d)."""
    X = np.asarray(X, dtype=np.float32)
    T_total, d = X.shape
    Xb = X.reshape(1, T_total, d)                      # single realization
    Wgraph = spinsvar_solver(Xb, lambda1=lambda1, lambda2=lambda2,
                             time_lag=p, omega=omega, epochs=epochs, T=T_total)
    W = Wgraph[:, :d].copy()                            # B_0
    A = np.vstack([Wgraph[:, (k + 1) * d:(k + 2) * d] for k in range(p)]) \
        if p >= 1 else np.zeros((0, d))                 # [B_1; ...; B_p]
    return W, A


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Utils import generate_svar_data, count_accuracy_svar
    X, B_true, A_list_true, _ = generate_svar_data(
        n_nodes=10, n_timesteps=1000, lag_order=1, instantaneous_edges=20,
        temporal_edges=20, temporal_strength=0.5, noise_scale=1.0,
        noise_type='ev', seed=42)
    W_true = B_true.T
    A_true = np.vstack([A.T for A in A_list_true])
    W_e, A_e = run_spinsvar(X, p=1, lambda1=0.05, lambda2=0.5, omega=0.3, epochs=1500)
    print(f"True W edges={int((np.abs(W_true)>0).sum())}, A edges={int((np.abs(A_true)>0).sum())}")
    for thr in [0.1, 0.2, 0.3]:
        mW = count_accuracy_svar(W_true, W_e, threshold=thr)
        mA = count_accuracy_svar(A_true, A_e, threshold=thr)
        print(f"  thr={thr}: W SHD={mW['W_shd']} TPR={mW['W_tpr']:.2f}  "
              f"A SHD={mA['W_shd']} TPR={mA['W_tpr']:.2f}")
