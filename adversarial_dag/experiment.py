"""
Adversarial vulnerability of DyCoLiDE.

Bi-level frame:
  outer:  δ* = argmax_{||δ||_F ≤ ε}  SEM_loss(Ŵ(X + δ); X + δ)
  inner:  Ŵ(X) = argmin_W  score(W; X) + sparsity + acyclicity
  goal:   see how fragile Ŵ is to small adversarial shifts in data, vs
          random perturbations of the same norm.

Pragmatic attack: we don't solve the full min-max (would require
unrolling DyCoLiDE). Instead we use a white-box surrogate: fit Ŵ_clean
once on clean data, then run projected gradient ascent on the linear-SEM
residual ‖X − XW‖² with that Ŵ fixed. The resulting δ* is a worst-case
direction under the surrogate, and we then *refit* DyCoLiDE on X + δ* to
measure the downstream damage. Random δ of equal norm is the baseline.

If adversarial δ collapses recovery at a budget where random δ barely
dents it, DyCoLiDE is non-robust in the sense the outer problem targets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from evolving_dag_experiment import generate_dag, compute_metrics
from Colide import colide_ev_batch


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def generate_data(W, n, sigma=1.0, rng=None):
    rng = rng or np.random.default_rng()
    d = W.shape[0]
    G = nx.DiGraph(W)
    order = list(nx.topological_sort(G))
    X = np.zeros((n, d))
    for t in range(n):
        for j in order:
            parents = list(G.predecessors(j))
            eta = X[t, parents].dot(W[parents, j]) if parents else 0.0
            X[t, j] = eta + rng.normal(scale=sigma)
    return X


def fit_dycolide(X, seed=42, lambda1=0.05,
                 n_iter_warm=10000, n_iter_final=25000, lr=0.001):
    """One DyCoLiDE fit. Returns weighted adjacency Ŵ."""
    model = colide_ev_batch(seed=seed)
    W_est, _ = model.fit(
        X=X,
        lambda1=lambda1,
        T=4,
        batch_size=min(100, len(X)),
        n_batches_warm=n_iter_warm,
        n_batches_final=n_iter_final,
        lr=lr,
    )
    return W_est


def surrogate_sem_loss_grad(X, W):
    """∇_X  (1/(2n)) ‖X − X W‖²_F  =  (1/n) (X(I − W)) (I − W)^T"""
    d = X.shape[1]
    n = X.shape[0]
    IW = np.eye(d) - W
    return (X @ IW) @ IW.T / n


def pgd_attack(X, W, epsilon, n_steps=40, step_frac=0.1):
    """Projected gradient ascent on surrogate SEM loss to find worst δ.

    Projection is onto the Frobenius ball ‖δ‖_F ≤ epsilon.
    """
    step_size = step_frac * epsilon
    delta = np.zeros_like(X)
    for _ in range(n_steps):
        grad = surrogate_sem_loss_grad(X + delta, W)
        # ascent: move in direction of grad
        delta = delta + step_size * grad / (np.linalg.norm(grad) + 1e-12)
        # project into epsilon-ball
        norm = np.linalg.norm(delta)
        if norm > epsilon:
            delta = delta * (epsilon / norm)
    return delta


def random_perturbation(X, epsilon, rng):
    """Frobenius-norm-matched isotropic Gaussian perturbation."""
    delta = rng.standard_normal(X.shape)
    delta = delta * (epsilon / (np.linalg.norm(delta) + 1e-12))
    return delta


# =============================================================================
# Main
# =============================================================================

def run(
    d=40,
    n=1000,
    expected_edges=160,
    eps_fractions=(0.01, 0.02, 0.05, 0.10, 0.20),
    seed=42,
    threshold=0.3,
    save_name='adversarial_dag.png',
):
    rng = np.random.default_rng(seed)

    # Ground truth
    W_true = generate_dag(d, expected_edges, seed=seed)
    X = generate_data(W_true, n, sigma=1.0, rng=rng)
    base_norm = np.linalg.norm(X)  # reference for "budget"

    print(f"Data: n={n}, d={d}, ‖X‖_F = {base_norm:.2f}")

    # Clean fit (reference W used as the PGD surrogate)
    print("\n[1/?] clean fit...")
    W_clean = fit_dycolide(X, seed=seed)
    m_clean = compute_metrics(W_true, W_clean, threshold=threshold)
    print(f"  clean SHD = {m_clean['shd']}, TPR = {m_clean['tpr']:.3f}, "
          f"FDR = {m_clean['fdr']:.3f}")

    rand_shd, rand_tpr, rand_fdr = [], [], []
    adv_shd,  adv_tpr,  adv_fdr  = [], [], []

    for i, frac in enumerate(eps_fractions):
        eps = frac * base_norm

        # ---- Random ----
        print(f"\n[random   ε={frac*100:.1f}% = {eps:.2f}] generating + fitting...")
        d_rand = random_perturbation(X, eps, rng)
        W_rand = fit_dycolide(X + d_rand, seed=seed)
        m_rand = compute_metrics(W_true, W_rand, threshold=threshold)
        rand_shd.append(m_rand['shd'])
        rand_tpr.append(m_rand['tpr'])
        rand_fdr.append(m_rand['fdr'])
        print(f"  random  → SHD={m_rand['shd']}, TPR={m_rand['tpr']:.3f}, FDR={m_rand['fdr']:.3f}")

        # ---- Adversarial (PGD surrogate) ----
        print(f"[adv      ε={frac*100:.1f}% = {eps:.2f}] attacking + fitting...")
        d_adv = pgd_attack(X, W_clean, eps, n_steps=40)
        W_adv = fit_dycolide(X + d_adv, seed=seed)
        m_adv = compute_metrics(W_true, W_adv, threshold=threshold)
        adv_shd.append(m_adv['shd'])
        adv_tpr.append(m_adv['tpr'])
        adv_fdr.append(m_adv['fdr'])
        print(f"  adv     → SHD={m_adv['shd']}, TPR={m_adv['tpr']:.3f}, FDR={m_adv['fdr']:.3f}")

    # =========================================================================
    # Plot
    # =========================================================================
    fracs = np.array(eps_fractions) * 100
    clean_shd = [m_clean['shd']] * len(fracs)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.plot(fracs, clean_shd, 'k:', label='clean (no attack)')
    ax.plot(fracs, rand_shd, 'b-o', label='random perturbation', markersize=6)
    ax.plot(fracs, adv_shd,  'r-s', label='adversarial (PGD)', markersize=6)
    ax.set_xlabel('perturbation budget (% of ‖X‖_F)')
    ax.set_ylabel('SHD (lower is better)')
    ax.set_title('DAG recovery degradation')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(fracs, [m_clean['tpr']] * len(fracs), 'k:', label='clean')
    ax.plot(fracs, rand_tpr, 'b-o', label='random', markersize=6)
    ax.plot(fracs, adv_tpr,  'r-s', label='adversarial', markersize=6)
    ax.set_xlabel('perturbation budget (%)')
    ax.set_ylabel('TPR (higher is better)')
    ax.set_title('Edge recall')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = axes[2]
    ax.plot(fracs, [m_clean['fdr']] * len(fracs), 'k:', label='clean')
    ax.plot(fracs, rand_fdr, 'b-o', label='random', markersize=6)
    ax.plot(fracs, adv_fdr,  'r-s', label='adversarial', markersize=6)
    ax.set_xlabel('perturbation budget (%)')
    ax.set_ylabel('FDR (lower is better)')
    ax.set_title('False discovery rate')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.suptitle('DyCoLiDE under adversarial vs. random data perturbations',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = RESULTS_DIR / save_name
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"\nplot saved to {save_path}")
    plt.show()

    return {
        'clean': m_clean,
        'eps_fractions': eps_fractions,
        'rand_shd': rand_shd, 'rand_tpr': rand_tpr, 'rand_fdr': rand_fdr,
        'adv_shd':  adv_shd,  'adv_tpr':  adv_tpr,  'adv_fdr':  adv_fdr,
    }


if __name__ == '__main__':
    run()
