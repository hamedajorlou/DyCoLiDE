"""
Bi-level NAS with CoLiDE-style DAG learning.

Problem:
  outer:  W* = argmin_W  L_val(θ*(W), W) + λ₁‖W‖₁ + ρ·h(W)
  inner:  θ*(W) = argmin_θ L_train(θ, W)

where W ∈ R^{N×N} is a learnable architecture adjacency over an *unordered*
pool of N candidate modules, and h(W) = -log det(sI − W⊙W) + d log s is
CoLiDE's differentiable acyclicity penalty.

This is the first "does it work" pass: simplest possible module pool
(identical Linear+ReLU blocks), MNIST, alternating SGD (DARTS-style
bi-level). No baselines yet — we just need to confirm that:
  (a) training converges,
  (b) h(W) → 0 (the learned DAG is actually acyclic),
  (c) ‖W‖₁ shrinks (sparse architecture emerges),
  (d) test accuracy is reasonable.
"""

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Model
# =============================================================================

class UnorderedNASCell(nn.Module):
    """N candidate modules with a learnable DAG W over them.

    Forward pass iterates h_i ← f_i(Σ_j W[j,i] h_j) for T steps. Node 0 is
    pinned to the projected input throughout. If W is acyclic and T ≥
    longest path, the output stabilizes; the acyclicity penalty drives W
    there during training.
    """

    def __init__(self, n_nodes=6, hidden=64, in_dim=784, out_dim=10, T=None):
        super().__init__()
        self.N = n_nodes
        self.H = hidden
        self.T = T if T is not None else n_nodes

        self.input_proj = nn.Linear(in_dim, hidden)
        self.modules_pool = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
            for _ in range(n_nodes)
        ])
        self.classifier = nn.Linear(hidden, out_dim)

        # Architecture DAG: N×N weighted adjacency. Diagonal zeroed out in forward.
        self.W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.05)

    def arch_params(self):
        return [self.W]

    def weight_params(self):
        return [p for name, p in self.named_parameters() if name != 'W']

    def forward(self, x):
        B = x.shape[0]
        W = self.W - torch.diag(torch.diag(self.W))  # zero diagonal (no self-loops)

        h0 = self.input_proj(x)                       # (B, H)
        h = torch.zeros(B, self.N, self.H, device=x.device)
        h[:, 0, :] = h0

        for _ in range(self.T):
            #  combined[b, i, :] = Σ_j W[j,i] * h[b, j, :]
            combined = torch.einsum('bjh,ji->bih', h, W)
            h_new = torch.stack(
                [self.modules_pool[i](combined[:, i, :]) for i in range(self.N)],
                dim=1,
            )
            h_new = h_new.clone()
            h_new[:, 0, :] = h0                       # pin node 0 to input
            h = h_new

        return self.classifier(h[:, -1, :])           # read output from last node


# =============================================================================
# CoLiDE penalties
# =============================================================================

def acyclicity_h(W, s=1.0):
    """DAGMA/CoLiDE acyclicity: h(W) = −log det(sI − W⊙W) + d·log s.
    Zero iff W corresponds to a DAG. Smooth, differentiable.
    """
    d = W.shape[0]
    M = s * torch.eye(d, device=W.device) - W * W
    # slogdet returns (sign, logabsdet); sign must be +1 for M to be PD
    sign, logabsdet = torch.slogdet(M)
    return -logabsdet + d * np.log(s)


def sparsity_l1(W):
    return W.abs().sum()


# =============================================================================
# Data
# =============================================================================

def get_mnist_loaders(batch_size=128, val_size=10000, data_root='./data'):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten 28x28 → 784
    ])
    full_train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tfm)

    n_train = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [n_train, val_size],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# =============================================================================
# Training
# =============================================================================

def train(
    n_nodes=6,
    hidden=64,
    epochs=10,
    batch_size=128,
    lr_w=0.05,       # module-weight LR (inner)
    lr_arch=0.01,    # architecture LR (outer)
    lambda1=0.005,   # sparsity weight
    rho=0.02,        # acyclicity weight
    s=1.0,
    seed=42,
    save_name='nas_bilevel.png',
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    model = UnorderedNASCell(n_nodes=n_nodes, hidden=hidden).to(device)

    opt_w    = torch.optim.SGD(model.weight_params(), lr=lr_w, momentum=0.9, weight_decay=3e-4)
    opt_arch = torch.optim.Adam(model.arch_params(), lr=lr_arch, betas=(0.5, 0.999), weight_decay=1e-3)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'test_acc': [],
        'h_W': [], 'l1_W': [], 'W_snapshots': [],
    }

    val_iter = iter(val_loader)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        train_losses = []
        val_losses = []

        for x_tr, y_tr in train_loader:
            x_tr, y_tr = x_tr.to(device), y_tr.to(device)

            # ---- Outer step: update W on validation batch ----
            try:
                x_val, y_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_val, y_val = next(val_iter)
            x_val, y_val = x_val.to(device), y_val.to(device)

            opt_arch.zero_grad()
            logits_val = model(x_val)
            ce_val = F.cross_entropy(logits_val, y_val)
            pen_l1 = lambda1 * sparsity_l1(model.W)
            pen_h  = rho * acyclicity_h(model.W, s=s)
            outer_loss = ce_val + pen_l1 + pen_h
            outer_loss.backward()
            opt_arch.step()
            val_losses.append(ce_val.item())

            # ---- Inner step: update module params on train batch ----
            opt_w.zero_grad()
            logits_tr = model(x_tr)
            ce_tr = F.cross_entropy(logits_tr, y_tr)
            ce_tr.backward()
            opt_w.step()
            train_losses.append(ce_tr.item())

        # End-of-epoch eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        test_acc = correct / total

        with torch.no_grad():
            W_now = model.W.detach().cpu().clone()
            W_no_diag = W_now - torch.diag(torch.diag(W_now))
            h_val  = acyclicity_h(W_no_diag.to(device), s=s).item()
            l1_val = sparsity_l1(W_no_diag).item()

        history['epoch'].append(epoch)
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['test_acc'].append(test_acc)
        history['h_W'].append(h_val)
        history['l1_W'].append(l1_val)
        history['W_snapshots'].append(W_no_diag.numpy())

        print(f"ep {epoch:2d}/{epochs} | train_ce={np.mean(train_losses):.4f} "
              f"val_ce={np.mean(val_losses):.4f} test_acc={test_acc:.4f} "
              f"h(W)={h_val:.4f} ‖W‖₁={l1_val:.3f} "
              f"time={time.time()-t0:.1f}s")

    # ====== Plot ======
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(history['epoch'], history['train_loss'], 'b-o', label='train CE')
    ax.plot(history['epoch'], history['val_loss'],   'r-s', label='val CE')
    ax.set_xlabel('epoch'); ax.set_ylabel('cross-entropy'); ax.set_title('Loss')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history['epoch'], history['test_acc'], 'g-o')
    ax.set_xlabel('epoch'); ax.set_ylabel('test accuracy'); ax.set_title('MNIST test accuracy')
    ax.set_ylim([0.5, 1.0]); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.semilogy(history['epoch'], history['h_W'], 'r-o')
    ax.set_xlabel('epoch'); ax.set_ylabel('h(W)  (log scale)')
    ax.set_title('Acyclicity violation h(W)\n(should → 0)')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1, 0]
    ax.plot(history['epoch'], history['l1_W'], 'k-o')
    ax.set_xlabel('epoch'); ax.set_ylabel('‖W‖₁'); ax.set_title('Sparsity (‖W‖₁)\n(should shrink)')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    W_final = history['W_snapshots'][-1]
    im = ax.imshow(W_final, cmap='RdBu_r',
                   vmin=-np.max(np.abs(W_final)), vmax=np.max(np.abs(W_final)))
    ax.set_title('Final W (weighted adjacency)')
    ax.set_xlabel('to node'); ax.set_ylabel('from node')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    W_bin = (np.abs(W_final) > 0.1).astype(int)
    im = ax.imshow(W_bin, cmap='Greys', vmin=0, vmax=1)
    ax.set_title('Thresholded DAG (|W| > 0.1)')
    ax.set_xlabel('to node'); ax.set_ylabel('from node')
    n_edges = W_bin.sum()
    ax.text(0.05, 0.95, f'edges: {n_edges}/{n_nodes*(n_nodes-1)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white'))

    plt.suptitle('Bi-level NAS with CoLiDE-style DAG learning (MNIST)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = RESULTS_DIR / save_name
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"\nplot saved to {save_path}")
    plt.show()

    return history


if __name__ == '__main__':
    train()
