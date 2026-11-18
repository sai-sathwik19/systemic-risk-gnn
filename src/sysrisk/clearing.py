"""Eisenberg-Noe (2001) clearing.

Conventions (used everywhere in this package)
--------------------------------------------
* ``L[i, j] >= 0`` is the nominal amount bank ``i`` (debtor) owes bank ``j`` (creditor); ``diag(L) = 0``.
* ``x[i] >= 0`` are external assets.
* ``pbar = L.sum(1)`` total nominal interbank liabilities, ``a = L.sum(0)`` total interbank assets.
* ``Pi[i, j] = L[i, j] / pbar[i]`` (0 if ``pbar[i] == 0``).
* A clearing vector solves ``p = min(pbar, x + Pi^T p)`` componentwise.  If every ``x_i > 0`` it is unique
  (Eisenberg & Noe 2001); in general all routines here return the *greatest* clearing vector.
* Total unpaid liabilities (the project's target): ``S = sum_i (pbar_i - p_i)``.

Two solvers are provided:

* :func:`clearing_fda` - exact "fictitious default algorithm" (numpy); used to create labels.
* :func:`picard_dense` / :func:`picard_sparse` - batched fixed-point iteration (torch, CPU/GPU); used for
  cross-checks, the K-step heuristic baseline and runtime comparisons.  Edge-list graphs follow the PyG
  convention ``edge_index = [src=debtor, dst=creditor]`` with ``L_vals[e] = L[src_e, dst_e]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ----------------------------------------------------------------------------------------------------------
# numpy: exact solver
# ----------------------------------------------------------------------------------------------------------
def relative_liabilities(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Pi, pbar)`` for a liability matrix ``L``."""
    L = np.asarray(L, dtype=np.float64)
    pbar = L.sum(axis=1)
    Pi = np.zeros_like(L)
    pos = pbar > 0
    Pi[pos] = L[pos] / pbar[pos, None]
    return Pi, pbar


@dataclass
class ClearingResult:
    p: np.ndarray          # clearing payments
    pbar: np.ndarray       # nominal liabilities
    default: np.ndarray    # bool mask, p_i < pbar_i
    rounds: int            # FDA rounds in which the default set grew ("contagion depth"); 0 = no default

    @property
    def u(self) -> np.ndarray:
        """Node-level unpaid liabilities ``pbar - p``."""
        return self.pbar - self.p

    @property
    def S(self) -> float:
        """Total unpaid liabilities."""
        return float(self.u.sum())

    @property
    def r(self) -> np.ndarray:
        """Payment ratio ``p / pbar`` (1 for banks without liabilities)."""
        out = np.ones_like(self.p)
        pos = self.pbar > 0
        out[pos] = self.p[pos] / self.pbar[pos]
        return out


def clearing_fda(L: np.ndarray, x: np.ndarray, tol: float = 1e-12) -> ClearingResult:
    """Exact greatest clearing vector via the fictitious default algorithm (Eisenberg & Noe 2001, Sec. 3).

    Starting from "everybody pays in full", banks whose inflow cannot cover ``pbar`` are declared in default;
    the payments of the defaulting set ``D`` are then obtained from the linear system

        (I - Pi_DD^T) p_D = x_D + Pi_{S,D}^T pbar_S          (S = solvent banks)

    and the default set is recomputed.  ``D`` only grows, so at most ``N`` rounds are needed.
    """
    Pi, pbar = relative_liabilities(L)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    PiT = Pi.T
    thresh = tol * max(1.0, float(pbar.max(initial=0.0)))

    p = pbar.copy()
    D = np.zeros(n, dtype=bool)
    rounds = 0
    while True:
        inflow = x + PiT @ p
        newD = (inflow < pbar - thresh) | D
        if newD.sum() == D.sum():
            break
        D = newD
        rounds += 1
        Sv = ~D
        A = PiT[np.ix_(D, D)]
        b = x[D] + PiT[np.ix_(D, Sv)] @ pbar[Sv]
        M = np.eye(int(D.sum())) - A
        try:
            pD = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:  # only possible for degenerate inputs (e.g. x_i = 0 on a closed cycle)
            pD = np.linalg.lstsq(M, b, rcond=None)[0]
        p = pbar.copy()
        p[D] = np.clip(pD, 0.0, pbar[D])

    default = p < pbar - thresh
    return ClearingResult(p=p, pbar=pbar, default=default, rounds=rounds)


def en_map(L: np.ndarray, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """One application of the clearing map ``Phi(p) = min(pbar, x + Pi^T p)``."""
    Pi, pbar = relative_liabilities(L)
    return np.minimum(pbar, np.asarray(x, dtype=np.float64) + Pi.T @ p)


def k_step_payments(L: np.ndarray, x: np.ndarray, K: int) -> np.ndarray:
    """``K`` Picard steps from ``p = pbar`` (numpy).  ``S_K = sum(pbar - p_K)`` increases to ``S`` as K grows."""
    Pi, pbar = relative_liabilities(L)
    x = np.asarray(x, dtype=np.float64)
    p = pbar.copy()
    for _ in range(K):
        p = np.minimum(pbar, x + Pi.T @ p)
    return p


# ----------------------------------------------------------------------------------------------------------
# torch: batched fixed-point iteration
# ----------------------------------------------------------------------------------------------------------
@dataclass
class PicardResult:
    p: torch.Tensor
    pbar: torch.Tensor
    iterations: int
    converged: bool


def picard_dense(
    L: torch.Tensor,
    x: torch.Tensor,
    start: str = "pbar",
    tol: float = 1e-10,
    max_iter: int = 100_000,
    n_iter: int | None = None,
    check_every: int = 10,
) -> PicardResult:
    """Batched Picard iteration on dense tensors ``L: (..., N, N)``, ``x: (..., N)``.

    ``start='pbar'`` converges from above to the greatest clearing vector, ``start='zero'`` from below
    (paper 2's code uses ``start='zero', n_iter=50``).  With ``n_iter`` given, exactly that many steps are run.
    """
    pbar = L.sum(-1)
    denom = torch.where(pbar > 0, pbar, torch.ones_like(pbar))
    PiT = (L / denom.unsqueeze(-1)).transpose(-1, -2)
    p = pbar.clone() if start == "pbar" else torch.zeros_like(pbar)
    if n_iter is not None:
        for _ in range(n_iter):
            p = torch.minimum(pbar, x + (PiT @ p.unsqueeze(-1)).squeeze(-1))
        return PicardResult(p, pbar, n_iter, False)
    scale = tol * (1.0 + float(pbar.max()) if pbar.numel() else 1.0)
    for it in range(1, max_iter + 1):
        p_new = torch.minimum(pbar, x + (PiT @ p.unsqueeze(-1)).squeeze(-1))
        if it % check_every == 0 and float((p_new - p).abs().max()) <= scale:
            return PicardResult(p_new, pbar, it, True)
        p = p_new
    return PicardResult(p, pbar, max_iter, False)


def nominal_from_edges(edge_index: torch.Tensor, L_vals: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(pbar, a)`` from an edge list."""
    src, dst = edge_index
    pbar = torch.zeros(num_nodes, dtype=L_vals.dtype, device=L_vals.device).index_add_(0, src, L_vals)
    a = torch.zeros(num_nodes, dtype=L_vals.dtype, device=L_vals.device).index_add_(0, dst, L_vals)
    return pbar, a


def picard_sparse(
    edge_index: torch.Tensor,
    L_vals: torch.Tensor,
    x: torch.Tensor,
    start: str = "pbar",
    tol: float = 1e-10,
    max_iter: int = 100_000,
    n_iter: int | None = None,
    check_every: int = 10,
) -> PicardResult:
    """Picard iteration on an edge list (works on a PyG mini-batch = disjoint union of graphs)."""
    src, dst = edge_index
    n = x.shape[0]
    pbar, _ = nominal_from_edges(edge_index, L_vals, n)
    pi_vals = L_vals / pbar[src]  # pbar[src] > 0 because src has an outgoing edge
    p = pbar.clone() if start == "pbar" else torch.zeros_like(pbar)

    def step(q: torch.Tensor) -> torch.Tensor:
        return torch.minimum(pbar, x.index_add(0, dst, pi_vals * q[src]))

    if n_iter is not None:
        for _ in range(n_iter):
            p = step(p)
        return PicardResult(p, pbar, n_iter, False)
    scale = tol * (1.0 + float(pbar.max()) if n else 1.0)
    for it in range(1, max_iter + 1):
        p_new = step(p)
        if it % check_every == 0 and float((p_new - p).abs().max()) <= scale:
            return PicardResult(p_new, pbar, it, True)
        p = p_new
    return PicardResult(p, pbar, max_iter, False)
