"""Fixed-dimensional graph-level features for the classical baselines.

* Set ``A`` - the 8 features of the mid-report: N, |E|, density, A_total, L_total, A_mean, L_mean, L_max.
* Set ``B`` - extended set with distributional, stress (first-round default) and network-centrality features.

All features are computed from the *inputs* (L, x) only - never from clearing payments - so there is no
target leakage.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from torch_geometric.data import Data

from .dataset import FEAT_A, FEAT_PBAR, FEAT_X

FEATURES_A = ["N", "n_edges", "density", "x_sum", "pbar_sum", "x_mean", "pbar_mean", "pbar_max"]
FEATURES_B = [
    "N", "n_edges", "density", "indeg_std", "outdeg_std",
    "x_sum", "x_mean", "x_std", "x_max",
    "pbar_sum", "pbar_mean", "pbar_std", "pbar_max",
    "a_std", "a_max",
    "buffer_ratio",          # sum(x) / sum(L)
    "e_min", "e_mean", "e_std", "frac_insolvent",   # net worth e = x + a - pbar (before contagion)
    "S1", "S1_rel",          # first-round shortfall sum(max(0, -e)) and its share of sum(L)
    "lev_mean",              # mean pbar / (x + a)
    "pr_max", "pr_gini",     # PageRank on the payment-flow graph (weights Pi)
]
FEATURE_SETS = {"A": FEATURES_A, "B": FEATURES_B}


def gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, dtype=np.float64))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def pagerank_flow(src: np.ndarray, dst: np.ndarray, pi_vals: np.ndarray, n: int, damping: float = 0.85,
                  iters: int = 100, tol: float = 1e-10) -> np.ndarray:
    """PageRank of the random walk that follows payments (debtor -> creditor with probability Pi)."""
    M = sparse.csr_matrix((pi_vals, (dst, src)), shape=(n, n))  # column-stochastic for non-dangling nodes
    dangling = np.ones(n, dtype=bool)
    dangling[src] = False
    v = np.full(n, 1.0 / n)
    for _ in range(iters):
        v_new = damping * (M @ v + v[dangling].sum() / n) + (1.0 - damping) / n
        if np.abs(v_new - v).sum() < tol:
            v = v_new
            break
        v = v_new
    return v / v.sum()


def graph_features(data: Data, feature_set: str = "B") -> np.ndarray:
    X = data.x.double().numpy()
    xe, a, pbar = X[:, FEAT_X], X[:, FEAT_A], X[:, FEAT_PBAR]
    src, dst = data.edge_index.numpy()
    Lv = data.edge_attr[:, 0].double().numpy()
    n = int(data.num_nodes)
    m = int(src.size)
    f: dict[str, float] = {
        "N": n,
        "n_edges": m,
        "density": m / (n * (n - 1)) if n > 1 else 0.0,
        "x_sum": xe.sum(),
        "pbar_sum": pbar.sum(),
        "x_mean": xe.mean(),
        "pbar_mean": pbar.mean(),
        "pbar_max": pbar.max(),
    }
    if feature_set == "B":
        outdeg = np.bincount(src, minlength=n)
        indeg = np.bincount(dst, minlength=n)
        e = xe + a - pbar
        L_tot = pbar.sum()
        S1 = np.maximum(0.0, -e).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            lev = np.where(xe + a > 0, pbar / (xe + a), 0.0)
        pi_vals = Lv / pbar[src] if m else Lv
        pr = pagerank_flow(src, dst, pi_vals, n) if m else np.full(n, 1.0 / n)
        f.update({
            "indeg_std": indeg.std(), "outdeg_std": outdeg.std(),
            "x_std": xe.std(), "x_max": xe.max(),
            "pbar_std": pbar.std(),
            "a_std": a.std(), "a_max": a.max(),
            "buffer_ratio": xe.sum() / L_tot if L_tot > 0 else 0.0,
            "e_min": e.min(), "e_mean": e.mean(), "e_std": e.std(),
            "frac_insolvent": float((e < 0).mean()),
            "S1": S1, "S1_rel": S1 / L_tot if L_tot > 0 else 0.0,
            "lev_mean": lev.mean(),
            "pr_max": pr.max(), "pr_gini": gini(pr),
        })
    return np.array([f[k] for k in FEATURE_SETS[feature_set]], dtype=np.float64)


def dataset_features(data_list: list[Data], feature_set: str = "B") -> np.ndarray:
    return np.stack([graph_features(d, feature_set) for d in data_list])
