"""Evaluation metrics.

Graph level (target S = total unpaid liabilities): MAE / RMSE / R^2 on S, the same on the scale-free
``S_rel = S / sum(L)`` (comparable across network sizes), bias, and MAPE on graphs with S > 0.
Stratified: error by stress bucket (S = 0, then terciles of S_rel among stressed graphs).
Node level (models that output payment ratios): default AUROC and per-graph Spearman rank correlation of
node shortfalls u_i (systemic-risk ranking quality).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _r2(t: np.ndarray, p: np.ndarray) -> float:
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / ss_tot if ss_tot > 0 else math.nan


def graph_metrics(S: np.ndarray, S_hat: np.ndarray, L_total: np.ndarray) -> dict[str, float]:
    S, S_hat, L = (np.asarray(v, dtype=np.float64) for v in (S, S_hat, L_total))
    err = S_hat - S
    y, yh = S / L, S_hat / L
    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": _r2(S, S_hat),
        "bias": float(err.mean()),
        "mae_rel": float(np.abs(yh - y).mean()),
        "rmse_rel": float(np.sqrt(((yh - y) ** 2).mean())),
        "r2_rel": _r2(y, yh),
        "S_mean": float(S.mean()),
        "S_rel_mean": float(y.mean()),
    }
    nz = S > 1e-6 * L
    out["mape_nonzero"] = float((np.abs(err[nz]) / S[nz]).mean()) if nz.any() else math.nan
    return out


def stratified_metrics(S: np.ndarray, S_hat: np.ndarray, L_total: np.ndarray) -> dict[str, dict[str, float]]:
    S, S_hat, L = (np.asarray(v, dtype=np.float64) for v in (S, S_hat, L_total))
    y, yh = S / L, S_hat / L
    zero = S <= 1e-9 * L
    buckets = {"zero": zero}
    if (~zero).sum() >= 3:
        q1, q2 = np.quantile(y[~zero], [1 / 3, 2 / 3])
        buckets["low"] = (~zero) & (y <= q1)
        buckets["mid"] = (~zero) & (y > q1) & (y <= q2)
        buckets["high"] = (~zero) & (y > q2)
    out = {}
    for k, m in buckets.items():
        if m.any():
            out[k] = {"count": int(m.sum()), "mae_rel": float(np.abs(yh[m] - y[m]).mean()),
                      "bias_rel": float((yh[m] - y[m]).mean())}
    return out


def node_metrics(r: np.ndarray, r_hat: np.ndarray, u: np.ndarray, u_hat: np.ndarray, graph_idx: np.ndarray,
                 pbar: np.ndarray) -> dict[str, float]:
    mask = pbar > 0
    default = r < 1.0 - 1e-7
    out: dict[str, float] = {"r_mae": float(np.abs(r_hat[mask] - r[mask]).mean())}
    d, score = default[mask], (1.0 - r_hat)[mask]
    out["default_auroc"] = float(roc_auc_score(d, score)) if 0 < d.sum() < d.size else math.nan
    rhos = []
    for g in np.unique(graph_idx):
        m = graph_idx == g
        if np.ptp(u[m]) > 1e-9 and np.ptp(u_hat[m]) > 1e-12:
            rho = spearmanr(u[m], u_hat[m]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
    out["spearman_u"] = float(np.mean(rhos)) if rhos else math.nan
    return out
