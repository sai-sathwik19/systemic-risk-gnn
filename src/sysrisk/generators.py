"""Synthetic interbank networks.

Defaults reproduce the generators of Gonon, Meyer-Brandis & Weber, "Computing Systemic Risk Measures with
Graph Neural Networks" (arXiv:2410.07222, Sec. 6.2; code: github.com/NiklasMWeber/ComputingSystemicRiskMeasures),
re-implemented here from the paper's specification and extended to arbitrary network size ``n``:

* ``er``  - directed Erdos-Renyi, p = 0.4, unit liabilities, assets 10 * Beta(2, 5) i.i.d.
* ``cp``  - core-periphery, 10% core; link probabilities [[.7, .3], [.3, .1]], liabilities [[10, 2], [2, 1]];
            Beta(2, 5) assets with Gaussian copula (rho = 0.5) scaled by 50 (core) / 10 (periphery);
            node order shuffled per sample.
* ``cpf`` - one fixed core-periphery liability matrix (drawn once from ``structure_seed``); only the
            assets are resampled.

Size modes (how parameters change when ``n`` differs from the reference size ``ref_n = 100``):

* ``fixed_p``      - keep link probabilities fixed (mean degree grows with n).  This is the mid-report design;
                     it changes density *and* stress level together with size.
* ``fixed_degree`` - keep the expected degree fixed (p = mean_degree / (n - 1); CP block probabilities scaled
                     by ref_n / n).  Local neighbourhoods look statistically the same at every n, so only the
                     global size changes.  This is the main size-generalisation mode.

At ``n = ref_n`` both modes coincide with the paper.  ``n`` may also be a list, in which case each sample
draws its size uniformly from the list (mixed-size training sets).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats

DEFAULTS: dict[str, dict[str, Any]] = {
    "er": {
        "topology": "er",
        "n": 100,
        "ref_n": 100,
        "size_mode": "fixed_p",
        "p": 0.4,
        "mean_degree": None,        # default p_ref * (ref_n - 1) = 39.6
        "weights": "unit",          # 'unit' (paper) | 'lognormal' (weighted variant)
        "weight_sigma": 1.0,        # lognormal sigma, mean of weights is 1
        "asset_factor": 10.0,
        "asset_rho": 0.0,
        "asset_multiplier": 1.0,    # extra stress knob: x <- multiplier * x
    },
    "cp": {
        "topology": "cp",
        "n": 100,
        "ref_n": 100,
        "size_mode": "fixed_p",
        "core_frac": 0.1,
        "probs": [[0.7, 0.3], [0.3, 0.1]],
        "liabs": [[10.0, 2.0], [2.0, 1.0]],
        "core_asset_factor": 50.0,
        "periphery_asset_factor": 10.0,
        "asset_rho": 0.5,
        "asset_multiplier": 1.0,
        "shuffle": True,
    },
}
DEFAULTS["cpf"] = {**DEFAULTS["cp"], "topology": "cpf", "shuffle": False, "structure_seed": 0}


def make_config(spec: dict[str, Any] | str) -> dict[str, Any]:
    """Merge a (partial) generator spec with the defaults of its topology."""
    if isinstance(spec, str):
        spec = {"topology": spec}
    topo = spec.get("topology", "er")
    if topo not in DEFAULTS:
        raise ValueError(f"unknown topology {topo!r}; choose from {list(DEFAULTS)}")
    cfg = {**DEFAULTS[topo], **spec}
    return cfg


# ----------------------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------------------
def copula_beta_assets(rng: np.random.Generator, n: int, rho: float = 0.0, a: float = 2.0, b: float = 5.0) -> np.ndarray:
    """Beta(a, b) marginals joined by an equicorrelated Gaussian copula (one-factor form, O(n))."""
    eps = rng.standard_normal(n)
    z = np.sqrt(rho) * rng.standard_normal() + np.sqrt(1.0 - rho) * eps if rho > 0 else eps
    u = np.clip(stats.norm.cdf(z), 1e-12, 1.0 - 1e-12)
    return stats.beta.ppf(u, a, b)


def er_edge_prob(cfg: dict[str, Any], n: int) -> float:
    if cfg["size_mode"] == "fixed_degree":
        k = cfg["mean_degree"] if cfg.get("mean_degree") is not None else cfg["p"] * (cfg["ref_n"] - 1)
        p = k / (n - 1)
    elif cfg["size_mode"] == "fixed_p":
        p = cfg["p"]
    else:
        raise ValueError(f"unknown size_mode {cfg['size_mode']!r}")
    if p > 1.0:
        warnings.warn(f"ER edge probability {p:.3f} > 1 at n={n}; capped at 1", stacklevel=2)
        p = 1.0
    return p


def cp_prob_scale(cfg: dict[str, Any], n: int) -> float:
    if cfg["size_mode"] == "fixed_degree":
        return cfg["ref_n"] / n
    if cfg["size_mode"] == "fixed_p":
        return 1.0
    raise ValueError(f"unknown size_mode {cfg['size_mode']!r}")


def cp_structure(rng: np.random.Generator, cfg: dict[str, Any], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Core-periphery liability matrix (unshuffled: core banks first) and group labels (0 = core)."""
    n_core = max(1, int(round(cfg["core_frac"] * n)))
    g = np.ones(n, dtype=np.int64)
    g[:n_core] = 0
    P = np.asarray(cfg["probs"], dtype=np.float64)[g[:, None], g[None, :]] * cp_prob_scale(cfg, n)
    P = np.minimum(P, 1.0)
    adj = rng.random((n, n)) < P
    np.fill_diagonal(adj, False)
    L = adj * np.asarray(cfg["liabs"], dtype=np.float64)[g[:, None], g[None, :]]
    return L, g


# ----------------------------------------------------------------------------------------------------------
# samplers
# ----------------------------------------------------------------------------------------------------------
def sample_er(rng: np.random.Generator, cfg: dict[str, Any], n: int):
    p = er_edge_prob(cfg, n)
    adj = rng.random((n, n)) < p
    np.fill_diagonal(adj, False)
    if cfg["weights"] == "unit":
        L = adj.astype(np.float64)
    elif cfg["weights"] == "lognormal":
        s = float(cfg["weight_sigma"])
        L = adj * rng.lognormal(mean=-0.5 * s * s, sigma=s, size=(n, n))
    else:
        raise ValueError(f"unknown weights {cfg['weights']!r}")
    x = cfg["asset_factor"] * copula_beta_assets(rng, n, cfg["asset_rho"])
    return L, x * cfg["asset_multiplier"], {}


def _cp_assets(rng: np.random.Generator, cfg: dict[str, Any], g: np.ndarray) -> np.ndarray:
    factors = np.where(g == 0, cfg["core_asset_factor"], cfg["periphery_asset_factor"])
    return factors * copula_beta_assets(rng, len(g), cfg["asset_rho"]) * cfg["asset_multiplier"]


def sample_cp(rng: np.random.Generator, cfg: dict[str, Any], n: int):
    L, g = cp_structure(rng, cfg, n)
    x = _cp_assets(rng, cfg, g)
    if cfg.get("shuffle", True):
        perm = rng.permutation(n)
        L, x, g = L[np.ix_(perm, perm)], x[perm], g[perm]
    return L, x, {"group": g}


_CPF_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def sample_cpf(rng: np.random.Generator, cfg: dict[str, Any], n: int):
    key = (n, cfg["structure_seed"], cfg["core_frac"], cfg["size_mode"], str(cfg["probs"]), str(cfg["liabs"]))
    if key not in _CPF_CACHE:
        _CPF_CACHE[key] = cp_structure(np.random.default_rng(cfg["structure_seed"]), cfg, n)
    L, g = _CPF_CACHE[key]
    x = _cp_assets(rng, cfg, g)
    return L, x, {"group": g}


SAMPLERS = {"er": sample_er, "cp": sample_cp, "cpf": sample_cpf}


def sample_network(spec: dict[str, Any] | str, rng: np.random.Generator):
    """Draw one network ``(L, x, meta)`` from a generator spec (see module docstring)."""
    cfg = make_config(spec)
    n = cfg["n"]
    if isinstance(n, (list, tuple)):
        n = int(rng.choice(n))
    return SAMPLERS[cfg["topology"]](rng, cfg, int(n))
