"""Model zoo: name -> spec.  Experiment configs refer to models by these names (overrides allowed)."""
from __future__ import annotations

import copy
from typing import Any

from .baselines import FeatureRegressor, KStepHeuristic
from .gnn_en import ENGNN
from .gnn_paper import PaperGNN

MODEL_ZOO: dict[str, dict[str, Any]] = {
    # non-learned classical heuristic (K rounds of EN contagion from full payment)
    "kstep1": {"type": "kstep", "K": 1},
    "kstep3": {"type": "kstep", "K": 3},
    # regressors on graph-level features
    "linreg_A": {"type": "feature", "kind": "linreg", "feature_set": "A"},
    "linreg_B": {"type": "feature", "kind": "linreg", "feature_set": "B"},
    "mlp_A": {"type": "feature", "kind": "mlp", "feature_set": "A"},
    "mlp_B": {"type": "feature", "kind": "mlp", "feature_set": "B"},
    "gbt_B": {"type": "feature", "kind": "gbt", "feature_set": "B"},
    # G1: paper-2 GNN port (code config: 3 layers) and paper-text config (5 layers)
    "g1_paper": {"type": "paper_gnn", "n_layers": 3, "hidden": 10, "out_dim": 7, "pad": 7, "activation": "sigmoid"},
    "g1_paper5": {"type": "paper_gnn", "n_layers": 5, "hidden": 10, "out_dim": 7, "pad": 7, "activation": "sigmoid"},
    # G2: EN-aligned recurrent GNN and its ablations
    "g2_en": {"type": "en_gnn", "hidden": 64, "T_min": 8, "T_max": 16, "T_eval": 16},
    "g2_en_mean": {"type": "en_gnn", "hidden": 64, "T_min": 8, "T_max": 16, "T_eval": 16, "aggr": "mean"},
    "g2_en_raw": {"type": "en_gnn", "hidden": 64, "T_min": 8, "T_max": 16, "T_eval": 16, "inputs": "raw"},
    "g2_en_untied": {"type": "en_gnn", "hidden": 64, "T_min": 16, "T_max": 16, "T_eval": 16, "tied": False},
    "g2_en_nodeaux": {"type": "en_gnn", "hidden": 64, "T_min": 8, "T_max": 16, "T_eval": 16, "node_aux": 1.0},
}

# keys that belong to the trainer rather than the model constructor
TRAINER_KEYS = {"type", "node_aux", "lr", "epochs", "batch_size", "patience", "eval_T"}


def model_spec(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in MODEL_ZOO:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_ZOO)}")
    spec = copy.deepcopy(MODEL_ZOO[name])
    spec.update(overrides or {})
    return spec


def build_model(spec: dict[str, Any], seed: int = 0):
    kw = {k: v for k, v in spec.items() if k not in TRAINER_KEYS}
    t = spec["type"]
    if t == "kstep":
        return KStepHeuristic(**kw)
    if t == "feature":
        return FeatureRegressor(seed=seed, **kw)
    if t == "paper_gnn":
        return PaperGNN(**kw)
    if t == "en_gnn":
        return ENGNN(**kw)
    raise ValueError(f"unknown model type {t!r}")


__all__ = ["MODEL_ZOO", "model_spec", "build_model", "KStepHeuristic", "FeatureRegressor", "PaperGNN", "ENGNN"]
