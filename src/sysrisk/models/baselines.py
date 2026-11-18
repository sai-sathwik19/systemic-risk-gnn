"""Classical baselines.

* :class:`KStepHeuristic` - no learning: run ``K`` rounds of Eisenberg-Noe contagion starting from full
  payment.  ``K = 1`` is the classical first-round ("fundamental default") shortfall
  ``sum_i max(0, pbar_i - x_i - a_i)``.  ``S_K`` is a lower bound that increases monotonically to ``S``.
* :class:`FeatureRegressor` - linear regression / MLP / gradient-boosted trees on fixed-dimensional graph
  features (:mod:`sysrisk.features`).  By default the regressors predict the scale-free target
  ``S / sum(L)`` and are rescaled by ``sum(L)`` - the same normalisation the GNNs use.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from ..clearing import picard_sparse
from ..dataset import FEAT_X


class KStepHeuristic:
    kind = "graph"

    def __init__(self, K: int = 1):
        self.K = int(K)

    def fit(self, *args, **kwargs) -> "KStepHeuristic":
        return self

    @torch.no_grad()
    def predict(self, data_list: list[Data], batch_size: int = 512, device: str = "cpu") -> dict[str, np.ndarray]:
        S, r, u = [], [], []
        for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            res = picard_sparse(batch.edge_index, batch.edge_attr[:, 0].double(), batch.x[:, FEAT_X].double(),
                                start="pbar", n_iter=self.K)
            uu = res.pbar - res.p
            S.append(global_add_pool(uu, batch.batch).cpu().numpy())
            rr = torch.where(res.pbar > 0, res.p / torch.where(res.pbar > 0, res.pbar, torch.ones_like(res.pbar)),
                             torch.ones_like(res.p))
            r.append(rr.cpu().numpy())
            u.append(uu.cpu().numpy())
        return {"S": np.concatenate(S), "r": np.concatenate(r), "u": np.concatenate(u)}


class FeatureRegressor:
    kind = "feature"

    def __init__(self, kind: str = "linreg", feature_set: str = "B", target: str = "S_rel", seed: int = 0):
        self.reg_kind = kind
        self.feature_set = feature_set
        self.target = target
        self.seed = seed
        self.model = self._make()

    def _make(self) -> TransformedTargetRegressor:
        if self.reg_kind == "linreg":
            reg = LinearRegression()
        elif self.reg_kind == "mlp":
            reg = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", learning_rate_init=1e-3,
                               max_iter=2000, early_stopping=True, validation_fraction=0.1,
                               n_iter_no_change=30, random_state=self.seed)
        elif self.reg_kind == "gbt":
            reg = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, early_stopping=True,
                                                validation_fraction=0.1, n_iter_no_change=50,
                                                random_state=self.seed)
        else:
            raise ValueError(f"unknown regressor {self.reg_kind!r}")
        pipe = Pipeline([("scale", StandardScaler()), ("reg", reg)])
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())

    def fit(self, X: np.ndarray, S: np.ndarray, L_total: np.ndarray) -> "FeatureRegressor":
        y = S / L_total if self.target == "S_rel" else S
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray, L_total: np.ndarray) -> dict[str, np.ndarray]:
        y = self.model.predict(X)
        return {"S": y * L_total if self.target == "S_rel" else y}
