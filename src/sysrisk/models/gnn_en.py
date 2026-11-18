"""G2 - EN-aligned recurrent GNN (this project's proposed architecture).

Motivation: one Eisenberg-Noe step written in payment ratios ``r_i = p_i / pbar_i`` is

    r_i  <-  min(1,  x_i / pbar_i  +  sum_{j -> i} w_ji r_j),       w_ji = L_ji / pbar_i,

i.e. a *sum* aggregation over debtors ``j`` with edge weight ``w_ji`` followed by a node-wise nonlinearity.
The update depends only on scale-free local quantities, so it is the same at every network size, and the
number of contagion rounds (not the number of layers) sets how far defaults travel.  The architecture
mirrors this:

* inputs ``u_i = [log1p(x_i / pbar_i), log1p(a_i / pbar_i)]`` (exactly invariant to rescaling all money),
* ``h^0 = enc(u)``, then ``T`` *weight-tied* rounds ``m_i = sum_j w_ji msg(h_j)``, ``h <- GRU([enc(u), m], h)``,
* node readout ``r_hat_i = sigmoid(out(h_i^T))`` in [0, 1],
* graph readout (weighted sum pooling, mirrors the definition of S):  ``S_hat = sum_i pbar_i (1 - r_hat_i)``.

``T`` is drawn from ``[T_min, T_max]`` per training batch and can be increased at test time.  Ablation flags:
``aggr='mean'``, ``inputs='raw'``, ``tied=False``.  Training uses only the graph label S (optionally plus a
node-level auxiliary loss on ``r``, controlled in the trainer).
"""
from __future__ import annotations

import random

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool

from ..dataset import FEAT_A, FEAT_PBAR, FEAT_X


def mlp(d_in: int, d_hidden: int, d_out: int, last_act: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out)]
    if last_act is not None:
        layers.append(last_act)
    return nn.Sequential(*layers)


class WeightedConv(MessagePassing):
    """``m_i = AGG_{j -> i} w_ji * z_j`` with ``z = msg(h)`` computed per node (cheap)."""

    def __init__(self, hidden: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.msg = mlp(hidden, hidden, hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, z=self.msg(h), w=w)

    def message(self, z_j: torch.Tensor, w: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return w.unsqueeze(-1) * z_j


class ENGNN(nn.Module):
    def __init__(self, hidden: int = 64, T_min: int = 8, T_max: int = 16, T_eval: int = 16, tied: bool = True,
                 aggr: str = "add", inputs: str = "ratio"):
        super().__init__()
        self.T_min, self.T_max, self.T_eval = T_min, T_max, T_eval
        self.tied = tied
        self.inputs = inputs
        n_cells = 1 if tied else T_max
        in_dim = 2 if inputs == "ratio" else 3
        self.enc = mlp(in_dim, hidden, hidden, nn.Tanh())
        self.convs = nn.ModuleList(WeightedConv(hidden, aggr) for _ in range(n_cells))
        self.cells = nn.ModuleList(nn.GRUCell(2 * hidden, hidden) for _ in range(n_cells))
        self.out = mlp(hidden, hidden, 1)
        self.register_buffer("money_scale", torch.tensor(1.0))
        self.register_buffer("edge_scale", torch.tensor(1.0))

    @torch.no_grad()
    def fit_normalizer(self, data_list: list[Data]) -> None:
        """Input scales (used with inputs='raw') and output-bias init from the *graph* labels only:
        start every bank at r = 1 - mean(S / sum L), so the initial prediction equals the mean target."""
        pbar = torch.cat([d.x[:, FEAT_PBAR] for d in data_list])
        L = torch.cat([d.edge_attr[:, 0] for d in data_list])
        self.money_scale.fill_(float(pbar.mean()))
        self.edge_scale.fill_(float(L.mean()))
        if all(hasattr(d, "y_rel") for d in data_list):
            r0 = 1.0 - float(torch.cat([d.y_rel for d in data_list]).mean())
            r0 = min(max(r0, 1e-3), 1 - 1e-3)
            self.out[-1].bias.fill_(float(torch.logit(torch.tensor(r0))))

    def _inputs(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        xe, a, pbar = data.x[:, FEAT_X], data.x[:, FEAT_A], data.x[:, FEAT_PBAR]
        src, dst = data.edge_index
        L = data.edge_attr[:, 0]
        if self.inputs == "ratio":
            pos = pbar > 0
            d = torch.where(pos, pbar, torch.ones_like(pbar))
            u = torch.stack([torch.log1p(xe / d), torch.log1p(a / d)], dim=-1)
            u = torch.where(pos.unsqueeze(-1), u, torch.zeros_like(u))
            w = torch.where(pos[dst], L / d[dst], torch.zeros_like(L))
        elif self.inputs == "raw":
            u = data.x / self.money_scale
            w = L / self.edge_scale
        else:
            raise ValueError(f"unknown inputs {self.inputs!r}")
        return u, w

    def forward(self, data: Data, T: int | None = None) -> dict[str, torch.Tensor]:
        batch = data.batch if getattr(data, "batch", None) is not None else \
            torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        if T is None:
            if not self.tied:
                T = self.T_max
            elif self.training:
                T = random.randint(self.T_min, self.T_max)
            else:
                T = self.T_eval
        u, w = self._inputs(data)
        e = self.enc(u)
        h = e
        for t in range(T):
            k = 0 if self.tied else min(t, len(self.cells) - 1)
            m = self.convs[k](h, data.edge_index, w)
            h = self.cells[k](torch.cat([e, m], dim=-1), h)
        pbar = data.x[:, FEAT_PBAR]
        r = torch.sigmoid(self.out(h).squeeze(-1))
        r = torch.where(pbar > 0, r, torch.ones_like(r))
        u_hat = pbar * (1.0 - r)
        S = global_add_pool(u_hat, batch)
        L_total = global_add_pool(pbar, batch)
        return {"S": S, "S_rel": S / L_total, "r": r, "u": u_hat}
