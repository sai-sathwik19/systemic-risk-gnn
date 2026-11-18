"""G1 - port of the GNN of Gonon, Meyer-Brandis & Weber (arXiv:2410.07222, Def. 5.4/5.5 and their code).

Original (DGL): ``SAGEConv(aggr='mean')`` layers with the raw liability ``L_ji`` as edge weight, messages
flowing debtor -> creditor, sigmoid activations, width 10, input ``[assets, incoming, outgoing liabilities]``
padded with 7 constant features, and a per-node linear readout on ``concat(inputs, h)``.

PyG equivalent: ``GraphConv(aggr='mean')`` computes ``W_root h_i + W_rel mean_j (L_ji h_j) + b`` - the same
function as DGL's weighted mean ``SAGEConv``.

Change for this project: the paper turns node scores into a bailout allocation (softmax).  Here the node
scores ``s_i`` are summed (``global_add_pool``) into the graph-level prediction ``S_hat = c * sum_i s_i``.
Normalisation is scale-only (no additive offset), so the readout stays additive across network sizes.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_add_pool

from ..dataset import FEAT_PBAR


class PaperGNN(nn.Module):
    def __init__(self, n_layers: int = 3, hidden: int = 10, out_dim: int = 7, pad: int = 7,
                 activation: str = "sigmoid", aggr: str = "mean"):
        super().__init__()
        self.pad = pad
        dims = [3 + pad] + [hidden] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(GraphConv(dims[i], dims[i + 1], aggr=aggr) for i in range(n_layers))
        self.readout = nn.Linear(out_dim + 3, 1)
        self.act = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        self.register_buffer("money_scale", torch.tensor(1.0))
        self.register_buffer("edge_scale", torch.tensor(1.0))

    @torch.no_grad()
    def fit_normalizer(self, data_list: list[Data]) -> None:
        pbar = torch.cat([d.x[:, FEAT_PBAR] for d in data_list])
        L = torch.cat([d.edge_attr[:, 0] for d in data_list])
        self.money_scale.fill_(float(pbar.mean()))
        self.edge_scale.fill_(float(L.mean()))
        if all(hasattr(d, "y") for d in data_list):  # init: node score = mean per-bank shortfall (graph labels only)
            per_bank = float(torch.cat([d.y for d in data_list]).sum()) / float(pbar.numel())
            self.readout.bias.fill_(per_bank / float(self.money_scale))

    def forward(self, data: Data, T: int | None = None) -> dict[str, torch.Tensor]:
        batch = data.batch if getattr(data, "batch", None) is not None else \
            torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        feats = data.x / self.money_scale
        h = torch.cat([feats, torch.ones(feats.shape[0], self.pad, device=feats.device)], dim=-1)
        ew = data.edge_attr[:, 0] / self.edge_scale
        for conv in self.convs[:-1]:
            h = self.act(conv(h, data.edge_index, ew))
        h = self.convs[-1](h, data.edge_index, ew)
        s = self.readout(torch.cat([feats, h], dim=-1)).squeeze(-1)
        S = global_add_pool(s, batch) * self.money_scale
        L_total = global_add_pool(data.x[:, FEAT_PBAR], batch)
        return {"S": S, "S_rel": S / L_total, "node": s}
