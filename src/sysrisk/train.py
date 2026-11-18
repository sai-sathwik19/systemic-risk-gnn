"""Training / inference loops for the GNNs (graph-level supervision on S / sum(L))."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .dataset import FEAT_PBAR


def _loss(out: dict[str, torch.Tensor], batch, y_var: float, node_aux: float) -> torch.Tensor:
    loss = ((out["S_rel"] - batch.y_rel) ** 2).mean() / y_var
    if node_aux > 0 and "r" in out:
        mask = batch.x[:, FEAT_PBAR] > 0
        loss = loss + node_aux * ((out["r"][mask] - batch.r[mask]) ** 2).mean()
    return loss


def train_gnn(model: torch.nn.Module, train_list: list[Data], val_list: list[Data], *, lr: float = 1e-3,
              epochs: int = 200, batch_size: int = 64, patience: int = 20, node_aux: float = 0.0,
              device: str = "cpu", verbose: bool = True, log_every: int = 10) -> dict:
    model.to(device)
    y = torch.tensor([float(d.y_rel) for d in train_list])
    y_var = max(float(y.var()), 1e-12)
    loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=256, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    hist: dict[str, list[float]] = {"train": [], "val": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            loss = _loss(model(batch), batch, y_var, node_aux)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * batch.num_graphs
            cnt += batch.num_graphs
        model.eval()
        with torch.no_grad():
            vt, vc = 0.0, 0
            for batch in val_loader:
                batch = batch.to(device)
                vt += float(((model(batch)["S_rel"] - batch.y_rel) ** 2).sum()) / y_var
                vc += batch.num_graphs
        tr, va = tot / cnt, vt / vc
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % log_every == 0 or ep == 1):
            print(f"    ep {ep:4d}  train {tr:.4f}  val {va:.4f}  best {best:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= patience:
            if verbose:
                print(f"    early stop at epoch {ep} (best val {best:.4f})", flush=True)
            break
    model.load_state_dict(best_state)
    hist["best_val"] = best
    hist["epochs_run"] = len(hist["train"])
    return hist


@torch.no_grad()
def predict_gnn(model: torch.nn.Module, data_list: list[Data], *, batch_size: int = 256, device: str = "cpu",
                T: int | None = None) -> dict[str, np.ndarray]:
    model.to(device).eval()
    S, r, u = [], [], []
    for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
        out = model(batch.to(device), T=T)
        S.append(out["S"].cpu().numpy())
        if "r" in out:
            r.append(out["r"].cpu().numpy())
            u.append(out["u"].cpu().numpy())
    res = {"S": np.concatenate(S).astype(np.float64)}
    if r:
        res["r"] = np.concatenate(r)
        res["u"] = np.concatenate(u)
    return res
