"""Labelled PyG datasets of interbank networks.

Each graph is a :class:`torch_geometric.data.Data` with

* ``x``          [N, 3]  raw node inputs ``[external assets, interbank assets a, nominal liabilities pbar]``
* ``edge_index`` [2, E]  debtor -> creditor (PyG source -> target), ``edge_attr`` [E, 1] = ``L_ij``
* node labels    ``p`` (clearing payments), ``r = p / pbar``, ``u = pbar - p``, ``default`` (bool)
* graph labels   ``y`` = S (total unpaid liabilities), ``y_rel = S / sum(L)``, ``L_total``, ``n_defaults``,
                 ``fda_rounds`` (contagion depth), ``n`` (number of banks)

Labels come from the exact fictitious-default algorithm (:func:`sysrisk.clearing.clearing_fda`).
Generation is deterministic in ``(generator spec, n_graphs, seed)``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .clearing import clearing_fda
from .generators import make_config, sample_network

FEAT_X, FEAT_A, FEAT_PBAR = 0, 1, 2


def network_to_data(L: np.ndarray, x: np.ndarray, meta: dict | None = None, solve: bool = True) -> Data:
    L = np.asarray(L, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    src, dst = np.nonzero(L)
    pbar = L.sum(axis=1)
    a = L.sum(axis=0)
    data = Data(
        x=torch.tensor(np.stack([x, a, pbar], axis=1), dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(L[src, dst], dtype=torch.float32).unsqueeze(-1),
        num_nodes=n,
    )
    if solve:
        res = clearing_fda(L, x)
        L_total = float(pbar.sum())
        data.p = torch.tensor(res.p, dtype=torch.float32)
        data.r = torch.tensor(res.r, dtype=torch.float32)
        data.u = torch.tensor(res.u, dtype=torch.float32)
        data.default = torch.tensor(res.default, dtype=torch.bool)
        data.y = torch.tensor([res.S], dtype=torch.float32)
        data.y_rel = torch.tensor([res.S / L_total if L_total > 0 else 0.0], dtype=torch.float32)
        data.L_total = torch.tensor([L_total], dtype=torch.float32)
        data.n_defaults = torch.tensor([int(res.default.sum())], dtype=torch.long)
        data.fda_rounds = torch.tensor([res.rounds], dtype=torch.long)
    data.n = torch.tensor([n], dtype=torch.long)
    if meta and "group" in meta:
        data.group = torch.tensor(meta["group"], dtype=torch.long)
    return data


def data_to_dense(data: Data) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(L, x)`` as numpy arrays from a single graph."""
    n = data.num_nodes
    L = np.zeros((n, n))
    src, dst = data.edge_index.numpy()
    L[src, dst] = data.edge_attr[:, 0].double().numpy()
    return L, data.x[:, FEAT_X].double().numpy()


def generate_dataset(spec: dict[str, Any] | str, n_graphs: int, seed: int, verbose: bool = False) -> list[Data]:
    """Generate ``n_graphs`` labelled graphs; graph ``i`` uses ``SeedSequence(seed).spawn(n_graphs)[i]``."""
    cfg = make_config(spec)
    children = np.random.SeedSequence(seed).spawn(n_graphs)
    out: list[Data] = []
    t0 = time.time()
    for i, child in enumerate(children):
        L, x, meta = sample_network(cfg, np.random.default_rng(child))
        out.append(network_to_data(L, x, meta))
        if verbose and (i + 1) % max(1, n_graphs // 10) == 0:
            print(f"  generated {i + 1}/{n_graphs} graphs ({time.time() - t0:.1f}s)", flush=True)
    return out


def spec_key(spec: dict[str, Any], n_graphs: int, seed: int) -> str:
    blob = json.dumps({"gen": make_config(spec), "n_graphs": n_graphs, "seed": seed}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def default_data_dir() -> Path:
    return Path(os.environ.get("SYSRISK_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def load_or_generate(
    spec: dict[str, Any] | str,
    n_graphs: int,
    seed: int,
    name: str = "dataset",
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> list[Data]:
    """Load a cached dataset or generate (and cache) it."""
    cache_dir = Path(cache_dir) if cache_dir is not None else default_data_dir()
    path = cache_dir / f"{name}_{spec_key(make_config(spec), n_graphs, seed)}.pt"
    if use_cache and path.exists():
        if verbose:
            print(f"[data] loading {path.name}", flush=True)
        return torch.load(path, weights_only=False)
    if verbose:
        print(f"[data] generating {name}: {n_graphs} graphs, spec={spec}", flush=True)
    t0 = time.time()
    data_list = generate_dataset(spec, n_graphs, seed, verbose=verbose)
    if verbose:
        print(f"[data] done in {time.time() - t0:.1f}s", flush=True)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, path)
    return data_list


def summarize(data_list: list[Data]) -> dict[str, float]:
    """Descriptive statistics of a labelled dataset (for sanity checks and the dataset card)."""
    S = np.array([float(d.y) for d in data_list])
    S_rel = np.array([float(d.y_rel) for d in data_list])
    n = np.array([int(d.n) for d in data_list])
    E = np.array([d.edge_index.shape[1] for d in data_list])
    nd = np.array([int(d.n_defaults) for d in data_list])
    rounds = np.array([int(d.fda_rounds) for d in data_list])
    return {
        "n_graphs": len(data_list),
        "n_mean": float(n.mean()),
        "edges_mean": float(E.mean()),
        "mean_degree": float((E / n).mean()),
        "S_mean": float(S.mean()),
        "S_std": float(S.std()),
        "S_rel_mean": float(S_rel.mean()),
        "S_zero_share": float((S <= 1e-9).mean()),
        "defaults_mean": float(nd.mean()),
        "default_share_mean": float((nd / n).mean()),
        "fda_rounds_mean": float(rounds.mean()),
        "fda_rounds_max": int(rounds.max()),
    }
