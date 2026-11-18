"""Config-driven experiment runner: build datasets -> train every model (several seeds) -> evaluate on every
test set -> write a JSON with all metrics and print a summary table.

Config (YAML) layout::

    name: e1_er
    data:
      train: {gen: {topology: er}, n_graphs: 10000, seed: 101}
      val:   {gen: {topology: er}, n_graphs: 2500,  seed: 102}
      test:
        er_n100: {gen: {topology: er}, n_graphs: 2500, seed: 103}
    models: [kstep1, linreg_A, gbt_B, g1_paper, g2_en]
    model_overrides: {g2_en: {eval_T: [16, 32]}}      # optional
    seeds: [0, 1, 2]
    train: {epochs: 200, batch_size: 64, lr: 0.001, patience: 20}
"""
from __future__ import annotations

import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import FEAT_PBAR, load_or_generate, summarize
from .features import dataset_features
from .metrics import graph_metrics, node_metrics, stratified_metrics
from .models import build_model, model_spec
from .train import predict_gnn, train_gnn
from .utils import set_seed

DETERMINISTIC = {"kstep"}


def quick_config(cfg: dict[str, Any], shrink: int = 50) -> dict[str, Any]:
    """Shrink an experiment for a fast end-to-end smoke test."""
    cfg = copy.deepcopy(cfg)
    cfg["name"] = cfg["name"] + "_quick"
    for split in ("train", "val"):
        cfg["data"][split]["n_graphs"] = max(20, cfg["data"][split]["n_graphs"] // shrink)
    for t in cfg["data"]["test"].values():
        t["n_graphs"] = max(10, t["n_graphs"] // shrink)
    cfg["seeds"] = cfg.get("seeds", [0])[:1]
    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = 2
    return cfg


def _labels(data_list) -> dict[str, np.ndarray]:
    return {
        "S": np.array([float(d.y) for d in data_list]),
        "L_total": np.array([float(d.L_total) for d in data_list]),
    }


def _node_arrays(data_list) -> dict[str, np.ndarray]:
    return {
        "r": np.concatenate([d.r.numpy() for d in data_list]),
        "u": np.concatenate([d.u.numpy() for d in data_list]),
        "pbar": np.concatenate([d.x[:, FEAT_PBAR].numpy() for d in data_list]),
        "graph_idx": np.concatenate([np.full(d.num_nodes, i) for i, d in enumerate(data_list)]),
    }


def run_experiment(cfg: dict[str, Any], out_dir: str | Path = "results/runs", device: str = "cpu",
                   use_cache: bool = True, save_checkpoints: bool = True) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    dcfg = cfg["data"]
    tcfg = {"epochs": 200, "batch_size": 64, "lr": 1e-3, "patience": 20, **cfg.get("train", {})}
    seeds = cfg.get("seeds", [0])
    print(f"=== experiment {name}  (device={device}) ===", flush=True)

    def load(tag: str, s: dict[str, Any]):
        return load_or_generate(s["gen"], s["n_graphs"], s["seed"], name=f"{name}_{tag}", use_cache=use_cache)

    train, val = load("train", dcfg["train"]), load("val", dcfg["val"])
    tests = {k: load(k, v) for k, v in dcfg["test"].items()}
    summaries = {"train": summarize(train), "val": summarize(val), **{k: summarize(v) for k, v in tests.items()}}
    print(pd.DataFrame(summaries).T[["n_graphs", "n_mean", "mean_degree", "S_mean", "S_rel_mean",
                                     "S_zero_share", "default_share_mean", "fda_rounds_mean"]].round(4).to_string(),
          flush=True)

    labels = {"train": _labels(train), **{k: _labels(v) for k, v in tests.items()}}
    nodes = {k: _node_arrays(v) for k, v in tests.items()}
    feat_cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}

    def feats(tag: str, data_list, fs: str):
        if (tag, fs) not in feat_cache:
            t0 = time.perf_counter()
            X = dataset_features(data_list, fs)
            feat_cache[(tag, fs)] = (X, (time.perf_counter() - t0) / len(data_list))
        return feat_cache[(tag, fs)]

    runs: list[dict[str, Any]] = []
    for mname in cfg["models"]:
        spec = model_spec(mname, (cfg.get("model_overrides") or {}).get(mname))
        mtype = spec["type"]
        is_det = mtype in DETERMINISTIC or (mtype == "feature" and spec.get("kind") == "linreg")
        for seed in ([0] if is_det else seeds):
            print(f"--- {mname} (seed {seed})", flush=True)
            set_seed(seed)
            model = build_model(spec, seed)
            hist: dict[str, Any] = {}
            t0 = time.perf_counter()
            if mtype == "feature":
                X, _ = feats("train", train, spec["feature_set"])
                model.fit(X, labels["train"]["S"], labels["train"]["L_total"])
            elif mtype in ("paper_gnn", "en_gnn"):
                model.fit_normalizer(train)
                hist = train_gnn(model, train, val, lr=spec.get("lr", tcfg["lr"]),
                                 epochs=spec.get("epochs", tcfg["epochs"]),
                                 batch_size=spec.get("batch_size", tcfg["batch_size"]),
                                 patience=spec.get("patience", tcfg["patience"]),
                                 node_aux=spec.get("node_aux", 0.0), device=device)
                if save_checkpoints:
                    ck = out_dir.parent / "checkpoints" / name
                    ck.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), ck / f"{mname}_seed{seed}.pt")
            train_time = time.perf_counter() - t0

            eval_Ts = spec.get("eval_T", [None]) if mtype == "en_gnn" else [None]
            for tname, tlist in tests.items():
                for T in eval_Ts:
                    feat_time = 0.0
                    t1 = time.perf_counter()
                    if mtype == "kstep":
                        pred = model.predict(tlist, device=device)
                    elif mtype == "feature":
                        X, feat_time = feats(tname, tlist, spec["feature_set"])
                        t1 = time.perf_counter()
                        pred = model.predict(X, labels[tname]["L_total"])
                    else:
                        pred = predict_gnn(model, tlist, device=device, T=T)
                    pred_time = (time.perf_counter() - t1) / len(tlist)
                    lab = labels[tname]
                    rec = {
                        "model": mname if T is None else f"{mname}@T{T}",
                        "base_model": mname, "seed": seed, "test_set": tname, "T": T,
                        "metrics": graph_metrics(lab["S"], pred["S"], lab["L_total"]),
                        "stratified": stratified_metrics(lab["S"], pred["S"], lab["L_total"]),
                        "train_time_s": train_time,
                        "predict_time_per_graph_s": pred_time,
                        "feature_time_per_graph_s": feat_time,
                        "epochs_run": hist.get("epochs_run"),
                        "best_val": hist.get("best_val"),
                    }
                    if "r" in pred:
                        nd = nodes[tname]
                        rec["node_metrics"] = node_metrics(nd["r"], pred["r"], nd["u"], pred["u"],
                                                           nd["graph_idx"], nd["pbar"])
                    runs.append(rec)
                    m = rec["metrics"]
                    print(f"    [{tname}{'' if T is None else f' T={T}'}] MAE_rel {m['mae_rel']:.5f}  "
                          f"R2_rel {m['r2_rel']:.4f}  MAE {m['mae']:.3f}  R2 {m['r2']:.4f}", flush=True)

    result = {"name": name, "config": cfg, "device": device, "datasets": summaries, "runs": runs}
    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    table = summary_table(result)
    print("\n" + table.to_string(), flush=True)
    table.to_csv(out_dir / f"{name}_summary.csv")
    print(f"\nwrote {path}", flush=True)
    return result


def summary_table(result: dict[str, Any], metrics: tuple[str, ...] = ("mae_rel", "r2_rel", "mae", "r2")) -> pd.DataFrame:
    """mean +/- std over seeds for every (test set, model)."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in result["runs"]:
        g = groups[(r["test_set"], r["model"])]
        for k in metrics:
            g[k].append(r["metrics"][k])
        if "node_metrics" in r:
            g["default_auroc"].append(r["node_metrics"]["default_auroc"])
            g["spearman_u"].append(r["node_metrics"]["spearman_u"])
    rows = []
    for (ts, model), g in groups.items():
        row = {"test_set": ts, "model": model}
        for k, vals in g.items():
            v = np.array(vals, dtype=np.float64)
            row[k] = f"{np.nanmean(v):.4f}" + (f" ± {np.nanstd(v):.4f}" if len(v) > 1 else "")
        rows.append(row)
    return pd.DataFrame(rows).set_index(["test_set", "model"])
