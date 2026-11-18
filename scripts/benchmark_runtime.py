"""Per-graph inference time vs network size: exact solver, batched Picard, GNNs and feature baselines.

    python scripts/benchmark_runtime.py --sizes 50 100 200 500 1000 --out results

All batched methods process the whole test batch at once on the given device (fair comparison with the
GNNs).  Picard runs in float32 with relative tolerance 1e-6 - far tighter than any learned model's error.
Times are medians over ``--repeats`` after one warm-up run; training time is NOT included.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from sysrisk.clearing import clearing_fda, picard_dense, picard_sparse
from sysrisk.dataset import FEAT_X, data_to_dense, generate_dataset
from sysrisk.features import dataset_features
from sysrisk.models import FeatureRegressor, build_model, model_spec
from sysrisk.utils import sync


def timeit(fn, device: str, repeats: int) -> float:
    fn()
    sync(device)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200, 500, 1000])
    ap.add_argument("--graphs", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

    rows = []
    for N in args.sizes:
        G = args.graphs if N <= 500 else max(20, args.graphs // 4)
        ds = generate_dataset({"topology": "er", "n": N, "size_mode": "fixed_degree"}, G, seed=999)
        dense = [data_to_dense(d) for d in ds]
        rec = {"N": N, "graphs": G}
        # exact solver, one graph at a time (numpy)
        t0 = time.perf_counter()
        for L, x in dense:
            clearing_fda(L, x)
        rec["fda_cpu_ms"] = 1e3 * (time.perf_counter() - t0) / G
        # feature extraction + GBT (fit on random targets: timing only)
        t0 = time.perf_counter()
        X = dataset_features(ds, "B")
        rec["features_B_cpu_ms"] = 1e3 * (time.perf_counter() - t0) / G
        Ltot = np.array([float(d.L_total) for d in ds])
        gbt = FeatureRegressor("gbt", "B").fit(X, np.random.rand(G) * Ltot, Ltot)
        rec["gbt_predict_cpu_ms"] = 1e3 * timeit(lambda: gbt.predict(X, Ltot), "cpu", args.repeats) / G

        for dev in devices:
            batch = Batch.from_data_list(ds).to(dev)
            Ld = torch.tensor(np.stack([L for L, _ in dense]), dtype=torch.float32, device=dev)
            xd = torch.tensor(np.stack([x for _, x in dense]), dtype=torch.float32, device=dev)
            res = picard_dense(Ld, xd, tol=1e-6)
            rec[f"picard_iters_{dev}"] = res.iterations
            rec[f"picard_dense_{dev}_ms"] = 1e3 * timeit(lambda: picard_dense(Ld, xd, tol=1e-6), dev, args.repeats) / G
            ev, Lv, xv = batch.edge_index, batch.edge_attr[:, 0], batch.x[:, FEAT_X]
            rec[f"picard_sparse_{dev}_ms"] = 1e3 * timeit(lambda: picard_sparse(ev, Lv, xv, tol=1e-6), dev, args.repeats) / G
            for name in ("g1_paper", "g2_en"):
                model = build_model(model_spec(name)).to(dev).eval()
                with torch.no_grad():
                    rec[f"{name}_{dev}_ms"] = 1e3 * timeit(lambda: model(batch), dev, args.repeats) / G
            del Ld, xd, batch
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runtime.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out / 'runtime.json'}")


if __name__ == "__main__":
    main()
