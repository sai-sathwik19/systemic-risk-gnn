"""Generate datasets and run the paper-2 sanity check.

Prints dataset statistics for the given topologies and compares the mean total unpaid liabilities with the
"None" (no bailout) column of Gonon, Meyer-Brandis & Weber (Table 2, test sets): ER 262.08, CP 238.34,
CPf 298.50.  It also re-computes S with the paper code's solver (Picard from p = 0, exactly 50 iterations)
to show how much a non-converged solver biases the labels.

    python scripts/generate_data.py --topologies er cp cpf --n-graphs 2500
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from sysrisk.clearing import picard_sparse
from sysrisk.dataset import FEAT_X, generate_dataset, summarize

PAPER_NONE = {"er": 262.08, "cp": 238.34, "cpf": 298.50}


def paper_solver_S(data_list, n_iter=50):
    out = []
    for d in data_list:
        res = picard_sparse(d.edge_index, d.edge_attr[:, 0].double(), d.x[:, FEAT_X].double(), start="zero",
                            n_iter=n_iter)
        out.append(float((res.pbar - res.p).sum()))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", nargs="+", default=["er", "cp", "cpf"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size-mode", default="fixed_p")
    ap.add_argument("--n-graphs", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    report = {}
    for topo in args.topologies:
        spec = {"topology": topo, "n": args.n, "size_mode": args.size_mode}
        t0 = time.time()
        ds = generate_dataset(spec, args.n_graphs, args.seed)
        gen_time = time.time() - t0
        s = summarize(ds)
        S = np.array([float(d.y) for d in ds])
        S50 = paper_solver_S(ds)
        s["S_sem"] = float(S.std() / np.sqrt(len(S)))
        s["S_mean_paper_solver_50it"] = float(S50.mean())
        s["paper_none"] = PAPER_NONE.get(topo) if args.n == 100 else None
        s["gen_seconds"] = gen_time
        report[topo] = s
        print(f"\n[{topo}] n={args.n} graphs={args.n_graphs}  generated in {gen_time:.1f}s")
        for k, v in s.items():
            print(f"  {k:28s} {v}")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
