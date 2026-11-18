"""Run one experiment config.

    python scripts/run_experiment.py --config configs/e1_er.yaml
    python scripts/run_experiment.py --config configs/e1_er.yaml --quick          # tiny smoke test
    python scripts/run_experiment.py --config configs/e2_size_er.yaml --models g2_en --seeds 0
"""
from __future__ import annotations

import argparse
import warnings

from sysrisk.experiment import quick_config, run_experiment
from sysrisk.utils import get_device, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="shrink data/epochs for a smoke test")
    ap.add_argument("--no-cache", action="store_true", help="do not read/write cached datasets")
    ap.add_argument("--models", nargs="+", help="override the model list")
    ap.add_argument("--seeds", nargs="+", type=int, help="override the seed list")
    ap.add_argument("--epochs", type=int, help="override max epochs")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = load_yaml(args.config)
    if args.models:
        cfg["models"] = args.models
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.epochs:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.quick:
        cfg = quick_config(cfg)
    run_experiment(cfg, out_dir=args.out, device=get_device(args.device), use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
