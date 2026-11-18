"""Turn experiment JSONs into markdown tables and size-generalisation plots.

    python scripts/make_tables.py --runs results/runs --out results/tables
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sysrisk.experiment import summary_table  # noqa: E402


def to_markdown(df: pd.DataFrame) -> str:
    df = df.reset_index()
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.values) + " |")
    return "\n".join(lines)


def size_plot(result: dict, path: Path, metric: str = "mae_rel") -> None:
    """MAE on S/sum(L) vs N for test sets named like '<prefix>_n<N>'."""
    recs = []
    for r in result["runs"]:
        m = re.match(r"(?P<mode>[a-z_]+)_n(?P<n>\d+)$", r["test_set"])
        if m:
            recs.append({"mode": m["mode"], "N": int(m["n"]), "model": r["model"], "v": r["metrics"][metric]})
    if not recs:
        return
    df = pd.DataFrame(recs).groupby(["mode", "model", "N"])["v"].agg(["mean", "std"]).reset_index()
    modes = sorted(df["mode"].unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(5.5 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = df[df["mode"] == mode]
        for model, g in sub.groupby("model"):
            g = g.sort_values("N")
            ax.errorbar(g["N"], g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, label=model)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of banks N")
        ax.set_ylabel(f"{metric} (S / sum L)")
        ax.set_title(f"{result['name']} - {mode}")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.runs).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        table = summary_table(result)
        md = f"## {result['name']}\n\n" + to_markdown(table) + "\n"
        (out / f"{result['name']}.md").write_text(md, encoding="utf-8")
        size_plot(result, out / f"{result['name']}_size.png")
        print(md)
    rt = Path(args.runs).parent / "runtime.json"
    if rt.exists():
        df = pd.DataFrame(json.loads(rt.read_text(encoding="utf-8"))).set_index("N")
        md = "## runtime (ms per graph)\n\n" + to_markdown(df.round(4)) + "\n"
        (out / "runtime.md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
