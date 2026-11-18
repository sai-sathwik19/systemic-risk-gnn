# GNNs for Systemic Financial Risk (AI in Finance, Project 12)

This project tests whether a graph neural network (GNN) can predict systemic risk in a banking network. The risk measure is the **total unpaid liabilities** under the Eisenberg–Noe (EN) clearing model:

```
p = min(p̄, x + Πᵀp)          (clearing payments)
S = Σᵢ (p̄ᵢ − pᵢ)             (target: total unpaid liabilities)
```

We compare the GNNs against classical baselines and network metrics. We also test whether they work on network sizes and topologies they were not trained on.

## What is in the repo

| Component | File | Notes |
|---|---|---|
| EN solver | `src/sysrisk/clearing.py` | Exact fictitious-default algorithm (labels) + batched Picard iteration in torch (dense / sparse, CPU / GPU) |
| Network generators | `src/sysrisk/generators.py` | Paper 2's ER, CP and CPf settings, re-implemented and extended to any size N (`fixed_p` / `fixed_degree`), plus a weighted-ER variant |
| Datasets | `src/sysrisk/dataset.py` | PyG graphs with node labels (p, r, u, default) and graph labels (S, S/ΣL, #defaults, contagion depth) |
| Baselines | `src/sysrisk/models/baselines.py`, `features.py` | K-step EN contagion heuristic (K=1 is the first-round shortfall); linear regression / MLP / gradient-boosted trees on feature set A (the mid-report's 8 features) or B (extended: stress and centrality) |
| G1 Paper-GNN | `src/sysrisk/models/gnn_paper.py` | Paper 2's architecture (weighted-mean GraphSAGE, sigmoid, width 10) with a **sum** readout |
| G2 EN-aligned GNN | `src/sysrisk/models/gnn_en.py` | Proposed model; see below |
| Runner | `src/sysrisk/experiment.py`, `scripts/` | YAML configs, multiple seeds, JSON results, tables and plots |

### G2: EN-aligned recurrent GNN
Written in payment ratios rᵢ = pᵢ/p̄ᵢ, one EN step is `rᵢ ← min(1, xᵢ/p̄ᵢ + Σⱼ wⱼᵢ rⱼ)` with wⱼᵢ = Lⱼᵢ/p̄ᵢ. That is a **sum** aggregation over a bank's debtors, using only scale-free local quantities. G2 copies this structure:
- ratio inputs [xᵢ/p̄ᵢ, aᵢ/p̄ᵢ];
- messages weighted by wⱼᵢ and summed;
- one GRU update with shared weights, applied T times (T can be larger at test time);
- per-bank output r̂ᵢ ∈ [0,1], combined exactly as **Ŝ = Σᵢ p̄ᵢ(1 − r̂ᵢ)**.

Both GNNs are trained **only on the graph label S** (loss = MSE on S/ΣL). Node labels are stored for evaluation only (default AUROC, Spearman rank of node shortfalls).

## Setup

**Local (development and tests, CPU):**
```bash
python -m venv C:\venvs\sysrisk          # keep the env outside OneDrive
C:\venvs\sysrisk\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy scipy scikit-learn networkx pyyaml pandas matplotlib pytest
pip install -e . --no-deps
pytest
```

**Colab (full experiments):** open `notebooks/colab_run.ipynb` and pick a T4 GPU. The notebook gets the code (git clone, or a copy of this folder on Drive) and installs PyG. It then runs the tests, the paper sanity check, E1, E2 and the runtime benchmark, and writes results to Drive.

## Running
**New to the repo? Follow [HOW_TO_RUN.md](HOW_TO_RUN.md).** It covers every step in order and explains why each is run.

```bash
python scripts/generate_data.py --n-graphs 2500                       # paper-2 sanity check
python scripts/run_experiment.py --config configs/e0_smoke.yaml        # ~1 min smoke test
python scripts/run_experiment.py --config configs/e1_er.yaml           # E1 in-distribution (also e1_cp, e1_cpf)
python scripts/run_experiment.py --config configs/e2_size_er.yaml      # E2 size generalisation (also e2_size_cp, e2_size_er_mixed)
python scripts/benchmark_runtime.py --out results                      # E5 runtime vs N
python scripts/make_tables.py --runs results/runs --out results/tables
```
Useful flags: `--quick` (tiny version of any config), `--models g2_en --seeds 0`, `--epochs 50`. Datasets are deterministic in (spec, size, seed) and are cached in `SYSRISK_DATA_DIR` (default `./data`).

## Experiments (Phase 1)

| ID | Question | Config |
|---|---|---|
| E1 | In-distribution accuracy at N=100 (ER, CP, CPf), all models | `e1_*.yaml` |
| E2 | Train at N=100, test at N ∈ {50, 200, 500, 1000}. `fd_*` keeps the expected degree fixed (clean size shift); `fp_*` keeps p fixed (the mid-report design, which also shifts density and stress) | `e2_size_*.yaml` |
| E5 | Inference time vs N: exact solver vs batched Picard vs GNNs vs feature models | `scripts/benchmark_runtime.py` |

Metrics: MAE / RMSE / R² on S and on the scale-free S/ΣL; errors by stress bucket; node-level default AUROC and Spearman ρ of shortfalls for models that output per-bank values.

## Validation results so far
- `pytest` passes 50 tests, including:
  - paper 2's stylised cascade/star networks (mean shortfall [N(N−1)/2 + (N−1)]/2, e.g. 27 at N=10);
  - the solver satisfies the clearing equation, 0 ≤ p ≤ p̄, conserves equity, scales with the inputs, and is unaffected by reordering banks;
  - exact and Picard solvers agree;
  - the GNNs give the same S when banks are reordered, and their per-bank outputs reorder with them.
- Generator check against paper 2 (2,500 graphs each, mean S without bailout):

  | | ours | paper 2 |
  |---|---|---|
  | ER | 263.6 ± 0.8 | 262.08 |
  | CP | 234.9 ± 2.3 | 238.34 |
  | CPf | 231.0 | 298.50 |

  CPf uses one fixed liability matrix. Ours is drawn from `structure_seed=0`, and across seeds 0–11 its mean S ranges from 155 to 249. The paper's specific matrix is not reproduced.
- In these settings contagion is shallow (at most 4–5 default rounds at N=100), and the paper's 50-iteration solver agrees with the exact one.

## Credits
- L. Eisenberg, T. H. Noe (2001), *Systemic Risk in Financial Systems*, Management Science 47(2).
- L. Gonon, T. Meyer-Brandis, N. Weber, *Computing Systemic Risk Measures with Graph Neural Networks*, arXiv:2410.07222. Our generator defaults and G1 follow their paper and code (github.com/NiklasMWeber/ComputingSystemicRiskMeasures, GPL-3.0). This repository re-implements them in PyTorch Geometric and does not copy their source.
