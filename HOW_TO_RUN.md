# How to run this project: step by step

Read this top to bottom the first time you open the repo. Each step says **what to run**, **why**, and **what you should see**. Do the steps in order: each one checks something the next one relies on.

---

## 0. The big picture (read once)

```
 generators.py          clearing.py              dataset.py              models/ + train.py        metrics.py
 make a random   --->   compute the TRUE   --->  package graph +  --->   baselines and GNNs  --->  compare predictions
 bank network           unpaid liabilities S     labels for PyG          learn to predict S        with the true S
 (L, x)                 (exact solver)
```

**Where the true labels come from.** They come from `src/sysrisk/clearing.py`, function `clearing_fda`. This is the exact Eisenberg–Noe "fictitious default algorithm". `dataset.py` calls it automatically for every generated network and stores S as the label.
- You never run it by hand.
- Every experiment generates its data and computes the labels itself.

**What configs are.** The files in `configs/` are **experiment recipes**, not label checks. Each YAML file says:
- which networks to generate (topology, size, how many, random seed) for train, validation and test;
- which models to train;
- how many seeds and epochs.

Running an experiment means running `scripts/run_experiment.py` with one of these files. Labels are checked by **tests** (Step 2) and by the **paper comparison** (Step 3), not by configs.

**The five scripts.** These are the only commands you need.

| Script | What it does | When you use it |
|---|---|---|
| `pytest` | Runs automatic correctness checks | Step 2, and after any code change |
| `scripts/generate_data.py` | Generates networks, prints statistics and compares them with paper 2 | Step 3 (once) |
| `scripts/run_experiment.py --config <file>` | Trains and evaluates all models for one recipe | Steps 4–6 |
| `scripts/benchmark_runtime.py` | Times solver vs GNNs vs baselines | Step 7 |
| `scripts/make_tables.py` | Turns result files into tables and plots for the report | Step 8 |

---

## 1. One-time setup (already done on Sangameshwar's laptop)

Open PowerShell in the project folder and activate the environment:

```powershell
C:\venvs\sysrisk\Scripts\activate
```

**Why:** it switches to the Python environment that has PyTorch (with GPU support), PyTorch Geometric and the project installed. Do this **every time** you open a new terminal.

<details><summary>Setting up on a new computer</summary>

```powershell
python -m venv C:\venvs\sysrisk                   # keep it OUTSIDE OneDrive
C:\venvs\sysrisk\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (use .../whl/cpu without one)
pip install torch_geometric numpy scipy scikit-learn networkx pyyaml pandas matplotlib pytest
pip install -e . --no-deps
```
</details>

Optional: put generated data outside OneDrive, so it doesn't sync gigabytes.

```powershell
$env:SYSRISK_DATA_DIR = "C:\venvs\sysrisk_data"
```

---

## 2. Check that the code is correct: `pytest` (about 1 minute)

```powershell
pytest
```

**Why:** if the true labels were wrong, every result would be wrong. The tests check that:
- the exact solver satisfies the clearing equation, keeps payments between 0 and what is owed, conserves total equity, and gives the same S when banks are reordered;
- it reproduces paper 2's hand-solvable networks (cascade and star, e.g. mean shortfall 27 at N = 10);
- the exact solver and the iterative solver give the same answer;
- the generators produce the right structures;
- the GNNs respect bank reordering (equivariance).

**Expected:** `50 passed`. If anything fails, stop and fix it before continuing.

---

## 3. Check that our data matches the paper: `generate_data.py` (about 30 seconds)

```powershell
python scripts/generate_data.py --n-graphs 2500
```

**Why:** we re-implemented paper 2's network generators. If our mean total unpaid liabilities match the paper's published numbers, our data is a faithful copy. Put this check in the report.

**Expected:**

| | our `S_mean` | paper's value |
|---|---|---|
| ER | ≈ 263 | 262.08 |
| CP | ≈ 235 | 238.34 |
| CPf | ≈ 231 | 298.50 |

CPf is *expected* to differ: it uses one fixed network, and ours is a different draw. This is documented in `docs/dataset_card.md`. `S_mean_paper_solver_50it` should equal `S_mean`, which shows the paper's 50-iteration solver had converged here.

---

## 4. Smoke test the full pipeline: `e0_smoke` (about 1–2 minutes)

```powershell
python scripts/run_experiment.py --config configs/e0_smoke.yaml
```

**Why:** it runs *every* model on tiny data with very short training, to prove that data, training, evaluation and saving all work before you spend an hour on a real run. **Ignore the accuracy numbers here**; the models are deliberately under-trained.

**Expected:** it finishes without errors and prints a summary table.

---

## 5. Experiment E1: accuracy at the training size (main comparison)

Run one topology at a time. Start with **one seed**, so the first pass is fast:

```powershell
python scripts/run_experiment.py --config configs/e1_er.yaml  --seeds 0
python scripts/run_experiment.py --config configs/e1_cp.yaml  --seeds 0
python scripts/run_experiment.py --config configs/e1_cpf.yaml --seeds 0
```

**Why:** this answers *"How accurately can GNNs estimate systemic risk compared with simpler methods?"* Every model is trained on the same 10,000 networks of 100 banks and tested on 2,500 new ones of the same kind.

**Models compared** (the names appear in the output):

| Name | What it is | Learns? |
|---|---|---|
| `kstep1`, `kstep3` | Run 1 or 3 rounds of default contagion and stop. `kstep1` is the classical "first-round losses" measure | no |
| `linreg_A` | Linear regression on the mid-report's 8 graph features | yes |
| `linreg_B`, `mlp_B`, `gbt_B` | Linear regression / MLP / gradient-boosted trees on 25 richer features (stress and centrality) | yes |
| `g1_paper` | Paper 2's GNN architecture, with a sum readout | yes |
| `g2_en` | Our EN-aligned recurrent GNN | yes |

**Time:** roughly 15–60 minutes per config on the RTX 3050. Most of it is the two GNNs.

**Final numbers:** remove `--seeds 0`. The configs then train each model with 5 seeds, and the report shows mean ± std. Run these overnight or on Colab.

---

## 6. Experiment E2: generalisation to unseen network sizes

```powershell
python scripts/run_experiment.py --config configs/e2_size_er.yaml --seeds 0
python scripts/run_experiment.py --config configs/e2_size_cp.yaml --seeds 0
python scripts/run_experiment.py --config configs/e2_size_er_mixed.yaml --seeds 0   # optional variant
```

**Why:** this answers the professor's research question: *"Do models generalise to network sizes not seen during training?"* Models are trained on 100 banks and tested, **without retraining**, on 50, 200, 500 and 1,000 banks.

**Reading the test-set names:**
- **`fd_nXXX`** ("fixed degree"): each bank keeps about the same number of counterparties as in training. This is the **clean** size test.
- **`fp_nXXX`** ("fixed probability"): the mid-report's original design. Banks get more counterparties and more debt as N grows, so stress and size change together. It is kept as a contrast to show why the design was changed.
- **`g2_en@T16` / `@T32` / `@T64`:** G2 evaluated with more message-passing rounds at test time.

**Variant (`e2_size_er_mixed`):** trains on a mix of 50/100/150-bank networks. It shows whether training on several sizes helps.

---

## 7. Runtime comparison: `benchmark_runtime.py` (about 5–10 minutes)

```powershell
python scripts/benchmark_runtime.py --out results
```

**Why:** this answers *"Can GNNs reduce inference time compared with the reference calculation?"* For networks of 50 to 1,000 banks it times, per network, on CPU and GPU:
- the exact solver (`fda_cpu_ms`);
- the batched iterative solver (`picard_dense_*`, `picard_sparse_*`);
- the two GNNs (`g1_paper_*`, `g2_en_*`);
- feature extraction and tree prediction (`features_B_cpu_ms`, `gbt_predict_cpu_ms`).

All GNNs and solvers process the whole batch at once, so the comparison is fair. Training time is **not** included; it is reported separately in the experiment results.

**Expected output:** `results/runtime.json`.

---

## 8. Build tables and plots for the report

```powershell
python scripts/make_tables.py --runs results/runs --out results/tables
```

**Why:** it turns all result files into report-ready markdown tables and size-generalisation plots.

**Expected output:**
- `results/tables/<experiment>.md`
- `results/tables/<experiment>_size.png`
- `results/tables/runtime.md`

---

## Where results are saved

| Path | Contents |
|---|---|
| `results/runs/<experiment>.json` | Everything: config, dataset statistics, every metric for every model, seed and test set |
| `results/runs/<experiment>_summary.csv` | The summary table printed at the end of the run |
| `results/checkpoints/<experiment>/` | Trained GNN weights |
| `results/runtime.json` | Timing benchmark |
| `results/tables/` | Report tables and plots (Step 8) |
| `$SYSRISK_DATA_DIR` or `data/` | Cached generated networks. Safe to delete; they regenerate identically |

## How to read the metrics

- **`mae_rel`, `r2_rel`**: error on S / total liabilities, i.e. the *fraction* of debt that goes unpaid. This allows comparison across network sizes. **This is the main metric.**
- **`mae`, `r2`**: error on raw S, in money units.
  - R² = 1 is perfect.
  - R² = 0 is no better than always predicting the average.
  - R² < 0 is worse than that.
- **`default_auroc`, `spearman_u`**: only for models that predict per-bank values (K-step, G2).
  - `default_auroc` measures how well the model spots the banks that default.
  - `spearman_u` measures how well it ranks banks by unpaid amount.

## Useful flags for `run_experiment.py`

| Flag | Effect |
|---|---|
| `--seeds 0` | One training run per model (fast first pass) |
| `--models g2_en gbt_B` | Run only some models |
| `--epochs 50` | Cap training length |
| `--quick` | Tiny version of any config, just to check it runs |
| `--device cpu` | Force the CPU (default: GPU if available) |

## When to use what

| You want to… | Run |
|---|---|
| Check nothing is broken after editing code | `pytest` |
| Show our data matches paper 2 | `scripts/generate_data.py` |
| Check an experiment runs before a long job | `run_experiment.py --config <file> --quick` |
| Compare accuracy of all models (report section: results) | E1 configs (Step 5) |
| Test generalisation to unseen sizes (report section: generalisation) | E2 configs (Step 6) |
| Compare speed | `benchmark_runtime.py` (Step 7) |
| Make tables and figures | `make_tables.py` (Step 8) |
