# Dataset card: synthetic Eisenberg–Noe interbank benchmark

## Graph contents
Each graph is one interbank network.

**Nodes** are banks. Raw inputs `x = [external assets, interbank assets a, nominal liabilities p̄]`.

**Edges** run from debtor to creditor, with `edge_attr = L_ij`, the amount bank i owes bank j.

**Labels** come from the exact fictitious-default algorithm.

| Level | Field | Meaning |
|---|---|---|
| node | `p` | clearing payment |
| node | `r` | p/p̄ (1 if p̄ = 0) |
| node | `u` | p̄ − p |
| node | `default` | true if p < p̄ |
| graph | `y` | S = Σu, the total unpaid liabilities (**target**) |
| graph | `y_rel` | S/ΣL |
| graph | `L_total` | ΣL |
| graph | `n_defaults` | number of defaulting banks |
| graph | `fda_rounds` | contagion depth: rounds in which the default set grew |
| graph | `n` | number of banks |

Every bank has external assets x > 0, so the clearing vector is unique.

## Generators
Defaults follow Gonon, Meyer-Brandis & Weber (arXiv:2410.07222, Sec. 6.2).

| Topology | Structure | Liabilities | External assets |
|---|---|---|---|
| `er` | directed ER, p = 0.4 | 1 per edge (`weights: lognormal` gives the weighted variant, mean 1) | 10·Beta(2,5) i.i.d. |
| `cp` | 10% core; link probabilities core–core 0.7, core–periphery 0.3, periphery–periphery 0.1; node order shuffled | 10 / 2 / 1 | Beta(2,5) with Gaussian copula ρ = 0.5, scaled ×50 for core and ×10 for periphery |
| `cpf` | one CP structure fixed by `structure_seed`, not shuffled | as CP | resampled per graph, as CP |

**Size modes** (reference size N = 100):
- `fixed_p` keeps the link probabilities fixed.
- `fixed_degree` keeps expected degrees fixed: ER uses p = 39.6/(N−1); CP block probabilities are scaled by 100/N.

**Extra knob:** `asset_multiplier` scales x to vary stress.

## Statistics at N = 100 (2,500 graphs, seed 12345)
| | edges | mean S | S/ΣL | share of graphs with S = 0 | share of banks defaulting | mean contagion depth (max) |
|---|---|---|---|---|---|---|
| ER | 3960 | 263.6 | 0.067 | 0 | 0.49 | 2.8 (4) |
| CP | 1404 | 234.9 | 0.094 | 0 | 0.41 | 2.5 (5) |
| CPf | 1422 | 231.0 | 0.093 | 0 | 0.49 | 2.5 (4) |

## Known limitations
- **Synthetic labels.** The labels are what the EN model predicts, not observed losses. The EN model also has no bankruptcy costs (a Rogers–Veraart extension is possible).
- **Shallow contagion.** In these settings contagion is shallow, so the first few EN rounds explain most of S, which makes the K-step heuristic a strong baseline. Deeper-cascade settings (sparser networks, lower buffers, chain structures) are a planned extension.
- **CPf differs from the paper.** Our CPf matrix is not the paper's. The paper's mean S of 298.5 is above the range we observe across structure seeds (155–249).
