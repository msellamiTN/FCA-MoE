# FCA-MoE v0.2 — Controlled FCA Recovery Tests

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_v02_Controls.ipynb)

## What changed

- **Positive control:** each fixed token window belongs to one of four overlapping four-expert communities. Every token still routes to exactly two experts.
- **Matched IID negative control:** exact permutation of the positive route-pair sequence. Every route-pair count is identical; only temporal/window organization changes.
- **Held-out router validation:** split at whole-window boundaries; train on 80% of windows and extract FCA contexts only from the held-out 20% predictions.
- **Exact unfiltered FCA:** enumerate the complete concept family, including both lattice bounds, before structural summaries.
- **False-positive-aware sensitivity:** vary window sizes 32/64/128 and thresholds 0.10/0.20/0.30; report expected-intent recall, order-four precision and unexpected intents.
- **Placement comparison:** compare concept-derived and pairwise affinities on the same virtual topology. This remains a simulation, not physical GPU communication evidence.

## Baseline acceptance at window=64, threshold=0.10

| Control | Held-out Top-2 exact match | Complete concepts | Order-4 concepts | Expected order-4 recall |
|---|---:|---:|---:|---:|
| Window-correlated positive | >= 0.98 | 10 | 4 | 1.00 |
| Marginal-matched IID negative | >= 0.98 | 1 | 0 | 0.00 |

These are synthetic acceptance criteria, not pretrained-model performance claims. The full sensitivity matrix deliberately exposes spurious concepts caused by binarization at short windows or higher thresholds.

Run locally:
```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_v02_controls.py --sensitivity
```
