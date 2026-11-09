# FCA-MoE v0.2 — Controlled FCA Recovery Tests

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_v02_Controls.ipynb)

## What changed

- Positive control: persistent overlapping expert communities per 64-token window.
- Negative IID control: same route-pair vocabulary, but no persistent community per window.
- Held-out router validation: train on 80% of complete windows; extract FCA contexts only from the held-out 20% predictions.
- Full unfiltered concept enumeration, including both lattice bounds.
- Sensitivity analysis for window sizes 32/64/128 and activation thresholds 0.10/0.20/0.30.
- Report both expected-intent recall and order-four precision/unexpected intents. Thresholding can create spurious concepts in IID data.
- Virtual placement remains a cost-model simulation, not a physical multi-GPU result.

## Baseline acceptance at window=64, threshold=0.10

| Control | Held-out Top-2 exact match | Full concepts | Order-4 concepts | Expected-intent recall |
|---|---:|---:|---:|---:|
| Window-correlated positive | >=0.98 | 10 | 4 | 1.00 |
| IID negative | >=0.98 | 1 | 0 | 0.00 |

These are controlled synthetic acceptance criteria, not pretrained-LLM performance claims.

Run locally:
```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_v02_controls.py --sensitivity
```
