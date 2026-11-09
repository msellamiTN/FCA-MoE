# FCA-MoE v0.2 validation

## Local CPU validation in the project-building environment

- Unit tests: **8 passed**.
- Positive-control held-out router exact Top-2 match (seed 17, 4096 samples, 120 epochs, window 64): approximately **99.64%**.
- Positive-control complete FCA lattice: **10 concepts**, all four expected overlapping order-four intents; precision and recall both **1.00** at threshold 0.10.
- IID negative control is an exact permutation of the positive-control route-pair sequence: identical empirical pair counts. At window 64 / threshold 0.10 it yields **1 complete concept**, **0 order-four concepts**, expected-intent recall **0.00**, and held-out router exact match approximately **99.52%**.

## Sensitivity warning

Thresholding can create spurious concepts even after marginal matching. In the local CPU sensitivity run, IID at window 32 / threshold 0.20 generated 196 concepts and 52 order-four intents; only 4 match the designed positive-control communities (precision about **7.69%**). At window 64 / threshold 0.20 it generated 120 concepts and 29 order-four intents, again with low precision. Recall alone is therefore insufficient.

## Hardware qualification

These are local CPU validation results, not Colab T4 measurements. The v0.2 notebook must be executed on the assigned T4 and its GPU, CUDA/PyTorch, timings and peak VRAM recorded. No pretrained LLM, Unsloth integration, hard GPU isolation or physical multi-GPU placement was tested.
