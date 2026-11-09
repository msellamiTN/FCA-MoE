# Validation report — FCA-MoE starter

Date: 2026-11-09

## Executed in the artifact-building environment

- Unit tests: **4 passed**.
- Python compilation: passed for `fca_moe`, `scripts`, and `tests`.
- Notebook JSON structure: valid Jupyter Notebook v4, 15 cells.
- ZIP integrity: passed.
- Synthetic smoke run: **CPU only** (`torch 2.10.0+cpu`; no CUDA GPU was exposed).

Synthetic smoke configuration: 1,024 token events, 8 experts, Top-2 routing, window size 64, 16 window objects. Observed router exact-match: 0.990234375. Exact FCA enumeration was complete and returned 8 concepts, including four order-4 intents:

- `{E0,E1,E2,E3}`
- `{E2,E3,E4,E5}`
- `{E4,E5,E6,E7}`
- `{E0,E1,E6,E7}`

These are expected from the deliberately constructed synthetic workload. They are **not pretrained-model results**. This smoke run does not validate T4 performance, Unsloth integration, GPU residency, or multi-GPU placement.

## Still requires execution in Colab

- Confirm the assigned device is actually a T4.
- Record CUDA, PyTorch, driver and package versions.
- Test Unsloth against a selected pretrained MoE.
- Measure GPU timing and peak VRAM.
- Capture genuine router traces from the pretrained model.