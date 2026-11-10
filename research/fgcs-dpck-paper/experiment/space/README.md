---
title: DPCK-MoE Trace Inspector
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# DPCK-MoE Trace Inspector

A Hugging Face Space template for validating and browsing exported OLMoE trace JSONL and the matching run manifest.

## Deploy

1. Create a new public or private Gradio Space in the intended Hugging Face account.
2. Copy `app.py`, `requirements.txt` and this `README.md` into the Space repository.
3. Commit and wait for the Space build to finish.
4. Upload a trace artifact and matching manifest produced by the Colab capture notebook.

No HF token or model download is needed. The app is intentionally CPU-first. A GPU upgrade is not required for trace inspection.

## Scope and limitations

- Validates required JSONL fields and summarizes sessions, layers, Top-K cardinalities and expert-assignment counts.
- Does not run OLMoE, FCA, DPCK, placement optimization or cluster benchmarks.
- Cannot independently verify that traces correspond to true runtime router indices; the Colab router-fidelity check is the evidence for that.
- Does not intentionally persist uploaded files. Do not upload private prompts or sensitive inference traces.
- Verify current Space hardware, plan requirements and pricing before enabling paid compute.
