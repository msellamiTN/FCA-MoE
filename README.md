## FCA–MoE v0.3 — Lattice / MoE / KV-cache research scaffold

[![Open v0.3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/research/fca-moe-v0.3/research/v03/notebooks/FCA_MoE_v03_scaffold.ipynb) [![v0.3 tests](https://github.com/msellamiTN/FCA-MoE/actions/workflows/v03-tests.yml/badge.svg)](https://github.com/msellamiTN/FCA-MoE/actions/workflows/v03-tests.yml)

The executable v0.3 project is in [research/v03](research/v03/README.md). It includes FCA algorithms, a small-E exact pairwise-MaxEnt oracle, real GPT-OSS-20B router instrumentation, resumable Parquet extraction, validation scripts, Colab notebooks, and CI.

**Real-model target:** OpenAI GPT-OSS-20B (24 MoE layers, 32 experts, Top-4). The capture pipeline is implemented but has **not yet been executed** to collect empirical traces. The current Hugging Face MXFP4 documentation specifies SM75+; a Colab T4 meets that minimum, but full-model fit is not guaranteed. The extractor uses CPU/disk offload and must pass a real inference pilot. See the [real-model extraction notebook](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/research/fca-moe-v0.3/research/v03/notebooks/GPT_OSS_20B_Real_Routing_Extraction.ipynb).

The scalable E=32 pairwise-MaxEnt sampler and 999-surrogate protocol remain pending validation. Synthetic routes are software tests only. No GPU or KV-cache speedup is claimed.

Local quick start:

```bash
cd research/v03
python -m pip install -e .
python -m pytest -q
python scripts/run_v03_smoke.py --tokens 4096
```

# FCA-MoE — Formal Concept-Guided Expert Locality

[![Open in Colab v0.2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_v02_Controls.ipynb) [![Tests](https://github.com/msellamiTN/FCA-MoE/actions/workflows/tests.yml/badge.svg)](https://github.com/msellamiTN/FCA-MoE/actions/workflows/tests.yml)

**Baseline:** v0.2 controlled synthetic recovery is retained as the baseline. v0.3 now contains a real GPT-OSS-20B trace-capture pipeline, but no pretrained-model traces have yet been collected.

## Open in Google Colab

Run the new [FCA-MoE v0.2 Positive/Negative Controls notebook](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_v02_Controls.ipynb).

## v0.2 research objective

Validate that the higher-order window context and FCA pipeline:
1. Recover four known overlapping order-four expert groups in a positive workload.
2. Do not invent those persistent groups in an IID negative control at the baseline window/threshold.
3. Preserve high Top-2 router accuracy on held-out windows.
4. Expose false concepts caused by window size and activation-threshold sensitivity.

The positive/negative controls use the same route-pair vocabulary. The difference is whether a community persists within a window.

## Expected baseline acceptance

At window size 64 and activation threshold 0.10:
- Positive: held-out Top-2 exact match >= 0.98; 10 complete concepts; four order-four intents; recall and precision 1.00.
- IID negative: held-out Top-2 exact match >= 0.98; one complete concept; no order-four intents; expected-intent recall 0.00.

These are synthetic software-validation criteria, not LLM results.

## Run locally

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_v02_controls.py --sensitivity
```

See [README_V02.md](README_V02.md) and [VALIDATION_V02.md](VALIDATION_V02.md).

## Scientific boundaries

- The original router remains authoritative.
- FCA uses held-out predicted Top-2 route events, aggregated into layer-local windows.
- The full unfiltered concept family is enumerated before summaries.
- The virtual placement optimizer is a simulation, not a physical multi-GPU benchmark.
- No one-container-per-expert GPU isolation or Unsloth fine-tuning claim is made.
