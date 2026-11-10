# FCA–MoE v0.3 — Executable Research Scaffold

This directory contains executable code for formal-context construction, FCA concept enumeration, structural diagnostics, and an exact pairwise-MaxEnt oracle for small expert counts.

## Scientific status and scope

This is a **runnable research scaffold**, not a completed 32-expert MaxEnt experiment and not evidence of GPU/KV-cache speedups.

- Experimental configuration: 32 experts, strict Top-2 routing.
- Chronological split: 60/20/20.
- Window sizes: 32, 64, 128.
- Activation thresholds: 0.10, 0.20, 0.30.
- FCA algorithm: NextClosure with a concept-count safety limit.
- Exact pairwise-MaxEnt oracle: E <= 16, for validating a future scalable sampler.
- Approved surrogate target: 999 repetitions; scalable E=32 sampler remains a pending implementation gate.

Synthetic routes are for software validation only, not pretrained-LLM evidence.

## Google Colab

Open [FCA_MoE_v03_scaffold.ipynb](notebooks/FCA_MoE_v03_scaffold.ipynb). The notebook clones this repository branch and runs from `research/v03`. CPU is sufficient for this scaffold; GPU is not required.

## Local installation and execution

From `research/v03`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest -q
python scripts/run_v03_smoke.py --tokens 4096
```

The smoke test writes `outputs/v03_smoke/smoke_report.json`.

## Modules

- `moe_fca/contexts.py`: strict route validation, token/window contexts, chronological split.
- `moe_fca/fca.py`: NextClosure, concept support and summaries.
- `moe_fca/metrics.py`: marginal/pairwise moments and concept profiles.
- `moe_fca/maxent_exact.py`: exact pairwise log-linear MaxEnt oracle for small E.
- `moe_fca/synthetic.py`: planted synthetic route generator.
- `configs/v03.yaml`: approved parameters and implementation boundaries.
- `tests/test_core.py`: unit tests.
- `scripts/run_v03_smoke.py`: reproducible end-to-end smoke run.

## Next implementation gate

Implement a scalable E=32 pairwise-MaxEnt sampler; validate it against the exact oracle at small E; report convergence, marginal/pairwise residuals, effective sample size, and initialization sensitivity before running the 999-surrogate protocol.

KV-cache locality is a separate systems hypothesis. It requires explicit cache policy, realistic sequence workloads, GPU instrumentation, and baseline comparisons. FCA concept statistics are not substitutes for cache hit rate, latency, throughput, memory, communication, or energy measurements.
