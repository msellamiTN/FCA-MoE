# FCA-MoE — Formal Concept-Guided Expert Locality

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_T4_Colab.ipynb) [![Tests](https://github.com/msellamiTN/FCA-MoE/actions/workflows/tests.yml/badge.svg)](https://github.com/msellamiTN/FCA-MoE/actions/workflows/tests.yml)

Research prototype for **Formal Concept Analysis (FCA)-guided discovery of expert co-activation structures and locality-aware execution** in sparse Mixture-of-Experts (MoE) models.

> **Research status:** this repository currently contains a synthetic Top-K MoE prototype and FCA algorithms. It does not yet integrate or benchmark a pretrained LLM. GPT-OSS-20B is a candidate baseline, not a model used in the current code. The Colab notebook has not yet been executed on an actual T4 runtime.

## Open in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_T4_Colab.ipynb)

Select a GPU runtime in Colab. The notebook verifies and records the actual assigned GPU; do not assume it is a T4 until its identity is printed.

## Research hypothesis

Can bounded FCA discover stable, overlapping, context-dependent higher-order structures in real MoE routing traces, and can those structures improve expert locality beyond pairwise co-activation and graph baselines without changing the model's learned Top-K router decisions?

## Current implementation

- Synthetic sparse MoE with 8 experts and Top-2 routing.
- Token-level routing telemetry (one event per token-layer routing decision).
- Window-level FCA context, enabling intents larger than Top-2.
- Exact NextClosure concept enumeration for small contexts.
- Explicit reporting of incomplete enumeration when a concept budget is reached.
- Concept-derived affinity and pairwise co-activation baseline.
- Capacity-constrained **virtual** placement objective.
- Unit tests and GitHub Actions workflow.

## Quick start

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_synthetic.py
```

## Important experimental boundaries

1. **No one-container-per-expert GPU isolation claim.** A Colab T4 does not provide MIG hardware partitioning. Experts are logical execution units in the prototype.
2. **No multi-GPU claims from a single T4.** Virtual multi-device placement is a cost-model simulation, not a physical all-to-all benchmark.
3. **No pretrained-model results yet.** Synthetic tests validate software logic, not gains on GPT-OSS-20B or another pretrained MoE.
4. **FCA does not modify routing.** The original neural router remains authoritative; locality mechanisms can only influence placement, residency or prefetch in the initial design.
5. **Higher-order context construction matters.** A token-level Top-2 formal context cannot reveal intents containing more than two experts. We retain token events and separately aggregate fixed-size, layer-local windows for higher-order discovery. Window size and activation thresholds must be reported and sensitivity-tested.

## Project layout

```text
fca_moe/          FCA context, enumeration, affinity, model, placement and telemetry
notebooks/        Colab T4 research notebook
tests/            correctness tests
docs/             design decisions and experiment protocol
configs/          experiment configuration
scripts/          reproducible command-line experiments
results/          small synthetic smoke outputs
.github/workflows/ CI tests
```

## Validation

The initial local validation ran four tests successfully (`4 passed`). See [VALIDATION.md](VALIDATION.md). This is not a Colab T4 result. Re-run tests in Colab and preserve the environment and experiment JSON before reporting GPU measurements.

## Roadmap

1. Execute the synthetic notebook on a real Colab GPU and capture the environment.
2. Integrate a compatible pretrained MoE and extract genuine token-layer router traces.
3. Compare FCA with pairwise co-activation, graph clustering and frequent itemset baselines.
4. Add measured CPU/GPU expert residency and asynchronous prefetching.
5. Validate multi-GPU placement and communication only on actual multi-GPU infrastructure.

## License

No license has been added yet. All rights and reuse permissions remain reserved until a license is selected.
