# FCA-MoE: Formal Concept-Guided Expert Locality for MoE Transformers

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_T4_Colab.ipynb)

**Status:** synthetic research prototype. No pretrained-LLM benchmark is present. GPT-OSS-20B is only a candidate; it has not been loaded or evaluated by this code.

## Research question
Can bounded Formal Concept Analysis discover stable, overlapping, context-dependent higher-order expert activation patterns and improve expert locality without changing the neural router's Top-K selections?

## Colab
[Open the Colab notebook](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/main/notebooks/FCA_MoE_T4_Colab.ipynb). Select a GPU runtime and inspect the actual allocated GPU before interpreting measurements.

## Current prototype
- Synthetic PyTorch MoE, eight experts, Top-2 routing.
- Token-level route telemetry plus layer-local window contexts.
- Exact NextClosure concept enumeration for small contexts.
- Explicit complete flag and incomplete reporting when a concept budget truncates enumeration.
- Concept-derived affinity and pairwise co-activation baseline.
- Capacity-constrained virtual placement (simulation only).
- Colab notebook and unit tests.

## Install and run
```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_synthetic.py
```

## Important boundaries
1. A token-level Top-K event cannot yield a non-empty concept intent larger than K. Higher-order FCA discovery therefore uses a separate window-level context. Window size and activation threshold require sensitivity analysis.
2. The learned neural router remains authoritative. FCA does not change selected experts in this prototype.
3. Per-expert Docker containers / hard GPU isolation are not provided on hosted Colab T4. Experts are logical execution units.
4. Virtual multi-device placement is not a physical multi-GPU benchmark.
5. Synthetic results do not establish speedups on a pretrained LLM.

See VALIDATION.md, docs/DECISIONS.md and docs/EXPERIMENT_PROTOCOL.md.
