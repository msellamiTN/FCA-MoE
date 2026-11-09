# Unsloth and NVIDIA T4 Integration Gate

Unsloth is not required by the synthetic FCA core. It is a later model-adapter and fine-tuning optimization option.

## T4 assumptions to verify at runtime

- Record GPU model and compute capability; Colab allocation is not guaranteed.
- Prefer FP16 as the initial T4 precision path.
- Do not assume BF16 hardware support on T4.
- Do not upgrade PyTorch/CUDA in the middle of a baseline experiment.
- Do not assume 4-bit BitsAndBytes MoE QLoRA compatibility.
- Confirm the exact model architecture and Unsloth loader support before attempting the load.

## Integration order

1. Run the Colab preflight and record versions.
2. Load a candidate pretrained MoE in inference-only mode.
3. Identify router modules, expert modules and the actual dispatch path.
4. Capture token-layer expert IDs without modifying routing.
5. Validate router trace counts against model forward outputs.
6. Only then introduce expert residency and prefetch experiments.
7. Treat Unsloth fine-tuning as a separate experiment, not a prerequisite for evaluating inference locality.

## Candidate model status

GPT-OSS-20B is a candidate, not yet loaded or benchmarked in this repository. T4 memory feasibility must be tested with a specific loader, precision, context length and offloading strategy. No T4 compatibility is claimed by this document.
