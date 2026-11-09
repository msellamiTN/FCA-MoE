# FCA-MoE v0.2 validation status

## Local CPU validation
- Unit tests: 7 passed.
- Positive-control held-out router exact Top-2 match, seed 17 / 4096 samples / 120 epochs / window 64: approximately 99.52%.
- Positive-control full FCA lattice: 10 concepts, including all four expected order-four intents; precision and recall 1.00 at threshold 0.10.
- IID negative control at window 64 / threshold 0.10: 1 complete concept, zero order-four concepts, expected-intent recall 0.00.
- Both router validation scores meet the 98% synthetic acceptance threshold.

## Sensitivity warning
At short windows and higher thresholds, the IID control may generate large numbers of spurious concepts. In the CPU sensitivity run, IID/window=32/threshold=0.20 generated 199 concepts and 56 order-four concepts; only 4 of those 56 matched the designed positive-control community patterns (precision about 7.14%). This is why recall alone is insufficient.

## Hardware qualification
The above measurements are from local CPU validation, not the user's Colab T4. The new notebook must be run on the actual T4 and must record GPU, CUDA/PyTorch, timings and peak VRAM. No pretrained LLM, Unsloth integration, physical multi-GPU topology, or expert hardware isolation is tested.
