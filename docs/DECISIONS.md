# FCA-MoE — Critical Design Decisions

Date: 2026-11-09

## 1. Unsloth is an optimization backend, not the research contribution

Keep FCA discovery, route telemetry, affinity construction, placement and evaluation independent of Unsloth. Use Unsloth only after the selected pretrained MoE, precision, kernels and package versions pass a smoke test. For a T4, plan around FP16; do not assume BF16 support or assume that 4-bit MoE QLoRA works with every backend.

References:
- https://unsloth.ai/docs/new
- https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-2507

## 2. No Docker container per expert in hosted Colab

Do not implement one Docker container per expert inside managed Colab. A T4 cannot expose MIG hardware instances. CUDA MPS is cooperative process multiplexing, not equivalent to independent hardware-isolated GPU instances. MPS v3 memory partitioning has host prerequisites that cannot be assumed in hosted Colab. Google's GPU Docker image documentation describes a local-runtime path with host-level control, not arbitrary containers inside hosted Colab.

References:
- https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html
- https://docs.nvidia.com/deploy/mps/latest/index.html
- https://docs.nvidia.com/deploy/mps/mpsv3-memory-partitioning.html
- https://research.google.com/colaboratory/local-runtimes.html
- https://research.google.com/colaboratory/intl/en-GB/faq.html

## 3. Logical expert isolation on T4

Represent experts as independently identified logical execution units with explicit metadata: memory footprint, residency state, last-use time, access count and prefetch state. This is a software abstraction, not hardware isolation.

## 4. Separate token and window contexts

A token-layer route event for Top-K routing contains at most K selected expert attributes. Its formal concept intents therefore cannot contain more than K experts (apart from the empty-extent concept). To study higher-order co-activation, retain token events and build a second, layer-local window context. Mark an expert attribute present when its activation rate within the window meets a declared threshold. Window size and threshold require sensitivity analysis.

## 5. Preserve neural routing

The model's original router remains authoritative. FCA may guide where experts are placed, retained or prefetched, but it must not alter the selected Top-K expert IDs in the first research version.

## 6. Count all costs

Measure FCA discovery, trace collection, lookup, scheduling, transfer, prefetch waste, cache misses and synchronization. A locality policy is useful only if its end-to-end costs improve over simpler alternatives.

## 7. Candidate pretrained model is not yet selected or tested

GPT-OSS-20B is a candidate to probe, not a model included in this starter. Do not report any pretrained-model result until a checkpoint loads successfully and genuine router traces are captured.
