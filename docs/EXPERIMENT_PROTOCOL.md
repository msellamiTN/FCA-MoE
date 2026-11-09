# FCA-MoE — Experimental Protocol

## Stage A: software correctness

- Verify derivation operators and closure.
- Compare NextClosure output with exhaustive enumeration for small random contexts.
- Verify concept-budget termination is labelled incomplete.
- Verify route traces preserve layer identity and event ordering.
- Verify window aggregation and activation thresholds.

## Stage B: structural discovery

Use real route traces when available. Split traces by time/workload into discovery and evaluation partitions to avoid using the same windows for both discovery and claimed generalization.

Compare:
1. Pairwise co-activation.
2. Graph community detection on the pairwise affinity graph.
3. Frequent itemset mining.
4. Exact FCA for small contexts.
5. Bounded FCA for larger contexts.

Report concept counts, intent-size distribution, extent support, redundancy, stability definition, discovery runtime, peak host memory, and out-of-sample coverage. Distinguish statistical/temporal stability from classical FCA extent stability.

## Stage C: locality execution

Keep router-selected experts identical across all policies. Compare:
- Original baseline placement/residency.
- LRU/LFU expert residency.
- Pairwise affinity-based locality.
- Graph placement.
- FCA affinity-based placement/residency.
- FCA prefetch (if implemented).

Measure transfer bytes, host-device transfer time, cache hit/miss, prefetch precision/recall, wasted bytes, memory occupancy, throughput and p50/p95/p99 latency. Include all controller overhead.

## Stage D: hardware claims

A single Colab T4 can validate a single-GPU prototype and host-to-device residency experiments. It cannot establish multi-GPU all-to-all reductions. Multi-GPU claims require physical multi-GPU runs with topology and device details reported. Virtual topology objectives must be labelled simulations.

## Mandatory reporting

Record:
- Model identifier and exact checkpoint revision.
- Dataset or prompt/workload, split and sample counts.
- Random seeds.
- GPU model, VRAM, driver, CUDA and PyTorch versions.
- Precision, batch size, sequence lengths and router Top-K.
- FCA context type, window size, threshold, filters and concept budget.
- Placement capacity constraints and topology.
- All wall-clock time, peak memory and controller/transfer overhead.
- Whether results are synthetic, simulated, CPU-only or measured on physical GPU hardware.

## Primary falsification test

Does FCA discover reusable higher-order structure that improves downstream locality over pairwise co-activation and graph baselines after all discovery and runtime overhead is counted? If not, FCA should not remain in the execution path.
