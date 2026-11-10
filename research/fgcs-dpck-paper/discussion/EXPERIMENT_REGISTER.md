# DPCK–MoE experiment register and go/no-go protocol

**Updated:** 2026-11-10  
**Primary repository branch:** `research/fgcs-dpck-paper`

## 1. Evidence ledger

| Evidence item | What exists | What it supports | What it does not support |
|---|---|---|---|
| FCA-MoE v0.2 synthetic controls | Prior local CPU run; 8 unit tests passed | Pipeline sanity, planted-overlap recovery, marginal-matched negative control | Real pretrained MoE behaviour, physical placement |
| Sensitivity matrix | Prior local CPU outputs | Threshold/window fragility and false-positive risk | Generalization to real task traces |
| Top-3 parity enumeration | Finite exhaustive 60-placement example and executable test | Pairwise-marginal insufficiency for that specified objective | Universal pairwise failure or GPU speedup |
| FGCS LaTeX draft | Compiled in the artifact-building environment | Manuscript structure and current formalization | Completed experimental evaluation |
| T4 notebook and capture script | Source prepared | Intended instrumentation workflow | A completed T4 execution |
| HF Space app | Source prepared | Intended trace schema/inspection interface | Deployed application or model serving |
| Residual-coverage idea | Mathematical candidate derivation | Potential lower-bound and residual-regret result under token-level assumptions | A validated or tight practical certificate |

## 2. Pilot question

At the same serialized-byte budget, does an FCA-derived pattern selection preserve held-out placement decisions under workload shift better than the strongest non-FCA summaries?

The pilot must be allowed to fail. A negative result is a valid research outcome and must not be converted into a success claim by changing metrics after evaluation.

## 3. Stage A — Colab T4 trace acquisition

### 3.1 Model candidate

Primary candidate: `allenai/OLMoE-1B-7B-0125`.

The capture script targets prefill routing and validates hooked expert indices against Top-K indices reconstructed from returned router logits. Confirm the model module paths against a pinned model/Transformers revision before confirmatory use.

### 3.2 Required manifest

Every run must record:
- model identifier, immutable model commit SHA and tokenizer revision;
- Transformers, PyTorch, CUDA, Python and relevant package versions;
- GPU name and device properties read from the runtime;
- precision/quantization configuration;
- prompt manifest hash, session IDs and task-family labels;
- capture mode (prefill only in the first experiment);
- router-index fidelity report;
- seeds and relevant inference configuration;
- trace artifact SHA-256.

The notebook must not assume a T4 was allocated. It must inspect the actual device and fail or label the run accurately if the requested runtime is different.

### 3.3 Trace fields

At minimum:
- token/session identifier or stable pseudonymous session mapping;
- token position and prefill/decode phase;
- layer identifier;
- exact Top-K expert indices;
- selected routing weights, when available;
- gate-score margin between the K-th and K+1-th expert, when available;
- timestamp/order key;
- model revision and precision.

On a single GPU, source rank is not a physically observed distributed rank. If simulated sources/topologies are used for offline analysis, mark them explicitly as simulated.

### 3.4 Data splitting

Split complete prompts/sessions, not tokens. Keep training, validation and test session sets disjoint. Lock the shifted workload mixtures before selection. Task labels may be used by the explicitly labelled task-aware comparator, but not by the label-free DPCK pipeline.

The bundled four-prompt CSV is an instrumentation smoke test only. It is not a confirmatory benchmark.

## 4. Candidate construction — token-level FCA

For each layer, construct context (K=(T,E,I)):

[
(t,e)in I iff ein R_t.
]

A formal concept (c=(A_c,B_c)) yields an intent (B_c) that is present in every token in (A_c). Concepts may overlap.

Do not assume the full lattice is compact. Record:
- total candidate count;
- enumeration completeness or truncation;
- construction time and peak memory;
- concept intent sizes and extent sizes;
- candidate serialization size;
- duplicate/identical extent handling;
- random/session-bootstrap persistence.

For scale, a filtered mining algorithm may be needed, but its completeness and pruning semantics must be disclosed.

## 5. Two candidate selection algorithms

### 5.1 DPCK-DR (current draft)

Selects concepts using reductions in validation placement regret per incremental serialized byte. It is a heuristic. It has no generic greedy approximation guarantee because regret need not be monotone or submodular.

### 5.2 DPCK-RC (candidate revision)

For selected concept set (S), define token coverage:

[
U_t(S)=igcup_{cin S: tin A_c}B_c.
]

The residual upper bound is:

[
Delta(S)=sum_t w_t|R_tsetminus U_t(S)|.
]

The coverage objective (F(S)=sum_t w_t|U_t(S)|) is monotone submodular on the fixed token-expert incidence ground set.

Implement two versions only if the study can clearly distinguish them. DPCK-RC's byte-cost guarantee requires additive per-concept costs or a carefully modelled modular serialization. Shared metadata and cross-concept encoding interactions must be measured.

## 6. Equal-budget baselines

All methods use the same:
- training/validation/test splits;
- layer and source conditioning;
- feasible placement candidate set;
- objective function;
- optimizer and solver budget;
- serialization/evaluation pipeline;
- model trace and workload episodes.

Required baselines:
1. Full weighted route histogram (exact static-cost reference).
2. Top-frequency complete route hyperedges.
3. Pairwise co-activation graph.
4. Support-ranked closed itemsets.
5. Kernel Concept Set (KCS) or faithful reproduced criteria.
6. Label-free clustering over activation/routing features.
7. Task-aware co-activation comparator, with access to task labels disclosed.
8. DPCK-DR and/or DPCK-RC, once implemented.

Primary budget is serialized bytes. Pattern count is secondary. Include all information needed to use the representation, not only the concept intents.

## 7. Primary metrics

- absolute held-out placement regret;
- relative held-out placement regret;
- optimal-placement agreement where the reference is certified;
- best-known-reference gap where optimality is not certified;
- destination devices contacted per token (offline proxy);
- exact residual certificate (Delta(S));
- certificate slack (Delta(S)-operatorname{Regret}(S));
- normalized certificate tightness;
- serialized bytes and bytes per layer;
- lattice construction, selection and decision latency;
- peak memory and failure rates;
- session-level bootstrap confidence intervals.

Do not call a best-known solver result an optimum unless optimality is certified.

## 8. Shift protocol

Predeclare:
- in-distribution test mixtures;
- task-family mixture shifts;
- held-out task/session episodes;
- model precision or quantization shifts;
- optional cross-model evaluation.

Train and select on training/validation only. Test mixture weights and sessions remain locked. Statistical units are independent prompts or sessions, not individual tokens.

## 9. Residual-certificate checks

The theorem is token-level and ex-post unless a separate statistical guarantee is derived.

For each test episode:
1. Compute actual exact placement objective from the retained route trace.
2. Compute (U_t(S)), (widehat J(pi;S)) and (Delta(S)).
3. Select (widehatpi_S) with the same feasible placement set.
4. Verify (J(widehatpi_S)-J(pi^*)leqDelta(S)) whenever the exact reference optimum is known.
5. Measure certificate slack and its dependence on Top-K, concept intent size, route repetition and workload shift.
6. Include the cost of retaining/matching token-to-concept membership; do not claim compression if this cost is omitted.

A certificate computed from known test routes is a retrospective diagnostic, not a pre-deployment guarantee.

## 10. Go/no-go criteria

**Go to full study only if:**
- router instrumentation is validated against the model's actual dispatch indices;
- independent session splits and locked shifts are respected;
- the concept candidates can be constructed and serialized reproducibly;
- an FCA method shows a practically meaningful equal-byte advantage or a substantially tighter useful certificate than non-FCA alternatives;
- selection and inference overhead do not erase the advantage.

**Stop or revise the claim if:**
- DPCK does not beat the strongest equal-byte baseline;
- advantage disappears under held-out workload shift;
- coverage bound is near-vacuous on real Top-8 traces;
- concept membership/metadata erases the byte savings;
- clustering or hyperedge frequency selection matches the result;
- the selection process uses future/test information.

## 11. Stage B — physical multi-GPU validation

Required only for systems-performance claims:
- real expert-parallel runtime and physical source ranks;
- actual all-to-all dispatch and combine;
- communication bytes, contention and link topology;
- TTFT, TPOT, throughput, p95/p99 latency;
- expert compute/memory imbalance;
- placement, replication and reconfiguration costs;
- multiple seeds and paired workload sessions.

A T4 does not substitute for this stage.

## 12. Execution state

As of 2026-11-10:
- Stage A: prepared, not executed.
- Full FCA/DPCK selection comparison: not implemented end-to-end.
- Stage B: not performed.
- Hugging Face Space: source prepared, not deployed.
