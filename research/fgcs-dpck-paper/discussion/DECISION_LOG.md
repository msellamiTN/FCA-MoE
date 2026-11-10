# DPCK–MoE decision log

**Updated:** 2026-11-10  
**Branch:** `research/fgcs-dpck-paper`

This file distinguishes decisions that are currently supported from candidate revisions that still require explicit approval or experimental validation.

## 1. Current scope

| Decision | Current status | Rationale |
|---|---|---|
| First paper studies one problem: static destination-deduplicated expert placement | **Recommended scope** | Avoids mixing placement, caching and scheduling objectives |
| Workload shift is a primary evaluation condition | **Retained** | Task-agnostic averages can conceal task-family-specific routing structure |
| Full weighted route histogram is the exact static-cost reference | **Mathematically established under separable event cost** | Exact objective-specific sufficiency |
| Pairwise graph is a baseline, not universally sufficient | **Mathematically supported by a constructed Top-3 counterexample** | Pairwise marginals need not determine higher-order destination sets |
| Full FCA lattice is not claimed to be compact | **Accepted correction** | No inherent compactness guarantee |
| Closed itemsets are not a distinct information representation from FCA intents | **Accepted correction** | Same binary context yields same closed patterns |
| DPCK must be evaluated at equal serialized-byte budgets | **Required** | Pattern count is not a storage budget |
| All methods use the same placement action set, cost evaluator and solver budget | **Required** | Prevents unfair comparisons |
| T4 is trace capture/offline analysis only | **Required** | No physical multi-GPU source ranks or all-to-all measurement |
| Physical distributed performance requires a separate expert-parallel cluster | **Required** | FGCS systems claim needs real systems evidence |
| Caching and scheduling are excluded from the first paper | **Recommended and retained for narrowing** | Different objectives and strong predictors/baselines |
| Multi-resolution FCA is removed from the first version | **Recommended** | No formal mapping among token, window and transition lattices |

## 2. Candidate revisions — not yet locked

### 2.1 New candidate title

*Residual-Certified FCA Pattern Selection for Workload-Shifted Mixture-of-Experts Placement*

Status: proposed; not yet approved as the final title.

### 2.2 Move from window-level to token-level FCA for the main theorem

The existing FGCS manuscript uses layer-wise window contexts. The proposed residual-coverage theorem requires token-level objects and expert attributes. A window-level concept intent is not guaranteed to be a subset of each individual token route in that window.

Status: required if the residual theorem is adopted; implementation and manuscript revision pending.

### 2.3 Residual-coverage certificate

Candidate definition:

[
U_t(S)=igcup_{cin S: tin A_c}B_c,
qquad U_t(S)subseteq R_t,
]

[
Delta(S)=sum_t w_t|R_tsetminus U_t(S)|.
]

For destination-deduplicated static per-event cost and a common placement set:

[
J(widehatpi_S)-J(pi^*)leqDelta(S)
]

when (widehatpi_S) minimizes the lower estimate induced by (U_t(S)).

Status: candidate theorem; requires formal integration, independent proof audit, precise assumptions and empirical tightness study.

### 2.4 Submodular coverage selection

[
F(S)=sum_t w_t|U_t(S)|
]

is monotone submodular for fixed token-expert incidences and nonnegative weights. Maximizing it is equivalent to minimizing the residual bound up to a constant.

Status: mathematical property of the stated construction. A cardinality constraint supports standard greedy (1-1/e); byte knapsack requires an appropriate algorithm and additive item costs. Shared serialization costs can invalidate the modular-cost assumption.

### 2.5 DPCK identity

Keep two concepts distinct in implementation and prose:

- **DPCK-DR:** the current draft's validation-regret-driven selector; a heuristic, with no generic submodular guarantee.
- **DPCK-RC:** the candidate residual-coverage selector; optimizes a residual certificate, not actual placement regret directly.

Do not conflate them. The final name and algorithm must be decided after the pilot.

## 3. Locked scientific boundaries

1. The exact route histogram sufficiency result applies only to a separable static objective whose per-event cost depends on layer, source rank, route set and placement.
2. The per-token destination-count objective is a proxy, not measured all-to-all latency.
3. Same pairwise marginals do not imply same higher-order route tuples or same optimal placement in general.
4. Concepts can overlap; they are not automatically partitions or semantic regimes.
5. Token-level intent inclusion is needed for the residual-coverage inequality.
6. An ex-post certificate computed from test routes is not a prospective guarantee for an unknown future workload.
7. A useful compression claim must include the cost of membership matching, feature extraction, coefficients, metadata and runtime artifacts.
8. A guarantee about residual coverage is not automatically a guarantee about physical latency.
9. No real OLMoE trace, Hugging Face Space deployment or physical multi-GPU result is currently claimed.

## 4. Submission positioning

- **FGCS:** retain as a conditional target if physical distributed systems validation is completed and the work demonstrates value in distributed resource management/communication.
- **FCA/data-mining/knowledge-representation venue:** plausible alternative if the final contribution remains a formal pattern-selection method with offline trace evaluation and no physical systems result.
- **Systems/ML venues:** require substantially stronger measured systems improvements or broader empirical evidence than the current draft contains.

No venue is guaranteed. The target must follow the evidence rather than dictate claims.

## 5. Change control

For each future change, append:
- date;
- decision changed;
- prior wording/assumption;
- new wording/assumption;
- proof or experiment that motivates the change;
- implementation commit;
- status: proposed, accepted, rejected or superseded.

Do not delete rejected hypotheses from the archive. Record why they were rejected.
