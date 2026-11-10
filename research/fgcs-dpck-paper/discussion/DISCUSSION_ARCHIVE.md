# DPCK–MoE research discussion archive

**Repository:** [msellamiTN/FCA-MoE](https://github.com/msellamiTN/FCA-MoE)  
**Branch:** `research/fgcs-dpck-paper`  
**Archive date:** 2026-11-10  
**Scope:** Consolidated research discussion and decision history available in this conversation context.

> **Archival qualification.** This is a structured, source-faithful consolidation of the available discussions, research notes, and decisions. It is not a verbatim export of every earlier ChatGPT conversation. Where exact wording from user-provided assessments is retained, it is marked as quoted assessment; elsewhere, the content is paraphrased and organized for continuity. No unexecuted experiment is represented as completed.

## 1. Executive research trajectory

The project evolved through five distinct stages:

1. **FCA-MoE v0.2 synthetic validation:** test whether FCA can recover deliberately planted overlapping expert communities and reject a negative control with exactly matched route-pair marginals.
2. **Objective-specific analysis:** determine which routing statistics are sufficient for which MoE resource-management objectives; reject the idea that FCA is universally better than graphs or hypergraphs.
3. **Decision-Preserving Concept Kernel (DPCK):** formulate selection of FCA-derived concepts under a representation budget, evaluated by downstream placement regret.
4. **FGCS manuscript and T4/Hugging Face experiment scaffold:** prepare a methodological LaTeX draft, a Tesla T4 router-trace capture path, a finite parity counterexample test, and a Gradio trace-inspection Space template.
5. **Reviewer-driven narrowing and residual-coverage theory:** reduce the first paper to static destination-deduplicated expert placement under workload shift; investigate a token-level residual-coverage certificate and its submodular selection structure.

The current position is deliberately provisional: the mathematical proposal is promising, but its practical value, certificate tightness, and performance under workload shift remain experimental questions.

## 2. Stage A — FCA-MoE v0.2: controlled synthetic methodology test

### 2.1 Intended question

Can FCA recover deliberately planted, overlapping expert co-activation communities from window-level routing contexts, while a negative control with the same global route-pair distribution does not exhibit those persistent window-level communities?

This was a controlled methodology test, not a pretrained MoE benchmark and not evidence of a placement speedup.

### 2.2 Experimental construction

- 8 experts and Top-2 routes.
- 4,096 synthetic token samples.
- Four overlapping four-expert communities:
  - {E0,E1,E2,E3}
  - {E2,E3,E4,E5}
  - {E4,E5,E6,E7}
  - {E0,E1,E6,E7}
- Positive control: one community governs each 64-token window; tokens are routed using pairs within that community.
- Negative control: exact permutation of the positive control's route-pair sequence. This preserves the exact global route-pair multiset and expert marginals while removing the persistent window membership.
- Complete-window train/test split; the router is trained on the first 80% of windows.
- FCA contexts are built from held-out router-predicted routes.
- Full NextClosure enumeration is used in the reported v0.2 validation.
- Sensitivity analysis varies window sizes 32/64/128 and activation thresholds 0.10/0.20/0.30.

### 2.3 Reported local CPU validation

These are prior local CPU results, not Colab T4 results.

| Condition | Held-out Top-2 exact match | Complete concepts | Order-4 concepts | Expected order-4 recall | Order-4 precision |
|---|---:|---:|---:|---:|---:|
| Window-correlated positive control | about 99.64% | 10 | 4 | 1.00 | 1.00 |
| Exact marginal-matched IID negative control | about 99.52% | 1 | 0 | 0.00 | not applicable |

Eight unit tests passed in the local project validation. The related repository commit was `4a8cd192c46bea565adb5a8e9770dd5584321e27`.

### 2.4 Sensitivity warning

Thresholding and binary window contexts can produce spurious patterns. In the local IID sensitivity run at window 32 and threshold 0.20, 196 concepts and 52 order-4 intents were observed; only four of the 52 order-4 intents matched the designed communities, approximately 7.69% precision. At window 64 and threshold 0.20, 120 concepts and 29 order-4 intents were observed.

**Interpretation:** recall alone is inadequate. False-discovery/precision, stability, window size, threshold and independent-session replication matter. The positive control is a planted-pattern recovery test, not emergent discovery.

### 2.5 What v0.2 does not establish

It does not validate:
- routing patterns in a pretrained MoE;
- physical expert placement;
- network communication or all-to-all latency;
- the physical validity of the virtual placement cost;
- FCA superiority over full route hypergraphs;
- robustness under real task-family distribution shift.

An older v1 smoke-test result (8 experts, Top-2, synthetic data) reported approximately 0.6709 route exact match and one concept. It is superseded by the v0.2 control experiment and should not be mixed with the v0.2 evidence.

## 3. Stage B — objective-specific sufficiency and the three application tracks

The discussion separated three possible MoE applications because their sufficient information differs.

### 3.1 Communication-aware expert placement

For a static additive per-event objective, source-conditioned exact routing-set frequencies are sufficient to evaluate any placement.

For destination-deduplicated Top-2, source-conditioned pair co-selection counts can be sufficient under the corresponding fixed dispatch semantics and additive cost. For general Top-K, pairwise marginals need not determine the union of remote destinations contacted by a token.

The central caution is that assignment-level dispatch and destination-deduplicated dispatch are different cost semantics. Neither should be silently treated as a complete physical all-to-all model.

### 3.2 Expert working-set/residency

FCA may describe recurring co-activation patterns, but binary incidence alone does not predict the future. Cache/residency decisions depend on order, expert sizes, transfer costs, GPU memory pressure, KV-cache competition and PCIe/network bottlenecks.

A credible study would need strong caching/prefetching baselines, including LRU/LFU and learned sequence predictors. This track was judged less suitable as the first paper and was dropped from the narrowed scope.

### 3.3 Temporal scheduling

Scheduling requires arrival times, queues, precedence, compute cost, communication cost, concurrency and causal state. A lattice is descriptive unless connected to a prediction/action policy. This track was dropped from the first paper.

### 3.4 No universal representation claim

There is no universal sufficient statistic for communication placement, caching and scheduling together. Every theorem must state the precise objective, event semantics, action space and assumptions under which its representation is sufficient.

## 4. Stage C — initial DPCK and the first critical assessment

The initial idea was to select a concept subset that preserves downstream decisions under a representation budget.

The general regret form was:

[
S_B^* = argmin_{Ssubseteqmathcal C, mathrm{Bytes}(S)le B}
mathbb E_Q[operatorname{Regret}_Q(S)].
]

The discussion identified the following critical weaknesses:

- Exact concept intents from a token-by-expert binary context are closed itemsets. Closed-itemset mining is therefore an equivalent pattern representation, not a fundamentally different information model.
- The full concept lattice is not guaranteed to be compact. A token route is a closed intent in the token-by-expert context, while intersections may add further intents.
- The exact weighted routing hypergraph/histogram retains higher-order route-set information needed for the stated static additive objective. FCA cannot be claimed to contain more information.
- Compactness must be measured on the selected, serialized artifact—not asserted for the whole lattice.
- A fixed byte budget must account for identifiers, extents or feature mappings, weights, offsets, model coefficients and metadata.
- Comparing each placement method under its own cost matrix or solver invalidates the comparison; all methods must share the same action bank, cost evaluator and solver budget.
- A regret guarantee against a full-data optimum may be circular if computing that optimum is itself the expensive operation.
- Non-monotone or non-submodular regret objectives do not inherit a greedy approximation guarantee.
- A concept extent is not automatically a disjoint workload regime; concepts overlap.
- Selection, matching, prediction and action execution are separate algorithmic steps.
- Causal policy decisions must not use future routing events.
- Window-level contexts can saturate as window size grows; threshold choices can manufacture spurious concepts.

### 4.1 First user-provided assessment — retained conclusion

The first assessment characterized the idea as coherent and testable but said that two claims were wrong as stated: universal compactness and a representational comparison against closed-itemset mining. It recommended merging:
- budgeted regret curves against equal-budget baselines; and
- label-free regime discovery,

while treating hardness and restricted guarantees as a modest theory section. It recommended dropping caching unless a pilot could beat learned predictors.

It proposed a pilot with two open MoE models, four or more task families, equal serialized budgets, full-hypergraph reference placement, shifted task mixes and held-out evaluation.

The assessment also noted that an exact route-set hypergraph is already sufficient for the specified static per-token cost. The correct research claim is therefore not “more information,” but potentially “near-equal decisions using fewer selected patterns.”

## 5. Stage D — first FGCS manuscript and experiment scaffold

The paper title used in the first draft was:

*Decision-Aware FCA Summaries for Workload-Shift-Robust Mixture-of-Experts Placement*

The draft includes:
- weighted route-histogram sufficiency for a separable static cost;
- a constructive Top-3 parity example;
- FCA contexts and concept descriptors;
- a byte-budgeted DPCK formulation;
- a conditional two-sided argmin-stability theorem;
- a byte-aware forward-selection heuristic, explicitly without a greedy ratio claim;
- equal-byte baselines;
- session-level held-out and shifted-workload methodology;
- an experimental distinction between T4 Stage A and physical multi-GPU Stage B;
- threats to validity and pending-result templates.

### 5.1 Exact weighted route-histogram result

Let an event be (r_t=(ell_t,s_t,R_t,w_t,	au_t)). If the cost is separable over events and depends only on ((ell,s,R)) and placement (pi), then:

[
J(pi;D)=sum_{ell,s,R}H_{ell,s}(R)c(ell,s,R,pi),
]

where (H_{ell,s}(R)) is the count of events with that layer, source and exact route set. Equal histograms imply equal objective values for every feasible placement and equal optimal-placement sets.

This is a sufficiency theorem under explicit assumptions, not a claim that every temporal or batched systems objective is captured by the histogram.

### 5.2 Top-3 parity counterexample

Six experts are arranged in three pairs:
((A_0,A_1),(B_0,B_1),(C_0,C_1)).

The even routes are:
- (A_0B_0C_0)
- (A_0B_1C_1)
- (A_1B_0C_1)
- (A_1B_1C_0)

The odd routes are:
- (A_0B_0C_1)
- (A_0B_1C_0)
- (A_1B_0C_0)
- (A_1B_1C_1)

Both traces have identical expert marginals and pairwise co-activation counts. With three labelled devices of capacities (3,2,1), source device 1, and cost equal to the number of distinct remote destination devices, exhaustive enumeration of 60 feasible placements gives:
- 12 optimal placements per trace;
- mean optimal cost 1.25;
- no common optimal placement;
- an even-optimal placement evaluated on the odd trace has mean cost 1.50.

This is a constructed finite counterexample, not a measurement from a physical MoE deployment. The executable test is in the repository.

### 5.3 T4 and Hugging Face roles

The selected trace-capture candidate is `allenai/OLMoE-1B-7B-0125`, with Top-8 routing. The notebook is intended to:
- check the actual allocated GPU rather than assume a T4;
- record model revision and environment;
- capture prefill routing;
- compare hooked router indices with independently reconstructed indices from router logits;
- export trace JSONL, manifest and fidelity report.

Quantization can alter hidden states and routing; precision is part of the recorded experimental configuration.

The Hugging Face Space is a CPU-first trace-inspection and schema-validation interface. It is not a model host, DPCK implementation, placement optimizer or physical systems benchmark.

### 5.4 Stage A versus Stage B

**Stage A (Colab T4):** actual model routing capture, instrumentation validation, offline placement analysis with explicitly simulated topology. It cannot measure real source ranks, multi-GPU all-to-all completion time or network contention.

**Stage B (separate expert-parallel cluster):** real source-rank dispatch, measured network bytes and latency, throughput, TTFT/TPOT, p95/p99, load imbalance, replication and reconfiguration costs.

No Stage B experiment has been performed.

## 6. Stage E — second critical review and scope reduction

The second assessment judged the full framework as titled to be too broad and not provable as a whole. It identified:

- Partial provability: structural facts and a conditional regret bound are provable; compactness and usefulness are empirical.
- Sound FCA foundations but no formally defined relation between token, window and transition lattices.
- A proxy communication objective that must be validated against measured all-to-all behavior.
- Loss of gate weights and routing-margin information in binary contexts.
- A mistaken framing of closed-itemset mining as a different representation.
- No reason to claim compactness for the full lattice.
- No predictive cache mechanism.
- No exact robust stability computation at scale.
- Low acceptance probability for systems venues without real cluster speedups.
- More plausible FCA/data-mining/knowledge-representation positioning if the result is mainly representational or methodological.

The review specifically warned that a Top-8 residual bound can be loose: if selected concepts contain only two or three experts, the uncovered-expert upper bound can permit five or more extra devices. In an eight-device deployment with one source device, the actual number of distinct remote destinations is at most seven; in general it is bounded by min(K, |D|-1). The numerical comparison therefore depends on the declared device count.

### 6.1 Expert recommendation retained

1. Drop caching and scheduling from the first paper.
2. Drop the undefined multi-resolution claim.
3. Focus on one decision problem: static destination-deduplicated expert placement.
4. Preserve the exact route histogram as the static-cost reference.
5. Compare methods at equal serialized-byte budgets.
6. Make workload shift a locked, held-out test condition.
7. Run the pilot before committing to the main novelty claim.
8. Submit to a systems venue only if the study has physical systems evidence; otherwise consider an FCA/data-mining/knowledge-representation venue.

## 7. Stage F — residual-coverage proposal and its corrected scope

The final discussion introduced a potentially stronger FCA-specific theoretical direction. It is a **candidate revision**, not yet an accepted or experimentally validated result.

### 7.1 Token-level context is required

Construct the formal context (K=(T,E,I)), with

[
(t,e)in I iff ein R_t.
]

For a concept (c=(A_c,B_c)), if (tin A_c), then (B_csubseteq R_t). For selected concepts (S), define:

[
U_t(S)=igcup_{cin S: tin A_c}B_c.
]

Thus (U_t(S)subseteq R_t).

**Important correction:** the currently drafted paper constructs layer-wise window contexts. Its window concepts do not automatically satisfy this per-token subset property. The residual-coverage theorem therefore requires a token-level formal context or a separately proven bridge. It cannot simply be asserted for the current window-level method.

### 7.2 Residual-coverage bound

For a non-replicated placement (pi), source (s_t), and destination-deduplicated dispatch, define:

[
D_t(X,pi)={pi(e):ein X, pi(e)
e s_t}.
]

The exact per-token cost is (c_t(pi)=|D_t(R_t,pi)|), and the selected-concept lower estimate is:

[
widehat c_t(pi;S)=|D_t(U_t(S),pi)|.
]

Since (U_t(S)subseteq R_t),

[
0le widehat c_t(pi;S)le c_t(pi),
]

and

[
0le c_t(pi)-widehat c_t(pi;S)
le |R_tsetminus U_t(S)|.
]

For nonnegative event weights (w_t), define:

[
Delta(S)=sum_t w_t|R_tsetminus U_t(S)|.
]

Then, under the stated static destination-count objective,

[
widehat J(pi;S)le J(pi)lewidehat J(pi;S)+Delta(S).
]

If (widehatpi_S) minimizes (widehat J(cdot;S)) over the same feasible placement set and (pi^*) minimizes (J), then:

[
J(widehatpi_S)-J(pi^*)leDelta(S).
]

This is a valid conditional residual-regret bound under the stated construction and objective. It is not a latency guarantee.

### 7.3 Submodularity claim and conditions

Define token-expert incidence coverage:

[
F(S)=sum_t w_t|U_t(S)|.
]

For a fixed trace and nonnegative weights, selecting concepts covers a union of token-expert incidences. Therefore (F) is monotone submodular. Since:

[
Delta(S)=sum_t w_t|R_t|-F(S),
]

minimizing this residual upper bound is equivalent to maximizing (F(S)).

Guarantees require careful constraints:
- Under a cardinality constraint, standard greedy achieves the usual (1-1/e) bound for monotone submodular maximization.
- Under a byte-cost knapsack constraint, use an appropriate knapsack-constrained submodular maximization algorithm; do not claim the guarantee for an arbitrary gain-per-byte heuristic.
- The cost function must be modular/additive in selected concept costs. Shared serialization overhead or interaction-dependent byte costs require separate treatment.

This result optimizes the residual coverage certificate, not necessarily the actual placement regret directly.

### 7.4 Critical unresolved limitations

The certificate can be loose, especially for Top-8 and small concept intents. Report:
- actual regret;
- (Delta(S));
- certificate slack (Delta(S)-operatorname{Regret}(S));
- normalized certificate ratio/slack;
- bytes and selection time.

There is also an operational issue: computing (U_t(S)) and (widehat J) requires knowing token routes or retaining an adequate token-to-concept membership representation. If the proposed summary does not retain enough information to compute its estimate, the claimed compression may disappear.

Further, a certificate calculated on held-out test routes is an ex-post certificate. It is not a prospective deployment guarantee unless we derive a statistical estimate and confidence bound for the future workload distribution. Distribution shift makes that an open problem.

### 7.5 Robustness to routing noise

Exact formal concept stability is computationally difficult; it should not be a mandatory pipeline operation. Candidate empirical checks:
- session-bootstrap persistence of selected intents;
- FP32 versus quantized routing comparison;
- sensitivity to the Top-K versus Top-(K+1) gate-score margin;
- context thresholds and window parameters where window contexts are retained as a secondary analysis.

Gate weights and margins should be retained even if the first theorem is about hard Top-K assignments.

## 8. Current narrowed paper candidate

**Candidate title (not yet approved):**

*Residual-Certified FCA Pattern Selection for Workload-Shifted Mixture-of-Experts Placement*

**Single decision problem:** static, destination-deduplicated expert placement under workload shift.

**Single main question:**

Can a byte-budgeted selection of token-level FCA patterns provide a useful residual-regret certificate and preserve held-out placement decisions better than equally budgeted non-FCA summaries?

**Candidate contribution set:**
1. Objective-specific exact route-histogram sufficiency for static additive placement.
2. Top-3 parity counterexample against pairwise-only summaries.
3. Token-level FCA descriptors with overlapping membership.
4. Residual-coverage certificate and its conditions.
5. Submodular coverage selection under explicit budget assumptions.
6. Equal-byte, workload-shift evaluation against hyperedge, pairwise, closed-itemset, KCS, clustering and task-aware baselines.

This is a research direction under evaluation, not a proven superiority claim. The title, final algorithm, and target venue remain provisional.

## 9. Current status at archive time

| Item | Status |
|---|---|
| FCA-MoE v0.2 controlled synthetic CPU experiment | Completed in prior local environment |
| v0.2 unit tests | 8 passed in prior local environment |
| Top-3 parity exhaustive test | Finite example verified locally; test source in repository |
| FGCS LaTeX manuscript | Compiled draft; not empirically complete |
| Colab T4 OLMoE trace capture | Not executed |
| Full token-level FCA/DPCK implementation | Not implemented end-to-end |
| Residual-coverage theorem | Candidate derivation; needs formal manuscript integration and independent proof audit |
| Byte-aware submodular selector | Not implemented |
| Equal-byte real-trace baseline experiment | Not executed |
| Held-out workload-shift evaluation | Not executed |
| Hugging Face Space | App template prepared; not deployed |
| Physical multi-GPU Stage B | Not performed |
| Journal submission | Not ready |

## 10. Research conduct rules retained from the discussion

- Do not invent empirical results or present simulated topology values as physical measurements.
- Distinguish proof, finite exhaustive computation, local CPU validation, Colab T4 validation and cluster validation.
- Keep a full exact route-histogram reference for the static additive objective.
- Use one shared placement action bank, cost evaluator and solver budget.
- Compare by serialized bytes, not only by number of patterns.
- Split data at independent session/prompt boundaries; do not randomly split tokens from the same session.
- Lock test sets and workload-shift scenarios before fitting or selecting features.
- Report failure cases, certificate slack and all selection overhead.
- Do not equate FCA concept extents with semantic expert specializations or disjoint workload regimes.
- Do not claim a physical distributed-systems improvement without Stage B.
- If DPCK does not outperform the strongest equal-byte non-FCA baseline, revise or reject the central claim rather than reinterpreting the evidence.

## 11. Related-work/source map from the discussion

The following works were repeatedly treated as important comparison points. Their inclusion here records the discussion's source map; the final paper must independently verify metadata, versions, claims and publication status before submission.

- TACG, task-aware co-activation grouping: https://arxiv.org/abs/2606.01007
- CAP, communication-aware MoE placement: https://arxiv.org/abs/2607.05116
- Replication in graph partitioning and scheduling: https://arxiv.org/abs/2605.00209
- MoETuner: https://arxiv.org/abs/2502.06643
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- ExpertFlow: https://arxiv.org/abs/2410.17954
- Kernel Concept Set Approach: https://inass.org/wp-content/uploads/2024/03/2024083142-2.pdf
- Kernel-concept mining with concept-set generation-cost optimization (2026): see bibliography entry and verify final publication metadata before submission.
- OLMoE model card: https://huggingface.co/allenai/OLMoE-1B-7B-0125
- Existing project: https://github.com/msellamiTN/FCA-MoE

---

**End of consolidated discussion archive.** Subsequent decisions and experiments should be appended to the decision log and experiment register rather than silently rewriting the historical record.
