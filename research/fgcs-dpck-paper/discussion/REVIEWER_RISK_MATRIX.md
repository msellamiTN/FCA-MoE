# Reviewer-risk matrix and claim discipline

**Updated:** 2026-11-10

This matrix records the most consequential scientific objections raised in the discussion and the action required to address them.

| Reviewer objection | Current answer | Required evidence or correction | Status |
|---|---|---|---|
| FCA intents are just closed itemsets | Acknowledged; same binary context, same closed patterns | Compare decision-driven selection, not pattern representation identity | Open |
| Full concept lattice is not compact | Acknowledged; no compactness claim | Measure selected serialized artifacts at equal bytes | Required |
| Hypergraph already contains exact higher-order routes | Acknowledged for static additive route-event cost | Keep full weighted route histogram as exact reference | Retained |
| Why not top-frequency hyperedges? | No answer without data | Equal-byte comparison under held-out and shifted workloads | Not executed |
| Why FCA rather than clustering? | Potential overlapping formal descriptors; not sufficient evidence | Compare label-free clustering, including its storage and predictor costs | Not executed |
| DPCK regret selection is circular | Existing draft uses validation-regret heuristic; no global certificate | State finite action bank and solver reference; separate candidate residual certificate | Open |
| The (2epsilon) theorem is generic | Correct | Treat as supporting lemma, not primary novelty | Accepted |
| Residual bound may be loose for Top-8 | Correct risk | Measure certificate slack on real traces; do not assume useful tightness | Not executed |
| Window concepts do not imply token subsets | Correct | Use token-level context for residual theorem or prove a separate aggregation relation | Required |
| Submodularity does not automatically imply knapsack greedy bound | Correct | Use correct knapsack-constrained algorithm; define additive byte costs | Required |
| Certificate uses test routes | Potentially ex-post | Derive a prospective statistical bound before claiming deployment guarantees | Open |
| Token-to-concept matching may erase compression | Serious operational issue | Count membership maps, matching cost and summary metadata | Open |
| Binary incidence ignores routing weights and margins | Partly acknowledged | Preserve gate weights/margins and run quantization/top-K margin sensitivity | Required |
| Devices-per-token is not measured all-to-all latency | Acknowledged | Validate against physical Stage B; avoid latency terminology in Stage A | Required |
| Task-aware grouping already addresses regimes | Acknowledged; task labels differ from label-free discovery | Compare TACG-style label-aware baseline and label-free clustering | Not executed |
| Caching uses predictors, not just patterns | Acknowledged | Exclude caching from first paper | Accepted |
| Multi-resolution is three disconnected contexts | Acknowledged | Remove from first paper | Accepted |
| Systems venues need real speedups | Acknowledged | Physical multi-GPU validation before systems claims | Not executed |
| FGCS fit without systems results is uncertain | Acknowledged | Retain as conditional target; venue decision follows evidence | Open |

## 1. Claim discipline

### Claims allowed now

- FCA concepts of the specified binary token-by-expert context correspond to closed itemsets.
- The weighted route histogram is sufficient for the specified separable static objective.
- The finite Top-3 parity construction demonstrates pairwise-marginal insufficiency for its specified cost and action space.
- The v0.2 synthetic CPU experiment has the previously recorded controlled results.
- The residual-coverage inequalities follow under the token-level membership and static destination-count assumptions, subject to independent formal review.

### Claims not allowed now

- FCA is more compact or more informative than a full routing hypergraph.
- DPCK outperforms hypergraph or pairwise selection.
- Concepts automatically discover semantic task regimes.
- DPCK improves real MoE latency or all-to-all throughput.
- T4 results validate multi-GPU systems performance.
- A test-trace residual certificate guarantees an unknown future workload.
- The new residual-selection algorithm has been implemented or experimentally validated.
- The Hugging Face Space has been deployed.

## 2. Conditions for strengthening the paper

The paper can only move toward a strong FGCS systems submission after:
1. the token-level method is implemented;
2. the residual theorem and knapsack assumptions are formally audited;
3. equal-byte comparisons beat strong baselines under locked workload shifts;
4. certificate tightness and all runtime/storage overheads are reported;
5. physical multi-GPU experiments connect the offline objective to measured network and end-to-end performance.

Otherwise, the honest destination is a narrower FCA/data-mining/knowledge-representation paper, provided its methodological contribution survives the same baseline tests.
