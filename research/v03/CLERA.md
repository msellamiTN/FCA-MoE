# CLERA: Cross-Layer Expert Role Alignment

## Objective

CLERA investigates whether expert routing profiles in one MoE layer have
reproducible correspondences with routing profiles in another layer, and
whether higher-order FCA intents learned in source sequences are transported
to held-out target sequences under the learned correspondence.

**Expert IDs are layer-local.** Expert 17 in one layer is not assumed to be
the same module as expert 17 in another layer. CLERA estimates a behavioural
correspondence from observed routing activation profiles; it does not compare
expert parameters or prove functional/semantic equivalence.

## Data and alignment

The initial experiment uses the existing Qwen1.5-MoE-A2.7B teacher-forced
prefill routing traces:

- 24 routed layers, 60 routed experts, Top-4;
- four sequences of 256 aligned token positions;
- source layer 11 and target layers 14 and 16;
- binary token-by-expert incidence matrices.

A source and target row refer to the same token position in the same sequence.
The alignment CSV and context NPZ are checked for complete, contiguous token
positions and exact Top-4 cardinality.

## Expert similarity and one-to-one mapping

For source expert i and target expert j, let x_i and y_j be their binary
activation profiles over aligned training token positions. Their phi
coefficient is the Pearson correlation of the two binary vectors:

```text
phi(i,j) = (n11*n00 - n10*n01) /
           sqrt((n1.*n0.)*(n.1*n.0))
```

Equivalently, it is the normalized dot product of the centered binary
profiles. If either profile is constant, its correlation is undefined; CLERA
sets that entry to zero and reports the fraction of matched pairs with both
profiles nonconstant.

A one-to-one assignment pi is estimated by the Hungarian algorithm:

```text
pi* = argmax_pi sum_i phi(i, pi(i))
```

The assignment is learned using three sequences only.

## Held-out validation

Leave-one-sequence-out (LOSO) validation is performed over the four sequences:

1. Hold out sequence s.
2. Concatenate the three remaining source profiles and target profiles.
3. Estimate the 60 x 60 phi matrix and Hungarian assignment on those training
   sequences.
4. Evaluate the frozen mapping on the held-out sequence.
5. Repeat for all four held-out sequences.

The primary held-out endpoint is the mean phi coefficient across the 60
matched expert pairs. Secondary descriptive endpoints are:

- median matched phi;
- fraction of matched pairs with positive phi;
- fraction with nonconstant source and target profiles;
- mapped Top-4 overlap: mean over tokens of the number of mapped source
  experts also selected in the target layer, divided by four;
- same-index identity mapping scores, to show whether the estimated
  correspondence improves over raw index equality.

Mapping stability is the mean fraction of source experts assigned to the same
target expert between pairs of LOSO folds.

## Circular-shift permutation reference

For each permutation and each target layer, independently circular-shift the
target-layer token rows within each sequence by a non-zero random offset.
This preserves the sequence's expert marginals, Top-4 states and circular
within-layer temporal order, while breaking token-position correspondence
with the fixed source layer (apart from the single circular boundary).

For every permuted dataset, CLERA re-runs the complete LOSO mapping selection
and held-out scoring procedure. This is important: the Hungarian assignment
is re-estimated inside every permutation, not held fixed from observed data.

The primary family-wise permutation statistic is the maximum held-out mean
matched phi over the target layers included in the run:

```text
T_obs = max_l observed_mean_matched_phi(l)
T_b   = max_l permuted_mean_matched_phi_b(l)
p     = (1 + count_b[T_b >= T_obs]) / (B + 1)
```

The same max construction is reported separately for mapped Top-4 overlap;
this is a secondary endpoint. The primary p-value adjusts selection among the
target layers supplied to the run. It does **not** correct selection of source
layer 11, selection of the 1% support threshold, or the preceding 24-layer
discovery. Therefore, these results remain exploratory.

Circular shifts are a test of aligned cross-layer routing correspondence.
They are not a null model for the occurrence of FCA tuples, because changing
row order does not change a formal context's extent/intent structure.

## Held-out FCA tuple transport

Using the same fold split:

- enumerate supported closed intents of cardinality at least three in each
  source training sequence at the 1% sequence support (3 of 256 objects);
- intersect those intent sets to obtain source tuples universal across the
  three training sequences;
- map source expert IDs through the training-only Hungarian assignment;
- enumerate supported higher-order closed intents in the held-out target
  sequence;
- measure exact transported-intent matches and best Jaccard overlap, both
  across arbitrary cardinalities and restricted to equal cardinality.

This is a held-out descriptive measure. It is **not** assigned a permutation
p-value by the circular-shift test, since circular shifts preserve the set of
closed intents. A separate pairwise-temporal generative null would be needed
to test tuple transport significance.

## Execution

From `research/v03`:

```bash
python scripts/clera_cross_layer_alignment.py \
  --root "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1" \
  --source-layer 11 \
  --target-layers 14 16 \
  --permutations 499 \
  --seed 20261119
```

Outputs are written to:

```text
sequence_replication/clera_layer11_targets_14_16_v1/
```

- `clera_summary.json`: configuration, inputs, scores, permutation results;
- `clera_fold_results.csv`: per-target, per-held-out-sequence scores and
  tuple transport diagnostics;
- `clera_expert_mappings.csv`: four folds x source experts x target layers;
- `clera_permutation_max_statistics.csv`: permutation max statistics.

## Interpretation limits

1. Four sequences are a small validation sample, not broad corpus-level
   generalization.
2. Source and target layers were selected from the earlier 24-layer scan.
   The primary permutation adjustment only covers target layers included in
   this CLERA run.
3. Activation-profile correspondence is behavioural routing similarity, not
   equivalence of expert parameters, computations, or semantics.
4. Tuple transport is descriptive until evaluated against a pairwise-temporal
   generative null.
5. The semantic context attachment supplied with the discovery contains
   placeholder token labels such as `tok_17`; it does not provide readable
   lexical content from which semantic specialisation can be inferred.
6. Confirmatory claims require new independent sequences and a frozen protocol
   that accounts for source-layer and target-layer selection.
