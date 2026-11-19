# FCA–MoE F2-E: nested parametric bootstrap

## Purpose

F2-D tested the higher-order FCA counts conditional on one fitted layer-0 pairwise MaxEnt distribution and one selected temporal Gibbs mixture. F2-E repeats fitting and temporal-weight selection inside an outer parametric bootstrap, so the global reference distribution includes these sources of variability under the specified null family.

The experiment analyses one explicitly selected routed layer of the Qwen1.5-MoE-A2.7B trace (layer 0 by default): four 256-token sequences, 60 routed experts, exact Top-4 contexts. The layer ID is part of the run fingerprint and output manifest.

## Null model

For each token, the selected expert set is a K-subset of E experts:

\[
P_\theta(S\mid |S|=K)=Z_K(\theta)^{-1}
\exp\left(\sum_{i<j}\theta_{ij}\,1[i,j\in S]\right).
\]

The fitted objective is

\[
J_\lambda(\theta)=\log Z_K(\theta)-\hat p^\top\theta
+\frac{\lambda}{2}\|\theta\|_2^2,
\]

where empirical pair moments are divided by the number of objects (not by \(\binom K2\)). Since every state has exactly K active experts, the sum of pair indicators is exactly \(\binom K2\). The ridge penalty fixes the additive gauge; the implementation centers the fitted parameter vector and reports its gauge sum.

For E=60,K=4, the state space has \(\binom{60}{4}=487,635\) states and 1,770 pair parameters. The fitter evaluates state energies without materializing a 487,635 × 1,770 dense feature matrix.

## Temporal kernel

Each transition chooses k retained experts from the current Top-K set, with k in {0,1,2,3}. For k=0 it samples a new state from the fitted stationary distribution. For k>0 it samples from the exact conditional distribution of a Top-K state containing the retained subset. A mixture over k with weights w is reversible with respect to the fitted pairwise distribution. The code constructs exact conditional completion CDFs.

Temporal weights are selected by minimizing the lag-profile RMSE over a simplex grid. The full protocol uses grid step 0.1 (286 candidates) and eight screening replicates per candidate, matching the earlier F2-C screening design. The pilot uses a deliberately coarser grid and fewer replicates and is not inferential.

## Test statistic

The predeclared family is:

- pooled higher-order closed-concept count at 1%, 2%, 5% relative support;
- mean per-sequence higher-order closed-concept count at 1%, 2%, 5% relative support.

For an observed context X, an independent inner null sample estimates each metric's upper-tail probability using the plus-one Monte Carlo estimator p_j = (1 + #{inner null values >= observed value})/(M+1). The per-metric extremeness score is -log(p_j), and the global statistic is the maximum score across all six predeclared metrics. Ties count as exceedances. This construction remains defined when an inner null sample is constant: if the observation exceeds that constant support, its estimated tail probability is 1/(M+1); if it equals the constant value, its estimated tail probability is 1. No metric is dropped because its sample SD is zero. The same complete six-metric family and the same inner replicate count are used in every outer replicate. The outer bootstrap calibrates the complete statistic, including finite-inner-sample discreteness.

## Nested parametric bootstrap

1. Fit the observed context using leave-one-sequence-out CV over the configured lambda grid, then refit on all four sequences.
2. Select temporal Gibbs weights using the observed ten-lag profile.
3. Generate an independent inner null sample and calculate the observed max-T statistic.
4. For each outer replicate, generate four 256-step chains from the original fitted null.
5. Re-select lambda by the same leave-one-chain-out CV; refit the pairwise model.
6. Re-select temporal Gibbs weights using that outer context's lag profile.
7. Generate an independent inner null sample from the outer refitted model and selected weights.
8. Calculate the outer context's six empirical upper-tail probabilities against its independent inner-null sample, transform them to -log(p), and calculate the maximum score.
9. Compare the observed max-T statistic with the outer max-T distribution. The plus-one Monte Carlo p-value is `(1 + # {T_b >= T_obs}) / (B + 1)`.

This is a plug-in parametric bootstrap conditional on the chosen null family. It does not protect against arbitrary null-model misspecification. Four original text sequences are not enough to support broad corpus-level generalization.

## Temporal leave-one-sequence-out validation

Before a confirmatory F2-E run, evaluate temporal predictive adequacy without using
the held-out sequence to select either lambda or temporal weights:

```bash
python scripts/f2e_temporal_loo_validation.py \
  --root "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1" \
  --output "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1/sequence_replication/layer_00/temporal/loo_confirmatory" \
  --lambdas 0.0003 0.001 0.003 --maxiter 400 \
  --weight-grid-step 0.1 --screen-replicates 8 \
  --predictive-replicates 200 --seed 20261120
```

For each held-out sequence, the script fits the pairwise model and selects temporal
weights from the other sequences only. It evaluates exact conditional transition
negative log likelihood on the held-out sequence for the training-selected kernel,
IID, and the fixed 90/10 reference. It also records held-out lag-profile predictive
discrepancies. The diagnostic is descriptive with four folds, not a hypothesis test.

The current predeclared adequacy flag requires the mean selected-kernel held-out
transition-NLL gain over IID to be positive and the selected kernel to outperform
IID in at least three of four folds. If this diagnostic does not pass, do not treat
the temporal null as validated; investigate model inadequacy before confirmatory
F2-E inference.

The transition likelihood uses the exact mixture probability
`P(T|S)=sum_k w_k mean_{R subset S, |R|=k}[1(R subset T) pi(T)/pi(R)]`,
with the k=0 term equal to pi(T). The first held-out state is conditioned on and
omitted from the transition score.

## Single-entry confirmatory campaign

The recommended one-shot entry point runs the LOO temporal prerequisite first and
launches F2-E only if the predeclared diagnostic passes. It also reuses a
hash/configuration-matched completed LOO result when resuming an interrupted
nested bootstrap:

```bash
python scripts/run_f2e_confirmatory_campaign.py \
  --root "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1" \
  --output-base "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1/sequence_replication/layer_00/temporal/f2e_confirmatory_campaign_v1" \
  --outer-replicates 999 --inner-replicates 100 \
  --screen-replicates 8 --weight-grid-step 0.1 \
  --lambdas 0.0003 0.001 0.003 --maxiter 400 \
  --predictive-replicates 200 --seed 20261120
```

After interruption, repeat the exact command with `--resume`. The runner reuses
the LOO summary only if its script hash, input hashes, lambda grid, optimizer
settings, weight grid, screening count, predictive count, and seed match. The
nested bootstrap independently enforces its own immutable configuration
fingerprint.

## Layer-aware execution

Both `f2e_temporal_loo_validation.py` and `f2e_nested_bootstrap.py` accept
`--layer-id 0..23`. The campaign runner also accepts this option and passes
the same layer ID to both stages. Keep a separate output directory for every
layer. A temporal LOO summary is reusable only when its layer ID, source hashes,
script hash, and all selection settings match. In particular, the previous
layer-0 LOO result must **not** be reused for layer 11.

Example: run the temporal predictive prerequisite for layer 11:

```bash
python scripts/f2e_temporal_loo_validation.py \
  --root "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1" \
  --output "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1/sequence_replication/layer_11/temporal/f2e_layer11_campaign_v1/temporal_loo" \
  --layer-id 11 \
  --lambdas 0.0003 0.001 0.003 --maxiter 400 \
  --weight-grid-step 0.1 --screen-replicates 8 \
  --predictive-replicates 200 --seed 20261120
```

The nested bootstrap command must then use the same `--layer-id 11` and a
separate, layer-specific output directory. Run it only after reviewing the
layer-11 LOO prerequisite and after freezing the inferential statistic and
metric family. The historical inner empirical-tail max statistic has a finite
ceiling when the inner sample is small; do not use it for a new layer-level
confirmatory claim without resolving that calibration limitation.

## Install and validate in Colab

Open `notebooks/FCA_MoE_F2E_Nested_Bootstrap.ipynb`. Mount Drive, clone this branch, install the v0.3 package, and run the tests. The notebook expects the existing trace files:

- `qwen_contexts_by_layer.npz`
- `sequence_alignment.csv`

under `FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1/`.

Run a smoke pilot first. Pilot parameters are software checks only and must not be reported as research inference.

## Suggested execution tiers

### Smoke pilot

```bash
python scripts/f2e_nested_bootstrap.py \
  --outer-replicates 2 --inner-replicates 10 \
  --screen-replicates 1 --weight-grid-step 0.5 \
  --lambdas 0.001 --maxiter 250
```

### Diagnostic run

```bash
python scripts/f2e_nested_bootstrap.py \
  --outer-replicates 20 --inner-replicates 30 \
  --screen-replicates 2 --weight-grid-step 0.5 \
  --lambdas 0.0003 0.001 0.003 --maxiter 400 --resume
```

### Publication-oriented run

```bash
python scripts/run_f2e_confirmatory_campaign.py \
  --root "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1" \
  --output-base "/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1/sequence_replication/layer_00/temporal/f2e_confirmatory_campaign_v1" \
  --outer-replicates 999 --inner-replicates 100 \
  --screen-replicates 8 --weight-grid-step 0.1 \
  --lambdas 0.0003 0.001 0.003 --maxiter 400 \
  --predictive-replicates 200 --seed 20261120
```

The publication-oriented run is computationally substantial. Use a high-RAM Colab runtime, keep Drive mounted, and resume with identical arguments and seed after interruptions. Do not change the seed or configuration while resuming a checkpoint.

## Outputs

Under `sequence_replication/layer_00/temporal/f2e_confirmatory_campaign_v1/`:

- `campaign_manifest.json` (campaign stage and temporal LOO result)
- `temporal_loo/` (fold selections, full weight screens, held-out transition likelihoods and lag profiles)
- `nested_bootstrap/run_config.json` (immutable statistical-configuration fingerprint)
- `nested_bootstrap/observed_fit_and_selection.json`
- `nested_bootstrap/observed_model_parameters.npz` and `observed_lambda_cv.csv`
- `nested_bootstrap/observed_weight_screen.csv`
- `nested_bootstrap/frozen_manifest.json` (input and code hashes, model/data identity, environment)
- `nested_bootstrap/observed_inner_calibration.json` and `observed_inner_metrics.csv`
- `nested_bootstrap/outer_bootstrap_replicates.csv` (checkpointed after each outer replicate)
- `nested_bootstrap/outer_model_parameters/replicate_XXXX.npz` and `outer_fit_XXXX.json`
- `nested_bootstrap/outer_weight_screens/replicate_XXXX.csv` (complete temporal grid selection)
- `nested_bootstrap/outer_inner_metrics/replicate_XXXX.csv` and corresponding diagnostics (raw inner samples)
- `nested_bootstrap/f2e_nested_bootstrap_summary.json`
- `SHA256SUMS` (newline-delimited SHA-256 checksums for all campaign files, excluding itself)

Failed outer replicates are retained with an error field. The summary reports the success and failure counts. Any failed outer replicate invalidates confirmatory reporting until resolved. The script fingerprints the statistical configuration and refuses to resume a checkpoint with changed settings. Use separate output directories for pilot, diagnostic, and publication runs. A run with material fit/sampling failures is not inferentially acceptable.

## Validation scope

The unit tests cover: finite-difference gradient agreement, fixed-cardinality fit and state ranking, conditional Gibbs generation, exact Top-K preservation, exhaustive small-context FCA closure counts, and empirical-tail max-statistic handling, including constant-null metrics. Passing these tests validates implementation components; it does not validate the Qwen trace or imply the full bootstrap has run.



---

## Optimized, chunked execution (F2-E v2)

The optimized branch preserves the confirmatory statistical settings. It changes implementation and checkpoint granularity, not the hypothesis, metric family, number of replicates, or temporal-weight search space.

### Exact-preserving optimizations

1. **Multi-threshold FCA mining.** Each pooled or sequence context is traversed once at its minimum requested support. During that traversal, closed concepts are counted separately at each support threshold (1%, 2%, and 5%). This is equivalent to the former three separate traversals because any concept satisfying a higher support threshold is also visited at the minimum threshold. Exact-equivalence tests compare the multi-threshold output with independent calls to `FCA.mine`.
2. **Cached MaxEnt structures.** The 60-expert/Top-4 combination table, pair-index matrix, conditional retained-set groups, and per-state group lookups are shared across model fits. Fitted probabilities and conditional CDFs remain model-specific.
3. **Cached Gibbs lookup.** A uniform draw over the cached subset columns selects one of the `C(K,k)` retained subsets. The selected group then uses the same exact conditional CDF as the original sampler. This changes the random-number sequence, but not the transition kernel or its stationary distribution.
4. **Resume-safe screening.** Temporal screens write a partial CSV every 25 of 286 grid candidates. The sidecar records a signature derived from the fitted parameters, target profiles, target pair moments, seed, and screen settings. Resume rejects mismatched signatures.
5. **Resume-safe inner nulls.** Inner simulations checkpoint every 10 replicates, including FCA metrics, lag profiles, and pair-moment diagnostics. Each checkpoint is tied to the model parameter hash, weights, seed, chain count, and replicate count.
6. **Outer chunks.** The cumulative outer CSV is checkpointed after every successful or failed replicate, and compact chunk CSVs are written every 25 outer replicates. The fitted model and selected screen are persisted before the inner simulations, so a resumed inner batch can reuse completed work.

The optimization tests check (a) equality of multi-threshold FCA counts with separate exact support traversals and (b) correctness of cached retained-subset mappings. The existing transition-kernel normalization, stationarity, detailed-balance, and transition-likelihood tests remain part of the v0.3 test suite. Runtime speedups must be benchmarked on the target Colab runtime; no fixed speedup factor is claimed here.

### Reusing the previous LOO result

The optimized notebook reuses the already-completed temporal LOO result from `f2e_confirmatory_campaign_v1/temporal_loo/loo_validation_summary.json`. The runner verifies the LOO script SHA-256, both input-file SHA-256 hashes, lambda candidates, optimizer iteration limit, temporal grid, screening count, predictive replicate count, and seed. It refuses reuse if any field differs.

The nested bootstrap itself runs from scratch under the optimized script hash and a separate output directory. Do not point the optimized run at the old nested-bootstrap directory: the code fingerprint is intentionally different.

### Optimized Colab notebook

Open:

`notebooks/FCA_MoE_F2E_Nested_Bootstrap.ipynb`

The notebook uses this branch and writes to:

`sequence_replication/layer_00/temporal/f2e_confirmatory_campaign_v2_optimized/`

It sets the statistical configuration to 999 outer replicates, 100 inner replicates, 8 screening replicates, a 0.1 simplex grid, and lambda candidates 0.0003, 0.001, 0.003. It passes `--reuse-temporal-loo-summary` to avoid repeating the completed LOO prerequisite.

Run the notebook's execution cell once with `RESUME = False`. After a runtime interruption, remount Drive, rerun setup cells, set `RESUME = True`, and execute the same cell with the same arguments. Do not edit the statistical configuration while resuming.

### Optimized outputs

The v2 output directory adds:

- `campaign_manifest.json` with explicit `nested_bootstrap_running` stage.
- `nested_bootstrap/progress.json` with outer-replicate status.
- `nested_bootstrap/observed_weight_screen.partial.csv` and its signature sidecar while observed temporal screening is in progress.
- `nested_bootstrap/observed_inner_metrics.csv`, `observed_inner_metrics.csv.progress.json`, and `observed_inner_metrics.csv.diagnostics.npz` for resumable observed inner simulations.
- `nested_bootstrap/outer_weight_screens/replicate_XXXX.partial.csv` plus signature sidecar while an outer temporal screen is in progress.
- `nested_bootstrap/outer_inner_metrics/replicate_XXXX.csv`, progress sidecar, and diagnostics sidecar for resumable outer inner simulations.
- `nested_bootstrap/outer_chunks/outer_XXXX_YYYY.csv` for groups of 25 outer replicates.

A partial file is a recovery checkpoint, not a completed replicate or a final statistical result. Interpret the global test only from the final summary after all 999 outer replicates are successful and the output package has been audited.
