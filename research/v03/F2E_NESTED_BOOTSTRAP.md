# FCA–MoE F2-E: nested parametric bootstrap

## Purpose

F2-D tested the higher-order FCA counts conditional on one fitted layer-0 pairwise MaxEnt distribution and one selected temporal Gibbs mixture. F2-E repeats fitting and temporal-weight selection inside an outer parametric bootstrap, so the global reference distribution includes these sources of variability under the specified null family.

The experiment is scoped to the existing Qwen1.5-MoE-A2.7B layer-0 trace: four 256-token sequences, 60 routed experts, exact Top-4 contexts.

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
python scripts/f2e_nested_bootstrap.py \
  --outer-replicates 999 --inner-replicates 100 \
  --screen-replicates 8 --weight-grid-step 0.1 \
  --lambdas 0.0003 0.001 0.003 --maxiter 600 \
  --checkpoint-every 1 --resume
```

The publication-oriented run is computationally substantial. Use a high-RAM Colab runtime, keep Drive mounted, and resume with identical arguments and seed after interruptions. Do not change the seed or configuration while resuming a checkpoint.

## Outputs

Under `sequence_replication/layer_00/temporal/f2e_nested_bootstrap/`:

- `run_config.json` (immutable statistical-configuration fingerprint)
- `observed_fit_and_selection.json`
- `observed_model_parameters.npz` and `observed_lambda_cv.csv`
- `observed_weight_screen.csv`
- `observed_inner_calibration.json`
- `outer_bootstrap_replicates.csv` (checkpointed after each outer replicate)
- `outer_model_parameters/replicate_XXXX.npz` and `outer_fit_XXXX.json` (successful outer fits)
- `f2e_nested_bootstrap_summary.json`

Failed outer replicates are retained with an error field. The summary reports the success and failure counts. Any failed outer replicate invalidates confirmatory reporting until resolved. The script fingerprints the statistical configuration and refuses to resume a checkpoint with changed settings. Use separate output directories for pilot, diagnostic, and publication runs. A run with material fit/sampling failures is not inferentially acceptable.

## Validation scope

The unit tests cover: finite-difference gradient agreement, fixed-cardinality fit and state ranking, conditional Gibbs generation, exact Top-K preservation, exhaustive small-context FCA closure counts, and empirical-tail max-statistic handling, including constant-null metrics. Passing these tests validates implementation components; it does not validate the Qwen trace or imply the full bootstrap has run.

