#!/usr/bin/env python3
"""Leave-one-sequence-out validation for the F2-E temporal pairwise null.

This is a predictive diagnostic, not a hypothesis test. For each held-out
sequence, lambda and temporal weights are selected using only the other
sequences. The held-out sequence is then evaluated by exact conditional
transition log likelihood and by lag-overlap profile discrepancy.

Run from the repository checkout:
  python scripts/f2e_temporal_loo_validation.py --root <contexts_v1> --output <loo_dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import hashlib
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from f2e_nested_bootstrap import (
    ConditionalPairwiseMaxEnt,
    fit_model,
    load_observed,
    select_weights,
    generate_dataset,
    LAGS,
)


def _conditional_group_masses(model):
    masses = {}
    for k in (1, 2, 3):
        q = model.cond[k]
        values = np.empty(len(q["starts"]), dtype=np.float64)
        for g, (a, b) in enumerate(zip(q["starts"], q["ends"])):
            values[g] = float(model.prob[q["ids"][int(a):int(b)]].sum())
        masses[k] = values
    return masses


def transition_probability(model, previous_state_id, next_state_id, weights,
                           group_masses=None):
    """Exact P(next_state | previous_state) for the conditional Gibbs mixture."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be a four-component probability vector for k=0,1,2,3")
    prev = model.states[int(previous_state_id)]
    nxt_id = int(next_state_id)
    nxt_set = set(map(int, model.states[nxt_id]))
    probability = float(w[0] * model.prob[nxt_id])
    if group_masses is None:
        group_masses = _conditional_group_masses(model)

    for k in (1, 2, 3):
        if w[k] == 0:
            continue
        q = model.cond[k]
        subset_total = math.comb(model.K, k)
        contribution = 0.0
        for positions in itertools.combinations(range(model.K), k):
            retained = tuple(int(prev[p]) for p in positions)
            if not all(x in nxt_set for x in retained):
                continue
            group = q["lookup"].get(model._mask(retained))
            if group is None:
                raise RuntimeError(f"Missing conditional group for {retained}")
            conditional_mass = float(group_masses[k][group])
            if conditional_mass <= 0 or not np.isfinite(conditional_mass):
                raise RuntimeError("Invalid conditional stationary mass")
            contribution += float(model.prob[nxt_id]) / conditional_mass
        probability += float(w[k]) * contribution / subset_total

    return probability


def transition_log_likelihood(model, context, weights):
    """Exact conditional log likelihood of observed transitions under the kernel.

    The first state is conditioned on and omitted. For current state S and next
    state T, P(T|S) is the mixture over k of the average conditional Gibbs
    resampling probability over all C(K,k) subsets of S.
    """
    X = np.asarray(context)
    if X.ndim != 2 or X.shape[1] != model.E:
        raise ValueError("context must be an N x E binary matrix")
    if not np.isin(X, [0, 1]).all() or not np.all(X.sum(axis=1) == model.K):
        raise ValueError("context must contain exact fixed-cardinality binary states")
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be a four-component probability vector for k=0,1,2,3")

    ids = model.state_ids(X)
    group_masses = _conditional_group_masses(model)
    values = []
    for t in range(1, len(ids)):
        p = transition_probability(
            model, int(ids[t - 1]), int(ids[t]), w,
            group_masses=group_masses
        )
        if p <= 0 or not np.isfinite(p):
            raise RuntimeError(
                f"Non-positive transition probability at position {t}: {p}"
            )
        values.append(math.log(p))

    if not values:
        raise ValueError("At least two observations are required")
    values = np.asarray(values, dtype=float)
    return {
        "transitions": int(len(values)),
        "negative_log_likelihood": float(-values.sum()),
        "mean_transition_nll": float(-values.mean()),
        "mean_transition_log_likelihood": float(values.mean()),
    }

def empirical_lag_profile(block):
    return np.asarray([
        float((block[:-lag] * block[lag:]).sum(axis=1).mean())
        for lag in range(1, LAGS + 1)
    ])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_validation(root, output, lambdas, maxiter, grid_step,
                   screen_replicates, predictive_replicates, seed):
    blocks = load_observed(root)
    if len(blocks) < 3:
        raise ValueError("At least three sequences are required for LOO validation")

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    profile_rows = []
    selected_rows = []

    for held in range(len(blocks)):
        train = [b for i, b in enumerate(blocks) if i != held]
        test = blocks[held]

        model, selected_lambda, lambda_cv = fit_model(train, lambdas, maxiter)
        train_profile = np.mean(
            np.vstack([empirical_lag_profile(b) for b in train]), axis=0
        )
        selected_weights, screen = select_weights(
            model=model,
            target_profile=train_profile,
            target_pair=model.fitted_pair_moments,
            grid_step=grid_step,
            screen_replicates=screen_replicates,
            chains=4,
            length=len(train[0]),
            seed_base=seed + 100000 + held * 1000000,
        )

        candidates = {
            "selected_on_training": np.asarray(selected_weights, dtype=float),
            "iid": np.array([1.0, 0.0, 0.0, 0.0]),
            "fixed_90_10": np.array([0.9, 0.1, 0.0, 0.0]),
        }

        unique = {}
        names_to_key = {}
        for name, weights in candidates.items():
            key = tuple(np.round(weights, 12))
            if key not in unique:
                unique[key] = weights
            names_to_key[name] = key

        test_profile = empirical_lag_profile(test)
        candidate_predictions = {}
        for ci, (key, weights) in enumerate(unique.items()):
            candidate_names = [
                n for n, k in names_to_key.items() if k == key
            ]
            # Exact held-out transition likelihood; no weight reselection on test.
            likelihood = transition_log_likelihood(model, test, weights)

            simulated_profiles = []
            for rep in range(predictive_replicates):
                _, profile = generate_dataset(
                    model, weights,
                    seed + 5000000 + held * 1000000 + ci * 10000 + rep,
                    chains=1, length=len(test)
                )
                simulated_profiles.append(profile)
            simulated_profiles = np.asarray(simulated_profiles)
            predicted_profile = simulated_profiles.mean(axis=0)
            profile_sd = simulated_profiles.std(axis=0, ddof=1)
            profile_rmse = float(np.sqrt(np.mean(
                (predicted_profile - test_profile) ** 2
            )))

            for candidate_name in candidate_names:
                rows.append({
                    "held_out_sequence": held,
                    "candidate": candidate_name,
                    "weights": json.dumps(weights.tolist()),
                    "selected_lambda_on_training": selected_lambda,
                    **likelihood,
                    "heldout_lag_profile_rmse": profile_rmse,
                })
                for lag in range(LAGS):
                    profile_rows.append({
                        "held_out_sequence": held,
                        "candidate": candidate_name,
                        "lag": lag + 1,
                        "observed": float(test_profile[lag]),
                        "predicted_mean": float(predicted_profile[lag]),
                        "predictive_sd": float(profile_sd[lag]),
                        "residual": float(predicted_profile[lag] - test_profile[lag]),
                    })

        screen_df = pd.DataFrame([
            {
                "w0": float(x["weights"][0]),
                "w1": float(x["weights"][1]),
                "w2": float(x["weights"][2]),
                "w3": float(x["weights"][3]),
                "lag_rmse": float(x["lag_rmse"]),
                "pair_rmse": float(x["pair_rmse"]),
            } for x in screen
        ])
        screen_df.to_csv(output / f"fold_{held}_training_weight_screen.csv", index=False)
        selected_rows.append({
            "held_out_sequence": held,
            "training_sequences": [i for i in range(len(blocks)) if i != held],
            "selected_lambda": selected_lambda,
            "selected_weights": selected_weights.tolist(),
            "training_lag_profile": train_profile.tolist(),
            "heldout_lag_profile": test_profile.tolist(),
            "lambda_cv": lambda_cv,
            "fit": model.fit_info,
        })
        print(
            f"LOO fold {held + 1}/{len(blocks)}: "
            f"lambda={selected_lambda}, weights={selected_weights.tolist()}",
            flush=True,
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output / "loo_transition_and_profile_metrics.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(
        output / "loo_lag_profiles_long.csv", index=False
    )
    (output / "loo_fold_selections.json").write_text(
        json.dumps(selected_rows, indent=2)
    )

    summary = (
        result_df.groupby("candidate")
        .agg(
            mean_transition_nll=("mean_transition_nll", "mean"),
            sd_transition_nll=("mean_transition_nll", "std"),
            mean_lag_rmse=("heldout_lag_profile_rmse", "mean"),
            folds=("held_out_sequence", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(output / "loo_candidate_summary.csv", index=False)

    iid = summary.loc[summary.candidate == "iid", "mean_transition_nll"]
    selected = summary.loc[
        summary.candidate == "selected_on_training", "mean_transition_nll"
    ]
    temporal_gain = None
    if len(iid) == 1 and len(selected) == 1:
        temporal_gain = float(iid.iloc[0] - selected.iloc[0])

    manifest = {
        "status": "completed",
        "purpose": "predictive temporal diagnostic; not a hypothesis test",
        "sequences": len(blocks),
        "sequence_length": int(len(blocks[0])),
        "lambda_candidates": list(map(float, lambdas)),
        "weight_grid_step": float(grid_step),
        "screen_replicates": int(screen_replicates),
        "predictive_replicates": int(predictive_replicates),
        "seed": int(seed),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "mean_transition_nll_gain_selected_vs_iid": temporal_gain,
        "interpretation": (
            "Positive gain means the training-selected temporal kernel has "
            "lower mean held-out transition NLL than IID. Four folds are "
            "descriptive and do not establish broad generalization."
        ),
        "elapsed_seconds": None,
    }
    (output / "loo_validation_summary.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return summary, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0003, 0.001, 0.003])
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--weight-grid-step", type=float, default=0.1)
    parser.add_argument("--screen-replicates", type=int, default=8)
    parser.add_argument("--predictive-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.weight_grid_step > 0.1 + 1e-12:
        raise ValueError("Confirmatory LOO validation requires grid step <= 0.1")
    if args.screen_replicates < 8:
        raise ValueError("Confirmatory LOO validation requires >= 8 screen replicates")
    if args.predictive_replicates < 100:
        raise ValueError("Use >= 100 predictive replicates")
    start = time.time()
    _, summary = run_validation(
        Path(args.root), Path(args.output), args.lambdas, args.maxiter,
        args.weight_grid_step, args.screen_replicates,
        args.predictive_replicates, args.seed
    )
    summary["elapsed_seconds"] = time.time() - start
    (Path(args.output) / "loo_validation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
