import itertools
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from f2e_nested_bootstrap import ConditionalPairwiseMaxEnt
from f2e_temporal_loo_validation import (
    transition_probability,
    transition_log_likelihood,
)


def _toy_model(seed=71):
    E, K = 7, 3
    model = ConditionalPairwiseMaxEnt(E, K)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(model.states), size=350)
    X = np.zeros((len(ids), E), dtype=np.uint8)
    X[np.arange(len(ids))[:, None], model.states[ids]] = 1
    model.fit(X, lam=0.04, maxiter=500).build_conditional_indexes()
    return model


def test_transition_kernel_probabilities_normalize_and_preserve_stationary_pi():
    model = _toy_model()
    weights = np.array([0.2, 0.4, 0.3, 0.1], dtype=float)
    n = len(model.states)
    P = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            P[i, j] = transition_probability(model, i, j, weights)

    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(model.prob @ P, model.prob, atol=1e-10)
    assert np.allclose(
        model.prob[:, None] * P,
        (model.prob[:, None] * P).T,
        atol=1e-10,
    )


def test_transition_log_likelihood_matches_transition_probability():
    model = _toy_model(seed=72)
    weights = np.array([0.9, 0.1, 0.0, 0.0], dtype=float)
    chosen = np.array([2, 11, 14, 20, 31, 3, 7, 17], dtype=int)
    X = np.zeros((len(chosen), model.E), dtype=np.uint8)
    X[np.arange(len(chosen))[:, None], model.states[chosen]] = 1

    result = transition_log_likelihood(model, X, weights)
    manual = []
    for t in range(1, len(chosen)):
        p = transition_probability(model, chosen[t - 1], chosen[t], weights)
        manual.append(-np.log(p))

    assert result["transitions"] == len(chosen) - 1
    assert np.isclose(
        result["negative_log_likelihood"],
        np.sum(manual),
        atol=1e-10,
    )
    assert np.isfinite(result["mean_transition_nll"])
