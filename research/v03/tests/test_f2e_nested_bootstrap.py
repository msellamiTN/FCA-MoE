import itertools
import sys
from pathlib import Path
import numpy as np
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from f2e_nested_bootstrap import (
    ConditionalPairwiseMaxEnt, FCA, max_t_from_inner, FAMILY,
    grid_weights, generate_dataset
)


def test_fixed_cardinality_pairwise_fit_and_rank_roundtrip():
    E, K = 8, 3
    model = ConditionalPairwiseMaxEnt(E, K)
    rng = np.random.default_rng(12)
    ids = rng.integers(0, len(model.states), size=240)
    X = np.zeros((len(ids), E), dtype=np.uint8)
    X[np.arange(len(ids))[:, None], model.states[ids]] = 1
    model.fit(X, lam=0.02, maxiter=500)
    assert np.array_equal(model.state_ids(X), ids)
    assert np.isclose(model.prob.sum(), 1.0)
    assert np.isclose(model.fitted_pair_moments.sum(), 3.0)
    assert abs(model.theta.sum()) < 1e-10


def test_conditional_gibbs_sampler_preserves_cardinality():
    E, K = 8, 3
    model = ConditionalPairwiseMaxEnt(E, K)
    rng = np.random.default_rng(13)
    ids = rng.integers(0, len(model.states), size=500)
    X = np.zeros((len(ids), E), dtype=np.uint8)
    X[np.arange(len(ids))[:, None], model.states[ids]] = 1
    model.fit(X, lam=0.03, maxiter=500).build_conditional_indexes()
    blocks, _ = generate_dataset(model, np.array([0.4, 0.4, 0.2, 0.0]), 42,
                                 chains=2, length=40)
    assert all(np.all(b.sum(axis=1) == K) for b in blocks)


def test_cbo_counts_match_exhaustive_closure_oracle():
    rng = np.random.default_rng(14)
    for _ in range(15):
        n, m = int(rng.integers(5, 10)), int(rng.integers(3, 8))
        X = (rng.random((n, m)) < .4).astype(np.uint8)
        minsup = int(rng.integers(1, n + 1))
        got = FCA.mine(X, minsup)
        expected = np.zeros(m + 1, dtype=np.int64)
        for mask in range(1 << m):
            attrs = [j for j in range(m) if (mask >> j) & 1]
            extent = np.all(X[:, attrs] == 1, axis=1) if attrs else np.ones(n, bool)
            if int(extent.sum()) < minsup:
                continue
            closure = [j for j in range(m) if np.all(X[extent, j] == 1)]
            closure_mask = sum(1 << j for j in closure)
            if closure_mask == mask:
                expected[len(attrs)] += 1
        assert np.array_equal(got["by_size"], expected)


def test_weight_grid_and_max_t_active_family():
    assert len(list(grid_weights(.5))) == 10
    inner = []
    for i in range(20):
        row = {name: float(i % 7) for name in FAMILY}
        row["pooled_higher_order_5pct"] = 0.0
        inner.append(row)
    obs = {name: 10.0 for name in FAMILY}
    obs["pooled_higher_order_5pct"] = 0.0
    result = max_t_from_inner(obs, inner)
    assert "pooled_higher_order_5pct" in result["constant"]
    assert result["max_t"] is not None
    # A metric declared active from the observed calibration may not silently
    # become degenerate in an outer replicate.
    try:
        max_t_from_inner(obs, [{**r, "pooled_higher_order_1pct": 1.0} for r in inner],
                         active_family=result["active"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected a degenerate active metric to fail")


def test_pairwise_objective_gradient_finite_difference():
    from scipy.special import logsumexp
    E, K, lam = 7, 3, 0.03
    model = ConditionalPairwiseMaxEnt(E, K)
    rng = np.random.default_rng(15)
    ids = rng.integers(0, len(model.states), size=100)
    X = np.zeros((len(ids), E), dtype=np.uint8)
    X[np.arange(len(ids))[:, None], model.states[ids]] = 1
    target = model.empirical_pair_moments(X)
    sp = model.state_pair
    theta = rng.normal(0, .05, model.n_params)
    def fg(t):
        energy = t[sp].sum(axis=1)
        logz = logsumexp(energy)
        p = np.exp(energy-logz)
        pred = np.bincount(sp.ravel(), weights=np.repeat(p, sp.shape[1]),
                           minlength=model.n_params)
        return logz-target@t+.5*lam*(t@t), pred-target+lam*t
    _, analytic = fg(theta)
    eps = 1e-6
    numeric = np.empty_like(theta)
    for j in range(len(theta)):
        delta = np.zeros_like(theta); delta[j] = eps
        numeric[j] = (fg(theta+delta)[0]-fg(theta-delta)[0])/(2*eps)
    assert np.max(np.abs(analytic-numeric)) < 1e-7


def test_conditional_gibbs_kernel_detailed_balance_on_small_state_space():
    import itertools
    E, K = 7, 3
    model = ConditionalPairwiseMaxEnt(E, K)
    rng = np.random.default_rng(16)
    ids = rng.integers(0, len(model.states), size=400)
    X = np.zeros((len(ids), E), dtype=np.uint8)
    X[np.arange(len(ids))[:, None], model.states[ids]] = 1
    model.fit(X, lam=.04, maxiter=500)
    states, pi = model.states, model.prob
    weights = np.array([.25, .25, .25, .25])
    P = np.zeros((len(states), len(states)), dtype=np.float64)
    for s_idx, S in enumerate(states):
        P[s_idx] += weights[0] * pi
        for k in (1, 2, 3):
            subsets = list(itertools.combinations(map(int, S), k))
            for R in subsets:
                candidates = np.asarray([j for j,T in enumerate(states)
                                         if set(R).issubset(set(map(int,T)))], dtype=int)
                cond = pi[candidates] / pi[candidates].sum()
                P[s_idx, candidates] += weights[k] * cond / len(subsets)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(pi @ P, pi, atol=1e-11)
    assert np.allclose(pi[:,None]*P, (pi[:,None]*P).T, atol=1e-11)

