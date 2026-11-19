import itertools
import sys
from pathlib import Path
import numpy as np
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from f2e_nested_bootstrap import (
    ConditionalPairwiseMaxEnt, FCA, max_t_from_inner, FAMILY,
    grid_weights, generate_dataset, load_observed
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


def test_multi_threshold_fca_matches_separate_exact_mining():
    rng = np.random.default_rng(20261116)
    for _ in range(30):
        n = int(rng.integers(12, 40))
        m = int(rng.integers(4, 10))
        X = (rng.random((n, m)) < rng.uniform(.2, .8)).astype(np.uint8)
        supports = sorted(set([
            max(1, int(np.ceil(.1 * n))),
            max(1, int(np.ceil(.25 * n))),
            max(1, int(np.ceil(.5 * n))),
        ]))
        combined = FCA.mine_multi_threshold(X, supports)
        for support in supports:
            separate = FCA.mine(X, support)
            assert combined[support]["higher"] == separate["higher"]
            assert combined[support]["triplets"] == int(separate["by_size"][3])
            expected_quads = int(separate["by_size"][4]) if m >= 4 else 0
            assert combined[support]["quadruplets"] == expected_quads


def test_cached_state_structures_are_shared_and_subset_maps_are_exact():
    first = ConditionalPairwiseMaxEnt(8, 3)
    second = ConditionalPairwiseMaxEnt(8, 3)
    assert first.states is second.states
    assert first.state_pair is second.state_pair

    rng = np.random.default_rng(91)
    ids = rng.integers(0, len(first.states), size=200)
    X = np.zeros((len(ids), 8), dtype=np.uint8)
    X[np.arange(len(ids))[:, None], first.states[ids]] = 1
    first.fit(X, lam=.03, maxiter=500).build_conditional_indexes()

    for k in (1, 2, 3):
        positions = list(itertools.combinations(range(first.K), k))
        q = first.cond[k]
        group_map = first.subset_group_ids[k]
        for state_id in (0, 1, len(first.states) // 2, len(first.states) - 1):
            for col, pos in enumerate(positions):
                retained = first.states[state_id, list(pos)]
                group = int(group_map[state_id, col])
                mask = first._mask(retained)
                assert q["lookup"][mask] == group
                a, b = int(q["starts"][group]), int(q["ends"][group])
                subset_states = first.states[q["ids"][a:b]]
                # Conditional groups retain the expert set, not its old slot positions.
                for expert in retained:
                    assert np.all(np.any(subset_states == expert, axis=1))


def test_weight_grid_and_max_t_retains_degenerate_metrics():
    assert len(list(grid_weights(.5))) == 10
    inner = []
    for i in range(20):
        row = {name: float(i % 7) for name in FAMILY}
        row["pooled_higher_order_5pct"] = 0.0
        row["mean_sequence_higher_order_5pct"] = 0.0
        inner.append(row)
    obs = {name: 10.0 for name in FAMILY}
    obs["pooled_higher_order_5pct"] = 0.0
    # Observed sequence-level 5% metric exceeds a constant-zero inner sample.
    obs["mean_sequence_higher_order_5pct"] = 0.25
    result = max_t_from_inner(obs, inner)
    assert result["active"] == FAMILY
    assert "mean_sequence_higher_order_5pct" in result["constant"]
    diagnostic = result["tail"]["mean_sequence_higher_order_5pct"]
    assert diagnostic["exceedances_ge"] == 0
    assert diagnostic["p_plus_one"] == 1.0 / 21.0
    assert np.isclose(diagnostic["extremeness_score"], -np.log(1.0 / 21.0))
    assert result["tail"]["pooled_higher_order_5pct"]["p_plus_one"] == 1.0
    # Every metric remains present when an outer replicate has a degenerate null.
    outer = max_t_from_inner(obs, inner, active_family=result["active"])
    assert set(outer["tail"]) == set(FAMILY)


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



def test_load_observed_selects_requested_layer(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(20261119)
    sequence_ids = np.repeat(np.arange(2), 256)
    token_positions = np.tile(np.arange(256), 2)

    def make_context(offset):
        context = np.zeros((512, 60), dtype=np.uint8)
        for row in range(512):
            chosen = rng.choice(60, size=4, replace=False)
            context[row, chosen] = 1
        if offset:
            context = np.roll(context, shift=offset, axis=1)
        return context

    layer_00 = make_context(0)
    layer_11 = make_context(7)
    np.savez_compressed(
        tmp_path / "qwen_contexts_by_layer.npz",
        layer_00=layer_00,
        layer_11=layer_11,
    )
    pd.DataFrame({
        "sequence_id": sequence_ids,
        "token_position": token_positions,
    }).to_csv(tmp_path / "sequence_alignment.csv", index=False)

    blocks = load_observed(tmp_path, layer_id=11)
    assert len(blocks) == 2
    assert all(block.shape == (256, 60) for block in blocks)
    assert all(np.all(block.sum(axis=1) == 4) for block in blocks)
    assert np.array_equal(np.vstack(blocks), layer_11)
    assert not np.array_equal(np.vstack(blocks), layer_00)


def test_load_observed_rejects_out_of_range_layer(tmp_path):
    import pandas as pd

    with np.testing.assert_raises(ValueError):
        load_observed(tmp_path, layer_id=24)
