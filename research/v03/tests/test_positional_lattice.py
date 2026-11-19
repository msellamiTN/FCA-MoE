from __future__ import annotations

import itertools

import numpy as np
import pytest

from moe_fca.contexts import (
    positional_attribute_labels,
    positional_window_context,
)
from moe_fca.fca import next_closure
from moe_fca.lattice import (
    closure_mapping_audit,
    cover_edges,
    derive_extent,
    derive_intent,
    intent_closure,
    positional_attribute_map,
)


def exhaustive_closed_intents(context):
    x = np.asarray(context, dtype=np.uint8)
    n_objects, n_attributes = x.shape
    closed = set()
    for mask in range(1 << n_attributes):
        intent = {j for j in range(n_attributes) if mask & (1 << j)}
        extent = derive_extent(x, intent)
        if derive_intent(x, extent) == frozenset(intent):
            closed.add(frozenset(intent))
    return closed


def test_positional_context_encodes_expert_and_relative_position():
    routes = np.array([
        [0, 1],
        [1, 2],
        [0, 2],
    ])
    context = positional_window_context(
        routes, window_size=2, n_experts=3, stride=1
    )
    assert context.shape == (2, 6)
    # Attribute index = expert * window_size + relative_position.
    assert set(np.flatnonzero(context[0])) == {0, 2, 3, 5}
    assert set(np.flatnonzero(context[1])) == {1, 2, 4, 5}
    assert np.all(context.sum(axis=1) == 4)
    assert positional_attribute_labels(2, 3) == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
    ]


def test_positional_context_window_count_and_stride():
    routes = np.tile(np.array([[0, 1, 2, 3]]), (10, 1))
    assert positional_window_context(routes, 4, n_experts=4, stride=1).shape == (7, 16)
    assert positional_window_context(routes, 4, n_experts=4, stride=2).shape == (4, 16)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0, "n_experts": 4, "stride": 1},
        {"window_size": 4, "n_experts": 4, "stride": 0},
        {"window_size": 11, "n_experts": 4, "stride": 1},
    ],
)
def test_positional_context_rejects_invalid_window_parameters(kwargs):
    routes = np.tile(np.array([[0, 1, 2, 3]]), (10, 1))
    with pytest.raises(ValueError):
        positional_window_context(routes, **kwargs)


def test_galois_and_closure_laws_on_random_small_contexts():
    rng = np.random.default_rng(20261119)
    for _ in range(40):
        n_objects = int(rng.integers(1, 8))
        n_attributes = int(rng.integers(1, 8))
        x = (rng.random((n_objects, n_attributes)) < rng.uniform(.15, .85)).astype(np.uint8)
        for _ in range(20):
            intent = {
                j for j in range(n_attributes) if rng.random() < .5
            }
            extent = derive_extent(x, intent)
            closed = derive_intent(x, extent)
            assert intent <= closed
            assert intent_closure(x, closed) == closed
            assert intent_closure(x, intent_closure(x, intent)) == closed


def test_next_closure_matches_exhaustive_oracle():
    rng = np.random.default_rng(20261118)
    for _ in range(50):
        n_objects = int(rng.integers(1, 8))
        n_attributes = int(rng.integers(1, 8))
        x = (rng.random((n_objects, n_attributes)) < rng.uniform(.1, .9)).astype(np.uint8)
        got = {frozenset(i for i in range(n_attributes) if mask & (1 << i))
               for mask in next_closure(x)}
        assert got == exhaustive_closed_intents(x)


def test_object_permutation_preserves_closed_intents_and_supports():
    x = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ], dtype=np.uint8)
    permutation = np.array([2, 0, 3, 1])
    y = x[permutation]
    intents_x = next_closure(x)
    intents_y = next_closure(y)
    assert intents_x == intents_y
    for intent in intents_x:
        assert len(derive_extent(x, {
            j for j in range(x.shape[1]) if intent & (1 << j)
        })) == len(derive_extent(y, {
            j for j in range(y.shape[1]) if intent & (1 << j)
        }))


def test_positional_attribute_map_preserves_relative_offsets():
    mapped = positional_attribute_map(
        {1, 8}, expert_mapping=[2, 0, 1], window_size=3
    )
    assert mapped == frozenset({7, 5})


def test_cover_edges_are_minimal_supersets():
    intents = [
        frozenset({0}),
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({0, 1, 2}),
    ]
    edges = cover_edges(intents)
    assert edges == {
        (frozenset({0}), frozenset({0, 1})),
        (frozenset({0}), frozenset({0, 2})),
        (frozenset({0, 1}), frozenset({0, 1, 2})),
        (frozenset({0, 2}), frozenset({0, 1, 2})),
    }


def test_closure_induced_mapping_is_weakly_monotone():
    source_context = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=np.uint8)
    target_context = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=np.uint8)
    source_intents = exhaustive_closed_intents(source_context)
    target_intents = exhaustive_closed_intents(target_context)
    result = closure_mapping_audit(
        source_intents,
        target_context,
        [2, 0, 1],
        min_support=1,
        target_intents=target_intents,
    )
    assert result["weak_intent_inclusion_preservation_rate"] == 1.0
    assert 0.0 <= result["closure_equality_rate"] <= 1.0
    assert 0.0 <= result["target_support_valid_rate"] <= 1.0
    assert result["n_unique_target_closures"] <= result["n_source_intents"]


def test_positional_closure_mapping_is_weakly_monotone():
    routes_source = np.array([
        [0, 1], [0, 2], [1, 2], [0, 1], [1, 2]
    ])
    routes_target = np.array([
        [2, 0], [2, 1], [0, 1], [2, 0], [0, 1]
    ])
    source_context = positional_window_context(routes_source, 2, 3)
    target_context = positional_window_context(routes_target, 2, 3)
    source_intents = exhaustive_closed_intents(source_context)
    target_intents = exhaustive_closed_intents(target_context)
    result = closure_mapping_audit(
        source_intents,
        target_context,
        [1, 2, 0],
        min_support=1,
        window_size=2,
        target_intents=target_intents,
    )
    assert result["weak_intent_inclusion_preservation_rate"] == 1.0


def test_order_reflection_rate_is_not_reported_as_violation_rate():
    # Two incomparable source intents collapse to the same target closure.
    target_context = np.array([[1, 1]], dtype=np.uint8)
    result = closure_mapping_audit(
        [frozenset({0}), frozenset({1})],
        target_context,
        [0, 1],
        min_support=1,
        target_intents=[frozenset({0, 1})],
    )
    assert result["order_reflection_rate"] == 0.0
    assert result["n_collapsed_source_pairs"] == 1
