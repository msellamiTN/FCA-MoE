from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from clera_cross_layer_alignment import (  # noqa: E402
    cross_validate_alignment_only,
    cross_validate_pair,
    extract_supported_higher_order_intents,
    hungarian_mapping,
    phi_similarity_blocks,
    mapped_topk_overlap,
    phi_similarity,
    score_mapping,
)


def test_phi_similarity_handles_permutation_and_constant_columns():
    a = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.uint8)
    # Target columns are source columns [1, 2, 0].
    b = a[:, [1, 2, 0]]
    similarity = phi_similarity(a, b)
    assert np.isclose(similarity[0, 2], 1.0)
    assert np.isclose(similarity[1, 0], 1.0)
    assert np.isclose(similarity[2, 1], 1.0)
    # No constant column in this fixture; test constants separately.
    constant = np.zeros((6, 1), dtype=np.uint8)
    assert np.all(phi_similarity(constant, a[:, :1]) == 0.0)


def test_hungarian_mapping_recovers_known_assignment():
    expected = np.array([2, 0, 3, 1])
    similarity = np.full((4, 4), -0.25)
    similarity[np.arange(4), expected] = 1.0
    got = hungarian_mapping(similarity)
    assert np.array_equal(got, expected)
    assert len(np.unique(got)) == 4


def test_mapped_topk_overlap_and_score():
    source = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=np.uint8)
    mapping = np.array([2, 3, 0, 1])
    target = np.zeros_like(source)
    for i, row in enumerate(source):
        target[i, mapping[np.flatnonzero(row)]] = 1
    assert mapped_topk_overlap(source, target, mapping) == 1.0
    result = score_mapping(source, target, mapping)
    assert np.isclose(result["matched_phi_mean"], 1.0)
    assert np.isclose(result["mapped_topk_overlap"], 1.0)


def _exhaustive_higher_order_intents(context, min_support):
    x = np.asarray(context, dtype=np.uint8)
    n, m = x.shape
    expected = set()
    for mask in range(1 << m):
        attrs = tuple(j for j in range(m) if mask & (1 << j))
        extent = (
            np.all(x[:, attrs] == 1, axis=1)
            if attrs else np.ones(n, dtype=bool)
        )
        if int(extent.sum()) < min_support:
            continue
        if not extent.any():
            closure = tuple(range(m))
        else:
            closure = tuple(
                j for j in range(m) if np.all(x[extent, j] == 1)
            )
        if closure == attrs and len(attrs) >= 3:
            expected.add(frozenset(attrs))
    return expected


def test_supported_higher_order_intents_match_exhaustive_oracle():
    rng = np.random.default_rng(20261119)
    for _ in range(30):
        n = int(rng.integers(5, 10))
        m = int(rng.integers(3, 8))
        x = (rng.random((n, m)) < rng.uniform(.2, .8)).astype(np.uint8)
        support = int(rng.integers(1, n + 1))
        assert extract_supported_higher_order_intents(
            x, support
        ) == _exhaustive_higher_order_intents(x, support)


def test_cross_validated_alignment_is_evaluated_on_heldout_sequences():
    rng = np.random.default_rng(20261118)
    n_sequences, n_tokens, n_experts, top_k = 4, 96, 10, 2
    true_mapping = np.array([3, 8, 1, 5, 0, 9, 4, 2, 7, 6])

    source_blocks = []
    target_blocks = []
    for _ in range(n_sequences):
        source = np.zeros((n_tokens, n_experts), dtype=np.uint8)
        for t in range(n_tokens):
            selected = rng.choice(n_experts, size=top_k, replace=False)
            source[t, selected] = 1
        target = np.zeros_like(source)
        for t, row in enumerate(source):
            target[t, true_mapping[np.flatnonzero(row)]] = 1
        source_blocks.append(source)
        target_blocks.append(target)

    summary, folds = cross_validate_pair(
        source_blocks, target_blocks, min_support=2
    )
    alignment_only = cross_validate_alignment_only(
        source_blocks, target_blocks
    )

    assert len(folds) == n_sequences
    assert summary["matched_phi_mean"] > 0.99
    assert summary["mapped_topk_overlap_mean"] == 1.0
    assert np.isclose(
        summary["matched_phi_mean"],
        alignment_only["matched_phi_mean"],
    )
    assert np.isclose(
        summary["mapped_topk_overlap_mean"],
        alignment_only["mapped_topk_overlap_mean"],
    )
    for fold in folds:
        mapping = np.asarray(__import__("json").loads(fold["mapping"]))
        assert np.array_equal(mapping, true_mapping)
        assert fold["exact_transport_rate"] is not None


def test_training_phi_is_centered_within_sequence_to_remove_sequence_confounding():
    # Within each sequence, A and B have exactly independent 2x2 counts.
    # Their marginal prevalence changes together between sequences, which
    # would create a spurious pooled correlation if rows were simply stacked.
    def independent_binary_block(n11, n10, n01, n00):
        a = np.array(
            [1] * (n11 + n10) + [0] * (n01 + n00),
            dtype=np.uint8
        )
        b = np.array(
            [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00,
            dtype=np.uint8
        )
        return np.column_stack([a, b])

    source_1 = independent_binary_block(64, 16, 16, 4)[:, [0]]
    target_1 = independent_binary_block(64, 16, 16, 4)[:, [1]]
    source_2 = independent_binary_block(4, 16, 16, 64)[:, [0]]
    target_2 = independent_binary_block(4, 16, 16, 64)[:, [1]]

    pooled = phi_similarity(
        np.vstack([source_1, source_2]),
        np.vstack([target_1, target_2])
    )[0, 0]
    within = phi_similarity_blocks(
        [source_1, source_2], [target_1, target_2]
    )[0, 0]

    assert pooled > 0.2
    assert abs(within) < 1e-12
