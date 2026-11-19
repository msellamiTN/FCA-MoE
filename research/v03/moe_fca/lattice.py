from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _binary_context(context: np.ndarray) -> np.ndarray:
    x = np.asarray(context)
    if x.ndim != 2 or not np.isin(x, [0, 1]).all():
        raise ValueError("context must be a binary 2-D matrix")
    return x.astype(np.uint8, copy=False)


def derive_extent(context: np.ndarray, intent: Iterable[int]) -> frozenset[int]:
    """Galois derivation B' for an attribute set B."""
    x = _binary_context(context)
    attributes = frozenset(int(a) for a in intent)
    if any(a < 0 or a >= x.shape[1] for a in attributes):
        raise ValueError("intent contains an attribute outside the context")
    if not attributes:
        return frozenset(range(x.shape[0]))
    columns = sorted(attributes)
    rows = np.flatnonzero(np.all(x[:, columns] == 1, axis=1))
    return frozenset(int(i) for i in rows)


def derive_intent(context: np.ndarray, extent: Iterable[int]) -> frozenset[int]:
    """Galois derivation A' for an object set A.

    The derivation of the empty extent is the full attribute set, as required
    by the universal quantifier in formal concept analysis.
    """
    x = _binary_context(context)
    objects = frozenset(int(g) for g in extent)
    if any(g < 0 or g >= x.shape[0] for g in objects):
        raise ValueError("extent contains an object outside the context")
    if not objects:
        return frozenset(range(x.shape[1]))
    rows = sorted(objects)
    columns = np.flatnonzero(np.all(x[rows, :] == 1, axis=0))
    return frozenset(int(j) for j in columns)


def intent_closure(context: np.ndarray, intent: Iterable[int]) -> frozenset[int]:
    """Compute B'' using the context's Galois connection."""
    return derive_intent(context, derive_extent(context, intent))


def positional_attribute_map(
    intent: Iterable[int],
    expert_mapping: Sequence[int],
    window_size: int,
) -> frozenset[int]:
    """Map expert-major positional attributes while preserving relative offset.

    Source attribute index is expert * window_size + offset. The one-to-one
    expert mapping is applied only to the expert coordinate.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    mapping = np.asarray(expert_mapping, dtype=int)
    if mapping.ndim != 1 or np.any(mapping < 0):
        raise ValueError("expert_mapping must be a non-negative 1-D array")
    if len(np.unique(mapping)) != len(mapping):
        raise ValueError("expert_mapping must be one-to-one")
    result: set[int] = set()
    for raw_index in intent:
        index = int(raw_index)
        if index < 0 or index >= len(mapping) * window_size:
            raise ValueError("positional attribute index outside source universe")
        expert, offset = divmod(index, window_size)
        result.add(int(mapping[expert]) * window_size + offset)
    return frozenset(result)


def cover_edges(intents: Iterable[Iterable[int]]) -> set[tuple[frozenset[int], frozenset[int]]]:
    """Return covers of the supplied intent family under ordinary inclusion.

    Edges are oriented from the smaller intent to its minimal strict superset.
    If the supplied family is support-filtered, these are covers of its
    induced subposet, not necessarily covers of the complete concept lattice.
    """
    family = {frozenset(int(v) for v in intent) for intent in intents}
    ordered = sorted(family, key=lambda item: (len(item), tuple(sorted(item))))
    edges: set[tuple[frozenset[int], frozenset[int]]] = set()

    for lower in ordered:
        candidates = [upper for upper in ordered if lower < upper]
        candidates.sort(key=lambda item: (len(item), tuple(sorted(item))))
        minimal_supersets: list[frozenset[int]] = []
        for candidate in candidates:
            if not any(existing < candidate for existing in minimal_supersets):
                minimal_supersets.append(candidate)
                edges.add((lower, candidate))
    return edges


def closure_mapping_audit(
    source_intents: Iterable[Iterable[int]],
    target_context: np.ndarray,
    expert_mapping: Sequence[int],
    *,
    min_support: int = 1,
    window_size: int | None = None,
    target_intents: Iterable[Iterable[int]] | None = None,
) -> dict:
    """Audit an expert (or positional-expert) map followed by target closure.

    The source_intents and optional target_intents define the evaluated
    concept families. When these are support-filtered, all order/cover metrics
    are explicitly limited to those induced subposets.

    For the direct set-image map pi and target FCA closure gamma_t,
    Phi(B)=gamma_t(pi(B)) is necessarily monotone under attribute-set
    inclusion. The reported weak inclusion rate is therefore a mathematical
    invariant and should equal 1 (up to implementation error).
    """
    target = _binary_context(target_context)
    if min_support < 1 or min_support > target.shape[0]:
        raise ValueError("min_support must be between 1 and target object count")

    mapping = np.asarray(expert_mapping, dtype=int)
    if mapping.ndim != 1 or np.any(mapping < 0):
        raise ValueError("expert_mapping must be a non-negative 1-D array")
    if len(np.unique(mapping)) != len(mapping):
        raise ValueError("expert_mapping must be one-to-one")
    if np.any(mapping >= (target.shape[1] if window_size is None else target.shape[1] // window_size)):
        raise ValueError("expert_mapping references a target expert outside the context")
    if window_size is not None:
        if window_size <= 0 or target.shape[1] != len(mapping) * window_size:
            raise ValueError("target attributes must equal n_experts * window_size")

    source = [frozenset(int(v) for v in item) for item in source_intents]
    if len(set(source)) != len(source):
        raise ValueError("source_intents contains duplicates")
    if any(v < 0 for item in source for v in item):
        raise ValueError("source intents must contain non-negative attributes")

    targets = None
    if target_intents is not None:
        targets = {frozenset(int(v) for v in item) for item in target_intents}
        if any(v < 0 or v >= target.shape[1] for item in targets for v in item):
            raise ValueError("target_intents contains an out-of-range attribute")

    mapped_raw: list[frozenset[int]] = []
    mapped_closed: list[frozenset[int]] = []
    supports: list[int] = []
    expansions: list[int] = []
    best_jaccards: list[float] = []
    best_overlaps: list[int] = []
    for intent in source:
        if window_size is None:
            if any(v >= len(mapping) for v in intent):
                raise ValueError("source intent references an unmapped expert")
            raw = frozenset(int(mapping[v]) for v in intent)
        else:
            raw = positional_attribute_map(intent, mapping, window_size)
        closed = intent_closure(target, raw)
        support = len(derive_extent(target, raw))
        mapped_raw.append(raw)
        mapped_closed.append(closed)
        supports.append(support)
        expansions.append(len(closed - raw))
        if targets:
            overlaps = [len(raw & candidate) for candidate in targets]
            unions = [len(raw | candidate) for candidate in targets]
            best_overlaps.append(max(overlaps, default=0))
            best_jaccards.append(
                max((o / u for o, u in zip(overlaps, unions) if u), default=0.0)
            )
        else:
            best_overlaps.append(0)
            best_jaccards.append(0.0)

    n_source = len(source)
    if n_source == 0:
        raise ValueError("source_intents must not be empty")

    source_pairs = [
        (i, j)
        for i, left in enumerate(source)
        for j, right in enumerate(source)
        if left < right
    ]
    weak_preserved = sum(mapped_closed[i] <= mapped_closed[j] for i, j in source_pairs)
    strict_preserved = sum(mapped_closed[i] < mapped_closed[j] for i, j in source_pairs)

    reflection_pairs = [
        (i, j)
        for i in range(n_source)
        for j in range(n_source)
        if i != j and mapped_closed[i] <= mapped_closed[j]
    ]
    reflected = sum(source[i] <= source[j] for i, j in reflection_pairs)

    source_covers = cover_edges(source)
    target_cover_set = cover_edges(targets) if targets is not None else None
    target_index = {intent: i for i, intent in enumerate(mapped_closed)}
    collapsed_pairs = sum(
        mapped_closed[i] == mapped_closed[j]
        for i in range(n_source)
        for j in range(i + 1, n_source)
    )

    cover_success = None
    if target_cover_set is not None:
        cover_success = 0
        for left, right in source_covers:
            mapped_left = intent_closure(target, (
                left if window_size is None
                else positional_attribute_map(left, mapping, window_size)
            )) if window_size is not None else intent_closure(
                target, frozenset(int(mapping[v]) for v in left)
            )
            mapped_right = intent_closure(target, (
                right if window_size is None
                else positional_attribute_map(right, mapping, window_size)
            )) if window_size is not None else intent_closure(
                target, frozenset(int(mapping[v]) for v in right)
            )
            if (mapped_left, mapped_right) in target_cover_set:
                cover_success += 1

    def rate(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    unique_images = set(mapped_closed)
    return {
        "n_source_intents": n_source,
        "n_unique_target_closures": len(unique_images),
        "closure_collision_fraction": float(1.0 - len(unique_images) / n_source),
        "n_collapsed_source_pairs": int(collapsed_pairs),
        "closure_equality_rate": float(np.mean([
            raw == closed for raw, closed in zip(mapped_raw, mapped_closed)
        ])),
        "target_support_valid_rate": float(np.mean(np.asarray(supports) >= min_support)),
        "target_frequent_intent_membership_rate": (
            float(np.mean([closed in targets for closed in mapped_closed]))
            if targets is not None else None
        ),
        "mean_closure_expansion_attributes": float(np.mean(expansions)),
        "mean_best_target_intent_jaccard": (
            float(np.mean(best_jaccards)) if targets is not None else None
        ),
        "partial_overlap_at_least_two_rate": (
            float(np.mean(np.asarray(best_overlaps) >= 2)) if targets is not None else None
        ),
        "weak_intent_inclusion_preservation_rate": rate(
            int(weak_preserved), len(source_pairs)
        ),
        "strict_intent_inclusion_preservation_rate": rate(
            int(strict_preserved), len(source_pairs)
        ),
        "order_reflection_rate": rate(len(reflection_pairs) - int(reflected), len(reflection_pairs)),
        "n_source_induced_cover_edges": len(source_covers),
        "cover_preservation_rate": (
            rate(cover_success, len(source_covers))
            if cover_success is not None else None
        ),
        "order_scope": "provided_intent_families; filtered inputs yield induced-subposet metrics",
        "support_counts": supports,
    }
