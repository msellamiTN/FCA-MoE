from __future__ import annotations
import numpy as np

def _closure(mask: int, objects: list[int], n_attributes: int) -> int:
    extent = [row for row in objects if (row & mask) == mask]
    if not extent:
        return (1 << n_attributes) - 1
    common = (1 << n_attributes) - 1
    for row in extent:
        common &= row
    return common

def next_closure(context: np.ndarray, max_concepts: int = 100_000) -> list[int]:
    """Enumerate formal intents with Ganter's NextClosure algorithm."""
    X = np.asarray(context, dtype=np.uint8)
    if X.ndim != 2 or not np.isin(X, [0, 1]).all():
        raise ValueError("context must be a binary 2-D matrix")
    n_objects, n_attributes = X.shape
    objects = []
    for row in X:
        mask = 0
        for j, value in enumerate(row):
            if value:
                mask |= 1 << j
        objects.append(mask)
    full = (1 << n_attributes) - 1
    current = _closure(0, objects, n_attributes)
    intents = [current]
    while current != full:
        found = False
        for i in range(n_attributes - 1, -1, -1):
            bit = 1 << i
            if current & bit:
                continue
            prefix = current & (bit - 1)
            candidate = _closure(prefix | bit, objects, n_attributes)
            if (candidate & (bit - 1)) == prefix:
                current = candidate
                intents.append(current)
                found = True
                break
        if not found:
            break
        if len(intents) > max_concepts:
            raise RuntimeError("max_concepts exceeded; reduce context or use bounded FCA")
    return intents

def intent_to_indices(intent: int, n_attributes: int) -> list[int]:
    return [i for i in range(n_attributes) if intent & (1 << i)]

def concept_support(context: np.ndarray, intent: int) -> int:
    X = np.asarray(context, dtype=np.uint8)
    cols = intent_to_indices(intent, X.shape[1])
    if not cols:
        return len(X)
    return int(np.all(X[:, cols] == 1, axis=1).sum())

def lattice_summary(context: np.ndarray, intents: list[int]) -> dict:
    n_attr = np.asarray(context).shape[1]
    sizes = [len(intent_to_indices(i, n_attr)) for i in intents]
    supports = [concept_support(context, i) for i in intents]
    return {
        "n_objects": int(len(context)), "n_attributes": int(n_attr),
        "n_concepts": int(len(intents)),
        "intent_size_histogram": {str(k): sizes.count(k) for k in sorted(set(sizes))},
        "supports": supports,
    }
