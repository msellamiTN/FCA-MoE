from __future__ import annotations
import numpy as np

def pairwise_moments(context: np.ndarray):
    """Return empirical first- and second-order activation moments."""
    X = np.asarray(context, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("context must be a finite matrix")
    return X.mean(axis=0), (X.T @ X) / max(len(X), 1)

def pairwise_moment_error(reference: np.ndarray, surrogate: np.ndarray) -> dict:
    p, p2 = pairwise_moments(reference)
    q, q2 = pairwise_moments(surrogate)
    return {
        "marginal_max_abs_error": float(np.max(np.abs(p-q))),
        "pairwise_max_abs_error": float(np.max(np.abs(p2-q2))),
        "marginal_rmse": float(np.sqrt(np.mean((p-q)**2))),
        "pairwise_rmse": float(np.sqrt(np.mean((p2-q2)**2))),
    }

def concept_count_profile(intents: list[int], n_attributes: int) -> dict:
    sizes = np.array([int(i).bit_count() for i in intents])
    return {
        "n_concepts": int(len(intents)),
        "n_singletons": int(np.sum(sizes == 1)),
        "n_order_ge_4": int(np.sum(sizes >= 4)),
        "mean_intent_size": float(sizes.mean()) if len(sizes) else 0.0,
        "max_intent_size": int(sizes.max()) if len(sizes) else 0,
    }
