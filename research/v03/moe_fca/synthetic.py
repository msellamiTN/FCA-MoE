from __future__ import annotations
import numpy as np

def planted_routes(n_tokens=12_000, n_experts=32, top_k=4, n_groups=8,
                   group_strength=.85, seed=7):
    """Synthetic planted-group routes for software checks, never model evidence."""
    if n_experts % n_groups:
        raise ValueError("n_experts must be divisible by n_groups")
    if not (0 <= group_strength <= 1):
        raise ValueError("group_strength must be in [0,1]")
    rng = np.random.default_rng(seed)
    groups = np.array_split(np.arange(n_experts), n_groups)
    out = np.empty((n_tokens, top_k), dtype=int)
    for t in range(n_tokens):
        group = groups[rng.integers(n_groups)]
        pool = group if rng.random() < group_strength else np.arange(n_experts)
        out[t] = rng.choice(pool, size=top_k, replace=False)
    return out
