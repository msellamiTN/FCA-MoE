from __future__ import annotations
import numpy as np

def validate_routes(routes: np.ndarray, n_experts: int = 32, top_k: int | None = None) -> np.ndarray:
    """Validate expert IDs and per-token uniqueness; infer Top-K from input width by default."""
    x = np.asarray(routes, dtype=int)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("routes must have shape (n_tokens, top_k) with top_k >= 1")
    if top_k is None:
        top_k = x.shape[1]
    if x.shape[1] != top_k:
        raise ValueError(f"routes must have shape (n_tokens, {top_k})")
    if np.any((x < 0) | (x >= n_experts)):
        raise ValueError("expert ID outside [0, n_experts)")
    if any(len(set(row.tolist())) != top_k for row in x):
        raise ValueError("duplicate expert within a token route")
    return x

def token_context(routes: np.ndarray, n_experts: int = 32) -> np.ndarray:
    """Binary token × expert incidence matrix; each row has exactly Top-K ones."""
    x = validate_routes(routes, n_experts=n_experts)
    out = np.zeros((len(x), n_experts), dtype=np.uint8)
    out[np.arange(len(x))[:, None], x] = 1
    return out

def window_context(routes: np.ndarray, window_size: int, threshold: float,
                   n_experts: int = 32) -> np.ndarray:
    """Non-overlapping window × expert context using activation-frequency threshold."""
    if window_size <= 0 or not (0 < threshold <= 1):
        raise ValueError("window_size must be positive and threshold in (0,1]")
    x = token_context(routes, n_experts)
    n_windows = len(x) // window_size
    if n_windows == 0:
        raise ValueError("not enough tokens for one complete window")
    freq = x[:n_windows * window_size].reshape(n_windows, window_size, n_experts).mean(axis=1)
    return (freq >= threshold).astype(np.uint8)

def chronological_split(data: np.ndarray, train=.60, validation=.20):
    """Chronological 60/20/20 split; no shuffling."""
    if not (0 < train < 1 and 0 < validation < 1 and train + validation < 1):
        raise ValueError("invalid split fractions")
    n = len(data)
    a, b = int(n * train), int(n * (train + validation))
    if min(a, b-a, n-b) == 0:
        raise ValueError("split creates an empty partition")
    return data[:a], data[a:b], data[b:]
