"""Controlled synthetic routing workloads for FCA-MoE v0.2.

The positive control has persistent window-level expert communities. The IID
negative control permutes exactly the same route-pair sequence, preserving
empirical pair marginals while removing persistent window membership.
"""
from __future__ import annotations
import torch

COMMUNITIES = [
    [(0, 1), (0, 2), (1, 3), (2, 3)],
    [(2, 3), (2, 4), (3, 5), (4, 5)],
    [(4, 5), (4, 6), (5, 7), (6, 7)],
    [(0, 1), (0, 6), (1, 7), (6, 7)],
]
EXPECTED_COMMUNITY_INTENTS = [
    frozenset({"E0", "E1", "E2", "E3"}),
    frozenset({"E2", "E3", "E4", "E5"}),
    frozenset({"E4", "E5", "E6", "E7"}),
    frozenset({"E0", "E1", "E6", "E7"}),
]

def make_correlated_routing_data(
    n_samples=4096, d_model=32, n_experts=8, *,
    seed=17, noise=0.20, window_size=64, mode="window_correlated"
):
    if mode not in {"window_correlated", "iid"}:
        raise ValueError("mode must be 'window_correlated' or 'iid'")
    if n_experts < 8 or d_model < n_experts:
        raise ValueError("fixture requires n_experts >= 8 and d_model >= n_experts")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    route_gen = torch.Generator().manual_seed(seed)
    patterns = sorted(set(pair for group in COMMUNITIES for pair in group))
    pattern_to_id = {p: i for i, p in enumerate(patterns)}
    by_community = [[pattern_to_id[p] for p in group] for group in COMMUNITIES]

    # Construct the positive-control labels first.
    labels = []
    for i in range(n_samples):
        community = (i // window_size) % len(COMMUNITIES)
        choices = by_community[community]
        labels.append(choices[int(torch.randint(len(choices), (1,), generator=route_gen).item())])

    # Matched negative control: same exact route-pair multiset, randomly permuted.
    if mode == "iid":
        shuffle_gen = torch.Generator().manual_seed(seed + 101)
        order = torch.randperm(n_samples, generator=shuffle_gen).tolist()
        labels = [labels[i] for i in order]

    targets = torch.zeros(n_samples, n_experts)
    for i, label in enumerate(labels):
        targets[i, list(patterns[label])] = 1.0

    # Matched latent feature basis and noise across both conditions.
    basis_gen = torch.Generator().manual_seed(seed + 202)
    noise_gen = torch.Generator().manual_seed(seed + 303)
    raw_basis = torch.randn(d_model, n_experts, generator=basis_gen)
    basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
    features = targets @ basis.T.contiguous() + noise * torch.randn(n_samples, d_model, generator=noise_gen)
    return features.float(), targets.float(), patterns
