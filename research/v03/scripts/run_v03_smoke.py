#!/usr/bin/env python3
"""Reproducible software smoke test for FCA-MoE v0.3."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from moe_fca.synthetic import planted_routes
from moe_fca.contexts import token_context, window_context, chronological_split
from moe_fca.fca import next_closure, lattice_summary
from moe_fca.metrics import concept_count_profile
from moe_fca.maxent_exact import fit_pairwise_maxent, sample_exact

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/v03_smoke")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    routes = planted_routes(n_tokens=args.tokens, n_experts=32, top_k=2, seed=args.seed)
    T = token_context(routes, n_experts=32)
    train, validation, test = chronological_split(T)
    report = {
        "experiment": "FCA-MoE v0.3 smoke test",
        "data_kind": "synthetic software-validation data; not model trace evidence",
        "routes_shape": list(routes.shape), "token_context_shape": list(T.shape),
        "split_shapes": [list(train.shape), list(validation.shape), list(test.shape)],
        "window_contexts": {},
    }
    for w in (32, 64, 128):
        for tau in (0.10, 0.20, 0.30):
            C = window_context(routes, w, tau, n_experts=32)
            report["window_contexts"][f"w{w}_tau{tau:.2f}"] = {
                "shape": list(C.shape),
                "mean_active_experts": float(C.sum(axis=1).mean()),
            }
    small = window_context(routes, 128, 0.20, n_experts=32)[:, :8]
    intents = next_closure(small, max_concepts=100_000)
    report["fca_8_expert_projection"] = {
        **lattice_summary(small, intents),
        **concept_count_profile(intents, small.shape[1]),
    }
    rng = np.random.default_rng(args.seed)
    X = (rng.random((600, 8)) < np.linspace(0.2, 0.55, 8)).astype(np.uint8)
    fit = fit_pairwise_maxent(X, max_experts=16)
    surrogate = sample_exact(fit, 5000, seed=args.seed + 1)
    report["small_e_maxent_oracle"] = {
        "experts": 8, "optimizer_success": fit["optimizer_success"],
        "fit_moment_max_abs_error": fit["moment_max_abs_error"],
        "sample_shape": list(surrogate.shape),
    }
    target = out / "smoke_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\\nWrote: {target}")

if __name__ == "__main__":
    main()
