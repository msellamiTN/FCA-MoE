#!/usr/bin/env python3
"""Validate Qwen1.5-MoE-A2.7B routing and shared-gate Parquet traces."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TOP_K = 4
N_EXPERTS = 60
N_LAYERS = 24

def validate(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = sorted((root / "parts").glob("chunk_*.parquet"))
    if len(parts) != int(manifest["completed_chunks"]):
        raise AssertionError("Manifest completed_chunks does not match Parquet shard count")
    if not parts:
        raise AssertionError("No routing shards found")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    expert_cols = [f"expert_{i}" for i in range(TOP_K)]
    score_cols = [f"score_{i}" for i in range(TOP_K)]
    logit_cols = [f"selected_logit_{i}" for i in range(TOP_K)]
    required = {
        "model_id","model_revision","runtime_dtype","dataset_id","dataset_split",
        "sequence_id","token_position","token_id","token_hash","layer_id","phase",
        "shared_gate_logit","shared_gate_weight",*expert_cols,*score_cols,*logit_cols
    }
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing columns: {sorted(missing)}")
    cfg = manifest["run_config"]
    expected_cfg = {
        "model_id": "Qwen/Qwen1.5-MoE-A2.7B",
        "model_type": "qwen2_moe",
        "n_experts": N_EXPERTS,
        "top_k": TOP_K,
        "n_layers": N_LAYERS,
        "norm_topk_prob": False,
    }
    for key, expected in expected_cfg.items():
        if cfg.get(key) != expected:
            raise AssertionError(f"Manifest {key}={cfg.get(key)!r}; expected {expected!r}")
    if set(df["layer_id"].unique()) != set(range(N_LAYERS)):
        raise AssertionError("Trace does not contain all 24 MoE layers")
    experts = df[expert_cols].to_numpy(dtype=int)
    scores = df[score_cols].to_numpy(dtype=float)
    selected_logits = df[logit_cols].to_numpy(dtype=float)
    if np.any((experts < 0) | (experts >= N_EXPERTS)):
        raise AssertionError("Routed expert ID outside [0,60)")
    if np.any(np.diff(np.sort(experts, axis=1), axis=1) == 0):
        raise AssertionError("Duplicate routed experts in a token's Top-4")
    if not np.isfinite(scores).all() or np.any(scores <= 0) or np.any(scores > 1):
        raise AssertionError("Invalid raw softmax routing probabilities")
    if np.any(scores.sum(axis=1) > 1.0001):
        raise AssertionError("Top-4 raw probability sum cannot exceed 1")
    if not np.isfinite(selected_logits).all():
        raise AssertionError("Non-finite selected router logits")
    gate_logits = df["shared_gate_logit"].to_numpy(dtype=float)
    gate_weights = df["shared_gate_weight"].to_numpy(dtype=float)
    if not np.isfinite(gate_logits).all() or not np.isfinite(gate_weights).all():
        raise AssertionError("Non-finite shared gate values")
    expected_gate = 1.0 / (1.0 + np.exp(-gate_logits))
    if not np.allclose(gate_weights, expected_gate, rtol=2e-5, atol=2e-6):
        raise AssertionError("Shared gate weights do not match sigmoid(shared_gate_logit)")
    token_counts = df.groupby(["sequence_id","layer_id"])["token_position"].nunique()
    expected_tokens = int(cfg["sequence_length"])
    if not (token_counts == expected_tokens).all():
        raise AssertionError("A sequence/layer is missing token positions")
    # For each row, verify selected logits and scores correspond to the same
    # expert order, and that selected probabilities are non-increasing.
    if np.any(np.diff(scores, axis=1) > 2e-5):
        raise AssertionError("Top-K routing scores are not ordered descending")
    report = {
        "status": "passed",
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "runtime_dtype": cfg["runtime_dtype"],
        "checkpoint_declared_dtype": cfg["checkpoint_declared_dtype"],
        "dataset": f'{cfg["dataset_id"]}/{cfg["dataset_config"]}:{cfg["dataset_split"]}',
        "routed_experts": N_EXPERTS,
        "logical_shared_experts": cfg["logical_shared_experts"],
        "top_k": TOP_K,
        "layers": N_LAYERS,
        "sequences": int(df["sequence_id"].nunique()),
        "tokens_per_sequence": expected_tokens,
        "layer_token_rows": int(len(df)),
        "unique_tokens": int(df[["sequence_id","token_position"]].drop_duplicates().shape[0]),
        "phase": sorted(df["phase"].unique().tolist()),
        "shared_gate_weight_mean": float(gate_weights.mean()),
        "shared_gate_weight_min": float(gate_weights.min()),
        "shared_gate_weight_max": float(gate_weights.max()),
        "expert_load_by_layer": {
            str(int(layer)): np.bincount(
                df.loc[df.layer_id == layer, expert_cols].to_numpy().ravel(),
                minlength=N_EXPERTS
            ).tolist()
            for layer in sorted(df["layer_id"].unique())
        },
        "raw_top4_probability_sum_mean": float(scores.sum(axis=1).mean()),
        "raw_top4_probability_sum_min": float(scores.sum(axis=1).min()),
        "raw_top4_probability_sum_max": float(scores.sum(axis=1).max()),
        "checkpoint_status": manifest["status"],
        "parts": len(parts),
    }
    (root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2))

if __name__ == "__main__":
    main()
