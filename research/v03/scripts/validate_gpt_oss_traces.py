#!/usr/bin/env python3
"""Validate captured real GPT-OSS routing shards and emit a compact audit report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EXPERT_COLS = [f"expert_{i}" for i in range(4)]
SCORE_COLS = [f"score_{i}" for i in range(4)]

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
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    required = {"model_id","model_revision","dataset_id","dataset_split","sequence_id",
                "token_position","token_id","token_hash","layer_id","phase",*EXPERT_COLS,*SCORE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing columns: {sorted(missing)}")
    cfg = manifest["run_config"]
    if cfg["model_id"] != "openai/gpt-oss-20b" or cfg["n_experts"] != 32 or cfg["top_k"] != 4 or cfg["n_layers"] != 24:
        raise AssertionError("Manifest architecture does not match approved GPT-OSS-20B 32-expert Top-4 configuration")
    if set(df["layer_id"].unique()) != set(range(24)):
        raise AssertionError("Trace does not contain all 24 MoE layers")
    experts = df[EXPERT_COLS].to_numpy(dtype=int)
    scores = df[SCORE_COLS].to_numpy(dtype=float)
    if np.any((experts < 0) | (experts >= 32)):
        raise AssertionError("Expert ID outside [0,32)")
    sorted_experts = np.sort(experts, axis=1)
    if np.any(np.diff(sorted_experts, axis=1) == 0):
        raise AssertionError("Duplicate experts in a token's Top-4 route")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise AssertionError("Invalid routing weights")
    if not np.allclose(scores.sum(axis=1), 1.0, atol=2e-4):
        raise AssertionError("Selected routing weights do not sum to one")
    token_counts = df.groupby(["sequence_id","layer_id"])["token_position"].nunique()
    expected = int(cfg["sequence_length"])
    if not (token_counts == expected).all():
        raise AssertionError("A sequence/layer is missing token positions")
    route_rows = len(df)
    report = {
        "status": "passed", "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "dataset": f'{cfg["dataset_id"]}/{cfg["dataset_config"]}:{cfg["dataset_split"]}',
        "experts": 32, "top_k": 4, "layers": 24,
        "sequences": int(df["sequence_id"].nunique()), "tokens_per_sequence": expected,
        "layer_token_rows": int(route_rows), "unique_tokens": int(df[["sequence_id","token_position"]].drop_duplicates().shape[0]),
        "phase": sorted(df["phase"].unique().tolist()),
        "expert_load_by_layer": {
            str(int(layer)): np.bincount(df.loc[df.layer_id == layer, EXPERT_COLS].to_numpy().ravel(), minlength=32).tolist()
            for layer in sorted(df["layer_id"].unique())
        },
        "routing_weight_sum_min": float(scores.sum(axis=1).min()),
        "routing_weight_sum_max": float(scores.sum(axis=1).max()),
        "checkpoint_status": manifest["status"], "parts": len(parts),
    }
    out = root / "validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2))

if __name__ == "__main__":
    main()
