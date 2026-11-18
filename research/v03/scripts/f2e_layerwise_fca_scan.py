#!/usr/bin/env python3
"""Descriptive FCA scan across all routed layers of a saved Qwen MoE trace.

This is an exploratory, non-inferential depth scan. It reuses the exact
multi-threshold FCA implementation and support definitions from F2-E.
No null simulations, p-values, layer selection, or significance claims
are produced by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from f2e_nested_bootstrap import evaluate_context  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_layer_blocks(root: Path, layer_id: int, expected_sequences: int,
                      sequence_length: int, n_experts: int,
                      top_k: int) -> tuple[list[np.ndarray], list[str]]:
    contexts_path = root / "qwen_contexts_by_layer.npz"
    alignment_path = root / "sequence_alignment.csv"

    alignment = pd.read_csv(alignment_path)
    required = {"sequence_id", "token_position"}
    if not required.issubset(alignment.columns):
        raise ValueError(f"Alignment is missing columns: {sorted(required - set(alignment.columns))}")

    order = alignment.sort_values(
        ["sequence_id", "token_position"], kind="stable"
    ).index.to_numpy()
    aligned = alignment.iloc[order].reset_index(drop=True)
    sequence_ids = list(pd.unique(aligned["sequence_id"]))

    if len(sequence_ids) != expected_sequences:
        raise ValueError(
            f"Expected {expected_sequences} sequences, found {len(sequence_ids)}"
        )

    with np.load(contexts_path) as archive:
        key = f"layer_{layer_id:02d}"
        if key not in archive.files:
            raise KeyError(f"Missing {key}; available keys: {archive.files}")
        X = archive[key].astype(np.uint8)

    if X.shape != (len(alignment), n_experts):
        raise ValueError(
            f"{key} shape {X.shape}; expected ({len(alignment)}, {n_experts})"
        )
    if not np.isin(X, [0, 1]).all():
        raise ValueError(f"{key} contains non-binary values")

    blocks = []
    for sid in sequence_ids:
        positions = np.flatnonzero(aligned["sequence_id"].to_numpy() == sid)
        block = X[order[positions]]
        token_positions = aligned.iloc[positions]["token_position"].to_numpy(dtype=int)

        if len(block) != sequence_length:
            raise ValueError(
                f"Sequence {sid!r} has {len(block)} rows; expected {sequence_length}"
            )
        if not np.array_equal(token_positions, np.arange(sequence_length)):
            raise ValueError(f"Sequence {sid!r} has missing/duplicate/out-of-order token positions")
        if not np.all(block.sum(axis=1) == top_k):
            raise ValueError(f"{key}, sequence {sid!r}: each row must contain exactly {top_k} experts")

        blocks.append(block)

    return blocks, [str(sid) for sid in sequence_ids]


def scan(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    contexts_path = root / "qwen_contexts_by_layer.npz"
    alignment_path = root / "sequence_alignment.csv"
    if not contexts_path.exists() or not alignment_path.exists():
        raise FileNotFoundError("Missing qwen_contexts_by_layer.npz or sequence_alignment.csv")

    rows = []
    for layer_id in range(args.n_layers):
        blocks, sequence_ids = load_layer_blocks(
            root, layer_id, args.sequences, args.sequence_length,
            args.n_experts, args.top_k
        )
        metrics = evaluate_context(blocks)
        row = {
            "layer_id": layer_id,
            "sequence_count": len(blocks),
            "tokens_per_sequence": args.sequence_length,
            "total_tokens": sum(len(b) for b in blocks),
            "unique_topk_states": int(len({
                tuple(np.flatnonzero(r).tolist())
                for block in blocks for r in block
            })),
            "expert_load_entropy_nats": float(
                -np.sum(
                    (lambda p: p[p > 0] * np.log(p[p > 0]))(
                        np.bincount(
                            np.concatenate([b.argmax(axis=1) for b in blocks]),
                            minlength=args.n_experts
                        ) / (sum(len(b) for b in blocks))
                    )
                )
            ),
        }
        row.update({k: float(v) for k, v in metrics.items()})
        rows.append(row)
        print(
            f"Layer {layer_id:02d}/{args.n_layers - 1:02d}: "
            f"pooled HO@1%={row['pooled_higher_order_1pct']}, "
            f"mean-sequence HO@1%={row['mean_sequence_higher_order_1pct']}, "
            f"pooled HO@2%={row['pooled_higher_order_2pct']}, "
            f"pooled HO@5%={row['pooled_higher_order_5pct']}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values("layer_id")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "layerwise_fca_metrics.csv"
    json_path = output / "layerwise_fca_scan_summary.json"
    frame.to_csv(csv_path, index=False)

    summary = {
        "status": "completed",
        "analysis_type": "descriptive_exploratory_layerwise_fca_scan",
        "inferential_status": "no_hypothesis_tests_or_p_values",
        "model": "Qwen/Qwen1.5-MoE-A2.7B",
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1:test",
        "routing": {
            "layers": args.n_layers,
            "experts": args.n_experts,
            "top_k": args.top_k,
            "sequences": args.sequences,
            "tokens_per_sequence": args.sequence_length,
        },
        "support_fractions": [0.01, 0.02, 0.05],
        "metric_definition": (
            "Exact F2-E multi-threshold CbO counts. Higher-order means "
            "closed intents with at least three attributes."
        ),
        "sequence_ids": load_layer_blocks(
            root, 0, args.sequences, args.sequence_length,
            args.n_experts, args.top_k
        )[1],
        "input_sha256": {
            contexts_path.name: sha256_file(contexts_path),
            alignment_path.name: sha256_file(alignment_path),
        },
        "results_csv": str(csv_path),
        "layers": rows,
        "interpretation_guardrail": (
            "Descriptive scan only. Do not interpret the largest layer, "
            "support threshold, or metric as statistically significant. "
            "Any selected layer requires independent validation with a "
            "predeclared family-wise procedure."
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {json_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--n-experts", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    args = parser.parse_args()
    if args.output is None:
        args.output = str(
            Path(args.root) / "sequence_replication/layerwise_fca_scan_v1"
        )
    scan(args)


if __name__ == "__main__":
    main()
