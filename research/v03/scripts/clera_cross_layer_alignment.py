#!/usr/bin/env python3
"""Cross-Layer Expert Role Alignment (CLERA).

Learns a one-to-one correspondence between experts in a source MoE layer
and experts in selected target layers using three sequences, then evaluates
that correspondence on the held-out sequence. The alignment null independently
circular-shifts each target-layer sequence, preserving its within-layer route
sequence and expert marginals while breaking token-position correspondence.

The primary statistic is held-out mean matched phi correlation across experts.
Token-level mapped Top-K overlap and FCA tuple transport are secondary
descriptive endpoints. The permutation max statistic adjusts the primary
alignment test across the target layers supplied to this run.

This is exploratory when source/target layers were selected from the same
24-layer scan. It does not establish functional equivalence of expert weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_layer_blocks(
    root: Path,
    layer_id: int,
    sequence_ids: list[object],
    order: np.ndarray,
    n_experts: int,
    top_k: int,
    sequence_length: int,
) -> list[np.ndarray]:
    with np.load(root / "qwen_contexts_by_layer.npz") as archive:
        key = f"layer_{layer_id:02d}"
        if key not in archive.files:
            raise KeyError(f"Missing {key}; available layers: {archive.files}")
        raw_matrix = archive[key]
        if not np.isin(raw_matrix, [0, 1]).all():
            raise ValueError(f"{key} is not a binary routing matrix")
        matrix = raw_matrix.astype(np.uint8)

    alignment = pd.read_csv(root / "sequence_alignment.csv")
    if matrix.shape != (len(alignment), n_experts):
        raise ValueError(
            f"{key} has shape {matrix.shape}; expected ({len(alignment)}, {n_experts})"
        )
    sorted_alignment = alignment.iloc[order].reset_index(drop=True)
    blocks: list[np.ndarray] = []
    for sid in sequence_ids:
        positions = np.flatnonzero(
            sorted_alignment["sequence_id"].to_numpy() == sid
        )
        block = matrix[order[positions]]
        token_positions = sorted_alignment.iloc[positions][
            "token_position"
        ].to_numpy(dtype=int)
        if len(block) != sequence_length:
            raise ValueError(
                f"Sequence {sid!r}, {key}: {len(block)} rows; "
                f"expected {sequence_length}"
            )
        if not np.array_equal(token_positions, np.arange(sequence_length)):
            raise ValueError(
                f"Sequence {sid!r}, {key}: token positions are not contiguous 0..{sequence_length-1}"
            )
        if not np.all(block.sum(axis=1) == top_k):
            raise ValueError(
                f"Sequence {sid!r}, {key}: every token must route to exactly {top_k} experts"
            )
        blocks.append(block)
    return blocks


def phi_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pairwise phi/Pearson correlation between binary expert profiles.

    Constant profiles have undefined correlation and are assigned zero,
    remaining visible in diagnostics rather than being silently dropped.
    """
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("source and target must be 2-D with equal token counts")
    if a.shape[0] < 2:
        raise ValueError("At least two aligned tokens are required")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Expert profiles must be binary")

    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    numerator = ac.T @ bc
    a_norm = np.sqrt(np.sum(ac * ac, axis=0))
    b_norm = np.sqrt(np.sum(bc * bc, axis=0))
    denominator = a_norm[:, None] * b_norm[None, :]
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1.0, 1.0)


def phi_similarity_blocks(
    source_blocks: list[np.ndarray], target_blocks: list[np.ndarray]
) -> np.ndarray:
    """Pooled within-sequence phi matrix.

    Each sequence is centered separately before accumulating cross-products.
    This prevents sequence-specific expert prevalence differences from
    generating spurious correspondence when training sequences are stacked.
    """
    if len(source_blocks) != len(target_blocks) or not source_blocks:
        raise ValueError("Provide equally many non-empty source/target blocks")
    n_experts = source_blocks[0].shape[1]
    numerator = np.zeros((n_experts, n_experts), dtype=float)
    source_ss = np.zeros(n_experts, dtype=float)
    target_ss = np.zeros(n_experts, dtype=float)

    for source, target in zip(source_blocks, target_blocks):
        a = np.asarray(source, dtype=float)
        b = np.asarray(target, dtype=float)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
            raise ValueError("Each paired block must be 2-D and token-aligned")
        if a.shape[1] != n_experts or b.shape[1] != n_experts:
            raise ValueError("All blocks must use the same expert count")
        if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
            raise ValueError("Expert profiles must be binary")
        ac = a - a.mean(axis=0, keepdims=True)
        bc = b - b.mean(axis=0, keepdims=True)
        numerator += ac.T @ bc
        source_ss += np.sum(ac * ac, axis=0)
        target_ss += np.sum(bc * bc, axis=0)

    denominator = np.sqrt(source_ss[:, None] * target_ss[None, :])
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1.0, 1.0)


def hungarian_mapping(similarity: np.ndarray) -> np.ndarray:
    """Return mapping[source_expert] = target_expert maximizing total phi."""
    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Similarity matrix must be square for one-to-one mapping")
    rows, cols = linear_sum_assignment(-matrix)
    mapping = np.empty(matrix.shape[0], dtype=int)
    mapping[rows] = cols
    if len(np.unique(mapping)) != len(mapping):
        raise AssertionError("Hungarian assignment did not produce a permutation")
    return mapping


def mapped_topk_overlap(
    source: np.ndarray, target: np.ndarray, mapping: np.ndarray
) -> float:
    """Mean fraction of mapped source Top-K experts found in target Top-K."""
    if source.shape != target.shape:
        raise ValueError("Source and target matrices must have equal shapes")
    top_k = int(source.sum(axis=1)[0])
    if not np.all(source.sum(axis=1) == top_k) or not np.all(
        target.sum(axis=1) == top_k
    ):
        raise ValueError("Source and target must have constant Top-K cardinality")
    values = np.empty(len(source), dtype=float)
    for token in range(len(source)):
        mapped = mapping[np.flatnonzero(source[token])]
        target_ids = np.flatnonzero(target[token])
        values[token] = len(np.intersect1d(mapped, target_ids)) / top_k
    return float(values.mean())


def score_mapping(
    source: np.ndarray, target: np.ndarray, mapping: np.ndarray
) -> dict:
    similarity = phi_similarity(source, target)
    matched = similarity[np.arange(similarity.shape[0]), mapping]
    valid_source = np.sum(source, axis=0) > 0
    valid_target = np.sum(target, axis=0) > 0
    valid = valid_source & valid_target[mapping]
    return {
        "matched_phi_mean": float(matched.mean()),
        "matched_phi_median": float(np.median(matched)),
        "matched_phi_positive_fraction": float(np.mean(matched > 0)),
        "matched_phi_valid_fraction": float(np.mean(valid)),
        "mapped_topk_overlap": mapped_topk_overlap(source, target, mapping),
        "identity_phi_mean": float(np.mean(np.diag(similarity))),
        "identity_topk_overlap": mapped_topk_overlap(
            source, target, np.arange(source.shape[1])
        ),
    }


def extract_supported_higher_order_intents(
    context: np.ndarray, min_support: int = 3
) -> set[frozenset[int]]:
    """Enumerate closed intents of cardinality >=3 at the specified support."""
    x = np.asarray(context, dtype=np.uint8)
    if x.ndim != 2 or not np.isin(x, [0, 1]).all():
        raise ValueError("FCA context must be a binary 2-D matrix")
    if min_support < 1 or min_support > len(x):
        raise ValueError("min_support must be within [1, number of objects]")

    n_objects, n_attributes = x.shape
    columns: list[int] = []
    for attribute in range(n_attributes):
        mask = 0
        for object_id in np.flatnonzero(x[:, attribute]):
            mask |= 1 << int(object_id)
        columns.append(mask)

    full_extent = (1 << n_objects) - 1

    def closure(extent: int) -> tuple[int, ...]:
        return tuple(
            attribute
            for attribute, column in enumerate(columns)
            if (extent & column) == extent
        )

    found: set[frozenset[int]] = set()

    def visit(intent: tuple[int, ...], extent: int, start: int) -> None:
        if len(intent) >= 3 and extent.bit_count() >= min_support:
            found.add(frozenset(intent))
        current = set(intent)
        for attribute in range(start, n_attributes):
            if attribute in current:
                continue
            new_extent = extent & columns[attribute]
            if new_extent.bit_count() < min_support:
                continue
            new_intent = closure(new_extent)
            # CbO canonicity test: reject if a smaller attribute would have
            # entered the closure but was not already in the current intent.
            if any(
                earlier < attribute and earlier not in current
                for earlier in new_intent
            ):
                continue
            visit(new_intent, new_extent, attribute + 1)

    visit(closure(full_extent), full_extent, 0)
    return found


def universal_intents(
    blocks: list[np.ndarray], min_support: int
) -> set[frozenset[int]]:
    per_sequence = [
        extract_supported_higher_order_intents(block, min_support)
        for block in blocks
    ]
    if not per_sequence:
        return set()
    result = set(per_sequence[0])
    for intents in per_sequence[1:]:
        result.intersection_update(intents)
    return result


def tuple_transport_diagnostics(
    source_train_blocks: list[np.ndarray],
    target_test_block: np.ndarray,
    mapping: np.ndarray,
    min_support: int,
) -> dict:
    source_universal = universal_intents(source_train_blocks, min_support)
    target_test_intents = extract_supported_higher_order_intents(
        target_test_block, min_support
    )
    if not source_universal:
        return {
            "source_train_universal_tuple_count": 0,
            "target_test_higher_order_tuple_count": len(target_test_intents),
            "exact_transport_count": 0,
            "exact_transport_rate": None,
            "mean_best_jaccard": None,
            "mean_same_cardinality_best_jaccard": None,
            "cognate_jaccard_ge_0_5_rate": None,
        }

    exact = 0
    best_jaccards = []
    same_cardinality_jaccards = []
    for intent in source_universal:
        mapped = frozenset(int(mapping[expert]) for expert in intent)
        if mapped in target_test_intents:
            exact += 1

        similarities = [
            len(mapped & candidate) / len(mapped | candidate)
            for candidate in target_test_intents
        ]
        best_jaccards.append(max(similarities, default=0.0))
        same_cardinality = [
            len(mapped & candidate) / len(mapped | candidate)
            for candidate in target_test_intents
            if len(candidate) == len(mapped)
        ]
        same_cardinality_jaccards.append(
            max(same_cardinality, default=0.0)
        )

    count = len(source_universal)
    return {
        "source_train_universal_tuple_count": count,
        "target_test_higher_order_tuple_count": len(target_test_intents),
        "exact_transport_count": exact,
        "exact_transport_rate": exact / count,
        "mean_best_jaccard": float(np.mean(best_jaccards)),
        "mean_same_cardinality_best_jaccard": float(
            np.mean(same_cardinality_jaccards)
        ),
        "cognate_jaccard_ge_0_5_rate": float(
            np.mean(np.asarray(best_jaccards) >= 0.5)
        ),
    }


def cross_validate_pair(
    source_blocks: list[np.ndarray],
    target_blocks: list[np.ndarray],
    min_support: int,
) -> tuple[dict, list[dict]]:
    if len(source_blocks) != len(target_blocks) or len(source_blocks) < 3:
        raise ValueError("At least three paired sequences are required")
    fold_records: list[dict] = []
    mappings: list[np.ndarray] = []
    n_sequences = len(source_blocks)

    for held_out in range(n_sequences):
        train_indices = [i for i in range(n_sequences) if i != held_out]
        source_train = [source_blocks[i] for i in train_indices]
        target_train = [target_blocks[i] for i in train_indices]
        similarity_train = phi_similarity_blocks(source_train, target_train)
        mapping = hungarian_mapping(similarity_train)
        mappings.append(mapping)

        source_test = source_blocks[held_out]
        target_test = target_blocks[held_out]
        scores = score_mapping(source_test, target_test, mapping)
        tuple_scores = tuple_transport_diagnostics(
            [source_blocks[i] for i in train_indices],
            target_test,
            mapping,
            min_support,
        )
        fold_records.append({
            "held_out_sequence_index": held_out,
            "training_sequence_indices": json.dumps(train_indices),
            **scores,
            **tuple_scores,
            "mapping": json.dumps(mapping.tolist()),
        })

    pairwise_stability = []
    for i in range(len(mappings)):
        for j in range(i + 1, len(mappings)):
            pairwise_stability.append(float(np.mean(mappings[i] == mappings[j])))

    summary = {
        "folds": len(fold_records),
        "matched_phi_mean": float(np.mean([
            row["matched_phi_mean"] for row in fold_records
        ])),
        "matched_phi_sd_across_folds": float(np.std([
            row["matched_phi_mean"] for row in fold_records
        ], ddof=1)),
        "matched_phi_positive_fraction_mean": float(np.mean([
            row["matched_phi_positive_fraction"] for row in fold_records
        ])),
        "matched_phi_valid_fraction_mean": float(np.mean([
            row["matched_phi_valid_fraction"] for row in fold_records
        ])),
        "mapped_topk_overlap_mean": float(np.mean([
            row["mapped_topk_overlap"] for row in fold_records
        ])),
        "identity_phi_mean": float(np.mean([
            row["identity_phi_mean"] for row in fold_records
        ])),
        "identity_topk_overlap_mean": float(np.mean([
            row["identity_topk_overlap"] for row in fold_records
        ])),
        "mean_mapping_pairwise_agreement": float(
            np.mean(pairwise_stability) if pairwise_stability else 1.0
        ),
        "folds_detail": fold_records,
    }
    return summary, fold_records


def cross_validate_alignment_only(
    source_blocks: list[np.ndarray],
    target_blocks: list[np.ndarray],
) -> dict:
    """LOSO routing-alignment scores only; avoids FCA mining in null loops."""
    if len(source_blocks) != len(target_blocks) or len(source_blocks) < 3:
        raise ValueError("At least three paired sequences are required")
    fold_scores = []
    for held_out in range(len(source_blocks)):
        train_indices = [i for i in range(len(source_blocks)) if i != held_out]
        source_train = [source_blocks[i] for i in train_indices]
        target_train = [target_blocks[i] for i in train_indices]
        mapping = hungarian_mapping(phi_similarity_blocks(source_train, target_train))
        fold_scores.append(score_mapping(
            source_blocks[held_out], target_blocks[held_out], mapping
        ))
    return {
        "matched_phi_mean": float(np.mean([
            row["matched_phi_mean"] for row in fold_scores
        ])),
        "mapped_topk_overlap_mean": float(np.mean([
            row["mapped_topk_overlap"] for row in fold_scores
        ])),
    }


def shifted_blocks(
    blocks: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Independent non-zero circular shift per sequence, preserving route series."""
    shifted = []
    for block in blocks:
        if len(block) < 2:
            raise ValueError("Cannot circularly shift a sequence shorter than 2")
        offset = int(rng.integers(1, len(block)))
        shifted.append(np.roll(block, shift=offset, axis=0))
    return shifted


def permutation_calibration(
    source_blocks: list[np.ndarray],
    targets: dict[int, list[np.ndarray]],
    observed: dict[int, dict],
    permutations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    target_layers = sorted(targets)
    null_phi = np.empty((permutations, len(target_layers)), dtype=float)
    null_overlap = np.empty_like(null_phi)

    for replicate in range(permutations):
        for col, target_layer in enumerate(target_layers):
            shifted = shifted_blocks(targets[target_layer], rng)
            scores = cross_validate_alignment_only(source_blocks, shifted)
            null_phi[replicate, col] = scores["matched_phi_mean"]
            null_overlap[replicate, col] = scores["mapped_topk_overlap_mean"]

        if (replicate + 1) % 25 == 0 or replicate + 1 == permutations:
            print(
                f"CLERA circular-shift null {replicate + 1}/{permutations}",
                flush=True,
            )

    obs_phi = np.array([
        observed[layer]["matched_phi_mean"] for layer in target_layers
    ])
    obs_overlap = np.array([
        observed[layer]["mapped_topk_overlap_mean"] for layer in target_layers
    ])

    # Max statistic controls selection among the target layers included here.
    obs_max_phi = float(obs_phi.max())
    null_max_phi = null_phi.max(axis=1)
    phi_exceed = int(np.sum(null_max_phi >= obs_max_phi))
    p_phi = (phi_exceed + 1) / (permutations + 1)

    obs_max_overlap = float(obs_overlap.max())
    null_max_overlap = null_overlap.max(axis=1)
    overlap_exceed = int(np.sum(null_max_overlap >= obs_max_overlap))
    p_overlap = (overlap_exceed + 1) / (permutations + 1)

    return {
        "permutations": permutations,
        "seed": seed,
        "null_scheme": (
            "Independent non-zero circular shift of each target-layer sequence; "
            "same shifts used for all token rows within that sequence; source "
            "sequences remain fixed; Hungarian mappings are refitted in every "
            "fold and permutation."
        ),
        "max_adjustment_scope": target_layers,
        "primary_metric": "mean held-out matched phi correlation",
        "observed_target_layer_scores_phi": {
            str(layer): float(observed[layer]["matched_phi_mean"])
            for layer in target_layers
        },
        "observed_max_phi": obs_max_phi,
        "null_max_phi_mean": float(null_max_phi.mean()),
        "null_max_phi_sd": float(null_max_phi.std(ddof=1)),
        "null_max_phi_95pct": float(np.quantile(null_max_phi, 0.95, method="higher")),
        "max_phi_exceedances": phi_exceed,
        "max_phi_p_plus_one": float(p_phi),
        "observed_target_layer_scores_overlap": {
            str(layer): float(observed[layer]["mapped_topk_overlap_mean"])
            for layer in target_layers
        },
        "observed_max_overlap": obs_max_overlap,
        "null_max_overlap_mean": float(null_max_overlap.mean()),
        "null_max_overlap_95pct": float(
            np.quantile(null_max_overlap, 0.95, method="higher")
        ),
        "max_overlap_exceedances": overlap_exceed,
        "max_overlap_p_plus_one": float(p_overlap),
        "individual_target_layer_phi": {
            str(layer): {
                "observed": float(obs_phi[i]),
                "null_mean": float(null_phi[:, i].mean()),
                "null_95pct": float(np.quantile(
                    null_phi[:, i], 0.95, method="higher"
                )),
                "exceedances": int(np.sum(null_phi[:, i] >= obs_phi[i])),
                "p_plus_one_unadjusted": float(
                    (1 + np.sum(null_phi[:, i] >= obs_phi[i]))
                    / (permutations + 1)
                ),
            }
            for i, layer in enumerate(target_layers)
        },
        "guardrail": (
            "Permutation p-values are exploratory and conditional on these "
            "four sequences and the specified circular-shift null. The max "
            "adjustment covers only the target layers included in this run; "
            "it does not correct selection of source layer, tokenization, "
            "support threshold, or the broader 24-layer discovery process."
        ),
        "null_max_phi_values": null_max_phi.tolist(),
        "null_max_overlap_values": null_max_overlap.tolist(),
    }


def run(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    contexts_path = root / "qwen_contexts_by_layer.npz"
    alignment_path = root / "sequence_alignment.csv"
    if not contexts_path.exists() or not alignment_path.exists():
        raise FileNotFoundError("Missing routing context NPZ or alignment CSV")

    alignment = pd.read_csv(alignment_path)
    required = {"sequence_id", "token_position"}
    if not required.issubset(alignment.columns):
        raise ValueError(f"Alignment missing columns: {sorted(required - set(alignment.columns))}")
    order = alignment.sort_values(
        ["sequence_id", "token_position"], kind="stable"
    ).index.to_numpy()
    sorted_alignment = alignment.iloc[order].reset_index(drop=True)
    sequence_ids = list(pd.unique(sorted_alignment["sequence_id"]))
    if len(sequence_ids) < 3:
        raise ValueError("At least three sequences are required for held-out alignment")
    counts = sorted_alignment.groupby("sequence_id")["token_position"].count()
    if not (counts == args.sequence_length).all():
        raise ValueError(
            f"Every sequence must have {args.sequence_length} tokens; found {counts.to_dict()}"
        )

    if args.source_layer in args.target_layers:
        raise ValueError("Source layer cannot also be a target layer")
    if len(set(args.target_layers)) != len(args.target_layers):
        raise ValueError("Target layers must be unique")
    if any(not 0 <= layer < args.n_layers for layer in [args.source_layer] + args.target_layers):
        raise ValueError("Layer IDs must be in the configured range")
    if args.permutations < 99:
        raise ValueError("Use at least 99 permutations for a useful exploratory calibration")

    source_blocks = load_layer_blocks(
        root, args.source_layer, sequence_ids, order,
        args.n_experts, args.top_k, args.sequence_length
    )
    target_blocks = {
        layer: load_layer_blocks(
            root, layer, sequence_ids, order,
            args.n_experts, args.top_k, args.sequence_length
        )
        for layer in args.target_layers
    }

    min_support = max(1, int(np.ceil(args.support_fraction * args.sequence_length - 1e-12)))
    print(
        f"Source layer {args.source_layer}; target layers {args.target_layers}; "
        f"sequences={len(sequence_ids)}, tokens/sequence={args.sequence_length}, "
        f"FCA min support={min_support}",
        flush=True,
    )

    observed: dict[int, dict] = {}
    fold_tables: list[pd.DataFrame] = []
    for target_layer, blocks in target_blocks.items():
        summary, folds = cross_validate_pair(
            source_blocks, blocks, min_support=min_support
        )
        observed[target_layer] = summary
        for fold in folds:
            fold_tables.append(pd.DataFrame([{
                "source_layer": args.source_layer,
                "target_layer": target_layer,
                "held_out_sequence_index": fold["held_out_sequence_index"],
                "training_sequence_indices": fold["training_sequence_indices"],
                "matched_phi_mean": fold["matched_phi_mean"],
                "matched_phi_median": fold["matched_phi_median"],
                "matched_phi_positive_fraction": fold["matched_phi_positive_fraction"],
                "matched_phi_valid_fraction": fold["matched_phi_valid_fraction"],
                "mapped_topk_overlap": fold["mapped_topk_overlap"],
                "identity_phi_mean": fold["identity_phi_mean"],
                "identity_topk_overlap": fold["identity_topk_overlap"],
                "source_train_universal_tuple_count": fold["source_train_universal_tuple_count"],
                "target_test_higher_order_tuple_count": fold["target_test_higher_order_tuple_count"],
                "exact_transport_count": fold["exact_transport_count"],
                "exact_transport_rate": fold["exact_transport_rate"],
                "mean_best_jaccard": fold["mean_best_jaccard"],
                "mean_same_cardinality_best_jaccard": fold["mean_same_cardinality_best_jaccard"],
                "cognate_jaccard_ge_0_5_rate": fold["cognate_jaccard_ge_0_5_rate"],
                "mapping": fold["mapping"],
            }]))
        tuple_rates = [
            row["exact_transport_rate"] for row in folds
            if row["exact_transport_rate"] is not None
        ]
        tuple_rate_label = (
            f"{np.mean(tuple_rates):.3f}" if tuple_rates else "n/a"
        )
        print(
            f"Layer {args.source_layer}->{target_layer}: "
            f"held-out mean phi={summary['matched_phi_mean']:.5f}, "
            f"mapped Top-K overlap={summary['mapped_topk_overlap_mean']:.5f}, "
            f"mean mapping agreement={summary['mean_mapping_pairwise_agreement']:.3f}, "
            f"tuple exact transport={tuple_rate_label}",
            flush=True,
        )

    fold_frame = pd.concat(fold_tables, ignore_index=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fold_csv = output / "clera_fold_results.csv"
    fold_frame.to_csv(fold_csv, index=False)

    # Save the 4 x target_layer x source_expert cross-validated correspondences.
    mapping_records = []
    for target_layer, summary in observed.items():
        for fold in summary["folds_detail"]:
            mapping = json.loads(fold["mapping"])
            for source_expert, target_expert in enumerate(mapping):
                mapping_records.append({
                    "source_layer": args.source_layer,
                    "target_layer": target_layer,
                    "held_out_sequence_index": fold["held_out_sequence_index"],
                    "source_expert": source_expert,
                    "target_expert": int(target_expert),
                })
    mapping_csv = output / "clera_expert_mappings.csv"
    pd.DataFrame(mapping_records).to_csv(mapping_csv, index=False)

    print("Starting circular-shift permutation calibration...", flush=True)
    permutation = permutation_calibration(
        source_blocks, target_blocks, observed,
        permutations=args.permutations, seed=args.seed
    )
    perm_csv = output / "clera_permutation_max_statistics.csv"
    pd.DataFrame({
        "replicate": np.arange(args.permutations),
        "max_phi": permutation.pop("null_max_phi_values"),
        "max_topk_overlap": permutation.pop("null_max_overlap_values"),
    }).to_csv(perm_csv, index=False)

    summary = {
        "status": "completed",
        "analysis": "cross_layer_expert_role_alignment",
        "inferential_status": "exploratory_permutation_calibration",
        "source_layer": args.source_layer,
        "target_layers": args.target_layers,
        "model": "Qwen/Qwen1.5-MoE-A2.7B",
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1:test",
        "sequence_ids": [str(value) for value in sequence_ids],
        "sequences": len(sequence_ids),
        "sequence_length": args.sequence_length,
        "experts": args.n_experts,
        "top_k": args.top_k,
        "support_fraction_for_tuple_transport": args.support_fraction,
        "min_support_objects_per_sequence": min_support,
        "expert_similarity": "binary phi correlation on training token positions",
        "assignment": "Hungarian maximum-weight one-to-one assignment",
        "validation": "leave-one-sequence-out; mapping learned on other sequences",
        "permutation": permutation,
        "target_layer_results": {
            str(layer): {
                key: value for key, value in metrics.items()
                if key != "folds_detail"
            }
            for layer, metrics in observed.items()
        },
        "artifacts": {
            "fold_results_csv": str(fold_csv),
            "expert_mappings_csv": str(mapping_csv),
            "permutation_max_statistics_csv": str(perm_csv),
        },
        "input_sha256": {
            contexts_path.name: sha256_file(contexts_path),
            alignment_path.name: sha256_file(alignment_path),
        },
        "interpretation_guardrail": (
            "Expert IDs are layer-local. A learned mapping indicates similarity "
            "of routing activation profiles on held-out sequences, not shared "
            "expert parameters, identical computations, or semantic equivalence. "
            "Tuple transport uses FCA intents learned from training sequences "
            "and evaluates exact/Jaccard matches on the held-out target sequence. "
            "Only target layers included in this run are covered by the max "
            "permutation adjustment. Independent sequences and a frozen "
            "protocol are required for confirmatory claims."
        ),
    }
    summary_path = output / "clera_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"\nSaved CLERA summary: {summary_path}")
    print(f"Saved fold results: {fold_csv}")
    print(f"Saved expert mappings: {mapping_csv}")
    print(f"Saved permutation statistics: {perm_csv}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--source-layer", type=int, default=11)
    parser.add_argument("--target-layers", type=int, nargs="+", default=[14, 16])
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--n-experts", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--support-fraction", type=float, default=0.01)
    args = parser.parse_args()

    if not 0 < args.support_fraction <= 1:
        parser.error("--support-fraction must be in (0, 1]")
    if args.output is None:
        args.output = str(
            Path(args.root) / (
                f"sequence_replication/clera_layer{args.source_layer:02d}_"
                f"targets_{'_'.join(map(str, args.target_layers))}_v1"
            )
        )
    run(args)


if __name__ == "__main__":
    main()
