#!/usr/bin/env python3
"""Single-entry F2-E confirmatory campaign.

Runs temporal LOO validation first. Only if the predeclared diagnostic passes
does it launch the nested parametric bootstrap. On --resume, a completed LOO
result is reused only when code, data, and all LOO settings match exactly.

Example:
  python scripts/run_f2e_confirmatory_campaign.py \
    --root /path/to/contexts_v1 \
    --output-base /path/to/f2e_confirmatory_campaign

After interruption, rerun the same command with --resume.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import f2e_nested_bootstrap as f2e
import f2e_temporal_loo_validation as loo


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def git_commit_sha():
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_sha256sums(root: Path):
    from hashlib import sha256
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            records.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(records) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--outer-replicates", type=int, default=999)
    parser.add_argument("--inner-replicates", type=int, default=100)
    parser.add_argument("--screen-replicates", type=int, default=8)
    parser.add_argument("--weight-grid-step", type=float, default=0.1)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0003, 0.001, 0.003])
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--predictive-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--screen-checkpoint-every", type=int, default=25)
    parser.add_argument("--inner-checkpoint-every", type=int, default=10)
    parser.add_argument("--outer-chunk-size", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-temporal-loo-summary",
        default=None,
        help="Reuse a completed, hash/configuration-matched temporal LOO summary from another campaign directory.",
    )
    args = parser.parse_args()

    if args.outer_replicates < 999:
        raise ValueError("Confirmatory campaign requires >= 999 outer replicates")
    if args.inner_replicates < 100:
        raise ValueError("Confirmatory campaign requires >= 100 inner replicates")
    if args.screen_replicates < 8:
        raise ValueError("Confirmatory campaign requires >= 8 screen replicates")
    if args.weight_grid_step > 0.1 + 1e-12:
        raise ValueError("Confirmatory campaign requires weight-grid step <= 0.1")
    if len(args.lambdas) < 3 or any(x <= 0 for x in args.lambdas):
        raise ValueError("Provide at least three positive lambda candidates")
    if args.predictive_replicates < 100:
        raise ValueError("Use at least 100 LOO predictive replicates")
    if args.screen_checkpoint_every < 1:
        raise ValueError("--screen-checkpoint-every must be >= 1")
    if args.inner_checkpoint_every < 1:
        raise ValueError("--inner-checkpoint-every must be >= 1")
    if args.outer_chunk_size < 1:
        raise ValueError("--outer-chunk-size must be >= 1")

    root = Path(args.root).resolve()
    base = Path(args.output_base).resolve()
    loo_dir = base / "temporal_loo"
    bootstrap_dir = base / "nested_bootstrap"
    loo_summary_path = loo_dir / "loo_validation_summary.json"
    campaign_manifest = base / "campaign_manifest.json"

    contexts_path = root / "qwen_contexts_by_layer.npz"
    alignment_path = root / "sequence_alignment.csv"
    if not contexts_path.exists() or not alignment_path.exists():
        raise FileNotFoundError(
            "Expected qwen_contexts_by_layer.npz and sequence_alignment.csv under --root"
        )

    expected_loo = {
        "lambda_candidates": list(map(float, args.lambdas)),
        "maxiter": int(args.maxiter),
        "weight_grid_step": float(args.weight_grid_step),
        "screen_replicates": int(args.screen_replicates),
        "predictive_replicates": int(args.predictive_replicates),
        "seed": int(args.seed),
        "loo_script_sha256": loo.sha256_file(Path(loo.__file__)),
        "contexts_sha256": loo.sha256_file(contexts_path),
        "alignment_sha256": loo.sha256_file(alignment_path),
    }

    if bootstrap_dir.joinpath("run_config.json").exists() and not args.resume:
        raise RuntimeError(
            f"{bootstrap_dir} already contains a nested-bootstrap run. "
            "Use --resume with identical arguments or select a new output-base."
        )

    start = time.time()
    reused_loo = False
    loo_result = None

    cached_loo_path = (
        Path(args.reuse_temporal_loo_summary).resolve()
        if args.reuse_temporal_loo_summary else loo_summary_path
    )
    if args.resume and loo_summary_path.exists():
        cached = json.loads(loo_summary_path.read_text())
        mismatches = {
            key: {"cached": cached.get(key), "expected": value}
            for key, value in expected_loo.items()
            if cached.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Cannot resume: cached temporal LOO summary does not match "
                f"the current code/data/configuration: {json.dumps(mismatches)}. "
                "Use a new output-base for a new statistical campaign."
            )
        if cached.get("status") != "completed":
            raise RuntimeError("Cached temporal LOO validation is not complete")
        loo_result = cached
        reused_loo = True
        print("Reusing hash- and configuration-matched temporal LOO result.", flush=True)
    elif args.reuse_temporal_loo_summary:
        if not cached_loo_path.exists():
            raise FileNotFoundError(
                f"Requested reusable LOO summary does not exist: {cached_loo_path}"
            )
        cached = json.loads(cached_loo_path.read_text())
        mismatches = {
            key: {"cached": cached.get(key), "expected": value}
            for key, value in expected_loo.items()
            if cached.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Cannot reuse temporal LOO summary: its code/data/configuration "
                f"does not match this campaign: {json.dumps(mismatches)}"
            )
        if cached.get("status") != "completed":
            raise RuntimeError("Reusable temporal LOO summary is not complete")
        loo_dir.mkdir(parents=True, exist_ok=True)
        save_json(loo_summary_path, cached)
        loo_result = cached
        reused_loo = True
        print(
            "Reusing external hash- and configuration-matched temporal LOO result.",
            flush=True,
        )
    else:
        loo_dir.mkdir(parents=True, exist_ok=True)
        _, loo_result = loo.run_validation(
            root=root,
            output=loo_dir,
            lambdas=args.lambdas,
            maxiter=args.maxiter,
            grid_step=args.weight_grid_step,
            screen_replicates=args.screen_replicates,
            predictive_replicates=args.predictive_replicates,
            seed=args.seed,
        )

    loo_pass = bool(loo_result.get("predeclared_loo_diagnostic_pass") is True)
    save_json(campaign_manifest, {
        "git_commit_sha": git_commit_sha(),
        "stage": "temporal_loo_completed",
        "status": "temporal_loo_passed" if loo_pass else "stopped_temporal_loo_failed",
        "root": str(root),
        "output_base": str(base),
        "temporal_loo_summary": str(loo_summary_path),
        "temporal_loo_result": loo_result,
        "temporal_loo_reused_from_cache": reused_loo,
        "confirmatory_configuration": {
            "outer_replicates": args.outer_replicates,
            "inner_replicates": args.inner_replicates,
            "screen_replicates": args.screen_replicates,
            "weight_grid_step": args.weight_grid_step,
            "lambdas": list(map(float, args.lambdas)),
            "maxiter": args.maxiter,
            "seed": args.seed,
        },
    })

    if not loo_pass:
        print(
            "STOP: temporal LOO diagnostic failed. The nested bootstrap was not run. "
            "Review loo_candidate_summary.csv and loo_transition_and_profile_metrics.csv.",
            flush=True,
        )
        raise SystemExit(2)

    save_json(campaign_manifest, {
        "git_commit_sha": git_commit_sha(),
        "stage": "nested_bootstrap_running",
        "status": "running",
        "root": str(root),
        "output_base": str(base),
        "temporal_loo_summary": str(loo_summary_path),
        "temporal_loo_result": loo_result,
        "temporal_loo_reused_from_cache": reused_loo,
        "nested_bootstrap_output": str(bootstrap_dir),
        "confirmatory_configuration": {
            "outer_replicates": args.outer_replicates,
            "inner_replicates": args.inner_replicates,
            "screen_replicates": args.screen_replicates,
            "weight_grid_step": args.weight_grid_step,
            "lambdas": list(map(float, args.lambdas)),
            "maxiter": args.maxiter,
            "seed": args.seed,
        },
    })
    print(
        f"Launching nested bootstrap; output directory: {bootstrap_dir}",
        flush=True,
    )

    # Invoke the F2-E CLI in-process so its validation, fingerprinting,
    # resumable checkpoints, and output files remain authoritative.
    argv = [
        "f2e_nested_bootstrap.py",
        "--root", str(root),
        "--output", str(bootstrap_dir),
        "--temporal-loo-summary", str(loo_summary_path),
        "--outer-replicates", str(args.outer_replicates),
        "--inner-replicates", str(args.inner_replicates),
        "--screen-replicates", str(args.screen_replicates),
        "--weight-grid-step", str(args.weight_grid_step),
        "--lambdas", *[str(x) for x in args.lambdas],
        "--maxiter", str(args.maxiter),
        "--seed", str(args.seed),
        "--checkpoint-every", "1",
        "--screen-checkpoint-every", str(args.screen_checkpoint_every),
        "--inner-checkpoint-every", str(args.inner_checkpoint_every),
        "--outer-chunk-size", str(args.outer_chunk_size),
    ]
    if args.resume:
        argv.append("--resume")

    original_argv = sys.argv
    try:
        sys.argv = argv
        f2e.main()
    finally:
        sys.argv = original_argv

    summary_path = bootstrap_dir / "f2e_nested_bootstrap_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Nested bootstrap returned without writing its summary")
    bootstrap_result = json.loads(summary_path.read_text())
    campaign_result = {
        "git_commit_sha": git_commit_sha(),
        "stage": "nested_bootstrap_completed",
        "status": bootstrap_result.get("status"),
        "inferential_status": bootstrap_result.get("inferential_status"),
        "temporal_loo_result": loo_result,
        "nested_bootstrap_summary": bootstrap_result,
        "elapsed_seconds": time.time() - start,
        "python": platform.python_version(),
    }
    save_json(campaign_manifest, campaign_result)
    write_sha256sums(base)
    print(json.dumps(campaign_result, indent=2), flush=True)

    if bootstrap_result.get("inferential_status") != "ready_for_review":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
