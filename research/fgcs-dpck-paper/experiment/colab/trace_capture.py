#!/usr/bin/env python3
"""Capture OLMoE prefill routing events with fail-closed router validation.

This script is a research scaffold. It has not been executed on Google Colab
in the current session. It deliberately records prefill only (no cached decode).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any

import torch

MODEL_ID_DEFAULT = "allenai/OLMoE-1B-7B-0125"
REQUIRED_CSV_COLUMNS = {"family", "prompt"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_prompt_rows(csv_path: Path, max_prompts: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - fields
        if missing:
            raise ValueError(f"Prompt CSV is missing columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            prompt = (row.get("prompt") or "").strip()
            family = (row.get("family") or "unspecified").strip() or "unspecified"
            if not prompt:
                continue
            session_id = (row.get("session_id") or "").strip()
            if not session_id:
                session_id = f"session-{idx:06d}-{sha256_text(prompt)[:10]}"
            rows.append({"session_id": session_id, "family": family, "prompt": prompt})
            if max_prompts is not None and len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError("No non-empty prompts were found in the CSV.")
    return rows


def install_router_hooks(model: Any) -> tuple[list[Any], dict[int, dict[str, torch.Tensor]]]:
    """Capture the exact top-k indices and weights returned by OLMoE router gates."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Unsupported model layout: expected model.model.layers.")

    captures: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is None:
            raise RuntimeError(f"No OLMoE router gate found at layer {layer_idx}.")

        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                if not isinstance(output, (tuple, list)) or len(output) < 3:
                    raise RuntimeError(
                        f"Router hook output at layer {idx} is not (logits, scores, indices)."
                    )
                logits, scores, indices = output[0], output[1], output[2]
                if not all(torch.is_tensor(x) for x in (logits, scores, indices)):
                    raise RuntimeError(f"Non-tensor router output at layer {idx}.")
                captures[idx] = {
                    "indices": indices.detach().to("cpu", dtype=torch.int64),
                    "scores": scores.detach().to("cpu", dtype=torch.float32),
                    "logits": logits.detach().to("cpu", dtype=torch.float32),
                }
            return hook

        handles.append(gate.register_forward_hook(make_hook(layer_idx)))
    return handles, captures


def verify_capture(
    router_logits: Any,
    captures: dict[int, dict[str, torch.Tensor]],
    expected_layers: int,
    top_k: int,
    num_experts: int,
) -> dict[str, Any]:
    if router_logits is None or not isinstance(router_logits, (tuple, list)):
        raise RuntimeError("Model did not return router_logits; refusing to export an unverified trace.")
    if len(router_logits) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} router-logit tensors, received {len(router_logits)}."
        )
    if set(captures) != set(range(expected_layers)):
        raise RuntimeError(
            f"Router hooks did not capture every layer: got {sorted(captures)}."
        )

    report: dict[str, Any] = {"layers": {}, "all_indices_match": True}
    for layer_idx in range(expected_layers):
        logits = router_logits[layer_idx].detach().to("cpu", dtype=torch.float32)
        direct = captures[layer_idx]["indices"]
        scores = captures[layer_idx]["scores"]
        if logits.ndim != 2 or logits.shape[1] != num_experts:
            raise RuntimeError(
                f"Unexpected router logits shape at layer {layer_idx}: {tuple(logits.shape)}"
            )
        if direct.ndim != 2 or direct.shape[1] != top_k:
            raise RuntimeError(
                f"Unexpected dispatched index shape at layer {layer_idx}: {tuple(direct.shape)}"
            )
        if scores.shape != direct.shape or logits.shape[0] != direct.shape[0]:
            raise RuntimeError(f"Router output shapes disagree at layer {layer_idx}.")

        recomputed = torch.topk(torch.softmax(logits, dim=-1), k=top_k, dim=-1).indices
        equal = bool(torch.equal(recomputed, direct))
        report["layers"][str(layer_idx)] = {
            "tokens": int(direct.shape[0]),
            "indices_match": equal,
            "hook_indices_shape": list(direct.shape),
            "logits_shape": list(logits.shape),
        }
        report["all_indices_match"] = report["all_indices_match"] and equal
        if not equal:
            mismatch = torch.nonzero(recomputed != direct, as_tuple=False)[:5].tolist()
            raise RuntimeError(
                f"Router-index mismatch at layer {layer_idx}; first mismatches: {mismatch}"
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="CSV with family,prompt[,session_id].")
    parser.add_argument("--output-dir", type=Path, default=Path("dpck_capture"))
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--revision", default=None, help="HF revision; pin the resolved commit for final runs.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--window-tokens", type=int, default=64)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--load-8bit", action="store_true", help="Load using bitsandbytes int8 quantization.")
    parser.add_argument("--require-t4", action="store_true", help="Abort unless device name contains Tesla T4.")
    parser.add_argument("--smoke-test", action="store_true", help="Mark this run as smoke-test data, not research data.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Select a GPU runtime in Colab.")
    device_name = torch.cuda.get_device_name(0)
    if args.require_t4 and "T4" not in device_name.upper():
        raise RuntimeError(f"This run requires a Tesla T4; detected: {device_name}")
    if args.max_length < 8 or args.window_tokens < 1:
        raise ValueError("max-length must be >=8 and window-tokens must be >=1.")

    prompts = load_prompt_rows(args.prompts, args.max_prompts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    load_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if args.load_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    torch.cuda.reset_peak_memory_stats(0)
    model.eval()
    model.config.output_router_logits = True

    num_layers = int(model.config.num_hidden_layers)
    num_experts = int(model.config.num_experts)
    top_k = int(model.config.num_experts_per_tok)
    handles, captures = install_router_hooks(model)
    run_id = str(uuid.uuid4())
    created_at = utc_now()
    input_device = model.get_input_embeddings().weight.device

    trace_path = args.output_dir / "trace.jsonl"
    fidelity: dict[str, Any] = {"prompts": {}, "all_indices_match": True}
    total_routed_tokens = 0

    try:
        with trace_path.open("w", encoding="utf-8") as trace_file:
            for prompt_idx, row in enumerate(prompts):
                text = row["prompt"]
                prompt_hash = sha256_text(text)
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                    padding=False,
                )
                input_ids = encoded["input_ids"].to(input_device)
                attention_mask = encoded["attention_mask"].to(input_device)
                n_tokens = int(input_ids.shape[1])
                if n_tokens == 0:
                    continue
                captures.clear()

                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                    "output_router_logits": True,
                }
                # Older Transformers versions may not expose logits_to_keep.
                # Use it only when it is explicitly part of the model signature.
                import inspect
                if "logits_to_keep" in inspect.signature(model.forward).parameters:
                    forward_kwargs["logits_to_keep"] = 1
                with torch.inference_mode():
                    outputs = model(**forward_kwargs)

                report = verify_capture(
                    outputs.router_logits, captures, num_layers, top_k, num_experts
                )
                fidelity[row["session_id"]] = report
                fidelity["all_indices_match"] = fidelity["all_indices_match"] and report["all_indices_match"]

                for layer_idx in range(num_layers):
                    ids = captures[layer_idx]["indices"]
                    weights = captures[layer_idx]["scores"]
                    for token_pos in range(n_tokens):
                        record = {
                            "run_id": run_id,
                            "created_at": created_at,
                            "model_id": args.model_id,
                            "model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
                            "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", None) or args.revision,
                            "prompt_hash": prompt_hash,
                            "session_id": row["session_id"],
                            "family": row["family"],
                            "window_id": f"{row['session_id']}:L{layer_idx}:W{token_pos // args.window_tokens}",
                            "token_position": token_pos,
                            "phase": "prefill",
                            "layer_id": layer_idx,
                            "source_rank": -1,
                            "source_rank_semantics": "unknown_single_gpu_sentinel_not_a_physical_rank",
                            "expert_ids": [int(x) for x in ids[token_pos].tolist()],
                            "routing_weights": [float(x) for x in weights[token_pos].tolist()],
                            "precision": str(next(model.parameters()).dtype),
                            "quantization": "bitsandbytes-int8" if args.load_8bit else "fp16-load",
                            "batch_id": f"single-prompt-{prompt_idx:06d}",
                            "runtime_device": device_name,
                            "token_count_in_prompt": n_tokens,
                        }
                        trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                total_routed_tokens += n_tokens
                del outputs, encoded, input_ids, attention_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{prompt_idx + 1}/{len(prompts)}] captured {n_tokens} tokens for {row['session_id']}")
    finally:
        for handle in handles:
            handle.remove()

    if not fidelity["all_indices_match"]:
        raise RuntimeError("A router-index check failed; trace must not be used.")

    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "model_id": args.model_id,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "requested_revision": args.revision,
        "model_config": {
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "num_experts_per_tok": top_k,
        },
        "capture_scope": "prefill_only_single_prompt_forward",
        "source_rank_semantics": "-1 means not observed; this is not an expert-parallel rank",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "cuda_runtime": torch.version.cuda,
            "device_name": device_name,
            "device_properties": str(torch.cuda.get_device_properties(0)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        },
        "precision": "fp16 compute; int8 weight loading" if args.load_8bit else "fp16 load",
        "window_tokens": args.window_tokens,
        "max_length": args.max_length,
        "prompt_count": len(prompts),
        "routed_tokens_per_layer": total_routed_tokens,
        "trace_records": total_routed_tokens * num_layers,
        "smoke_test_only": bool(args.smoke_test),
        "raw_prompt_text_exported": False,
        "router_fidelity": fidelity,
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fidelity_path = args.output_dir / "router_fidelity.json"
    fidelity_path.write_text(json.dumps(fidelity, indent=2), encoding="utf-8")

    print(json.dumps({
        "trace": str(trace_path),
        "manifest": str(manifest_path),
        "model_revision_resolved": manifest["model_revision_resolved"],
        "device": device_name,
        "records": manifest["trace_records"],
        "router_indices_match": fidelity["all_indices_match"],
        "trace_sha256": manifest["trace_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()