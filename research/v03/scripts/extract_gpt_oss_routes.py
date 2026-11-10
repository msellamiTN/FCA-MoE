#!/usr/bin/env python3
"""Capture actual GPT-OSS-20B router decisions with resumable Parquet checkpoints.

This script intentionally refuses unsupported GPU architectures. MXFP4 expert
kernels require NVIDIA compute capability 7.5+ per the upstream deployment guide.
CPU offload is a memory-placement strategy; it cannot make an unsupported GPU kernel run.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe_fca.trace_io import ParquetTraceWriter

MODEL_ID = "openai/gpt-oss-20b"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"

def hardware_preflight(min_cpu_gib: float = 24.0) -> dict:
    import torch, psutil
    info = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_ram_available_gib": round(psutil.virtual_memory().available / 2**30, 2),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. GPT-OSS-20B MXFP4 extraction requires a supported CUDA GPU; CPU offload alone is not a validated execution backend.")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    info.update({
        "gpu_name": name, "compute_capability": f"{capability[0]}.{capability[1]}",
        "gpu_vram_free_gib": round(free / 2**30, 2),
        "gpu_vram_total_gib": round(total / 2**30, 2),
    })
    if capability < (7, 5):
        raise RuntimeError(
            f"Unsupported GPU: {name} has compute capability {capability[0]}.{capability[1]}. "
            "The current Hugging Face MXFP4 documentation specifies NVIDIA compute capability 7.5 or newer. "
            "A Colab T4 (SM75) meets the documented minimum; memory placement and actual kernel loading must still be tested. "
            "If kernel loading or inference fails, retain the diagnostic and evaluate the documented higher-precision fallback or a different runtime without silently changing the experiment."
        )
    if info["cpu_ram_available_gib"] < min_cpu_gib:
        raise RuntimeError(
            f"Only {info['cpu_ram_available_gib']} GiB host RAM is available; "
            f"at least {min_cpu_gib} GiB free is required for the configured offload plan."
        )
    return info

def token_chunks(dataset, tokenizer, sequence_length: int):
    """Pack corpus tokens into deterministic fixed-length sequences."""
    buffer: list[int] = []
    for row in dataset:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            buffer.extend(ids)
        while len(buffer) >= sequence_length:
            chunk, buffer = buffer[:sequence_length], buffer[sequence_length:]
            yield np.asarray(chunk, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sequence-length", type=int, default=256)
    ap.add_argument("--max-chunks", type=int, default=4, help="maximum total committed chunks in this output directory; 0 means the complete streaming split")
    ap.add_argument("--gpu-memory", default="12GiB")
    ap.add_argument("--cpu-memory", default="8GiB")
    ap.add_argument("--offload-folder", default="/content/hf_offload")
    ap.add_argument("--revision", default=None, help="Optional immutable Hugging Face model commit SHA")
    ap.add_argument("--dataset-revision", default="main", help="Hugging Face dataset revision (pin a commit SHA for strict reproducibility)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--min-cpu-ram-gib", type=float, default=8.0)
    args = ap.parse_args()
    if args.sequence_length < 16:
        ap.error("--sequence-length must be >= 16")

    hw = hardware_preflight(args.min_cpu_ram_gib)
    print("Hardware preflight:", json.dumps(hw, indent=2))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    model_kwargs = {
        "device_map": "auto", "torch_dtype": "auto",
        "max_memory": {0: args.gpu_memory, "cpu": args.cpu_memory},
        "offload_folder": args.offload_folder,
        "offload_state_dict": True, "low_cpu_mem_usage": True,
        "revision": args.revision,
    }
    Path(model_kwargs["offload_folder"]).mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    cfg = model.config
    quant_cfg = getattr(cfg, "quantization_config", None)
    if hasattr(quant_cfg, "to_dict"):
        quant_cfg = quant_cfg.to_dict()
    quant_method = quant_cfg.get("quant_method") if isinstance(quant_cfg, dict) else None
    if quant_method != "mxfp4":
        raise RuntimeError(
            f"Expected the checkpoint's native MXFP4 quantization, got {quant_method!r}. "
            "Refusing an implicit precision fallback; inspect the load logs and approve any precision change explicitly."
        )
    n_experts = int(getattr(cfg, "num_local_experts", -1))
    top_k = int(getattr(cfg, "num_experts_per_tok", -1))
    n_layers = int(getattr(cfg, "num_hidden_layers", -1))
    if (n_experts, top_k, n_layers) != (32, 4, 24):
        raise RuntimeError(f"Unexpected architecture {(n_experts, top_k, n_layers)}; expected (32 experts, Top-4, 24 layers).")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Unexpected Transformers GPT-OSS module layout; router instrumentation cannot be attached safely.")

    resolved_revision = args.revision or getattr(cfg, "_commit_hash", None) or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if not resolved_revision:
        raise RuntimeError("Could not resolve an immutable Hugging Face model commit. Pass --revision with a commit SHA.")
    run_config = {
        "model_id": MODEL_ID, "model_revision": resolved_revision,
        "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG, "dataset_split": DATASET_SPLIT, "dataset_revision": args.dataset_revision,
        "sequence_length": args.sequence_length, "n_experts": n_experts, "top_k": top_k,
        "n_layers": n_layers, "phase": "prefill_teacher_forced",
        "instrumentation": "forward_hook_on_each_GptOssTopKRouter",
    }
    writer = ParquetTraceWriter(args.output_dir, run_config, resume=not args.no_resume)
    captures: dict[int, dict[str, np.ndarray]] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(module, inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise RuntimeError(f"Router output contract changed at layer {layer_id}; expected (logits, scores, indices).")
            logits, scores, indices = output[:3]
            if logits.ndim != 2 or logits.shape[-1] != n_experts:
                raise RuntimeError(f"Unexpected router logits shape at layer {layer_id}: {tuple(logits.shape)}")
            idx = indices.detach().to("cpu", dtype=torch.int16).numpy()
            sc = scores.detach().to("cpu", dtype=torch.float32).numpy()
            selected_logits = torch.gather(logits.detach(), -1, indices).to("cpu", dtype=torch.float32).numpy()
            captures[layer_id] = {"expert_ids": idx, "scores": sc, "selected_logits": selected_logits}
        return hook

    for layer_id, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "router"):
            raise RuntimeError(f"Missing GPT-OSS router at transformer layer {layer_id}.")
        handles.append(layer.mlp.router.register_forward_hook(make_hook(layer_id)))

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, revision=args.dataset_revision, streaming=True)
    start_chunk = writer.next_chunk_id
    print(f"Resume point: chunk {start_chunk}; output={args.output_dir}")
    processed_this_run = 0
    try:
        for chunk_id, token_ids in enumerate(token_chunks(dataset, tokenizer, args.sequence_length)):
            if args.max_chunks and chunk_id >= args.max_chunks:
                break
            if chunk_id < start_chunk:
                continue
            captures.clear()
            input_device = model.get_input_embeddings().weight.device
            if input_device.type == "meta":
                input_device = torch.device("cuda:0")
            input_ids = torch.as_tensor(token_ids, dtype=torch.long, device=input_device).unsqueeze(0)
            with torch.inference_mode():
                _ = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                          use_cache=False, output_router_logits=False)
            if len(captures) != n_layers:
                raise RuntimeError(f"Captured {len(captures)} router layers; expected {n_layers}.")
            expected_shape = (args.sequence_length, top_k)
            rows = []
            token_hash = hashlib.sha256(token_ids.tobytes()).hexdigest()
            for layer_id in range(n_layers):
                cap = captures[layer_id]
                if cap["expert_ids"].shape != expected_shape or cap["scores"].shape != expected_shape:
                    raise RuntimeError(f"Bad capture shape in layer {layer_id}: {cap['expert_ids'].shape}")
                ids, scores, selected_logits = cap["expert_ids"], cap["scores"], cap["selected_logits"]
                if np.any((ids < 0) | (ids >= n_experts)):
                    raise RuntimeError(f"Out-of-range expert ID in layer {layer_id}")
                if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
                    raise RuntimeError(f"Duplicate expert selected within a token at layer {layer_id}")
                if not np.allclose(scores.sum(axis=1), 1.0, atol=2e-4):
                    raise RuntimeError(f"Selected routing weights do not sum to 1 in layer {layer_id}")
                part = pd.DataFrame({
                    "model_id": MODEL_ID, "model_revision": run_config["model_revision"] or "unresolved",
                    "dataset_id": DATASET_ID, "dataset_split": DATASET_SPLIT,
                    "sequence_id": chunk_id, "token_position": np.arange(args.sequence_length, dtype=np.int32),
                    "token_id": token_ids.astype(np.int32), "token_hash": token_hash,
                    "layer_id": layer_id, "phase": "prefill_teacher_forced",
                    **{f"expert_{j}": ids[:, j].astype(np.int16) for j in range(top_k)},
                    **{f"score_{j}": scores[:, j].astype(np.float32) for j in range(top_k)},
                    **{f"selected_logit_{j}": selected_logits[:, j].astype(np.float32) for j in range(top_k)},
                })
                rows.append(part)
            frame = pd.concat(rows, ignore_index=True)
            path = writer.write_chunk(chunk_id, frame)
            processed_this_run += 1
            print(f"Committed chunk={chunk_id} rows={len(frame)} file={path.name}")
    except KeyboardInterrupt:
        writer.finish("interrupted")
        print("Interrupted safely; rerun with the same output directory to resume.")
        return
    finally:
        for handle in handles:
            handle.remove()
    writer.finish("completed" if args.max_chunks == 0 or writer.next_chunk_id >= args.max_chunks else "partial")
    print(json.dumps(writer.manifest, indent=2))

if __name__ == "__main__":
    main()
