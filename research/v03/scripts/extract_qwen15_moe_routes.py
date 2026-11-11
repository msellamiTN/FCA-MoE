#!/usr/bin/env python3
"""Capture actual Qwen1.5-MoE-A2.7B router outputs with resumable Parquet shards.

The routed Top-4 decisions are captured from each Qwen2MoeTopKRouter forward
output. The shared expert gate is captured separately. This script records
teacher-forced prefill only; it does not generate decode traces.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe_fca.trace_io import ParquetTraceWriter

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"

def hardware_preflight(min_cpu_gib: float = 6.0) -> dict:
    import torch, psutil
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_ram_available_gib": round(psutil.virtual_memory().available / 2**30, 2),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this Colab pilot.")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    info.update({
        "gpu_name": name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "gpu_vram_free_gib": round(free / 2**30, 2),
        "gpu_vram_total_gib": round(total / 2**30, 2),
    })
    if capability < (7, 5):
        raise RuntimeError(f"GPU {name} has compute capability {capability}; this pilot requires SM75 or newer.")
    if info["cpu_ram_available_gib"] < min_cpu_gib:
        raise RuntimeError(
            f"Only {info['cpu_ram_available_gib']} GiB host RAM is available; "
            f"at least {min_cpu_gib} GiB is required by this conservative preflight."
        )
    return info

def token_chunks(dataset, tokenizer, sequence_length: int):
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
    ap.add_argument("--max-chunks", type=int, default=4,
                    help="maximum total committed chunks; 0 means the full streaming split")
    ap.add_argument("--gpu-memory", default="12GiB")
    ap.add_argument("--cpu-memory", default="6GiB")
    ap.add_argument("--offload-folder", default="/content/qwen_hf_offload")
    ap.add_argument("--revision", default=None, help="Optional immutable Hugging Face model commit SHA")
    ap.add_argument("--dataset-revision", default="main",
                    help="Dataset revision; pin a commit SHA for strict reproducibility")
    ap.add_argument("--min-cpu-ram-gib", type=float, default=6.0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.sequence_length < 16:
        ap.error("--sequence-length must be >= 16")
    hw = hardware_preflight(args.min_cpu_ram_gib)
    print("Hardware preflight:", json.dumps(hw, indent=2), flush=True)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    config = AutoConfig.from_pretrained(MODEL_ID, revision=args.revision)
    expected = {
        "model_type": "qwen2_moe",
        "num_hidden_layers": 24,
        "num_experts": 60,
        "num_experts_per_tok": 4,
    }
    actual = {k: getattr(config, k, None) for k in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected Qwen architecture config: {actual}; expected {expected}")
    if bool(getattr(config, "norm_topk_prob", False)):
        raise RuntimeError("This trace schema expects the checkpoint's norm_topk_prob=False behavior.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    Path(args.offload_folder).mkdir(parents=True, exist_ok=True)

    # T4 is not a native BF16 compute target. Explicit FP16 conversion is
    # recorded in the manifest; routes therefore describe this FP16 runtime.
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "max_memory": {0: args.gpu_memory, "cpu": args.cpu_memory},
        "offload_folder": args.offload_folder,
        "offload_state_dict": True,
        "low_cpu_mem_usage": True,
        "revision": args.revision,
    }
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    cfg = model.config
    n_experts = int(cfg.num_experts)
    top_k = int(cfg.num_experts_per_tok)
    n_layers = int(cfg.num_hidden_layers)
    resolved_revision = args.revision or getattr(cfg, "_commit_hash", None) or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if not resolved_revision:
        raise RuntimeError("Could not resolve immutable model revision; pass --revision with a commit SHA.")

    run_config = {
        "model_id": MODEL_ID,
        "model_revision": resolved_revision,
        "model_type": cfg.model_type,
        "runtime_dtype": "float16",
        "checkpoint_declared_dtype": str(getattr(cfg, "torch_dtype", "unknown")),
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": args.dataset_revision,
        "sequence_length": args.sequence_length,
        "n_experts": n_experts,
        "top_k": top_k,
        "n_layers": n_layers,
        "norm_topk_prob": bool(getattr(cfg, "norm_topk_prob", False)),
        "logical_shared_experts": 4,
        "shared_expert_implementation": "merged_shared_expert_MLP_with_scalar_sigmoid_gate",
        "phase": "prefill_teacher_forced",
        "instrumentation": "forward_hooks_on_Qwen2MoeTopKRouter_and_shared_expert_gate",
    }

    # Inspect and attach hooks only to actual sparse MoE layers.
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != n_layers:
        raise RuntimeError("Unexpected Qwen model layout: model.model.layers not found or wrong layer count.")
    captures: dict[int, dict[str, np.ndarray]] = {}
    handles = []

    def make_router_hook(layer_id: int):
        def hook(module, inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise RuntimeError(f"Router output contract changed at layer {layer_id}.")
            logits, scores, indices = output[:3]
            if logits.ndim != 2 or logits.shape[-1] != n_experts:
                raise RuntimeError(f"Unexpected router logits shape at layer {layer_id}: {tuple(logits.shape)}")
            captures.setdefault(layer_id, {})
            captures[layer_id]["expert_ids"] = indices.detach().to("cpu", dtype=torch.int16).numpy()
            captures[layer_id]["scores"] = scores.detach().to("cpu", dtype=torch.float32).numpy()
            captures[layer_id]["logits"] = logits.detach().to("cpu", dtype=torch.float32).numpy()
        return hook

    def make_shared_gate_hook(layer_id: int):
        def hook(module, inputs, output):
            captures.setdefault(layer_id, {})
            gate_logits = output.detach().to("cpu", dtype=torch.float32).reshape(-1).numpy()
            captures[layer_id]["shared_gate_logits"] = gate_logits
            captures[layer_id]["shared_gate_weights"] = (1.0 / (1.0 + np.exp(-gate_logits))).astype(np.float32)
        return hook

    for layer_id, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "gate", None)
        shared_gate = getattr(mlp, "shared_expert_gate", None)
        shared_expert = getattr(mlp, "shared_expert", None)
        if router is None or shared_gate is None or shared_expert is None:
            raise RuntimeError(
                f"Layer {layer_id} does not expose expected Qwen MoE modules "
                "(mlp.gate, mlp.shared_expert_gate, mlp.shared_expert)."
            )
        handles.append(router.register_forward_hook(make_router_hook(layer_id)))
        handles.append(shared_gate.register_forward_hook(make_shared_gate_hook(layer_id)))

    writer = None
    try:
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT,
                               revision=args.dataset_revision, streaming=True)
        writer = ParquetTraceWriter(args.output_dir, run_config, resume=not args.no_resume)
        start_chunk = writer.next_chunk_id
        print(f"Resume point: chunk {start_chunk}; output={args.output_dir}", flush=True)
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
                _ = model(input_ids=input_ids,
                          attention_mask=torch.ones_like(input_ids),
                          use_cache=False,
                          output_router_logits=False)
            if len(captures) != n_layers:
                raise RuntimeError(f"Captured {len(captures)} MoE layers; expected {n_layers}.")
            rows = []
            token_hash = hashlib.sha256(token_ids.tobytes()).hexdigest()
            for layer_id in range(n_layers):
                cap = captures[layer_id]
                ids = cap["expert_ids"]
                scores = cap["scores"]
                logits = cap["logits"]
                shared_logits = cap["shared_gate_logits"]
                shared_weights = cap["shared_gate_weights"]
                expected_shape = (args.sequence_length, top_k)
                if ids.shape != expected_shape or scores.shape != expected_shape:
                    raise RuntimeError(f"Bad route shape at layer {layer_id}: ids={ids.shape}, scores={scores.shape}")
                if logits.shape != (args.sequence_length, n_experts):
                    raise RuntimeError(f"Bad router-logit shape at layer {layer_id}: {logits.shape}")
                if shared_logits.shape != (args.sequence_length,) or shared_weights.shape != (args.sequence_length,):
                    raise RuntimeError(f"Bad shared-gate shape at layer {layer_id}.")
                if np.any((ids < 0) | (ids >= n_experts)):
                    raise RuntimeError(f"Out-of-range expert ID at layer {layer_id}.")
                if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
                    raise RuntimeError(f"Duplicate routed experts at layer {layer_id}.")
                # Qwen1.5 config has norm_topk_prob=False: selected weights are
                # raw softmax probabilities and generally do NOT sum to one.
                probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                expected_scores = np.take_along_axis(probs, ids.astype(np.int64), axis=1)
                if not np.allclose(scores, expected_scores, rtol=3e-3, atol=3e-5):
                    raise RuntimeError(f"Captured routing scores disagree with softmax(logits) at layer {layer_id}.")
                expected_ids = np.argsort(-probs, axis=1)[:, :top_k]
                # Compare selected sets; top-k order is score-descending.
                if not np.array_equal(ids.astype(np.int64), expected_ids):
                    raise RuntimeError(f"Captured expert indices disagree with recomputed Top-K at layer {layer_id}.")
                if not np.isfinite(shared_weights).all() or np.any((shared_weights <= 0) | (shared_weights >= 1)):
                    raise RuntimeError(f"Invalid shared expert sigmoid gate at layer {layer_id}.")
                part = pd.DataFrame({
                    "model_id": MODEL_ID,
                    "model_revision": resolved_revision,
                    "runtime_dtype": "float16",
                    "dataset_id": DATASET_ID,
                    "dataset_split": DATASET_SPLIT,
                    "sequence_id": chunk_id,
                    "token_position": np.arange(args.sequence_length, dtype=np.int32),
                    "token_id": token_ids.astype(np.int32),
                    "token_hash": token_hash,
                    "layer_id": layer_id,
                    "phase": "prefill_teacher_forced",
                    **{f"expert_{j}": ids[:, j].astype(np.int16) for j in range(top_k)},
                    **{f"score_{j}": scores[:, j].astype(np.float32) for j in range(top_k)},
                    **{f"selected_logit_{j}": np.take_along_axis(logits, ids[:, j:j+1].astype(np.int64), axis=1).reshape(-1).astype(np.float32) for j in range(top_k)},
                    "shared_gate_logit": shared_logits.astype(np.float32),
                    "shared_gate_weight": shared_weights.astype(np.float32),
                })
                rows.append(part)
            frame = pd.concat(rows, ignore_index=True)
            path = writer.write_chunk(chunk_id, frame)
            print(f"Committed chunk={chunk_id} rows={len(frame)} file={path.name}", flush=True)
        writer.finish("completed" if args.max_chunks == 0 or writer.next_chunk_id >= args.max_chunks else "partial")
        print(json.dumps(writer.manifest, indent=2), flush=True)
    except KeyboardInterrupt:
        if writer is not None:
            writer.finish("interrupted")
        print("Interrupted safely; rerun with the same output directory/configuration to resume.", flush=True)
    finally:
        for handle in handles:
            handle.remove()

if __name__ == "__main__":
    main()
