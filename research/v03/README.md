# FCA–MoE v0.3 — Executable Research Scaffold

This branch contains FCA algorithms and real-model routing-trace capture pipelines for **OpenAI GPT-OSS-20B** and **Qwen1.5-MoE-A2.7B**.

## Approved real-model target

- Model: `openai/gpt-oss-20b`
- Architecture: 24 MoE layers, 32 experts, Top-4 per token
- Empirical source: actual router outputs from the pretrained model
- Initial corpus: `Salesforce/wikitext`, `wikitext-2-raw-v1`, test split
- Capture phase: teacher-forced prefill (not autoregressive decode)
- Storage: one atomic Parquet shard per 256-token sequence, with JSON manifest and resumable checkpoints
- Instrumentation: forward hooks on each `GptOssTopKRouter`, capturing selected expert IDs, selected routing weights and selected router logits

## Critical hardware constraint — T4 memory and offload

GPT-OSS-20B expert weights are distributed in MXFP4. Current Hugging Face Transformers MXFP4 documentation specifies NVIDIA compute capability 7.5 or newer. A Colab T4 is SM75 and therefore **meets the documented minimum**. This does not guarantee that the full model fits in T4 VRAM: the pipeline caps GPU memory, uses CPU and local-disk offload, and validates actual model loading/inference. The earlier OpenAI deployment cookbook described Hopper+ for its optimized path; current Hugging Face docs clarify the broader MXFP4 compatibility threshold. Treat T4 execution as a pilot to be measured, not guaranteed.

The extraction script checks compute capability and available host RAM **before downloading model weights**. A T4 passes the documented SM75 minimum; whether the model loads and completes inference with the configured GPU/CPU/disk offload is an empirical pilot gate. Do not claim real traces until the Parquet shards pass validation.

References:
- [OpenAI GPT-OSS model card](https://deploymentsafety.openai.com/gpt-oss)
- [OpenAI guide: run gpt-oss with Transformers](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers)
- [Transformers GPT-OSS documentation](https://huggingface.co/docs/transformers/en/model_doc/gpt_oss)

## Additional real-model target: Qwen1.5-MoE-A2.7B

- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Architecture: 24 MoE layers, 60 routed experts, Top-4 routed experts per token, plus four logical shared experts.
- Important implementation detail: the Transformers Qwen2-MoE block exposes a merged `shared_expert` MLP and a scalar `shared_expert_gate`; the extractor records the actual scalar gate logit and sigmoid weight. It does not claim to capture four separately instantiated shared modules.
- Checkpoint dtype: BF16. The T4 notebook explicitly converts/loads weights as FP16 because T4 is not a native BF16 compute target. The traces are therefore labelled FP16-runtime traces and are not claimed to be bit-identical to BF16 execution.
- The checkpoint contains about 14.3B total parameters, although about 2.7B are active per token. Active parameter count is not the storage footprint; CPU and disk offload are required for the T4 pilot.
- Router scores are raw selected softmax probabilities because `norm_topk_prob=false`; they generally do **not** sum to one. The extractor verifies selected weights against softmax(router logits), rather than applying GPT-OSS's normalized-score check.
- Capture phase: teacher-forced prefill only.

Files:
- `scripts/extract_qwen15_moe_routes.py`
- `scripts/validate_qwen15_moe_traces.py`
- `configs/qwen15_moe_a27_colab.yaml`
- `notebooks/Qwen1_5_MoE_A2_7B_Real_Routing_Extraction.ipynb`

Qwen references:
- [Official Hugging Face model configuration](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json)
- [Qwen1.5-MoE architecture description](https://qwenlm.github.io/blog/qwen-moe/)
- [Transformers Qwen2-MoE implementation documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen2_moe)

## Colab notebooks

1. [Real GPT-OSS-20B routing extraction](notebooks/GPT_OSS_20B_Real_Routing_Extraction.ipynb) — hardware preflight, Drive mount, model instrumentation, 4-chunk pilot and resumable Parquet output.
2. [Trace validation and audit](notebooks/GPT_OSS_20B_Trace_Validation.ipynb) — validates persisted shards without reloading the model.
3. [Qwen1.5-MoE-A2.7B real routing extraction](notebooks/Qwen1_5_MoE_A2_7B_Real_Routing_Extraction.ipynb) — 60 routed experts, Top-4, shared-gate capture, FP16 T4 pilot and Parquet validation.
4. [FCA-MoE v0.3 scaffold](notebooks/FCA_MoE_v03_scaffold.ipynb) — FCA and small-E exact MaxEnt software checks.

Open directly in Colab:
- [Extraction notebook](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/research/fca-moe-v0.3/research/v03/notebooks/GPT_OSS_20B_Real_Routing_Extraction.ipynb)
- [Validation notebook](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/research/fca-moe-v0.3/research/v03/notebooks/GPT_OSS_20B_Trace_Validation.ipynb)
- [Qwen1.5-MoE extraction notebook](https://colab.research.google.com/github/msellamiTN/FCA-MoE/blob/research/fca-moe-v0.3/research/v03/notebooks/Qwen1_5_MoE_A2_7B_Real_Routing_Extraction.ipynb)

## Local installation and tests

From `research/v03`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
python scripts/run_v03_smoke.py --tokens 4096
```

For the model runtime, install a compatible PyTorch/CUDA stack plus Transformers, Accelerate, Datasets, Triton and the Hugging Face `kernels` package as described in the upstream deployment guide. The Colab notebook installs these runtime packages without replacing Colab's preinstalled PyTorch.

## CLI: capture and resume

On a supported SM90+ runtime:

```bash
python scripts/extract_gpt_oss_routes.py \\
  --output-dir /content/drive/MyDrive/FCA_MoE_v03/gpt_oss_20b_wikitext_test \\
  --sequence-length 256 --max-chunks 4 \\
  --gpu-memory 12GiB --cpu-memory 8GiB \\
  --offload-folder /content/hf_offload
```

Rerun with the same arguments and output directory after interruption. The writer resumes from the next contiguous committed shard. Increase `--max-chunks` for a larger run; use `0` to stream the complete dataset split.

Validate:

```bash
python scripts/validate_gpt_oss_traces.py --output-dir /path/to/gpt_oss_20b_wikitext_test
```

## Trace schema

Each row represents one (sequence, token, MoE layer) observation and includes model/revision, corpus/split, sequence ID, token position and token ID, layer ID, four selected expert IDs, four selected routing weights, and four selected router logits. The full 32-way logits are not persisted to keep storage manageable. The manifest records the architecture, instrumentation method, sequence length, completed shards, and run fingerprint.

## Scientific guardrails

- Synthetic routes are software tests only; they are never mixed into empirical traces.
- Router decisions are captured from the original pretrained model. No router is replaced or modified.
- This first extractor measures teacher-forced prefill. Decode-phase traces require a separate implementation and label.
- Routing co-activation does not establish KV-cache reuse or speedup.
- The scalable E=32 pairwise-MaxEnt sampler, 999-surrogate protocol, bounded incremental FCA, and GPU KV-cache systems benchmark remain separate pending milestones.
