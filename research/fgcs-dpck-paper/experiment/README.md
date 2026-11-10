# DPCK–MoE experiment scaffold

## 1. Implementation status

| Component | Status |
|---|---|
| OLMoE prefill router capture | Notebook/script prepared; not run on Colab here |
| Hooked router-index vs returned router-logit Top-K check | Fail-closed code prepared; runtime confirmation pending |
| Trace JSONL, run manifest and SHA-256 | Implemented in capture script; no real-run artifacts yet |
| Top-3 parity counterexample | Executable exhaustive test; locally verified |
| Hugging Face Space trace inspector | Gradio app template prepared; not deployed |
| Full FCA miner, DPCK selection and equal-byte baselines | Specified in manuscript; end-to-end implementation still required |
| Physical expert-parallel Stage B | Not performed; requires a separate multi-GPU runtime |

Prepared scripts are not empirical validation.

## 2. Colab T4 trace capture

1. Open `colab/DPCK_T4_Trace_Capture.ipynb` in Google Colab.
2. Select a Tesla T4 runtime. The notebook checks the allocated device; Colab availability is dynamic.
3. Upload `trace_capture.py` and a controlled prompt CSV with `family,prompt,session_id` columns. The bundled `prompts_smoke.csv` is for instrumentation smoke testing only.
4. Run the install, load, capture and export cells. The default uses int8 weight loading to reduce memory pressure. Quantization can alter hidden states and routing, so precision is recorded and a separate sensitivity study is required.
5. Pin the resolved model commit SHA and exact package versions before confirmatory runs.
6. Inspect `manifest.json`, `router_fidelity.json` and `trace.jsonl`. The script does not export prompt bodies; it stores prompt hashes and session IDs. Keep session identifiers pseudonymous.
7. Retain the ZIP immutably and record its SHA-256.

The capture is prefill-only. `source_rank=-1` means unobserved on single-GPU execution and must not be interpreted as a physically measured expert-parallel rank.

## 3. Router instrumentation contract

The script targets the OLMoE module layout `model.model.layers[i].mlp.gate`. It validates captured indices against Top-K indices reconstructed from the model's returned router logits. The script aborts when the two disagree. Re-check this contract against the pinned Transformers/model revision before a final run because model internals may change.

## 4. Hugging Face Space

The Space is a trace viewer and schema validator, not a model host, FCA miner or placement optimizer. It is designed to run CPU-first.

1. Create a new Gradio Space in the desired Hugging Face account.
2. Copy `space/app.py`, `space/requirements.txt` and `space/README.md` into the Space repository.
3. Keep the Space private while handling non-public traces. Do not upload raw prompts or private inference traces to a public Space.
4. Upload a trace and matching manifest to validate schema, model configuration, hashes, layer counts and expert-assignment summaries.
5. Treat uploaded data as temporary; do not use the Space as durable experiment storage.

No HF token or model download is required by this viewer. Verify current Space hardware and pricing before enabling any paid resources.

## 5. Parity example test

Run:

```bash
python tests/test_parity.py
```

Expected output:

```text
PASS: 60 placements; pairwise signatures equal; 12 disjoint optima each; cross-cost 1.50.
```

This verifies a finite constructed example. It is not an MoE workload experiment.

## 6. Next implementation milestone

Before confirmatory Stage A, implement a deterministic offline pipeline that:

1. Validates and aggregates the real trace into session/window/layer records.
2. Constructs layer-wise FCA contexts from training sessions only.
3. Records concept-miner completeness and candidate truncation.
4. Freezes the same feasible placement bank for every baseline.
5. Serializes each deployable summary and predictor to measure actual bytes.
6. Fits predictors on training sessions and selects DPCK using validation episodes only.
7. Evaluates held-out placement regret on locked test sessions, including build, selection and decision overhead.

The best full-route-histogram result in a finite candidate bank is a reference, not necessarily a global optimum.
