# DPCK–MoE: FGCS manuscript and experiment blueprint

This directory contains the pre-experiment manuscript and reproducibility scaffold for *Decision-Aware FCA Summaries for Workload-Shift-Robust Mixture-of-Experts Placement*.

## Package map

- `paper/main.tex` — Elsevier `elsarticle` review manuscript with TikZ figures, propositions and proofs, an algorithm, methodology, threats to validity and pending-result tables.
- `paper/references.bib` — editable bibliographic records. The current manuscript includes a manually maintained bibliography and does not require BibTeX to compile.
- `experiment/colab/DPCK_T4_Trace_Capture.ipynb` — Colab notebook for OLMoE prefill router-trace capture and router-index fidelity checks.
- `experiment/colab/trace_capture.py` — standalone trace-capture implementation.
- `experiment/colab/prompts_smoke.csv` — four-prompt instrumentation smoke test only; not confirmatory data.
- `experiment/colab/prompts_template.csv` — prompt-manifest template.
- `experiment/tests/test_parity.py` — exhaustive 60-placement test for the Top-3 parity counterexample.
- `experiment/space/` — Gradio trace-inspection application template for a Hugging Face Space.
- `experiment/README.md` — operational steps, data contract and deployment notes.

## Scientific evidence boundary

The manuscript distinguishes (1) formal results proved under stated assumptions, (2) the finite parity example that was exhaustively enumerated, and (3) empirical claims that remain untested. Pending result cells must not be populated with illustrative or simulated values.

The Colab T4 stage is intended to validate real-model router instrumentation and conduct offline decision analysis under a declared *simulated* topology. It cannot measure physical all-to-all communication, network contention or real multi-GPU latency. A separate expert-parallel cluster experiment is required before making those systems claims.

The current notebook covers trace capture and fidelity validation. The complete FCA candidate miner, equal-byte baseline implementations, DPCK selector and cluster runtime integration remain to be implemented and executed. The paper specifies the intended method and acceptance criteria; it does not claim those experiments have already been performed.

## Compile the manuscript

Requires a TeX installation with `elsarticle`, TikZ, `algorithm`, `algpseudocode`, `tabularx` and the other packages declared in `main.tex`.

From `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Two passes resolve cross-references. The current build uses the embedded manual bibliography; `references.bib` is provided as an editable source record for later BibTeX integration.

## Required submission gates

1. Run the Colab notebook on an actual Tesla T4 and save the model commit SHA, environment manifest, router-fidelity report and trace hash.
2. Validate the trace schema and independent-session splits. The four-row smoke CSV is not confirmatory data.
3. Implement and execute the FCA candidate miner, byte-accurate artifact serializer, DPCK selector and all baselines under identical action sets and solver budgets.
4. Run held-out and shifted-workload experiments with session-level statistical analysis.
5. Complete physical multi-GPU Stage B for any FGCS systems-performance claim.
6. Populate result tables only from retained raw logs; resolve author declarations and independently verify references and the current FGCS Guide for Authors.

## Current status

This branch contains a research draft and experiment scaffold, not an empirical submission-ready paper. No real OLMoE trace has yet been captured on Colab T4, the Hugging Face Space has not been deployed, and no physical multi-GPU experiment has been performed.
