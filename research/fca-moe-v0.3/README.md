# FCA-MoE v0.3 — Colab T4 package

This branch publishes the modular v0.3 experiment package as an XZ archive encoded in ordered Base64 text segments. This avoids GitHub's raw-blob upload-size constraints while keeping the full source reconstructable.

## Reconstruct the package

Download all eight `fca_moe_v03.tar.xz.b64.part00` ... `part07` files from this directory into the same folder, then run:

```bash
cat fca_moe_v03.tar.xz.b64.part* | base64 -d > fca_moe_v03.tar.xz
tar -xJf fca_moe_v03.tar.xz
cd fca_moe_v03
pip install -r requirements.txt
pytest -q
```

The parts must be concatenated in lexical order (part00 through part07).

## Package scope

- Modular FCA-MoE v0.3 Python package, scripts, tests and Colab notebook
- 32 routed experts; native Top-16 routing retained
- Fixed-cardinality pairwise MaxEnt null on the exact Top-K support
- Router trace capture, FCA contexts/lattice analysis, null diagnostics, checkpointing and statistical evaluation
- BH and BY multiple-testing corrections (BY is the conservative result under dependent concept tests)
- Colab Tesla T4 runtime and Google Drive checkpoint configuration

## Validation status

**Local validation:** 8 unit tests passed. Notebook Python cells passed static syntax checking after removing Colab magic and shell lines.

**Real T4 benchmark:** the real-model capture and complete analysis on an actual Google Colab Tesla T4 have **not yet been executed**. This branch publishes runnable implementation and protocol, not completed real-model T4 measurements. Synthetic controls are calibration tests only and must not be presented as evidence of real-model discovery.

## Real-model target and scope limitation

The configured target is `bayliner1980/Qwen3-0.6B-MoE-A0.5B`, whose published config specifies 32 local experts and Top-16. This is a converted/distilled research checkpoint derived from dense Qwen3-0.6B, not an independently pretrained large-scale production MoE model. Results from it can validate the trace-to-analysis methodology on real model-generated routes; they cannot alone establish generality across production-scale MoE families.

See `THEORETICAL_AND_EXPERIMENTAL_PROTOCOL.md` inside the reconstructed archive for assumptions, mathematical definitions, controls, fit diagnostics, acceptance criteria and the experimental sequence.
