from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import gradio as gr
import pandas as pd

MAX_TRACE_BYTES = 100 * 1024 * 1024
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_RECORDS = 2_000_000
REQUIRED = {
    "run_id", "model_id", "prompt_hash", "session_id", "window_id",
    "token_position", "phase", "layer_id", "source_rank", "expert_ids",
    "routing_weights", "precision", "quantization"
}


def _file_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return Path(file_obj)
    return Path(file_obj.name)


def inspect_trace(trace_file, manifest_file):
    trace_path = _file_path(trace_file)
    manifest_path = _file_path(manifest_file)
    if trace_path is None:
        raise gr.Error("Upload a trace.jsonl file.")
    if not trace_path.exists() or trace_path.stat().st_size > MAX_TRACE_BYTES:
        raise gr.Error("Trace is missing or exceeds the 100 MiB inspection limit.")

    manifest = {}
    if manifest_path is not None:
        if not manifest_path.exists() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise gr.Error("Manifest is missing or exceeds the 5 MiB inspection limit.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise gr.Error(f"Could not parse manifest JSON: {exc}") from exc

    rows = 0
    sessions = set()
    families = Counter()
    layers = defaultdict(lambda: {"events": 0, "assignments": 0, "tokens": set(), "top_k": set()})
    expert_counts = defaultdict(Counter)
    phases = Counter()
    run_ids = set()
    model_ids = set()
    cardinality_errors = 0
    range_errors = 0

    try:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows += 1
                if rows > MAX_RECORDS:
                    raise gr.Error(f"Trace exceeds the {MAX_RECORDS:,}-record inspection limit.")
                try:
                    rec = json.loads(line)
                except Exception as exc:
                    raise gr.Error(f"Invalid JSON on line {line_no}: {exc}") from exc
                missing = REQUIRED - set(rec)
                if missing:
                    raise gr.Error(f"Line {line_no} misses fields: {sorted(missing)}")
                ids = rec.get("expert_ids")
                weights = rec.get("routing_weights")
                if not isinstance(ids, list) or not isinstance(weights, list) or len(ids) != len(weights):
                    cardinality_errors += 1
                    continue
                try:
                    ids = [int(x) for x in ids]
                    layer_id = int(rec["layer_id"])
                    token_position = int(rec["token_position"])
                except Exception:
                    range_errors += 1
                    continue
                if len(set(ids)) != len(ids) or any(x < 0 for x in ids):
                    range_errors += 1
                    continue

                rows_by_layer = layers[layer_id]
                rows_by_layer["events"] += 1
                rows_by_layer["assignments"] += len(ids)
                rows_by_layer["tokens"].add((rec["session_id"], token_position))
                rows_by_layer["top_k"].add(len(ids))
                for expert_id in ids:
                    expert_counts[layer_id][expert_id] += 1
                sessions.add(str(rec["session_id"]))
                families[str(rec.get("family", "unspecified"))] += 1
                phases[str(rec.get("phase", "unknown"))] += 1
                run_ids.add(str(rec["run_id"]))
                model_ids.add(str(rec["model_id"]))
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Trace read failed: {exc}") from exc

    if rows == 0:
        raise gr.Error("No trace records were found.")

    layer_rows = []
    expert_rows = []
    for layer_id in sorted(layers):
        stat = layers[layer_id]
        layer_rows.append({
            "layer": layer_id,
            "records": stat["events"],
            "distinct session-token pairs": len(stat["tokens"]),
            "routing assignments": stat["assignments"],
            "observed Top-K cardinalities": ", ".join(map(str, sorted(stat["top_k"]))),
        })
        total = max(stat["assignments"], 1)
        for expert_id, count in sorted(expert_counts[layer_id].items()):
            expert_rows.append({
                "layer": layer_id,
                "expert": expert_id,
                "assignments": count,
                "assignment share": count / total,
            })

    manifest_summary = {
        "run_ids": sorted(run_ids),
        "model_ids": sorted(model_ids),
        "records": rows,
        "sessions": len(sessions),
        "layers": len(layers),
        "families (record counts)": dict(families),
        "phases (record counts)": dict(phases),
        "cardinality errors": cardinality_errors,
        "expert index / type errors": range_errors,
        "manifest model revision": manifest.get("model_revision_resolved"),
        "capture scope": manifest.get("capture_scope"),
        "smoke test only": manifest.get("smoke_test_only"),
        "router index fidelity": manifest.get("router_fidelity", {}).get("all_indices_match"),
        "note": "This app validates and summarizes exported traces; it does not execute FCA/DPCK or measure distributed systems performance."
    }
    return (
        json.dumps(manifest_summary, indent=2),
        pd.DataFrame(layer_rows),
        pd.DataFrame(expert_rows),
    )


with gr.Blocks(title="DPCK–MoE Trace Inspector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # DPCK–MoE Trace Inspector
    Inspect a pre-exported OLMoE router trace and its manifest. The Space is an artifact viewer and schema sanity checker; it does **not** run the model, mine the concept lattice, choose placements, or report GPU/network performance.

    Upload a `trace.jsonl` file and its matching `manifest.json`. Files are processed for this request only; this app does not intentionally write uploads to persistent storage.
    """)
    with gr.Row():
        trace_input = gr.File(label="trace.jsonl", file_types=[".jsonl"], type="filepath")
        manifest_input = gr.File(label="manifest.json (optional)", file_types=[".json"], type="filepath")
    inspect_button = gr.Button("Validate and summarize", variant="primary")
    summary_output = gr.Code(label="Trace summary", language="json")
    layer_output = gr.Dataframe(label="Layer summary", interactive=False)
    expert_output = gr.Dataframe(label="Per-layer expert assignment counts", interactive=False)
    inspect_button.click(
        fn=inspect_trace,
        inputs=[trace_input, manifest_input],
        outputs=[summary_output, layer_output, expert_output],
    )
    gr.Markdown("**Research integrity:** a valid JSONL file is not proof that routing indices match the runtime. Use the router-fidelity result from the capture manifest and retain the exact model revision and raw artifact hash.")

if __name__ == "__main__":
    demo.launch()