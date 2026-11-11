from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V03_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V03_ROOT))

from moe_fca.trace_io import ParquetTraceWriter
from scripts.validate_qwen15_moe_traces import validate


def _valid_run_config():
    return {
        "model_id": "Qwen/Qwen1.5-MoE-A2.7B",
        "model_revision": "unit-test-revision",
        "model_type": "qwen2_moe",
        "runtime_dtype": "float16",
        "checkpoint_declared_dtype": "bfloat16",
        "dataset_id": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_split": "test",
        "dataset_revision": "unit-test-dataset-revision",
        "sequence_length": 16,
        "n_experts": 60,
        "top_k": 4,
        "n_layers": 24,
        "norm_topk_prob": False,
        "logical_shared_experts": 4,
        "shared_expert_implementation": "merged_shared_expert_MLP_with_scalar_sigmoid_gate",
        "phase": "prefill_teacher_forced",
        "instrumentation": "unit_test_fixture",
    }


def _valid_frame(sequence_id=0):
    rows = []
    for layer_id in range(24):
        for pos in range(16):
            rows.append({
                "model_id": "Qwen/Qwen1.5-MoE-A2.7B",
                "model_revision": "unit-test-revision",
                "runtime_dtype": "float16",
                "dataset_id": "Salesforce/wikitext",
                "dataset_split": "test",
                "sequence_id": sequence_id,
                "token_position": pos,
                "token_id": 1000 + pos,
                "token_hash": "unit-test-token-hash",
                "layer_id": layer_id,
                "phase": "prefill_teacher_forced",
                "expert_0": 0,
                "expert_1": 1,
                "expert_2": 2,
                "expert_3": 3,
                "score_0": 0.40,
                "score_1": 0.30,
                "score_2": 0.20,
                "score_3": 0.05,
                "selected_logit_0": 4.0,
                "selected_logit_1": 3.0,
                "selected_logit_2": 2.0,
                "selected_logit_3": 1.0,
                "shared_gate_logit": 0.0,
                "shared_gate_weight": 0.5,
            })
    return pd.DataFrame(rows)


def test_qwen_trace_validator_accepts_complete_fixture(tmp_path):
    writer = ParquetTraceWriter(tmp_path, _valid_run_config(), resume=False)
    writer.write_chunk(0, _valid_frame())
    writer.finish("completed")

    report = validate(tmp_path)

    assert report["status"] == "passed"
    assert report["routed_experts"] == 60
    assert report["top_k"] == 4
    assert report["layers"] == 24
    assert report["layer_token_rows"] == 24 * 16


def test_qwen_trace_validator_rejects_duplicate_token_rows(tmp_path):
    writer = ParquetTraceWriter(tmp_path, _valid_run_config(), resume=False)
    frame = _valid_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    writer.write_chunk(0, frame)
    writer.finish("completed")

    try:
        validate(tmp_path)
    except AssertionError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Validator accepted a duplicate token/layer row")
