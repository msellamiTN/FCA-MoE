"""Atomic, resumable Parquet storage for real MoE routing traces."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any
import pandas as pd

SCHEMA_VERSION = "1.0"

def fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

class ParquetTraceWriter:
    """One immutable Parquet shard per sequence chunk; resume by contiguous shard IDs."""
    def __init__(self, output_dir: str | Path, run_config: dict[str, Any], resume: bool = True):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "parts"
        self.data_dir.mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.config = run_config
        self.config_hash = fingerprint(run_config)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_hash") != self.config_hash:
                raise ValueError("Existing output belongs to a different run configuration; choose a new output directory.")
            if not resume:
                raise FileExistsError("Output exists. Use a new output directory or resume=True.")
            self.manifest = manifest
        else:
            existing = list(self.data_dir.glob("chunk_*.parquet"))
            if existing:
                raise RuntimeError("Parquet shards exist but manifest is missing; refusing unsafe resume.")
            self.manifest = {
                "schema_version": SCHEMA_VERSION, "run_config": run_config,
                "config_hash": self.config_hash, "completed_chunks": 0,
                "rows_written": 0, "status": "running", "parts": []
            }
            atomic_json(self.manifest_path, self.manifest)
        self.next_chunk_id = self._discover_next_chunk()

    def _discover_next_chunk(self) -> int:
        ids = sorted(int(p.stem.split("_")[1]) for p in self.data_dir.glob("chunk_*.parquet"))
        expected = list(range(len(ids)))
        if ids != expected:
            raise RuntimeError(f"Non-contiguous checkpoint shards: {ids[:10]}")
        if len(ids) < int(self.manifest.get("completed_chunks", 0)):
            raise RuntimeError("Manifest claims more completed chunks than exist on disk.")
        if len(ids) > int(self.manifest.get("completed_chunks", 0)):
            # A shard may have been atomically committed immediately before a runtime interruption.
            for idx in range(int(self.manifest.get("completed_chunks", 0)), len(ids)):
                p = self.data_dir / f"chunk_{idx:08d}.parquet"
                df = pd.read_parquet(p)
                self.manifest["rows_written"] = int(self.manifest.get("rows_written", 0)) + len(df)
                self.manifest.setdefault("parts", []).append(p.name)
            self.manifest["completed_chunks"] = len(ids)
            atomic_json(self.manifest_path, self.manifest)
        return len(ids)

    def write_chunk(self, chunk_id: int, frame: pd.DataFrame) -> Path:
        if chunk_id != self.next_chunk_id:
            raise ValueError(f"Expected chunk {self.next_chunk_id}, got {chunk_id}")
        if frame.empty:
            raise ValueError("Refusing to write an empty trace shard")
        final = self.data_dir / f"chunk_{chunk_id:08d}.parquet"
        tmp = self.data_dir / f"chunk_{chunk_id:08d}.tmp.parquet"
        frame.to_parquet(tmp, index=False, compression="zstd")
        # Read-after-write validation before publishing the shard.
        check = pd.read_parquet(tmp)
        if len(check) != len(frame) or list(check.columns) != list(frame.columns):
            tmp.unlink(missing_ok=True)
            raise IOError("Parquet read-after-write validation failed")
        os.replace(tmp, final)
        self.manifest["completed_chunks"] = chunk_id + 1
        self.manifest["rows_written"] = int(self.manifest.get("rows_written", 0)) + len(frame)
        self.manifest.setdefault("parts", []).append(final.name)
        self.manifest["status"] = "running"
        atomic_json(self.manifest_path, self.manifest)
        self.next_chunk_id += 1
        return final

    def finish(self, status: str = "completed") -> None:
        self.manifest["status"] = status
        atomic_json(self.manifest_path, self.manifest)
