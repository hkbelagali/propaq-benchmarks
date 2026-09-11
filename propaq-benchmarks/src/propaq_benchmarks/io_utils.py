"""
Shared JSONL checkpoint and npz (de)serialization helpers used across benchmarks
"""
from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_JSON_KEYS_COLUMN = "__json_keys__"


def save_records_npz(records: list[dict[str, Any]], path: str) -> None:
    keys = sorted({k for r in records for k in r})
    arrays: dict[str, np.ndarray] = {}
    json_keys: list[str] = []
    for key in keys:
        values = [r.get(key) for r in records]
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, bool) for v in present):
            arrays[key] = np.array([bool(v) for v in values], dtype=bool)
        elif present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
            arrays[key] = np.array([float(v) if v is not None else np.nan for v in values], dtype=np.float64)
        elif present and all(isinstance(v, str) for v in present):
            arrays[key] = np.array([v if v is not None else "" for v in values], dtype=object)
        else:
            json_keys.append(key)
            arrays[key] = np.array([json.dumps(v) for v in values], dtype=object)
    arrays[_JSON_KEYS_COLUMN] = np.array(json_keys, dtype=object)
    np.savez_compressed(path, **arrays)


def load_records_npz(path: str) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    json_keys = set(data[_JSON_KEYS_COLUMN].tolist()) if _JSON_KEYS_COLUMN in data.files else set()
    keys = [k for k in data.files if k != _JSON_KEYS_COLUMN]
    n = len(data[keys[0]]) if keys else 0
    records: list[dict[str, Any]] = []
    for i in range(n):
        rec: dict[str, Any] = {}
        for key in keys:
            arr = data[key]
            v = arr[i]
            if key in json_keys:
                rec[key] = json.loads(str(v))
            elif arr.dtype == bool:
                rec[key] = bool(v)
            elif np.issubdtype(arr.dtype, np.floating):
                rec[key] = None if np.isnan(v) else float(v)
            else:
                rec[key] = str(v)
        records.append(rec)
    return records


def load_experiment_results(experiment_dir: str | Path) -> list[dict[str, Any]]:
    experiment_dir = Path(experiment_dir)
    records: list[dict[str, Any]] = []
    seen_backends: set[str] = set()
    for npz_path in sorted(glob(str(experiment_dir / "results_*.npz"))):
        backend = Path(npz_path).stem.removeprefix("results_")
        seen_backends.add(backend)
        records.extend(load_records_npz(npz_path))
    for jsonl_path in sorted(glob(str(experiment_dir / "results_*.jsonl"))):
        backend = Path(jsonl_path).stem.removeprefix("results_")
        if backend in seen_backends:
            continue
        records.extend(load_jsonl(jsonl_path))
    return records
