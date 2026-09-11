"""
Shared helper that every experiment's Python backend runner file calls.
"""
from __future__ import annotations

import re
import resource
import time
from pathlib import Path
from typing import Any, Callable

import json

from . import io_utils
from .circuit_ir import ProblemIR


def _natural_sort_key(path: Path) -> list[Any]:
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", path.stem)]


def _sorted_circuit_paths(circuits_dir: Path) -> list[Path]:
    return sorted(circuits_dir.glob("*.json"), key=_natural_sort_key)


def load_circuits(circuits_dir: str | Path) -> list[tuple[str, ProblemIR]]:
    """Load every saved circuit in circuits_dir, in filename order, as (label, ir) pairs."""
    circuits_dir = Path(circuits_dir)
    return [(path.stem, ProblemIR.load(str(path))) for path in _sorted_circuit_paths(circuits_dir)]


def load_fermionic_circuits(circuits_dir: str | Path) -> list[tuple[str, dict[str, Any]]]:
    circuits_dir = Path(circuits_dir)
    out = []
    for path in _sorted_circuit_paths(circuits_dir):
        with open(path) as f:
            out.append((path.stem, json.load(f)))
    return out


def run_on_saved_fermionic_circuits(
    circuits_dir: str | Path,
    experiment_dir: str | Path,
    backend: str,
    propagate: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    experiment_dir = Path(experiment_dir)
    checkpoint = experiment_dir / f"results_{backend}.jsonl"
    npz_path = experiment_dir / f"results_{backend}.npz"
    done = {r["label"] for r in io_utils.load_jsonl(str(checkpoint)) if r.get("ok")}

    circuits = load_fermionic_circuits(circuits_dir)
    if not circuits:
        raise SystemExit(f"no saved circuits in {circuits_dir}, run generate_circuits.py first")

    for label, source in circuits:
        if label in done:
            print(f"  {label} ... skip (already done)")
            continue
        t0 = time.perf_counter()
        try:
            record = propagate(source)
            record.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            record = {"ok": False, "error": str(exc)}
        record.setdefault("wall_time_s", time.perf_counter() - t0)
        record["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        record["label"] = label
        record["backend"] = backend
        record["basis"] = "majorana"
        record.setdefault("problem", source["problem"])
        record.setdefault("params", source["params"])
        status = "OK" if record["ok"] else "FAIL"
        print(f"  {label} ... {status} ({record['wall_time_s']:.3f}s)")
        if not record["ok"]:
            print(f"    {record.get('error')}")
        io_utils.append_jsonl(str(checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))


def run_on_saved_circuits(
    circuits_dir: str | Path,
    experiment_dir: str | Path,
    backend: str,
    basis: str,
    propagate: Callable[[ProblemIR], dict[str, Any]],
) -> None:
    experiment_dir = Path(experiment_dir)
    checkpoint = experiment_dir / f"results_{backend}.jsonl"
    npz_path = experiment_dir / f"results_{backend}.npz"
    done = {r["label"] for r in io_utils.load_jsonl(str(checkpoint))
            if r.get("ok") and r.get("basis") == basis}

    circuits = load_circuits(circuits_dir)
    if not circuits:
        raise SystemExit(f"no saved circuits in {circuits_dir}, run generate_circuits.py first")

    for label, ir in circuits:
        if label in done:
            print(f"  {label} ... skip (already done)")
            continue
        t0 = time.perf_counter()
        try:
            record = propagate(ir)
            record.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            record = {"ok": False, "error": str(exc)}
        record.setdefault("wall_time_s", time.perf_counter() - t0)
        record["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        record["label"] = label
        record["backend"] = backend
        record["basis"] = basis
        record.setdefault("problem", ir.problem)
        record.setdefault("n_qubits", ir.n_qubits)
        record.setdefault("gate_count", ir.gate_count())
        record.setdefault("params", ir.params)
        status = "OK" if record["ok"] else "FAIL"
        print(f"  {label} ... {status} ({record['wall_time_s']:.3f}s)")
        if not record["ok"]:
            print(f"    {record.get('error')}")
        io_utils.append_jsonl(str(checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))
