"""Shared helper that every experiment's Python backend runner file calls.

Each experiment lives in its own folder under experiments/. A generate_circuits.py script
in that folder builds and saves every circuit instance once, as plain JSON, into a
circuits/ subfolder. Each run_<backend>.py file in that same folder then calls
run_on_saved_circuits once per basis it exercises, which loads every saved circuit, calls
the caller's propagate function on each, and checkpoints the result to
results_<backend>.jsonl and results_<backend>.npz right there in the experiment folder.

There is no cross-language orchestrator process spawning these as subprocesses anymore.
Running python3 run_propaq.py (or julia run_pauli_propagation_jl.jl) directly is the whole
interface. Wall time and peak RSS are measured in-process (time.perf_counter and
resource.getrusage), around the whole run, not the isolated per-subprocess measurement the
old orchestrate.py got from wrapping every call in /usr/bin/time -v. Peak RSS is therefore
a cumulative high-water mark across every circuit already run by that process, not a clean
per-circuit isolated peak, which is an acceptable tradeoff for not needing a subprocess
wrapper at all.
"""
from __future__ import annotations

import re
import resource
import time
from pathlib import Path
from typing import Any, Callable

import json

from common import io_utils
from common.circuit_ir import ProblemIR


def _natural_sort_key(path: Path) -> list[Any]:
    """Splits a filename into text/number chunks so e.g. steps2 sorts before steps10.

    Plain alphabetical sort puts steps10 before steps2, which processes a fine step curve
    out of numeric order and makes a partial (killed or still-running) run's progress
    confusing to read. This has no effect on the final results once a run completes, only
    on the order circuits are visited in.
    """
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", path.stem)]


def _sorted_circuit_paths(circuits_dir: Path) -> list[Path]:
    return sorted(circuits_dir.glob("*.json"), key=_natural_sort_key)


def load_circuits(circuits_dir: str | Path) -> list[tuple[str, ProblemIR]]:
    """Load every saved circuit in circuits_dir, in natural filename order, as (label, ir) pairs."""
    circuits_dir = Path(circuits_dir)
    return [(path.stem, ProblemIR.load(str(path))) for path in _sorted_circuit_paths(circuits_dir)]


def load_fermionic_circuits(circuits_dir: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """Load every saved native-fermionic problem (common/problems_fermionic.py's
    FermionicProblem, just {"problem": ..., "params": ...}) in circuits_dir, as (label, dict)
    pairs. This is a different, simpler file shape than ProblemIR, used only by the native
    Majorana side of hubbard_trotter and random_fermionic_circuit, which has no qubit gate
    list or observable to save (MajoranaPropagation.jl, propaq, and monoprop each build the
    circuit and observable directly from these physical parameters)."""
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
    """Same contract as run_on_saved_circuits, for the native-fermionic problem shape.

    propagate receives the raw {"problem": ..., "params": ...} dict loaded from one saved
    file (see load_fermionic_circuits) instead of a ProblemIR, and basis is always
    "majorana" since every native-fermionic backend in this suite is Majorana-only.
    """
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
        except Exception as exc:  # noqa: BLE001 - a single bad circuit should not abort the run
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
    """Run propagate(ir) once per saved circuit in circuits_dir, and checkpoint the result.

    propagate must return a dict of result fields (expectation_value, n_terms_final, and so
    on). wall_time_s is filled in automatically around the call unless propagate already
    set it. A circuit whose label is already recorded with ok=True in the existing
    checkpoint is skipped, so a repeated run only fills in what previously failed or is new.

    Results land in experiment_dir as results_<backend>.jsonl (the checkpoint) and
    results_<backend>.npz (a snapshot of the same records, regenerated after every
    circuit, which is what every plotting script reads).
    """
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
        except Exception as exc:  # noqa: BLE001 - a single bad circuit should not abort the run
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
