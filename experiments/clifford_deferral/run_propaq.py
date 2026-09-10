#!/usr/bin/env python3
"""Run propaq's PauliPropagator with and without Clifford deferral on every saved
clifford_deferral circuit, recording both timings and the on/off agreement check.

Deferral is toggled via the PROPAQ_DISABLE_CLIFFORD_DEFERRAL env var, read at the start
of every propagation call in crates/pauli/src/engine.rs and crates/majorana/src/engine.rs.
It forces defer_cliffords = false, so every Clifford rotation takes the generic branching
path instead of being absorbed into the frame tableau in O(1).
Each circuit produces one result record with both timings paired together, not two
independent runs, so this uses a custom loop instead of common.experiment_runner's
run_on_saved_circuits.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import io_utils  # noqa: E402
from propaq_benchmarks.circuit_ir import ProblemIR  # noqa: E402
from propaq_benchmarks.experiment_runner import load_circuits  # noqa: E402

warnings.simplefilter("ignore")  # propaq's transpile-fallback UserWarnings go to stderr

REPEATS = 5
N_THREADS = 64
BENCHMARK_ID = "clifford_deferral_ablation_v1"


def time_run(circuit, obs, trunc, n_threads: int, repeats: int):
    from propaq.propagators import PauliPropagator

    prop = PauliPropagator(None, trunc, n_threads=n_threads)
    prop.expectation_value(obs, circuit, initial_state=0)  # warmup
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = prop.expectation_value(obs, circuit, initial_state=0)
        times.append(time.perf_counter() - t0)
    return min(times), result.expectation_value, result.n_terms[-1]


def propagate(ir: ProblemIR) -> dict:
    from propaq.circuits import PauliCircuit
    from propaq.datatypes import PauliTermSum
    from propaq.noise import TruncationPolicy

    circuit = PauliCircuit.from_qiskit(ir.to_qiskit())
    obs = PauliTermSum.from_sparse_pauli_op(ir.observable.to_sparse_pauli_op())
    trunc = TruncationPolicy(weight_cutoff=1_000_000, coeff_cutoff=ir.params["coeff_cutoff"])

    os.environ.pop("PROPAQ_DISABLE_CLIFFORD_DEFERRAL", None)
    t_on, v_on, n_on = time_run(circuit, obs, trunc, N_THREADS, REPEATS)

    os.environ["PROPAQ_DISABLE_CLIFFORD_DEFERRAL"] = "1"
    t_off, v_off, n_off = time_run(circuit, obs, trunc, N_THREADS, REPEATS)
    os.environ.pop("PROPAQ_DISABLE_CLIFFORD_DEFERRAL", None)

    # Deferral is exact. It changes only how many Clifford steps are folded into the frame
    # tableau versus branched into the term store, never the physics.
    # n_terms legitimately differs between the two, so only the expectation value is
    # asserted, not the term count.
    agreement = abs(v_on - v_off)
    if agreement > 1e-6:
        raise RuntimeError(f"p={ir.params['p']}: deferral changed the expectation value by {agreement:.3e}")

    return {
        "benchmark": BENCHMARK_ID,
        "n_qubits": ir.n_qubits,
        "total_layers": ir.params["total_layers"],
        "n_threads": N_THREADS,
        "repeats": REPEATS,
        "p": ir.params["p"],
        "t_on_ms": t_on * 1e3,
        "t_off_ms": t_off * 1e3,
        "n_on": n_on,
        "n_off": n_off,
        "speedup": t_off / t_on,
        "v_on": v_on,
        "v_off": v_off,
    }


def main() -> None:
    circuits_dir = HERE / "circuits"
    checkpoint = HERE / "results_propaq.jsonl"
    npz_path = HERE / "results_propaq.npz"
    done = {r["p"] for r in io_utils.load_jsonl(str(checkpoint)) if r.get("ok")}

    circuits = load_circuits(circuits_dir)
    if not circuits:
        raise SystemExit(f"no saved circuits in {circuits_dir}, run generate_circuits.py first")

    for label, ir in circuits:
        p = ir.params["p"]
        if p in done:
            print(f"  {label} ... skip (already done)")
            continue
        try:
            record = propagate(ir)
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001 - a single bad circuit should not abort the run
            record = {"ok": False, "error": str(exc), "p": p}
        record["label"] = label
        record["backend"] = "propaq"
        status = "OK" if record["ok"] else "FAIL"
        print(f"  {label} ... {status}")
        if not record["ok"]:
            print(f"    {record.get('error')}")
        io_utils.append_jsonl(str(checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))


if __name__ == "__main__":
    main()
