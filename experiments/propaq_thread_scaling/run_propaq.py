#!/usr/bin/env python3
"""Thread-scaling benchmark: propaq's Pauli-propagation runtime vs number of Trotter
steps, swept across thread counts, on the same 2D transverse-field Ising Trotter circuit
propaq's own CHANGELOG benchmarks against monoprop (see
propaq_benchmarks/problems_qubit.py:ising_trotter_problem). Runtime grows with the step count (more
RZZ/RX layers to propagate through), which is the axis this experiment holds fixed per
point and sweeps to get a range of problem sizes at each thread count.

n_threads is a real constructor kwarg on propaq's propagators, so every thread count in
THREAD_SWEEP is swept in-process for each saved circuit, with one result row per
(steps, n_threads) pair.
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

THREAD_SWEEP = [1, 2, 4, 8, 16, 32, 64]
REPEATS = 5
COEFF_CUTOFF = 1e-8
BENCHMARK_ID = "thread_scaling_ising_trotter_v1"


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


def run_point(ir: ProblemIR, n_threads: int) -> dict:
    from propaq.circuits import PauliCircuit
    from propaq.datatypes import PauliTermSum
    from propaq.noise import TruncationPolicy

    circuit = PauliCircuit.from_qiskit(ir.to_qiskit())
    obs = PauliTermSum.from_sparse_pauli_op(ir.observable.to_sparse_pauli_op())
    trunc = TruncationPolicy(weight_cutoff=1_000_000, coeff_cutoff=COEFF_CUTOFF)

    t, v, n = time_run(circuit, obs, trunc, n_threads, REPEATS)
    return {
        "benchmark": BENCHMARK_ID,
        "nx": ir.params["nx"], "ny": ir.params["ny"], "J": ir.params["J"],
        "h": ir.params["h"], "dt": ir.params["dt"],
        "steps": ir.params["steps"], "n_threads": n_threads, "repeats": REPEATS,
        "coeff_cutoff": COEFF_CUTOFF,
        "wall_time_s": t, "expectation_value": v, "n_terms": n,
    }


def main() -> None:
    circuits_dir = HERE / "circuits"
    checkpoint = HERE / "results_propaq.jsonl"
    npz_path = HERE / "results_propaq.npz"

    circuits = load_circuits(circuits_dir)
    if not circuits:
        raise SystemExit(f"no saved circuits in {circuits_dir}, run generate_circuits.py first")

    done = {
        (r["steps"], r["n_threads"]) for r in io_utils.load_jsonl(str(checkpoint))
        if r.get("ok") and r.get("benchmark") == BENCHMARK_ID
    }

    print(f"Thread scaling: propaq-only sweep over steps={[ir.params['steps'] for _, ir in circuits]}, threads={THREAD_SWEEP}")
    print(f"{'steps':>6} {'threads':>8} {'wall(s)':>10} {'n_terms':>10}")
    for label, ir in circuits:
        steps = ir.params["steps"]
        for n_threads in THREAD_SWEEP:
            if (steps, n_threads) in done:
                print(f"{steps:6d} {n_threads:8d}  skip (already checkpointed)")
                continue
            try:
                record = run_point(ir, n_threads)
                record["ok"] = True
            except Exception as exc:  # noqa: BLE001 - a single bad point should not abort the sweep
                record = {"ok": False, "error": str(exc), "steps": steps, "n_threads": n_threads, "benchmark": BENCHMARK_ID}
            record["label"] = label
            record["backend"] = "propaq"
            if record["ok"]:
                print(f"{steps:6d} {n_threads:8d} {record['wall_time_s']:10.4f} {record['n_terms']:10d}", flush=True)
            else:
                print(f"{steps:6d} {n_threads:8d}  FAIL ({record.get('error')})")
            io_utils.append_jsonl(str(checkpoint), record)
            io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))


if __name__ == "__main__":
    main()
