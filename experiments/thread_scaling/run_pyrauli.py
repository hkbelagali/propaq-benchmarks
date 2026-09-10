#!/usr/bin/env python3
"""Sweep pyrauli's OpenMP thread count across THREAD_SWEEP on the random_circuit instance.

pyrauli's thread count is read from OMP_NUM_THREADS once, at its first parallel region, so
it cannot be changed inside a running process. This file re-executes itself as a fresh
subprocess once per thread count, with OMP_NUM_THREADS set before that subprocess starts,
which is the only subprocess use left anywhere in this suite and exists solely because of
this OpenMP constraint, not as a general dispatch mechanism.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import io_utils  # noqa: E402
from propaq_benchmarks.circuit_ir import ProblemIR  # noqa: E402

THREAD_SWEEP = [1, 2, 4, 8, 16, 32]
MIN_ABS_COEFF = 1e-6


def propagate(ir: ProblemIR, n_threads: int) -> dict:
    import pyrauli

    qc = ir.to_qiskit()
    obs = ir.observable.to_sparse_pauli_op()
    pcirc = pyrauli.from_qiskit(qc)
    pcirc.set_truncator(pyrauli.CoefficientTruncator(MIN_ABS_COEFF))
    pcirc.set_truncate_policy(pyrauli.AlwaysAfterSplittingPolicy())
    pobs = pyrauli.from_qiskit(obs, reverse=True)
    runtime = pyrauli.seq if n_threads <= 1 else pyrauli.par
    ev, err = pcirc.expectation_value(pobs, runtime=runtime)
    return {"expectation_value": float(ev), "truncation_onenorm": float(err)}


def run_one_thread_count(n_threads: int) -> None:
    """Called in the child subprocess, where OMP_NUM_THREADS is already set."""
    ir = ProblemIR.load(str(HERE / "circuits" / "random_circuit.json"))
    t0 = time.perf_counter()
    try:
        record = propagate(ir, n_threads)
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001
        record = {"ok": False, "error": str(exc)}
    record.setdefault("wall_time_s", time.perf_counter() - t0)
    record["label"] = "random_circuit"
    record["basis"] = "pauli"
    record["backend"] = "pyrauli"
    record["n_threads"] = n_threads
    record["problem"] = ir.problem
    record["n_qubits"] = ir.n_qubits
    print(f"  random_circuit n_threads={n_threads} ... {'OK' if record['ok'] else 'FAIL'} "
          f"({record['wall_time_s']:.3f}s)")
    checkpoint = HERE / "results_pyrauli.jsonl"
    io_utils.append_jsonl(str(checkpoint), record)
    io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(HERE / "results_pyrauli.npz"))


def main() -> None:
    checkpoint = HERE / "results_pyrauli.jsonl"
    done = {r["n_threads"] for r in io_utils.load_jsonl(str(checkpoint))
            if r.get("ok") and r.get("label") == "random_circuit"}
    for n_threads in THREAD_SWEEP:
        if n_threads in done:
            print(f"  random_circuit n_threads={n_threads} ... skip (already done)")
            continue
        env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(n_threads)],
            env=env, check=True,
        )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        run_one_thread_count(int(sys.argv[2]))
    else:
        main()
