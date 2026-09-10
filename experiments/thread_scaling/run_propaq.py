#!/usr/bin/env python3
"""Sweep propaq's n_threads across THREAD_SWEEP on two fixed circuit instances.

n_threads is a real constructor kwarg on propaq's propagators (a dedicated per-instance
Rayon thread pool), so this file can sweep every thread count in one process, unlike the
pyrauli and Julia runners in this folder, which each need a fresh process per thread count.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import io_utils  # noqa: E402
from propaq_benchmarks.circuit_ir import ProblemIR  # noqa: E402

warnings.simplefilter("ignore")

THREAD_SWEEP = [1, 2, 4, 8, 16, 32]
MIN_ABS_COEFF = 1e-6


def propagate_pauli(ir: ProblemIR, n_threads: int) -> dict:
    from propaq.circuits import PauliCircuit
    from propaq.datatypes import PauliTermSum
    from propaq.noise import TruncationPolicy
    from propaq.propagators import PauliPropagator

    circuit = PauliCircuit.from_qiskit(ir.to_qiskit())
    obs = PauliTermSum.from_sparse_pauli_op(ir.observable.to_sparse_pauli_op())
    trunc = TruncationPolicy(coeff_cutoff=MIN_ABS_COEFF)
    prop = PauliPropagator(None, trunc, n_threads=n_threads)
    result = prop.expectation_value(obs, circuit, initial_state=0)
    return {
        "expectation_value": float(result.expectation_value),
        "n_terms_final": int(result.n_terms[-1]) if result.n_terms else None,
    }


def propagate_majorana(ir: ProblemIR, n_threads: int) -> dict:
    from propaq.circuits import MajoranaCircuit
    from propaq.datatypes import MajoranaTermSum
    from propaq.noise import TruncationPolicy
    from propaq.propagators import MajoranaPropagator

    circuit = MajoranaCircuit.from_qiskit(ir.to_qiskit(), n_modes=2 * ir.n_qubits)
    obs = MajoranaTermSum.from_sparse_pauli_op(ir.observable.to_sparse_pauli_op())
    trunc = TruncationPolicy(coeff_cutoff=MIN_ABS_COEFF)
    prop = MajoranaPropagator(None, trunc, n_threads=n_threads)
    result = prop.expectation_value(obs, circuit, initial_state=0)
    return {
        "expectation_value": float(result.expectation_value),
        "n_terms_final": int(result.n_terms[-1]) if result.n_terms else None,
    }


def run(label: str, ir: ProblemIR, basis: str, propagate) -> None:
    import time

    checkpoint = HERE / "results_propaq.jsonl"
    npz_path = HERE / "results_propaq.npz"
    done = {
        (r["label"], r["basis"], r["n_threads"])
        for r in io_utils.load_jsonl(str(checkpoint)) if r.get("ok")
    }
    for n_threads in THREAD_SWEEP:
        if (label, basis, n_threads) in done:
            print(f"  {label} basis={basis} n_threads={n_threads} ... skip (already done)")
            continue
        t0 = time.perf_counter()
        try:
            record = propagate(ir, n_threads)
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001
            record = {"ok": False, "error": str(exc)}
        record.setdefault("wall_time_s", time.perf_counter() - t0)
        record["label"] = label
        record["basis"] = basis
        record["backend"] = "propaq"
        record["n_threads"] = n_threads
        record["problem"] = ir.problem
        record["n_qubits"] = ir.n_qubits
        status = "OK" if record["ok"] else "FAIL"
        print(f"  {label} basis={basis} n_threads={n_threads} ... {status} ({record['wall_time_s']:.3f}s)")
        io_utils.append_jsonl(str(checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))


if __name__ == "__main__":
    random_circuit = ProblemIR.load(str(HERE / "circuits" / "random_circuit.json"))
    run("random_circuit", random_circuit, "pauli", propagate_pauli)
    run("random_circuit", random_circuit, "majorana", propagate_majorana)

    hubbard_qubit = ProblemIR.load(str(HERE / "circuits" / "hubbard_qubit.json"))
    run("hubbard_qubit", hubbard_qubit, "majorana", propagate_majorana)
