#!/usr/bin/env python3
"""Run pauli-prop on the random_circuit instance, as a flat single-thread reference point.

pauli-prop is confirmed single-threaded and has no thread-count knob, so it is run once
here rather than swept, as a flat reference line on the thread-scaling plot.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import io_utils  # noqa: E402
from common.circuit_ir import ProblemIR  # noqa: E402

MIN_ABS_COEFF = 1e-6
MAX_TERMS = 2_000_000_000


def propagate(ir: ProblemIR) -> dict:
    import pauli_prop

    qc = ir.to_qiskit()
    obs = ir.observable.to_sparse_pauli_op()
    cliff, residual = pauli_prop.evolve_through_cliffords(qc)
    evolved, one_norm = pauli_prop.propagate_through_circuit(
        obs, residual, max_terms=MAX_TERMS, atol=MIN_ABS_COEFF, frame="h",
    )
    evolved.paulis = evolved.paulis.evolve(cliff, frame="h")
    expval = float(evolved.coeffs[~evolved.paulis.x.any(axis=1)].sum())
    return {
        "expectation_value": expval,
        "n_terms_final": int(len(evolved)),
        "truncation_onenorm": float(one_norm),
        "n_threads": 1,
    }


if __name__ == "__main__":
    checkpoint = HERE / "results_pauli_prop.jsonl"
    npz_path = HERE / "results_pauli_prop.npz"
    done = {r["label"] for r in io_utils.load_jsonl(str(checkpoint)) if r.get("ok")}
    if "random_circuit" in done:
        print("  random_circuit ... skip (already done)")
    else:
        import time

        ir = ProblemIR.load(str(HERE / "circuits" / "random_circuit.json"))
        t0 = time.perf_counter()
        record = propagate(ir)
        record["ok"] = True
        record["wall_time_s"] = time.perf_counter() - t0
        record["label"] = "random_circuit"
        record["backend"] = "pauli_prop"
        record["basis"] = "pauli"
        record["problem"] = ir.problem
        record["n_qubits"] = ir.n_qubits
        print(f"  random_circuit ... OK ({record['wall_time_s']:.3f}s)")
        io_utils.append_jsonl(str(checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))
