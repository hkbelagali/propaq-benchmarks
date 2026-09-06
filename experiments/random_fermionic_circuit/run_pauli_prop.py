#!/usr/bin/env python3
"""Run pauli-prop (Pauli basis, single-threaded) on every saved random_fermionic_circuit circuit."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.circuit_ir import ProblemIR  # noqa: E402
from common.experiment_runner import run_on_saved_circuits  # noqa: E402

MIN_ABS_COEFF = 1e-6
MAX_TERMS = 2_000_000_000  # a required non-None placeholder, see pauli-prop's known issue below


def propagate(ir: ProblemIR) -> dict:
    import pauli_prop

    qc = ir.to_qiskit()
    obs = ir.observable.to_sparse_pauli_op()

    t0 = time.perf_counter()
    cliff, residual = pauli_prop.evolve_through_cliffords(qc)
    # pauli-prop's propagate_through_circuit docstring says max_terms=None disables the
    # term-count cap, but it crashes with TypeError on None instead. MAX_TERMS is set high
    # enough to never actually bind, matching this backend's truncation to every other
    # backend's coefficient-cutoff-only truncation.
    evolved, one_norm = pauli_prop.propagate_through_circuit(
        obs, residual, max_terms=MAX_TERMS, atol=MIN_ABS_COEFF, frame="h",
    )
    evolved.paulis = evolved.paulis.evolve(cliff, frame="h")
    expval = float(evolved.coeffs[~evolved.paulis.x.any(axis=1)].sum())
    wall_time_s = time.perf_counter() - t0

    return {
        "wall_time_s": wall_time_s,
        "expectation_value": expval,
        "n_terms_final": int(len(evolved)),
        "truncation_onenorm": float(one_norm),
        "max_terms": MAX_TERMS,
        "min_abs_coeff": MIN_ABS_COEFF,
        "n_threads": 1,
    }


if __name__ == "__main__":
    run_on_saved_circuits(HERE / "circuits", HERE, "pauli_prop", "pauli", propagate)
