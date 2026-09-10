#!/usr/bin/env python3
"""Run pyrauli (Pauli basis) on every saved heisenberg_chain_trotter circuit.

pyrauli.from_qiskit needs reverse=True on the observable. pyrauli's own Observable string
convention is big-endian (leftmost char = qubit 0), the opposite of Qiskit's little-endian
convention, and from_qiskit does not correct for this by default. Without it, expectation
values come out silently wrong (confirmed, exactly 0 for entangled circuits, sign-flipped
for product states).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks.circuit_ir import ProblemIR  # noqa: E402
from propaq_benchmarks.experiment_runner import run_on_saved_circuits  # noqa: E402

MIN_ABS_COEFF = 1e-6
N_THREADS = 64


def propagate(ir: ProblemIR) -> dict:
    import pyrauli

    qc = ir.to_qiskit()
    obs = ir.observable.to_sparse_pauli_op()

    pcirc = pyrauli.from_qiskit(qc)
    pcirc.set_truncator(pyrauli.CoefficientTruncator(MIN_ABS_COEFF))
    pcirc.set_truncate_policy(pyrauli.AlwaysAfterSplittingPolicy())
    pobs = pyrauli.from_qiskit(obs, reverse=True)

    runtime = pyrauli.par if N_THREADS > 1 else pyrauli.seq
    ev, err = pcirc.expectation_value(pobs, runtime=runtime)

    return {
        "expectation_value": float(ev),
        "truncation_onenorm": float(err),
        "min_abs_coeff": MIN_ABS_COEFF,
        "n_threads": N_THREADS,
    }


if __name__ == "__main__":
    import os

    # pyrauli's OpenMP thread count is process-wide via OMP_NUM_THREADS, read once at its
    # first parallel region, so it must be set before pyrauli's C++ extension is imported.
    os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
    run_on_saved_circuits(HERE / "circuits", HERE, "pyrauli", "pauli", propagate)
