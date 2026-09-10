#!/usr/bin/env python3
"""Run propaq, both Pauli and Majorana basis, on every saved random_fermionic_circuit circuit.

Majorana basis reaches this circuit through Jordan-Wigner (MajoranaCircuit.from_qiskit
on the same qubit circuit), the same qubit circuit the Pauli basis above consumes, giving a
same-package, same-input Pauli-vs-Majorana comparison point in addition to the
cross-package one.
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

from propaq_benchmarks.circuit_ir import ProblemIR  # noqa: E402
from propaq_benchmarks.experiment_runner import run_on_saved_circuits  # noqa: E402

warnings.simplefilter("ignore")  # propaq's transpile-fallback UserWarnings go to stderr

MIN_ABS_COEFF = 1e-6
N_THREADS = 64


def propagate_pauli(ir: ProblemIR) -> dict:
    from propaq.circuits import PauliCircuit
    from propaq.datatypes import PauliTermSum
    from propaq.noise import TruncationPolicy
    from propaq.propagators import PauliPropagator

    circuit = PauliCircuit.from_qiskit(ir.to_qiskit())
    obs = PauliTermSum.from_sparse_pauli_op(ir.observable.to_sparse_pauli_op())
    trunc = TruncationPolicy(coeff_cutoff=MIN_ABS_COEFF)
    prop = PauliPropagator(None, trunc, n_threads=N_THREADS)
    result = prop.expectation_value(obs, circuit, initial_state=0)

    return {
        "expectation_value": float(result.expectation_value),
        "n_terms_final": int(result.n_terms[-1]) if result.n_terms else None,
        "min_abs_coeff": MIN_ABS_COEFF,
        "n_threads": N_THREADS,
    }


def propagate_majorana(ir: ProblemIR) -> dict:
    from propaq.circuits import MajoranaCircuit
    from propaq.datatypes import MajoranaTermSum
    from propaq.noise import TruncationPolicy
    from propaq.propagators import MajoranaPropagator

    circuit = MajoranaCircuit.from_qiskit(ir.to_qiskit(), n_modes=2 * ir.n_qubits)
    obs = MajoranaTermSum.from_sparse_pauli_op(ir.observable.to_sparse_pauli_op())
    trunc = TruncationPolicy(coeff_cutoff=MIN_ABS_COEFF)
    prop = MajoranaPropagator(None, trunc, n_threads=N_THREADS)
    result = prop.expectation_value(obs, circuit, initial_state=0)

    return {
        "expectation_value": float(result.expectation_value),
        "n_terms_final": int(result.n_terms[-1]) if result.n_terms else None,
        "min_abs_coeff": MIN_ABS_COEFF,
        "n_threads": N_THREADS,
    }


if __name__ == "__main__":
    run_on_saved_circuits(HERE / "circuits", HERE, "propaq", "pauli", propagate_pauli)
    run_on_saved_circuits(HERE / "circuits", HERE, "propaq", "majorana", propagate_majorana)
