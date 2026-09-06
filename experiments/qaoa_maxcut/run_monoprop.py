#!/usr/bin/env python3
"""Run monoprop (Pauli basis) on every saved qaoa_maxcut circuit.

MonoProp's Qiskit adapter accepts PauliEvolution gates and Pauli rotations, but not the
saved circuits' Clifford h/cx instructions. This decomposes exactly into {rx, ry, rz, rxx}
first, which MonoProp accepts, whenever the direct conversion is rejected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.circuit_ir import ProblemIR  # noqa: E402
from common.experiment_runner import run_on_saved_circuits  # noqa: E402

MIN_ABS_COEFF = 1e-6
N_THREADS = 64
MONOPROP_BASIS = ["rx", "ry", "rz", "rxx"]


def occupied_qubits(bitstring: int, n_qubits: int) -> list[int]:
    """Decode ProblemIR's little-endian computational-basis integer."""
    return [q for q in range(n_qubits) if bitstring & (1 << q)]


def propagate(ir: ProblemIR) -> dict:
    from qiskit import transpile
    from monoprop import PauliPropagator
    from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator

    initial_state = occupied_qubits(ir.initial_state, ir.n_qubits)
    qc = ir.to_qiskit()
    try:
        circuit = from_qiskit_circuit(qc, initial_state)
        conversion_mode = "direct"
    except ValueError as direct_error:
        try:
            qc = transpile(qc, basis_gates=MONOPROP_BASIS, optimization_level=0)
            circuit = from_qiskit_circuit(qc, initial_state)
        except ValueError:
            raise direct_error
        conversion_mode = "decomposed"
    observable = from_qiskit_operator(ir.observable.to_sparse_pauli_op())

    # A cutoff equal to register width is structurally exact. Only the coefficient
    # threshold below truncates.
    prop = PauliPropagator(observable, initial_state, cutoff=ir.n_qubits, lower_atol=MIN_ABS_COEFF)
    prop.propagate(circuit)

    return {
        "expectation_value": float(prop.expectation_value()),
        "n_terms_final": int(prop.size()),
        "conversion_mode": conversion_mode,
        "min_abs_coeff": MIN_ABS_COEFF,
        "n_threads": N_THREADS,
    }


if __name__ == "__main__":
    # MonoProp reads this when it constructs the C++ propagator. "auto" partitioning then
    # uses one partition per physical core, capped here to the suite-wide thread request.
    os.environ["monoprop_NUM_THREADS"] = str(N_THREADS)
    run_on_saved_circuits(HERE / "circuits", HERE, "monoprop", "pauli", propagate)
