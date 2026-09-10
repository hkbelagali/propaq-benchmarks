#!/usr/bin/env python3
"""Build and save the two fixed TFIM circuit blocks the hybrid_mps_heisenberg experiment
compares against, plus the shared local observable.

mps_preparation is a six-layer nearest-neighbour TFIM block well suited to an MPS
Schrodinger calculation, and heisenberg_readout is a three-layer local TFIM block meant to
be propagated backwards in the Heisenberg picture. Both are saved once as ProblemIR here
since they never change across the weight-cutoff sweep run_propaq.py performs.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from propaq_benchmarks.circuit_ir import qiskit_to_ir  # noqa: E402

N_QUBITS = 12
MPS_LAYERS = 10
HEISENBERG_LAYERS = 4


def tfim_block(n_layers: int, coupling_base: float) -> QuantumCircuit:
    """Build a shallow 1D TFIM block with nearest-neighbour entanglement only."""
    circuit = QuantumCircuit(N_QUBITS)
    for layer in range(n_layers):
        for qubit in range(N_QUBITS - 1):
            circuit.rzz(coupling_base + 0.03 * ((qubit + layer) % 7), qubit, qubit + 1)
        for qubit in range(N_QUBITS):
            circuit.rx(0.22 + 0.02 * ((qubit + layer) % 5), qubit)
    return circuit


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    observable = SparsePauliOp("I" * (N_QUBITS // 2) + "Z" + "I" * (N_QUBITS // 2 - 1))

    mps_preparation = tfim_block(MPS_LAYERS, 0.41)
    ir = qiskit_to_ir(
        mps_preparation, observable, "hybrid_mps_heisenberg",
        {"n_qubits": N_QUBITS, "mps_layers": MPS_LAYERS, "heisenberg_layers": HEISENBERG_LAYERS, "role": "mps_preparation"},
    )
    path = circuits_dir / "mps_preparation.json"
    ir.save(str(path))
    print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")

    heisenberg_readout = tfim_block(HEISENBERG_LAYERS, 0.55)
    ir = qiskit_to_ir(
        heisenberg_readout, observable, "hybrid_mps_heisenberg",
        {"n_qubits": N_QUBITS, "mps_layers": MPS_LAYERS, "heisenberg_layers": HEISENBERG_LAYERS, "role": "heisenberg_readout"},
    )
    path = circuits_dir / "heisenberg_readout.json"
    ir.save(str(path))
    print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
