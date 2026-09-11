#!/usr/bin/env python3
"""
Build the circuits at each T gate density for the Clifford deferral benchmarks
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from propaq_benchmarks.circuit_ir import qiskit_to_ir  # noqa: E402

N_QUBITS = 64
TOTAL_LAYERS = 80
PROPORTIONS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
SEED = 123
COEFF_CUTOFF = 1e-9


def build_circuit(n_qubits: int, total_layers: int, p: float, seed: int) -> QuantumCircuit:
    """Build a randomized Clifford-round circuit with a T gate inserted per layer with probability p."""
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(total_layers):
        for q in range(n_qubits):
            getattr(qc, rng.choice(["h", "s", "x", "sdg"]))(q)
        qubits = list(range(n_qubits))
        rng.shuffle(qubits)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(qubits[i], qubits[i + 1])
        if rng.random() < p:
            qc.t(rng.randrange(n_qubits))
    return qc


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    observable = SparsePauliOp("Z" + "I" * (N_QUBITS - 1))
    for p in PROPORTIONS:
        qc = build_circuit(N_QUBITS, TOTAL_LAYERS, p, SEED)
        ir = qiskit_to_ir(
            qc, observable, "clifford_deferral",
            {
                "n_qubits": N_QUBITS,
                "total_layers": TOTAL_LAYERS,
                "p": p,
                "seed": SEED,
                "coeff_cutoff": COEFF_CUTOFF,
            },
            canonicalize_circuit=False,
        )
        label = f"p{p}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
