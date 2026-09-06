#!/usr/bin/env python3
"""Save the fixed QAOA MaxCut problem parameters for the surrogate_optimization experiment.

The QAOA ansatz here is parameterized with a live Qiskit ParameterVector, which propaq's
SurrogatePauliCircuit compiles symbolically. ProblemIR can only serialize a circuit with
concrete float gate angles, not an unbound symbolic parameter, so the actual circuit is
rebuilt in run_propaq.py from these saved descriptive parameters (build_problem is fully
deterministic, so this loses nothing).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

N_QUBITS = 12
N_LAYERS = 2
# C_12 plus opposite vertices, giving every vertex degree three.
MAXCUT_EDGES = [(qubit, (qubit + 1) % N_QUBITS) for qubit in range(N_QUBITS)] + [
    (qubit, qubit + N_QUBITS // 2) for qubit in range(N_QUBITS // 2)
]
INITIAL_POINT = [0.31, -0.22, 0.17, 0.28]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    path = circuits_dir / "qaoa_maxcut.json"
    with open(path, "w") as f:
        json.dump(
            {
                "problem": "qaoa_maxcut_surrogate_optimization",
                "n_qubits": N_QUBITS,
                "n_layers": N_LAYERS,
                "maxcut_edges": MAXCUT_EDGES,
                "initial_point": INITIAL_POINT,
            },
            f,
        )
    print(f"saved {path} ({N_QUBITS} qubits, {N_LAYERS} layers, {len(MAXCUT_EDGES)} edges)")


if __name__ == "__main__":
    main()
