#!/usr/bin/env python3
"""Save the layer-count sweep for the custom_decomposition experiment.

The 10-qubit brickwork circuit contains a custom UnitaryGate (the fixed non-Clifford
exchange gate) that ProblemIR's gate list cannot serialize, since ProblemIR only stores a
gate name plus a single angle, not an arbitrary matrix. The circuit is otherwise
deterministic and cheap to rebuild from just its layer count, so each saved file here holds
only that sweep parameter, and run_propaq.py rebuilds the actual Qiskit circuit from it.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

N_QUBITS = 10
DEFAULT_LAYERS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for n_layers in DEFAULT_LAYERS:
        path = circuits_dir / f"layers{n_layers}.json"
        with open(path, "w") as f:
            json.dump({"problem": "custom_decomposition", "n_qubits": N_QUBITS, "layers": n_layers}, f)
        print(f"saved {path} (layers={n_layers})")


if __name__ == "__main__":
    main()
