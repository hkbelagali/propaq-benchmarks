#!/usr/bin/env python3
"""Build and save every random_near_clifford circuit instance this experiment compares
backends on.

Run this once (or again after changing SIZES below) before any run_<backend> file in this
folder. Every run_<backend> file then loads whatever is saved in circuits/ and does not
build circuits itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import problems_qubit  # noqa: E402

# Fixed 13-qubit, depth-18 Clifford skeleton, swept over t_density to probe how each
# backend's term count scales with non-Cliffordness.
N_QUBITS = 13
DEPTH = 18
SIZES = [0.1, 0.3, 0.6, 1.0]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for t_density in SIZES:
        ir = problems_qubit.random_near_clifford_problem(
            n_qubits=N_QUBITS, depth=DEPTH, t_density=t_density, seed=0,
        )
        label = f"t_density{t_density}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
