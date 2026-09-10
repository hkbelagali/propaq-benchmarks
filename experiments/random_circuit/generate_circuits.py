#!/usr/bin/env python3
"""Build and save every random_circuit circuit instance this experiment compares backends on.

Run this once (or again after changing SIZES below) before any run_<backend> file in this
folder. Every run_<backend> file then loads whatever is saved in circuits/ and does not
build circuits itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import problems_qubit  # noqa: E402

# (n_qubits, depth): a small warm-up size plus growing qubit count at fixed depth 20, then a
# fixed 12-qubit size grown through depth, to separate width scaling from depth scaling.
SIZES = [(10, 20), (12, 20), (12, 28), (12, 36)]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for n_qubits, depth in SIZES:
        ir = problems_qubit.random_circuit_problem(n_qubits=n_qubits, depth=depth, seed=0)
        label = f"n{n_qubits}_depth{depth}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
