#!/usr/bin/env python3
"""Build and save every heisenberg_chain_trotter circuit instance this experiment compares
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

# (n_qubits, steps): a small warm-up size growing to a 30-qubit chain, with the Trotter step
# count grown alongside the chain length so the circuit depth keeps scaling too.
SIZES = [(12, 2), (20, 4), (26, 6), (30, 8)]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for n_qubits, steps in SIZES:
        ir = problems_qubit.heisenberg_chain_trotter_problem(n_qubits=n_qubits, steps=steps)
        label = f"n{n_qubits}_steps{steps}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
