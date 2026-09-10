#!/usr/bin/env python3
"""Build and save every qaoa_maxcut circuit instance this experiment compares backends on.

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

# (n_qubits, p): a small warm-up size growing to a 36-qubit random 3-regular graph, with the
# QAOA layer count p grown alongside the graph size so circuit depth keeps scaling too.
SIZES = [(10, 2), (16, 3), (24, 4), (36, 5)]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for n_qubits, p in SIZES:
        ir = problems_qubit.qaoa_maxcut_problem(n_qubits=n_qubits, p=p, seed=0)
        label = f"n{n_qubits}_p{p}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
