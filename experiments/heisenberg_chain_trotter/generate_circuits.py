#!/usr/bin/env python3
"""
Build and save the Heisenberg chain Trotter circuits for benchmarks
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import problems_qubit  # noqa: E402

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
