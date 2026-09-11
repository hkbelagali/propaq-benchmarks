#!/usr/bin/env python3
"""
Build and save 2D Ising Trotter circuits
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import problems_qubit  # noqa: E402

NX, NY = 3, 3
J = 1.0
H = 0.5
DT = 0.1
STEPS_SWEEP = [1, 2, 3, 4, 5, 6]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for steps in STEPS_SWEEP:
        ir = problems_qubit.ising_trotter_problem(nx=NX, ny=NY, J=J, h=H, dt=DT, steps=steps)
        label = f"steps{steps}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
