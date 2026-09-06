#!/usr/bin/env python3
"""Build and save every ising_trotter circuit instance this experiment compares backends on.

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

# (nx, ny, steps): a 3x3 warm-up size plus the 6x6=36 qubit lattice from the original request,
# grown through the Trotter step count since 6x6 was already near the practical runtime ceiling
# for the slowest backend at this lattice size.
SIZES = [(3, 3, 10), (4, 4, 12), (6, 6, 15), (6, 6, 20)]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for nx, ny, steps in SIZES:
        ir = problems_qubit.ising_trotter_problem(nx=nx, ny=ny, steps=steps)
        label = f"{nx}x{ny}_steps{steps}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
