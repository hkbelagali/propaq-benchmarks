#!/usr/bin/env python3
"""
Build and save the circuits used for the Ising Trotter benchmarks.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import problems_qubit  # noqa: E402

WARMUP_SIZES = [(3, 3, 10), (4, 4, 12)]
FINE_CURVE_NX, FINE_CURVE_NY = 6, 6
FINE_CURVE_STEPS = range(1, 26)


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    sizes = [(nx, ny, steps) for nx, ny, steps in WARMUP_SIZES]
    sizes += [(FINE_CURVE_NX, FINE_CURVE_NY, steps) for steps in FINE_CURVE_STEPS]
    for nx, ny, steps in sizes:
        ir = problems_qubit.ising_trotter_problem(nx=nx, ny=ny, steps=steps)
        label = f"{nx}x{ny}_steps{steps}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
