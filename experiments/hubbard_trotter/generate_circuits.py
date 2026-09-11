#!/usr/bin/env python3
"""
Build and save every hubbard_trotter circuit instance this experiment compares backends on.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import problems_fermionic, problems_qubit  # noqa: E402

SIZES = [(2, 2, 1), (3, 3, 2), (4, 4, 3), (5, 5, 4)]

FINE_CURVE_NX, FINE_CURVE_NY = 3, 3
FINE_CURVE_STEPS = range(1, 26)


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    circuits_native_dir = HERE / "circuits_native"
    circuits_native_dir.mkdir(exist_ok=True)

    for nx, ny, steps in SIZES:
        label = f"{nx}x{ny}_steps{steps}"

        ir = problems_qubit.hubbard_trotter_problem(nx=nx, ny=ny, steps=steps)
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")

        fp = problems_fermionic.hubbard_trotter_fermionic(nx=nx, ny=ny, steps=steps)
        native_path = circuits_native_dir / f"{label}.json"
        fp.save(str(native_path))
        print(f"saved {native_path} (nx={nx}, ny={ny}, steps={steps})")

    for steps in FINE_CURVE_STEPS:
        label = f"{FINE_CURVE_NX}x{FINE_CURVE_NY}_steps{steps}"
        fp = problems_fermionic.hubbard_trotter_fermionic(nx=FINE_CURVE_NX, ny=FINE_CURVE_NY, steps=steps)
        native_path = circuits_native_dir / f"{label}.json"
        fp.save(str(native_path))
        print(f"saved {native_path} (nx={FINE_CURVE_NX}, ny={FINE_CURVE_NY}, steps={steps})")


if __name__ == "__main__":
    main()
