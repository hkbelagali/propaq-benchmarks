#!/usr/bin/env python3
"""Build and save every hubbard_trotter circuit instance this experiment compares backends on.

Run this once (or again after changing SIZES below) before any run_<backend> file in this
folder. Every run_<backend> file then loads whatever is saved in circuits/ or circuits_native/
and does not build circuits itself.

This experiment has two independently-constructed sides that describe the same nominal
(nx, ny, steps) instance without being bit-identical circuits. circuits/ holds the qubit-suite
Jordan-Wigner-mapped ProblemIR (common.problems_qubit.hubbard_trotter_problem), consumed by the
5 qubit-side run_<backend> files. circuits_native/ holds the much simpler native-fermionic
FermionicProblem (common.problems_fermionic.hubbard_trotter_fermionic, just a params dict with
no gate list or observable), consumed by the 3 native-side run_<backend> files that build their
own circuit directly from physical parameters.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import problems_fermionic, problems_qubit  # noqa: E402

# (nx, ny, steps): shared across both sides so each pair names the same nominal instance
# even though the two circuits are built along entirely different paths.
SIZES = [(2, 2, 1), (3, 3, 2), (4, 4, 3), (5, 5, 4)]


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


if __name__ == "__main__":
    main()
